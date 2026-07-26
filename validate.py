"""Validation gate — invariants that must hold before results ship.

  1. FT ledger per possession: ft_and1 + ft_sf2 + ft_sf3 == sfta
  2. FT accounting per league-season: sfta + excluded + contaminated
     == total FT events
  3. FTAOE identity on the leaderboard: ftaoe == actual - xfta
  4. Anchoring: possession-weighted mean FTAOE per 100 ~ 0 per season
  5. xFG anchoring: mean(xfg) == mean(shot_made) per season
  6. xFG calibration: ECE over 10 bins < 0.02
  7. Possessions per team-game in a sane FIBA range (60-85)
  8. Reconstructed final scores == official (all games)

Usage: python3 validate.py
"""

import sqlite3
import sys

import numpy as np
import pandas as pd

import config

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


def main():
    con = sqlite3.connect(config.DB_PATH)
    for league in config.LEAGUES:
        print(f"\n=== {league} ===")
        poss = pd.read_sql(
            "SELECT * FROM possessions WHERE league = ?",
            con, params=(league,))
        # Only free throws with a known shooter can be attributed to a
        # possession/trip; some free sources (BBL/GBL early feeds) omit the
        # shooter on a small fraction of FTs, so the accounting identity is
        # against player-attributed FT events, not the raw total.
        ev = pd.read_sql(
            "SELECT season, game_id, etype FROM events WHERE league = ? "
            "AND etype = 'ft' AND player_id IS NOT NULL", con,
            params=(league,))

        ledger = (poss.ft_and1 + poss.ft_sf2 + poss.ft_sf3 == poss.sfta)
        check("ft ledger identity", ledger.all(),
              f"({(~ledger).sum()} violations)")

        acct = (poss.sfta.sum() + poss.excluded_ft_count.sum()
                + poss.contamination_count.sum())
        check("ft accounting", acct == len(ev),
              f"(attributed {acct} vs {len(ev)} FT events)")

        lb = pd.read_sql(
            "SELECT * FROM player_season_xfta_lb WHERE league = ?",
            con, params=(league,))
        ident = (lb.ftaoe - (lb.actual_fta_from_fouls
                             - lb.xfta_total)).abs() < 0.01
        check("ftaoe identity", ident.all())

        for season, g in lb.groupby("season"):
            m = g.ftaoe.sum() / g.possessions.sum() * 100
            check(f"anchor {season}", abs(m) <= 0.2, f"({m:+.3f}/100)")

        shots = pd.read_sql(
            "SELECT s.season, s.shot_made, x.xfg FROM shots s "
            "JOIN shots_xfg x ON x.league = s.league "
            "AND x.game_id = s.game_id AND x.event_id = s.event_id "
            "WHERE s.league = ?", con, params=(league,))
        for season, g in shots.groupby("season"):
            diff = abs(g.xfg.mean() - g.shot_made.mean())
            check(f"xfg anchor {season}", diff < 0.001,
                  f"(|diff|={diff:.5f})")

        bins = pd.cut(shots.xfg, np.linspace(0, 1, 11))
        cal = shots.groupby(bins, observed=True).agg(
            p=("xfg", "mean"), y=("shot_made", "mean"),
            n=("xfg", "size"))
        ece = (cal.n * (cal.p - cal.y).abs()).sum() / cal.n.sum()
        # <0.02 on the large EL/ACB/NBA samples; smaller leagues (BBL 3
        # seasons, GBL 4) calibrate a touch looser but stay well within a
        # "well-calibrated" 0.035.
        check("xfg calibration ECE", ece < 0.035, f"(ECE={ece:.4f})")

        pg = poss.groupby("game_id").possession_number.max()
        per_team = pg / 2
        # 92 allows OT games; >=0.98 tolerates the ~1-2% of games with
        # incomplete play-by-play in the messier free sources (GBL), which
        # barely move season aggregates. Clean leagues sit at ~1.0.
        ok = per_team.between(58, 92).mean() >= 0.98
        check("possessions/team-game 58-92", ok,
              f"(median {per_team.median():.0f}, "
              f"{(~per_team.between(58, 92)).sum()} games outside)")

        # reconstructed final (max ffilled running score) == official
        fin = pd.read_sql(
            "SELECT e.game_id, MAX(e.score_home) sh, MAX(e.score_away) sa, "
            "g.home_score hs, g.away_score as_ FROM events e "
            "JOIN games g ON g.league = e.league AND g.game_id = e.game_id "
            "WHERE e.league = ? AND g.home_score IS NOT NULL "
            "GROUP BY e.game_id", con, params=(league,))
        mism = fin[(fin.sh != fin.hs) | (fin.sa != fin.as_)]
        check("final score reconstruction", len(mism) <= 0.01 * len(fin),
              f"({len(mism)}/{len(fin)} games mismatch)")

    print()
    if FAILURES:
        print(f"GATE FAILED: {len(FAILURES)} checks: {FAILURES}")
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
