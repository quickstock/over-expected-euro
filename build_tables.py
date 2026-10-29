"""cache/ -> euro_oe.db base tables: games, events, shots,
player_season, player_season_ft.

Usage: python3 build_tables.py [--league EL|ACB]
"""

import argparse
import math
import sqlite3

import pandas as pd

import config
from leagues import ADAPTERS, acb, eurocup, euroleague  # noqa: F401 (used below)

# Leagues whose adapter also exposes player_seasons() -> bio rows (height,
# position) from their own bios endpoint, same shape as EuroLeague's.
_BIO_SOURCE = {"EL": euroleague, "EUC": eurocup}


# ------------------------------------------------------------- geometry

def zone_basic(x, y, shot_value):
    """FIBA shot zones from meters-from-hoop coords, named like the NBA
    zones so downstream code reads the same."""
    d = math.hypot(x, y)
    if shot_value == 3:
        if d > 12.0:
            return "Backcourt"
        if y <= config.FIBA_CORNER_Y_MAX:
            return "Left Corner 3" if x < 0 else "Right Corner 3"
        return "Above the Break 3"
    if d <= config.FIBA_RA_RADIUS:
        return "Restricted Area"
    if (abs(x) <= config.FIBA_PAINT_HALF_WIDTH
            and -config.FIBA_HOOP_TO_BASELINE <= y
            <= config.FIBA_PAINT_DEPTH_FROM_HOOP):
        return "In The Paint (Non-RA)"
    return "Mid-Range"


def zone_area(x, y):
    # atan2 handles y <= 0 natively; behind-the-hoop shots get > 90 deg
    a = math.degrees(math.atan2(abs(x), y))
    if a <= 18:
        return "Center"
    side = "Left" if x < 0 else "Right"
    if a <= 60:
        return f"{side} Side Center"
    return f"{side} Side"


# ---------------------------------------------------------------- build

def build_league(league, con):
    adapter = ADAPTERS[league]
    all_games, all_events, all_shots = [], [], []
    skipped = []

    for season in config.SEASONS:
        games = adapter.list_games(season)
        for g in games:
            try:
                events, shots = adapter.normalize_game(season, g)
            except (FileNotFoundError, ValueError):  # missing / mid-write
                skipped.append(g["game_id"])
                continue
            if league == "ACB":
                g["ref1"], g["ref2"], g["ref3"] = acb.referees(g)

            # running score: ffill over event order, 0 at game start.
            # EL stamps only the scoring team's total, so ffill each
            # side independently. A score can never legitimately DECREASE
            # within a game; a regression means the source feed served a
            # duplicated/out-of-order event block (confirmed on some GBL
            # games — a stale or re-submitted run of events sharing
            # actionNumbers with the real sequence). Treat that game as
            # corrupted and skip it, same as a missing/mid-write game.
            sh = sa = 0
            corrupted = False
            for e in events:
                if e["score_home"] is not None:
                    if e["score_home"] < sh:
                        corrupted = True
                        break
                    sh = e["score_home"]
                if e["score_away"] is not None:
                    if e["score_away"] < sa:
                        corrupted = True
                        break
                    sa = e["score_away"]
                e["score_home"], e["score_away"] = sh, sa
                e["league"], e["season"] = league, season
            if corrupted:
                skipped.append(g["game_id"])
                continue

            # sanity: reconstructed final == official (both sides known)
            if (g["home_score"] is not None
                    and (sh, sa) != (g["home_score"], g["away_score"])):
                print(f"  score mismatch {g['game_id']}: "
                      f"events {sh}-{sa} vs official "
                      f"{g['home_score']}-{g['away_score']}")

            margin = {e["event_id"]: e["score_home"] - e["score_away"]
                      for e in events}
            for s in shots:
                s["league"], s["season"] = league, season
                if s["y_m"] is None:
                    # Depth unknown but at or below the hoop line (ACB
                    # clamps it there). Distance and angle stay unknown.
                    # Zones are cut at the bound y = 0: below the line
                    # every zone is fixed by x alone except restricted
                    # area vs paint within 1.25 m, and in the leagues that
                    # report below-line depth 97.7% of those 2PA are in
                    # the restricted area, which is what y = 0 gives.
                    s["distance_m"] = s["angle_deg"] = None
                    y_zone = 0.0
                else:
                    s["distance_m"] = math.hypot(s["x_m"], s["y_m"])
                    s["angle_deg"] = math.degrees(
                        math.atan2(abs(s["x_m"]), s["y_m"]))
                    y_zone = s["y_m"]
                s["zone_basic"] = zone_basic(s["x_m"], y_zone,
                                             s["shot_value"])
                s["zone_area"] = zone_area(s["x_m"], y_zone)
                s["shot_type"] = f"{s['shot_value']}pt"
                m = margin.get(s["event_id"], 0)
                is_home = s["team_id"] == g["home_team_id"]
                s["score_margin"] = m if is_home else -m
                s["home_or_away"] = "home" if is_home else "away"
            all_games.append(g)
            all_events.extend(events)
            all_shots.extend(shots)
        print(f"[{league} {season}] {len(games)} games normalized",
              flush=True)

    if skipped:
        print(f"[{league}] {len(skipped)} games missing from cache "
              f"(pull again): {skipped[:5]}...")

    gdf = pd.DataFrame(all_games).drop(columns=["game_code"])
    edf = pd.DataFrame(all_events)
    sdf = pd.DataFrame(all_shots)

    # -------- player_season: names from PBP, bios from EL people
    def _mode(s):
        m = s.dropna().mode()
        return m.iat[0] if len(m) else None

    pev = edf[edf.player_id.notna()]
    ps = (pev.groupby(["season", "player_id"])
             .agg(player_name=("player_name", _mode),
                  team_id=("team_id", _mode))
             .reset_index())
    ps["league"] = league
    ps["height_cm"] = None
    ps["position"] = None
    if league in _BIO_SOURCE:
        bio_module = _BIO_SOURCE[league]
        bios = pd.DataFrame([r for season in config.SEASONS
                             for r in bio_module.player_seasons(season)])
        bios = bios.drop_duplicates(["season", "player_id"])
        ps = ps.merge(
            bios[["season", "player_id", "height_cm", "position"]],
            on=["season", "player_id"], how="left", suffixes=("_x", ""))
        ps = ps.drop(columns=["height_cm_x", "position_x"])
        cov = ps.height_cm.notna().mean()
        print(f"[{league}] bio join coverage: {cov:.1%}")

    # -------- player_season_ft from FT events
    ft = edf[(edf.etype == "ft") & edf.player_id.notna()]
    psf = (ft.groupby(["season", "player_id"])
             .agg(ftm=("shot_made", "sum"), fta=("shot_made", "size"))
             .reset_index())
    psf["ft_pct"] = psf.ftm / psf.fta
    psf["league"] = league

    for name, df in (("games", gdf), ("events", edf), ("shots", sdf),
                     ("player_season", ps), ("player_season_ft", psf)):
        con.execute(f"DELETE FROM {name} WHERE league = ?", (league,)) \
            if _table_exists(con, name) else None
        df.to_sql(name, con, if_exists="append", index=False)
        print(f"  {name}: +{len(df)} rows")


def _table_exists(con, name):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone() is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=list(ADAPTERS), default=None)
    args = ap.parse_args()
    leagues = [args.league] if args.league else config.LEAGUES

    con = sqlite3.connect(config.DB_PATH)
    for league in leagues:
        build_league(league, con)
    con.execute("CREATE INDEX IF NOT EXISTS ix_events_game "
                "ON events(league, game_id, event_id)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_shots_game "
                "ON shots(league, game_id, event_id)")
    con.commit()
    con.close()
    print("build_tables done")


if __name__ == "__main__":
    main()
