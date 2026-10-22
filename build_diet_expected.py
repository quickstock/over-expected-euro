"""Expected free throws from shot diet: the FTAOE baseline since 2026-10-22.

Same construction as ../NBA_OE/build_diet_expected.py, per league. FTAOE =
actual drawn free throws minus this expectation (train_xfta.py builds the
board from it). The context model in train_xfta.py still feeds the team and
referee views, but it explained ~1% of the spread between players, so FTAOE
on that baseline was r 0.96-0.98 with the raw free-throw rate in every league.

Per league-season, zones rim / paint / mid / three:
  attempts  = FGA in the zone, plus fouled misses, which are not FGAs:
              3-shot trips go to three; 2-shot trips are spread over the
              player's own 2-pt zones in proportion to his FGA there times
              the league's and-1-per-FGA propensity for that zone
  rate      = league FTs per attempt in the zone, leaving the player out
  expected  = sum over zones of attempts x rate, anchored per season

Known limit, stated on the methodology page: FIBA play-by-play does not
separate a bonus free-throw pair from a shooting foul on a miss, so 2-shot
trips include bonus FTs and are treated as fouled 2-pt attempts. Splitting
them out by bonus state was measured and rejected: the bonus half then goes
unadjusted and FTAOE falls back to r ~0.91 with the raw rate. The cost of
keeping them in is a mild tilt against tall finishers (height r about -0.2
in EuroLeague and EuroCup, the two leagues with heights).

Writes diet_game(league, game_id, player_id, season, xft) and
diet_rates(league, season, zone, attempts, fts, rate, propensity, anchor).
"""
import argparse
import sqlite3

import numpy as np
import pandas as pd

import config

ZONE = {"Restricted Area": "rim", "In The Paint (Non-RA)": "paint",
        "Mid-Range": "mid", "Above the Break 3": "three",
        "Left Corner 3": "three", "Right Corner 3": "three"}
TWO = ["rim", "paint", "mid"]
ZONES = TWO + ["three"]


def load(con, league):
    shots = pd.read_sql(
        """SELECT game_id, event_id, player_id, season, zone_basic AS zone
           FROM shots WHERE league = ?""", con, params=(league,))
    shots["z"] = shots.zone.map(ZONE)
    shots = shots[shots.z.notna() & shots.player_id.notna()].copy()
    a1 = pd.read_sql("SELECT game_id, event_id FROM and1_shots WHERE league = ?",
                     con, params=(league,))
    shots["and1"] = shots.set_index(["game_id", "event_id"]).index.isin(
        a1.set_index(["game_id", "event_id"]).index).astype(int)
    poss = pd.read_sql(
        """SELECT game_id, finisher_player_id AS player_id, season,
                  ft_sf2, ft_sf3
           FROM possessions
           WHERE league = ? AND finisher_player_id IS NOT NULL""",
        con, params=(league,))
    return shots, poss


def per_game(shots, poss):
    """(game, player, season) rows: FGA and and-1s by zone, 2- and 3-shot trips."""
    key = ["game_id", "player_id", "season"]
    fga = shots.pivot_table(index=key, columns="z", values="and1",
                            aggfunc="size", fill_value=0)
    an1 = shots.pivot_table(index=key, columns="z", values="and1",
                            aggfunc="sum", fill_value=0)
    trips = poss.groupby(key).agg(trips2=("ft_sf2", "sum"),
                                  trips3=("ft_sf3", "sum"))
    trips["trips2"] /= 2
    trips["trips3"] /= 3
    g = (fga.reindex(columns=ZONES, fill_value=0).add_prefix("fga_")
         .join(an1.reindex(columns=ZONES, fill_value=0).add_prefix("a1_"),
               how="outer")
         .join(trips, how="outer").fillna(0.0))
    return g.reset_index()


def season_model(g):
    """Per player-season allocation weights and leave-player-out zone rates
    for one season's per-game rows. Returns (player table, league rates)."""
    p = g.groupby("player_id").sum(numeric_only=True)
    prop = {z: p["a1_" + z].sum() / max(p["fga_" + z].sum(), 1) for z in TWO}
    w = np.stack([p["fga_" + z].to_numpy() * prop[z] for z in TWO], axis=1)
    tot = w.sum(axis=1, keepdims=True)
    league_w = np.array([prop[z] for z in TWO]) / sum(prop.values())
    w = np.where(tot > 0, w / np.where(tot > 0, tot, 1), league_w)
    for i, z in enumerate(TWO):
        p["w_" + z] = w[:, i]
        p["A_" + z] = p["fga_" + z] + p.trips2 * w[:, i]
        p["F_" + z] = p["a1_" + z] + 2 * p.trips2 * w[:, i]
    p["A_three"] = p.fga_three + p.trips3
    p["F_three"] = p.a1_three + 3 * p.trips3
    rates = {}
    for z in ZONES:
        A, F = p["A_" + z].sum(), p["F_" + z].sum()
        rates[z] = {"attempts": A, "fts": F, "rate": F / A,
                    "propensity": prop.get(z)}
        p["r_" + z] = (F - p["F_" + z]) / np.maximum(A - p["A_" + z], 1)
    expected = sum(p["A_" + z] * p["r_" + z] for z in ZONES)
    actual = sum(p["F_" + z] for z in ZONES)
    anchor = actual.sum() / expected.sum()
    return p, rates, anchor


def build(con, league):
    shots, poss = load(con, league)
    g = per_game(shots, poss)
    out, rate_rows = [], []
    for season, gs in g.groupby("season"):
        p, rates, anchor = season_model(gs)
        m = gs.join(p[[c for c in p.columns if c[:2] in ("w_", "r_")]],
                    on="player_id")
        xft = m.fga_three.add(m.trips3) * m.r_three
        for z in TWO:
            xft += (m["fga_" + z] + m.trips2 * m["w_" + z]) * m["r_" + z]
        out.append(pd.DataFrame({"league": league, "game_id": m.game_id,
                                 "player_id": m.player_id, "season": season,
                                 "xft": xft * anchor}))
        for z, r in rates.items():
            rate_rows.append({"league": league, "season": season, "zone": z,
                              **r, "anchor": anchor})
        print(f"  [{league} {season}] anchor x{anchor:.4f}  " + "  ".join(
            f"{z} {r['rate']:.3f}" for z, r in rates.items()), flush=True)
    return pd.concat(out, ignore_index=True), pd.DataFrame(rate_rows)


def _replace(con, name, league, df):
    if con.execute("SELECT 1 FROM sqlite_master WHERE name=?",
                   (name,)).fetchone():
        con.execute(f"DELETE FROM {name} WHERE league = ?", (league,))
    df.to_sql(name, con, if_exists="append", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=config.LEAGUES, default=None)
    args = ap.parse_args()
    leagues = [args.league] if args.league else config.LEAGUES
    con = sqlite3.connect(config.DB_PATH)
    for league in leagues:
        diet, rates = build(con, league)
        _replace(con, "diet_game", league, diet)
        _replace(con, "diet_rates", league, rates)
        print(f"[{league}] diet_game: {len(diet):,} player-games")
    con.commit()
    con.close()


if __name__ == "__main__":
    main()
