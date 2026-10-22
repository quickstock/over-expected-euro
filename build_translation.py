"""What a domestic-league player would shoot in EuroLeague.

Shot-making over expected (FG% - xFG%, percentage points) is measured against
each league's own xFG model, so a +3 in the BBL and a +3 in EuroLeague are not
the same thing. EuroLeague clubs also play their domestic leagues, so the same
player, in the same season, has a number in both. Those pairs measure the step.

Projection, for every domestic player-season:

    EL shot-making = trust x domestic shot-making + step[league]
    trust          = FGA / (FGA + K)

`trust` discounts a domestic season by its own sample, and what is trusted
carries one-for-one. Measured on 2020-26 before shipping (leave-one-season-out
over 576 same-season pairs): RMSE 4.43 pp, against 4.79 for "a league-average
EuroLeague player" and 6.14 for "he shoots the same"; on 90 players who moved
from a domestic league to EuroLeague the next season, 4.30 against 4.65 flat.
K = 300 was the best of a grid on that test, and fitting a free slope came out
at 1.09, so the slope is fixed at 1. The step is refit each run from all pairs.

The run fails if the projection stops beating the flat baseline out of sample,
so the site never shows a translation that has stopped predicting.

Players are matched across competitions by normalised name within a season
(the feeds use different ids); a name shared by two players in one
league-season is dropped rather than guessed.

Writes translation(league, season, player_id, el_proj, el_lo, el_hi,
el_actual, el_fga) and translation_meta(league, meta JSON).
"""
import json
import re
import sqlite3
import unicodedata

import numpy as np
import pandas as pd

import config

DOMESTIC = ["ACB", "LBA", "BBL", "ABA", "GBL"]
MIN_FGA = 50       # both sides of a same-season pair, and movers' EL season
MOVER_FGA = 100    # movers' domestic season
K = 300


def _key(name):
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore")
    return " ".join(sorted(re.sub(r"[^a-z ]", " ", n.decode().lower()).split()))


def _offsets(pairs):
    x = pairs.make_d * pairs.fga_d / (pairs.fga_d + K)
    return {lg: float(np.average(d.make_e - x[d.index], weights=d.w))
            for lg, d in pairs.groupby("league_d")}


def _rmse(err, w):
    return round(float(np.sqrt(np.average(err ** 2, weights=w))), 3)


def main():
    con = sqlite3.connect(config.DB_PATH)
    sv = pd.read_sql(
        "SELECT league, season, player_id, player_name, fga, shot_making_oe "
        "AS make FROM shot_value WHERE league IN (%s)"
        % ",".join("?" * (len(DOMESTIC) + 1)), con, params=DOMESTIC + ["EL"])
    sv["key"] = sv.player_name.map(_key)
    ambiguous = sv.groupby(["league", "season", "key"]).player_id \
        .transform("nunique") > 1
    sv = sv[~ambiguous]
    el = sv[sv.league == "EL"]
    dom = sv[sv.league != "EL"]

    pairs = dom.merge(el, on=["season", "key"], suffixes=("_d", "_e"))
    pairs = pairs[(pairs.fga_d >= MIN_FGA) & (pairs.fga_e >= MIN_FGA)].copy()
    pairs["w"] = 1 / (1 / pairs.fga_d + 1 / pairs.fga_e)

    # leave-one-season-out: the projection against the two simple answers
    err, err_flat, err_naive, w = [], [], [], []
    for season in sorted(pairs.season.unique()):
        tr, te = pairs[pairs.season != season], pairs[pairs.season == season]
        step = _offsets(tr)
        flat = {lg: np.average(d.make_e, weights=d.w)
                for lg, d in tr.groupby("league_d")}
        trust = te.fga_d / (te.fga_d + K)
        err.append(te.make_e - (te.league_d.map(step) + trust * te.make_d))
        err_flat.append(te.make_e - te.league_d.map(flat))
        err_naive.append(te.make_e - te.make_d)
        w.append(te.w)
    err, w = pd.concat(err), pd.concat(w)
    rmse = _rmse(err, w)
    rmse_flat = _rmse(pd.concat(err_flat), w)
    rmse_naive = _rmse(pd.concat(err_naive), w)
    if not rmse < rmse_flat:
        raise SystemExit(f"translation RMSE {rmse} no longer beats the flat "
                         f"baseline {rmse_flat}; not shipping it")
    lo_q, hi_q = (float(q) for q in np.quantile(err, [0.1, 0.9]))

    step = _offsets(pairs)

    # movers: domestic season t -> EuroLeague t+1, not in EuroLeague at t
    seasons = sorted(sv.season.unique())
    nxt = dict(zip(seasons, seasons[1:]))
    in_el = set(zip(el.season, el.key))
    d0 = dom[(dom.fga >= MOVER_FGA) & dom.season.isin(nxt)].copy()
    d0["season_n"] = d0.season.map(nxt)
    d0 = d0[[(s, k) not in in_el for s, k in zip(d0.season, d0.key)]]
    mv = d0.merge(el[el.fga >= MIN_FGA], left_on=["season_n", "key"],
                  right_on=["season", "key"], suffixes=("_d", "_e"))
    mv_w = 1 / (1 / mv.fga_d + 1 / mv.fga_e)
    mv_pred = mv.league_d.map(step) + mv.make_d * mv.fga_d / (mv.fga_d + K)
    flat_all = {lg: np.average(d.make_e, weights=d.w)
                for lg, d in pairs.groupby("league_d")}

    # projections for every domestic player-season
    proj = dom.copy()
    proj["el_proj"] = proj.league.map(step) + proj.make * proj.fga / (proj.fga + K)
    proj["el_lo"] = proj.el_proj + lo_q
    proj["el_hi"] = proj.el_proj + hi_q
    actual = pairs[["league_d", "season", "player_id_d", "make_e", "fga_e"]] \
        .rename(columns={"league_d": "league", "player_id_d": "player_id",
                         "make_e": "el_actual", "fga_e": "el_fga"})
    proj = proj.merge(actual, on=["league", "season", "player_id"], how="left")
    out = proj[["league", "season", "player_id", "el_proj", "el_lo", "el_hi",
                "el_actual", "el_fga"]]

    shared = {
        "K": K, "minFga": MIN_FGA, "nPairs": int(len(pairs)),
        "oosRmse": rmse, "flatRmse": rmse_flat, "naiveRmse": rmse_naive,
        "rangeLo": round(lo_q, 2), "rangeHi": round(hi_q, 2),
        "moversN": int(len(mv)),
        "moversRmse": _rmse(mv.make_e - mv_pred, mv_w),
        "moversFlatRmse": _rmse(mv.make_e - mv.league_d.map(flat_all), mv_w),
        "steps": {lg: round(v, 2) for lg, v in step.items()},
    }
    meta = [(lg, json.dumps({**shared, "step": round(step[lg], 2),
                              "nLeaguePairs": int((pairs.league_d == lg).sum())}))
            for lg in DOMESTIC if lg in step]
    con.execute("DROP TABLE IF EXISTS translation")
    out.to_sql("translation", con, index=False)
    con.execute("DROP TABLE IF EXISTS translation_meta")
    con.execute("CREATE TABLE translation_meta (league TEXT, meta TEXT)")
    con.executemany("INSERT INTO translation_meta VALUES (?, ?)", meta)
    con.commit()
    con.close()
    print(f"translation: {len(pairs)} pairs, oos RMSE {rmse} vs flat "
          f"{rmse_flat} vs naive {rmse_naive}; movers n={len(mv)} "
          f"{shared['moversRmse']} vs flat {shared['moversFlatRmse']}")
    print("  steps:", shared["steps"], f" 80% range {lo_q:+.1f} / {hi_q:+.1f}")


if __name__ == "__main__":
    main()
