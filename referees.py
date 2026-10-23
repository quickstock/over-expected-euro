"""Referee views: the expected whistle, shrunk to what the data can support.

Mirrored in ../NBA_OE/referees.py (same construction, every league); keep the
two in step. Each export builds the three inputs from its own database.

The expected whistle of a game is what its players would draw from where they
shot: each finisher's shot-diet expectation (diet_game) times his own
foul-drawing over that expectation in the season's other games (shrunk by K
free throws), summed over the game and rescaled so the season's expected total
equals its actual total. An official's number is free throws over that
expectation per 100 possessions in the games he worked.

Measured before building (NBA live seasons, 2026-10-23): the players on the
floor explain game-to-game free throws better than shot diet alone (per-game r
0.39 -> 0.45), but barely move officials (r 0.97 with the shot-diet-only
version), because assignments rotate across teams. What does not move is
noise: one season of an official is 0.37 reliable (odd vs even games), a
career of 4+ seasons 0.50, and quarter and game-script leans do not repeat at
all, even pooled over a career (Q4 0.01, close games -0.16). So every number
is shrunk toward the league by empirical Bayes, with a 90% interval, and the
reliabilities are measured here on every export and shipped for the page to
quote.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_PLAYER = 25.0          # pseudo free throws shrinking a player's own multiplier
Z90 = 1.645
QUARTERS = ["Q1", "Q2", "Q3", "Q4", "OT"]
SCRIPTS = ["close", "mid", "blowout"]


def _ratio_se(num_g: pd.Series, den_g: pd.Series, rate: float) -> float:
    """SE (per 100) of sum(num)/sum(den) from per-game residuals, which keeps
    the overdispersion of free throws arriving in pairs and trips."""
    den = den_g.sum()
    return float(np.sqrt(((num_g - rate * den_g) ** 2).sum()) / den * 100) if den else np.nan


def _spearman_brown(a: pd.Series, b: pd.Series, min_n: int) -> float | None:
    ok = a.notna() & b.notna()
    if ok.sum() < min_n:
        return None
    r = float(np.corrcoef(a[ok], b[ok])[0, 1])
    # a negative half-vs-half correlation is noise around zero, not negative
    # reliability: report it as none
    return round(max(0.0, 2 * r / (1 + r)), 2)


def _shrink(diff: pd.Series, se: pd.Series, tau2: float):
    b = tau2 / (tau2 + se ** 2)
    est = b * diff
    sd = np.sqrt(b) * se
    return est, est - Z90 * sd, est + Z90 * sd


MIN_TAU_N = 20


def _tau2(diff: pd.Series, se: pd.Series) -> float | None:
    """True-spread variance: observed spread minus the average sampling noise.
    None below MIN_TAU_N officials (EuroCup has 7 qualified official-seasons,
    GBL 15): too few to tell signal from noise, so nothing is shrunk or
    published as an estimate there."""
    if len(diff) < MIN_TAU_N:
        return None
    return max(0.0, float(diff.var() - (se ** 2).mean()))


def build(poss: pd.DataFrame, diet: pd.DataFrame, refs: pd.DataFrame, qualify_games: int):
    """poss: one row per possession: game_id, season, q, bucket, sfta, pid
             (finisher id, NaN when none).
    diet:  game_id, pid, xft (shot-diet expected free throws, player-game).
    refs:  game_id, ref_id, ref_name (one row per official per game).
    Returns (referees_out, ref_profiles, meta)."""
    # ---- expected whistle per game -------------------------------------
    pg = (poss[poss.pid.notna()].groupby(["season", "game_id", "pid"]).sfta.sum()
          .rename("fta").reset_index())
    pg = pg.merge(diet, on=["game_id", "pid"], how="left").fillna({"xft": 0.0})
    tot = pg.groupby(["season", "pid"])[["fta", "xft"]].transform("sum")
    pg["exp"] = pg.xft * (tot.fta - pg.fta + K_PLAYER) / (tot.xft - pg.xft + K_PLAYER)
    games = (poss.groupby(["season", "game_id"])
             .agg(poss=("sfta", "size"), sfta=("sfta", "sum"), bucket=("bucket", "first"))
             .reset_index())
    ge = pg.groupby("game_id")[["fta", "exp"]].sum()
    games = games.join(ge, on="game_id").fillna({"fta": 0.0, "exp": 0.0})
    scale = games.groupby("season").fta.transform("sum") / games.groupby("season").exp.transform("sum")
    games["exp"] *= scale
    games["res"] = games.fta - games.exp

    lg_season = games.groupby("season").apply(lambda g: g.sfta.sum() / g.poss.sum() * 100)
    qp = poss.groupby(["season", "game_id", "q"]).agg(poss=("sfta", "size"), sfta=("sfta", "sum")).reset_index()
    lg_q = qp.groupby(["season", "q"]).apply(lambda g: g.sfta.sum() / g.poss.sum() * 100)
    lg_b = games.groupby(["season", "bucket"]).apply(lambda g: g.sfta.sum() / g.poss.sum() * 100)

    # ---- official-seasons -------------------------------------------------
    ref_name = refs.groupby("ref_id").ref_name.agg(lambda s: s.mode().iat[0])
    rg = refs[["game_id", "ref_id"]].drop_duplicates().merge(games, on="game_id")

    def per_official(df: pd.DataFrame, keys: list) -> pd.DataFrame:
        rows = []
        for k, g in df.groupby(keys):
            d = g.res.sum() / g.poss.sum()
            rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))),
                         "games": g.game_id.nunique(), "poss": g.poss.sum(),
                         "sfta": g.sfta.sum(), "diff": d * 100,
                         "se": _ratio_se(g.res, g.poss, d)})
        return pd.DataFrame(rows)

    os_ = per_official(rg, ["ref_id", "season"])
    os_["qual"] = os_.games >= qualify_games
    q = os_[os_.qual]
    tau_s = _tau2(q["diff"], q.se)
    if tau_s is None:
        os_["est"] = os_["lo"] = os_["hi"] = np.nan
    else:
        os_["est"], os_["lo"], os_["hi"] = _shrink(os_["diff"], os_.se, tau_s)

    # careers: every game an official worked in this export, for officials
    # with two or more qualified seasons
    nq = q.groupby("ref_id").size()
    multi = nq[nq >= 2].index
    car = per_official(rg[rg.ref_id.isin(multi)], ["ref_id"]) if len(multi) else pd.DataFrame()
    tau_c = _tau2(car["diff"], car.se) if len(car) else None
    if tau_c is not None:
        car["est"], car["lo"], car["hi"] = _shrink(car["diff"], car.se, tau_c)

    # ---- reliabilities, measured on this export ---------------------------
    order = rg.sort_values("game_id")
    order["half"] = order.groupby(["ref_id", "season"]).cumcount() % 2
    halves = [per_official(order[order.half == h], ["ref_id", "season"]).set_index(["ref_id", "season"])
              for h in (0, 1)]
    hj = halves[0].join(halves[1], lsuffix="_a", rsuffix="_b", how="inner")
    hj = hj[(hj.games_a >= qualify_games / 2) & (hj.games_b >= qualify_games / 2)]
    rel_season = _spearman_brown(hj.diff_a, hj.diff_b, 20)

    seasons = sorted(q.season.unique())
    par = {s: i % 2 for i, s in enumerate(seasons)}
    long_careers = nq[nq >= 4].index
    rel_career = None
    if len(long_careers) >= 20:
        cg = rg[rg.ref_id.isin(long_careers)].assign(par=lambda d: d.season.map(par))
        ce = per_official(cg, ["ref_id", "par"]).pivot(index="ref_id", columns="par", values="diff")
        rel_career = _spearman_brown(ce[0], ce[1], 20)

    # splits: each official-season's lean in a quarter or game script, over
    # the league's own rate there and his own overall level, odd vs even games
    qpg = qp.merge(refs[["game_id", "ref_id"]].drop_duplicates(), on="game_id")
    qpg = qpg.merge(order[["game_id", "ref_id", "season", "half"]], on=["game_id", "ref_id", "season"])
    leans = []
    for h in (0, 1):
        sub = qpg[qpg.half == h]
        lvl = order[order.half == h].groupby(["ref_id", "season"]).apply(
            lambda g: g.sfta.sum() / g.poss.sum() * 100).rename("lvl")
        lgs = lg_season.rename("lgs")
        cell = sub.groupby(["ref_id", "season", "q"]).apply(
            lambda g: g.sfta.sum() / g.poss.sum() * 100).rename("rate").reset_index()
        cell = cell.join(lvl, on=["ref_id", "season"]).join(lgs, on="season")
        cell["lgq"] = [lg_q[(s, qq)] for s, qq in zip(cell.season, cell.q)]
        cell[f"lean{h}"] = (cell.rate - cell.lgq) - (cell.lvl - cell.lgs)
        leans.append(cell.set_index(["ref_id", "season", "q"])[f"lean{h}"])
    qualified = set(zip(q.ref_id, q.season))

    def lean_reliability(halves_: list) -> float | None:
        lj = pd.concat(halves_, axis=1).dropna()
        lj = lj[[(r, s) in qualified for r, s, _ in lj.index]]
        return _spearman_brown(lj.lean0, lj.lean1, 40)

    rel_quarters = lean_reliability(leans)
    # the same for game script: each game is one bucket, so the cells are games
    sleans = []
    for h in (0, 1):
        sub = order[order.half == h]
        lvl = sub.groupby(["ref_id", "season"]).apply(
            lambda g: g.sfta.sum() / g.poss.sum() * 100).rename("lvl")
        cell = sub.groupby(["ref_id", "season", "bucket"]).apply(
            lambda g: g.sfta.sum() / g.poss.sum() * 100).rename("rate").reset_index()
        cell = cell.join(lvl, on=["ref_id", "season"]).join(lg_season.rename("lgs"), on="season")
        cell["lgb"] = [lg_b[(s, b)] for s, b in zip(cell.season, cell.bucket)]
        cell[f"lean{h}"] = (cell.rate - cell.lgb) - (cell.lvl - cell.lgs)
        sleans.append(cell.set_index(["ref_id", "season", "bucket"])[f"lean{h}"])
    rel_script = lean_reliability(sleans)

    # ---- splits per official-season, with intervals -----------------------
    def split_rows(g: pd.DataFrame, key: str, order_: list, lg: pd.Series, season: str) -> list:
        out = []
        for k in order_:
            s = g[g[key] == k]
            if s.empty:
                continue
            p, f = s.poss.sum(), s.sfta.sum()
            rate = f / p
            se = _ratio_se(s.sfta, s.poss, rate)
            row = {key if key != "bucket" else "b": k, "poss": int(p), "per100": round(rate * 100, 2),
                   "lg": round(float(lg[(season, k)]), 2),
                   "lo": round(rate * 100 - Z90 * se, 2), "hi": round(rate * 100 + Z90 * se, 2)}
            if key == "bucket":
                row["games"] = int(s.game_id.nunique())
            out.append(row)
        return out

    by_q = dict(tuple(qpg.groupby(["ref_id", "season"])))
    by_g = dict(tuple(rg.groupby(["ref_id", "season"])))
    referees_out: dict = {}
    ref_profiles: dict = {}
    for r in os_[os_.qual].itertuples():
        rid, season = r.ref_id, r.season
        lg = float(lg_season[season])
        shrunk = tau_s is not None
        row = {"id": rid, "name": ref_name[rid], "games": int(r.games),
               "per100": round(r.sfta / r.poss * 100, 2), "diff": round(r.diff, 2),
               "est": round(r.est, 2) if shrunk else None,
               "lo": round(r.lo, 2) if shrunk else None,
               "hi": round(r.hi, 2) if shrunk else None}
        referees_out.setdefault(season, []).append(row)
        prof = ref_profiles.setdefault(str(rid), {"name": ref_name[rid], "seasons": [], "detail": {}, "careers": []})
        prof["seasons"].append(season)
        gq, gb = by_q[(rid, season)], by_g[(rid, season)]
        prof["detail"][season] = {
            "games": int(r.games), "poss": int(r.poss), "fta": int(r.sfta),
            "per100": row["per100"], "lg": round(lg, 2),
            "diff": row["diff"], "est": row["est"], "lo": row["lo"], "hi": row["hi"],
            "quarters": split_rows(gq, "q", QUARTERS, lg_q, season),
            "script": split_rows(gb, "bucket", SCRIPTS, lg_b, season),
        }
    for rows in referees_out.values():
        rows.sort(key=lambda x: -(x["est"] if x["est"] is not None else x["diff"]))
    if tau_c is not None:
        span = q.groupby("ref_id").season.agg(["min", "max", "size"])
        for c in car.itertuples():
            ref_profiles[str(c.ref_id)]["careers"].append({
                "from": span.loc[c.ref_id, "min"], "to": span.loc[c.ref_id, "max"],
                "seasons": int(span.loc[c.ref_id, "size"]), "games": int(c.games),
                "diff": round(c.diff, 2), "est": round(c.est, 2),
                "lo": round(c.lo, 2), "hi": round(c.hi, 2)})
    for prof in ref_profiles.values():
        prof["seasons"] = sorted(prof["seasons"])

    meta = {
        "qualifyGames": qualify_games,
        "reliability": {"season": rel_season, "career": rel_career,
                        "quarters": rel_quarters, "script": rel_script},
        # the rate per 100 over every possession of the season, the pool every
        # official's raw rate is measured on
        "league": {s: round(float(v), 2) for s, v in lg_season.items()},
        "spread": {"season": None if tau_s is None else round(float(np.sqrt(tau_s)), 2),
                   "career": None if tau_c is None else round(float(np.sqrt(tau_c)), 2)},
    }
    return referees_out, ref_profiles, meta
