"""Per-game pages: the expected score of a game and how often a side wins it.

Byte-identical in ../Euro_OE/games.py (a gate runs cmp on the two); keep them
in step. Pure functions, numpy and pandas only. The exporters build the shot
and free-throw frames from their own databases and call these.

A game's expected score is what its shots were worth before they were taken:
each field goal is worth value x xfg, each free throw its shooter's
free-throw percentage shrunk toward the league by K pseudo free throws (so a
2-for-2 night does not read as a 100% shooter). Win
share replays the game: every shot and free throw is redrawn as an independent
Bernoulli at its expected probability, both sides are summed, and the share of
replays the home side wins is the figure, with an exact tie counting half.
Calibration checks that share against what happened: games in decile bins of
predicted win share, bin mean against the bin's actual home win rate, and the
n-weighted mean absolute gap between the two (ECE).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

K_FT = 50.0              # pseudo free throws shrinking a shooter toward the league
HOME, AWAY = 0, 1


def week_key(d: dt.date) -> str:
    """ISO date of the Monday of d's week."""
    return (d - dt.timedelta(days=d.weekday())).isoformat()


def shrunk_ft_pct(ftm: int, fta: int, league_pct: float, k: float = K_FT) -> float:
    """Free-throw percentage shrunk toward the league by k pseudo attempts."""
    return (ftm + k * league_pct) / (fta + k)


def side_totals(shots: pd.DataFrame, fts: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Actual and expected points for each side.

    shots: side, value, made, xfg. fts: side, made, pct. A side with no rows
    gets zeros. fg/xfg are field-goal points, ftm/xft free throws made/expected.
    """
    out = {}
    for s in (HOME, AWAY):
        sh = shots[shots["side"] == s]
        ft = fts[fts["side"] == s]
        value = sh["value"].to_numpy(dtype=float)
        out[s] = {
            "fg": float((value * sh["made"].to_numpy(dtype=float)).sum()),
            "xfg": float((value * sh["xfg"].to_numpy(dtype=float)).sum()),
            "ftm": float(ft["made"].to_numpy(dtype=float).sum()),
            "xft": float(ft["pct"].to_numpy(dtype=float).sum()),
        }
    return out


def _simulate_side(rng: np.random.Generator, shots: pd.DataFrame, fts: pd.DataFrame,
                   side: int, n: int) -> np.ndarray:
    """n replayed scores for one side: an (n, attempts) Bernoulli matrix times points."""
    sh = shots[shots["side"] == side]
    ft = fts[fts["side"] == side]
    p = np.concatenate([sh["xfg"].to_numpy(dtype=float), ft["pct"].to_numpy(dtype=float)])
    pts = np.concatenate([sh["value"].to_numpy(dtype=float), np.ones(len(ft))])
    return (rng.random((n, len(p))) < p).astype(float) @ pts


def win_share(shots: pd.DataFrame, fts: pd.DataFrame, n: int = 10_000, seed: int = 0) -> float:
    """Share of n replays the home side wins, an exact tie counting half."""
    rng = np.random.default_rng(seed)
    home = _simulate_side(rng, shots, fts, HOME, n)
    away = _simulate_side(rng, shots, fts, AWAY, n)
    return float(((home > away).sum() + 0.5 * (home == away).sum()) / n)


def calibration(win: np.ndarray, home_won: np.ndarray, bins: int = 10) -> tuple[list[dict], float]:
    """Equal-count bins of predicted win share against the home win rate.

    Returns the bins ({"mean", "actual", "n"}, lowest prediction first) and the
    ECE, the n-weighted mean of |bin mean - bin actual|.
    """
    win = np.asarray(win, dtype=float)
    won = np.asarray(home_won, dtype=float)
    order = np.argsort(win, kind="stable")
    rows = [{"mean": float(win[i].mean()), "actual": float(won[i].mean()), "n": int(len(i))}
            for i in np.array_split(order, bins) if len(i)]
    total = sum(r["n"] for r in rows)
    ece = sum(r["n"] * abs(r["mean"] - r["actual"]) for r in rows) / total
    return rows, float(ece)
