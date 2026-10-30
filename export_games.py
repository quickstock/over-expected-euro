"""Per-game pages for the eight Euro_OE leagues -> site/public/games-{LG}-{season}.json
and gameweek-{LG}-{week}.json.

Same files and schema as ../NBA_OE/export_games.py: one index per league-season
(every game: teams, final score, the score its shots were worth, the home side's
win share) and one detail chunk per Monday-keyed week (each game's shots, free
throws, player lines and officials). The math lives in games.py, shared with
NBA_OE; this file only assembles each game's frames from this database and
applies the gates. Seasons are each league's exported data-{LG}.json's own.

Sources. Shots: shots joined to shots_xfg (the leak-free, season-anchored xFG the
shot-making lens uses), each worth xFG x its value. Free throws: events with
etype 'ft', each valued at its shooter's season FT% (player_season_ft) shrunk
toward the league's. Final scores: games.home_score / away_score. Player FGA / FG
points / expected FG points start from player_game_shot_value (the form chart's
numbers). Names: player_season, then the events' own. Officials: games.ref1-3,
ids and display names as the league's referee pages have them.

Field goals the shot chart lacks. The shots table is the shot-chart feed; the
play-by-play carries a few hundred more field goals a league (EuroLeague 175,
171 of them made) with no chart row, so no location and no xFG. Leaving them out
left 5-10% of EuroLeague, EuroCup and Liga ACB games unable to rebuild their
final. They stay in the game as the NBA's unlocated shots do: no coordinates, the
xFG the league-season makes shots of that value at, counted at that xFG in the
totals and the player lines.

Shots and free throws with no shooter. The BBL and GBL feeds list some with a team
and no player. They are real points, so they stay: each side gets one
"Unattributed" row in the game's player list, which the shot and free-throw
columns index like any player. Its id starts with "?", which no league's ids do.

A game is published only when its events rebuild the official final: each side's
made field goals at their value plus its made free throws must equal the final
score. A game that does not is left out and counted. A game with no shot events
and no free-throw events at all was not played (the ABA's 2020-21 awarded 20-0
forfeits): it is skipped and counted apart, as meta.forfeits, not as left out.

Gates (exit 1, nothing written):
  * per league-season, no more than max(1% of the games played, 3) left out; the
    GBL's feed drops scoring rows in 6-24% of its games, so its limit is 25%;
  * per league-season, expected FG points within 0.5% of actual FG points (2% for
    the in-progress season, config.SEASONS[-1]: the anchor is on makes, not
    points, so a season a few games old drifts in points while every completed
    season sits inside 0.5%). The anchor measures the shots that have a model xFG:
    unlocated shots carry an imputed league rate instead (and are nearly all made,
    which would bias it), so they stay in every total and player line but out of
    the anchor.
Calibration of the win share is per league, pooled over its seasons, and not a
failure: when the ECE is not under 0.05, or a decile with 100+ games is more than
8 points from its mean, the league's win share is withheld (meta.winShare false,
w null) and its pages show the expected score alone.

The GBL publishes no game dates (its dates are interpolated from Wayback ids), so
its index rows carry d null; the week key, which only chunks the files, is still
read from the interpolated date.

A chunk's per-game detail is columnar (a shot's side is its player's side):
  p  [[pid, name, side, fga, fgPts, xFgPts, fta, ftm]]  every player in the game
  s  {p: index into p, q: period, c: seconds left, x, y: cm or null,
      m: "0110", v: "2323", e: xFG in thousandths}
  f  {p, q, c, m: "0110", e: shrunk FT% in thousandths}
  r  [[refId, name]]

Usage: python3 export_games.py [LG ...]   (default: every league in config.LEAGUES)
"""
import argparse
import json
import sys
import zlib

import numpy as np
import pandas as pd

import config
from export_site_data import OUT_DIR, conn, ref_key, team_abbrev_map
from games import calibration, shrunk_ft_pct, side_totals, week_key, win_share

FG_TOL = 0.005                 # |expected - actual| FG points / actual, completed seasons
FG_TOL_IN_PROGRESS = 0.02
LEFT_OUT_PERCENT = {"GBL": 25}   # a season may leave out this share of its games (1% unless listed) ...
LEFT_OUT_PERCENT_DEFAULT = 1
LEFT_OUT_MIN = 3               # ... but never fewer than this many
XFG_TIE = 0.05                 # a side's player xFgPts may differ from its expected FG points by this much
ECE_MAX = 0.05
DECILE_MAX_GAP = 0.08
DECILE_MIN_N = 100
INTERPOLATED_DATES = ("GBL",)  # leagues whose game dates are not published
UNATTRIBUTED = {0: "?home", 1: "?away"}   # side -> the id of its "no shooter listed" row
UNATTRIBUTED_NAME = "Unattributed"
SHOT_COLS = ["player_id", "event_id", "period", "sec", "x", "y", "made", "value", "xfg", "xpts", "side"]


def cm(values) -> list:
    """Centimetres as ints, null where a shot has no location (or, for y, no depth)."""
    return [None if v != v else int(v) for v in values]


def make_rates(located: pd.DataFrame) -> dict[int, float]:
    """The league's make rate by shot value (2 or 3) over a season's located shots."""
    return {int(v): float(r) for v, r in located.groupby("value")["made"].mean().items()}


def place_unlocated(un: pd.DataFrame, rates: dict[int, float]) -> pd.DataFrame:
    """Give shots that have no location the xFG the league makes shots of their
    value at (value column filled), and no coordinates."""
    un = un.copy()
    un["xfg"] = un["value"].map(rates)
    un["xpts"] = un["xfg"] * un["value"]
    un["x"] = np.nan
    un["y"] = np.nan
    return un


def shot_block(sh: pd.DataFrame, idx: dict) -> dict:
    """A game's shots as columns, chronological. idx maps player id -> row of the game's p list."""
    return {
        "p": [idx[p] for p in sh["player_id"]],
        "q": sh["period"].astype(int).tolist(),
        "c": sh["sec"].astype(int).tolist(),
        "x": cm(sh["x"]),
        "y": cm(sh["y"]),
        "m": "".join(str(int(m)) for m in sh["made"]),
        "v": "".join(str(int(v)) for v in sh["value"]),
        "e": np.rint(sh["xfg"].to_numpy(dtype=float) * 1000).astype(int).tolist(),
    }


def ft_block(fts: pd.DataFrame, idx: dict) -> dict:
    """A game's free throws as columns, in play order; e is the shooter's shrunk FT% in thousandths."""
    return {
        "p": [idx[p] for p in fts["pid"]],
        "q": fts["period"].astype(int).tolist(),
        "c": fts["sec"].astype(int).tolist(),
        "m": "".join("1" if m else "0" for m in fts["made"]),
        "e": np.rint(fts["pct"].to_numpy(dtype=float) * 1000).astype(int).tolist(),
    }


def anchor_tolerance(season: str) -> float:
    """How far a season's expected FG points may sit from its actual FG points: 0.5%,
    or 2% for the in-progress season (the anchor is on makes, not points)."""
    return FG_TOL_IN_PROGRESS if season == config.SEASONS[-1] else FG_TOL


def left_out_limit(n_games: int, league: str) -> int:
    """The most games a season may leave out: 1% of its games played (25% for the GBL,
    whose feed drops scoring rows), never fewer than three."""
    return max(n_games * LEFT_OUT_PERCENT.get(league, LEFT_OUT_PERCENT_DEFAULT) // 100, LEFT_OUT_MIN)


def shooter_sides(sh: pd.DataFrame, fts: pd.DataFrame) -> "dict | None":
    """Player id -> side for everyone who shot or took a free throw, or None when a
    player's free throws are on a different team than his shots."""
    side_of = dict(zip(sh["player_id"], sh["side"]))
    for pid, side in zip(fts["pid"], fts["side"]):
        if side_of.get(pid, side) != side:
            return None
        side_of[pid] = side
    return side_of


def official_names(games: pd.DataFrame) -> dict:
    """Official id -> the name to show, for every official who has a Latin spelling in
    the games' ref1-ref3: the most common one. The referee export's own rule
    (export_site_data.export_league): the GBL lists a person in Greek in some games and
    Latin in others, and ref_key puts both on one id, shown in Latin."""
    names = pd.concat([games[c] for c in ("ref1", "ref2", "ref3")], ignore_index=True)
    names = names[names.notna() & (names != "")]
    latin = names[names.map(lambda n: n.isascii())]
    return latin.groupby(latin.map(ref_key)).agg(lambda n: n.mode().iat[0]).to_dict()


def officials(refs, shown: dict) -> list:
    """[[id, name]] for each official a game lists (its ref1-ref3, blank or null skipped),
    the id and display name the league's referee pages use."""
    out = []
    for name in refs:
        if isinstance(name, str) and name:
            rid = ref_key(name)
            out.append([rid, shown.get(rid, name)])
    return out


def player_lines(gid: str, sh: pd.DataFrame, extra: "pd.DataFrame | None", fts: pd.DataFrame,
                 pgsv: "pd.DataFrame | None", side_of: dict, names: dict, event_names: dict,
                 expected_fg: dict) -> tuple[list, int]:
    """A game's [pid, name, side, fga, fgPts, xFgPts, fta, ftm] rows, and how many names
    came from the events. FG numbers start from player_game_shot_value (pgsv; the form
    chart's), plus the shots it does not hold (extra: the unlocated ones and those with no
    shooter) at their xpts; free throws from the events. Everyone in the game's shots or
    free throws gets a row. Raises when a side's xFgPts do not add up to expected_fg[side],
    the expected FG points its shots are worth."""
    line = {}
    if pgsv is not None:
        for r in pgsv.itertuples():
            line[r.player_id] = [int(r.fga), int(round(r.act_fg_pts)), float(r.exp_fg_pts), 0, 0]
    for pid in sh["player_id"]:
        line.setdefault(pid, [0, 0, 0.0, 0, 0])
    if extra is not None:
        for pid, made, value, xpts in zip(extra["player_id"], extra["made"], extra["value"], extra["xpts"]):
            row = line[pid]
            row[0] += 1
            row[1] += int(made) * int(value)
            row[2] += float(xpts)
    for pid, made in zip(fts["pid"], fts["made"]):
        row = line.setdefault(pid, [0, 0, 0.0, 0, 0])
        row[3] += 1
        row[4] += int(made)
    for side in (0, 1):
        got = sum(row[2] for pid, row in line.items() if side_of[pid] == side)
        if abs(got - expected_fg[side]) > XFG_TIE:
            raise ValueError(f"{gid}: side {side} players' expected FG points {got:.3f} "
                             f"differ from the side's {expected_fg[side]:.3f}")
    players, fallbacks = [], 0
    for pid, (fga, pts, xpts, fta, ftm) in line.items():
        if pid in names:
            name = names[pid]
        else:
            name = event_names[pid]
            fallbacks += 1
        players.append([pid, name, int(side_of[pid]), fga, pts, round(xpts, 2), fta, ftm])
    players.sort(key=lambda p: (p[2], -(p[4] + p[7]), p[0]))
    return players, fallbacks


def chrono(df: pd.DataFrame) -> pd.DataFrame:
    """Play order within each game. The events' own ids are not in clock order in every
    league (the ABA's run team by team), so the clock leads and the id breaks ties."""
    return df.sort_values(["game_id", "period", "sec", "event_id"],
                          ascending=[True, True, False, True], kind="stable")


def with_shooter(df: pd.DataFrame, side: pd.Series) -> pd.Series:
    """The shooter's id, or his side's "Unattributed" id where the feed lists none."""
    return df["player_id"].replace("", np.nan).fillna(side.map(UNATTRIBUTED))


def load_located(conn, league: str, season: str) -> pd.DataFrame:
    """The season's shots that have an xFG: coordinates in centimetres (NaN where the feed
    has none: ACB's depth below the hoop line is NULL, its x is real), value, xFG and xpts,
    and the shooter's side (0 home, 1 away) from the shot's home_or_away."""
    df = pd.read_sql(
        """SELECT s.game_id, s.event_id, s.player_id, s.period, s.secs_remaining AS sec,
                  s.shot_made AS made, s.shot_value AS value, s.x_m, s.y_m, s.home_or_away, x.xfg
           FROM shots s
           JOIN shots_xfg x ON x.league = s.league AND x.game_id = s.game_id AND x.event_id = s.event_id
           WHERE s.league = ? AND s.season = ?""", conn, params=(league, season))
    side = df["home_or_away"].map({"home": 0, "away": 1})
    if side.isna().any():
        raise ValueError(f"{league} {season}: {side.isna().sum()} shots with no home or away side")
    df["side"] = side.astype(int)
    df["player_id"] = with_shooter(df, df["side"])
    df["sec"] = np.rint(df["sec"]).astype(int)
    df["value"] = df["value"].astype(int)
    df["x"] = np.rint(df["x_m"] * 100)
    df["y"] = np.rint(df["y_m"] * 100)
    df["xpts"] = df["xfg"] * df["value"]
    return chrono(df)


def load_events(conn, league: str, season: str, located: pd.DataFrame,
                games: pd.DataFrame, ft_pct: dict, unseen_pct: float) -> tuple:
    """(unlocated, free throws, names) of the season from the play-by-play events.
    Unlocated: field goals with no row in the located shots (value filled, side from
    the team, -1 where the team is not in its game). Free throws: side, shooter, clock,
    made and the shooter's shrunk FT%. Names: player id -> the events' name for him."""
    ev = pd.read_sql(
        """SELECT game_id, event_id, period, secs_remaining AS sec, team_id, player_id, player_name,
                  etype, shot_value AS value, shot_made AS made
           FROM events WHERE league = ? AND season = ? AND etype IN ('fg', 'ft')""",
        conn, params=(league, season))
    ev = ev.merge(games[["game_id", "home_team_id", "away_team_id"]], on="game_id")
    ev["side"] = np.select([ev["team_id"] == ev["home_team_id"], ev["team_id"] == ev["away_team_id"]],
                           [0, 1], default=-1)
    ev["player_id"] = with_shooter(ev, ev["side"])
    ev["sec"] = np.rint(ev["sec"]).astype(int)
    ev["made"] = ev["made"].astype(int)

    fg = ev[ev["etype"] == "fg"]
    if fg["value"].isna().any():
        raise ValueError(f"{league} {season}: {fg['value'].isna().sum()} field goals with no value")
    key = ["game_id", "event_id"]
    unlocated = fg.merge(located[key], on=key, how="left", indicator=True)
    unlocated = unlocated[unlocated["_merge"] == "left_only"].drop(columns="_merge")
    unlocated["value"] = unlocated["value"].astype(int)

    ft = ev[ev["etype"] == "ft"]
    fts = pd.DataFrame({
        "game_id": ft["game_id"], "event_id": ft["event_id"], "side": ft["side"],
        "pid": ft["player_id"], "period": ft["period"], "sec": ft["sec"],
        "made": ft["made"].astype(bool), "pct": ft["player_id"].map(ft_pct).fillna(unseen_pct)})
    named = ev[ev["player_name"].notna()]
    return unlocated, chrono(fts), dict(zip(named["player_id"], named["player_name"]))


def export_season(conn, league: str, season: str, abbrev: dict, shown: dict) -> dict:
    """Everything a league-season publishes except the win-share-dependent fields:
    index rows (win share unrounded), chunk entries by week, and the sums the gates
    read (the FG anchor over the shots that have a model xFG). abbrev is team_abbrev_map(league), shown official_names of the league's games."""
    games = pd.read_sql(
        "SELECT game_id, date, home_team_id, away_team_id, home_score, away_score, ref1, ref2, ref3 "
        "FROM games WHERE league = ? AND season = ? ORDER BY date, game_id",
        conn, params=(league, season))
    games["date"] = pd.to_datetime(games["date"], errors="coerce").dt.date   # '' (a forfeit) -> NaT

    ft_season = pd.read_sql(
        "SELECT player_id, ftm, fta FROM player_season_ft WHERE league = ? AND season = ?",
        conn, params=(league, season))
    league_ft = ft_season["ftm"].sum() / ft_season["fta"].sum()
    ft_pct = {r.player_id: shrunk_ft_pct(int(r.ftm), int(r.fta), league_ft) for r in ft_season.itertuples()}
    unseen_pct = shrunk_ft_pct(0, 0, league_ft)

    located = load_located(conn, league, season)
    rates = make_rates(located)
    unlocated, fts_all, event_names = load_events(conn, league, season, located, games, ft_pct, unseen_pct)
    unlocated = place_unlocated(unlocated, rates)
    shots_by_game = {g: d[SHOT_COLS] for g, d in located.groupby("game_id", sort=False)}
    unlocated_by_game = {g: d[SHOT_COLS] for g, d in unlocated.groupby("game_id", sort=False)}
    fts_by_game = {g: d for g, d in fts_all.groupby("game_id", sort=False)}

    names = dict(pd.read_sql(
        "SELECT player_id, player_name FROM player_season WHERE league = ? AND season = ?", conn,
        params=(league, season)).itertuples(index=False, name=None))
    names.update({pid: UNATTRIBUTED_NAME for pid in UNATTRIBUTED.values()})
    lines = pd.read_sql(
        "SELECT player_id, game_id, fga, act_fg_pts, exp_fg_pts "
        "FROM player_game_shot_value WHERE league = ? AND season = ?", conn, params=(league, season))
    lines_by_game = {g: d for g, d in lines.groupby("game_id", sort=False)}

    rows, chunks = [], {}
    wins, home_won = [], []
    xpts_sum = act_sum = 0.0
    out = {"no_shots": 0, "no_team": 0, "side": 0, "mismatch": 0}
    n_unlocated = n_unattributed = name_fallbacks = forfeits = 0

    for g in games.itertuples(index=False):
        gid, home, away = g.game_id, g.home_team_id, g.away_team_id
        if gid not in shots_by_game and gid not in unlocated_by_game:
            if gid in fts_by_game:
                out["no_shots"] += 1
            else:
                forfeits += 1      # no shot and no free throw: it was not played
            continue
        fts = fts_by_game.get(gid, fts_all.iloc[:0])
        un = unlocated_by_game.get(gid)
        if (fts["side"] < 0).any() or (un is not None and (un["side"] < 0).any()):
            out["no_team"] += 1
            continue

        sh = shots_by_game.get(gid, located.iloc[:0][SHOT_COLS])
        priced = sh        # the shots with a model xFG: the anchor's
        extra = sh[sh["player_id"].isin(UNATTRIBUTED.values())]    # pgsv holds no shooterless shots
        if un is not None:
            extra = pd.concat([extra, un])
            sh = pd.concat([sh, un], ignore_index=True)
            sh = sh.sort_values(["period", "sec", "event_id"], ascending=[True, False, True], kind="stable")

        # A player has one side: the team of his shots, which his free throws must agree with.
        side_of = shooter_sides(sh, fts)
        if side_of is None:
            out["side"] += 1
            continue

        tot = side_totals(sh, fts)
        hs, as_ = int(g.home_score), int(g.away_score)
        if (tot[0]["fg"] + tot[0]["ftm"], tot[1]["fg"] + tot[1]["ftm"]) != (hs, as_):
            out["mismatch"] += 1
            continue
        if pd.isna(g.date):
            raise ValueError(f"{league} {gid}: a game that rebuilds its final has no date")

        xpts_sum += float(priced["xpts"].sum())
        act_sum += float((priced["value"] * priced["made"]).sum())
        n_unlocated += 0 if un is None else len(un)
        n_unattributed += int(sh["player_id"].isin(UNATTRIBUTED.values()).sum())
        share = win_share(sh, fts, seed=zlib.crc32(f"{league}{gid}".encode()))
        wins.append(share)
        home_won.append(1.0 if hs > as_ else 0.0 if hs < as_ else 0.5)

        players, fallbacks = player_lines(
            gid, sh, extra if len(extra) else None, fts, lines_by_game.get(gid), side_of, names,
            event_names, {0: tot[0]["xfg"], 1: tot[1]["xfg"]})
        name_fallbacks += fallbacks
        idx = {p[0]: i for i, p in enumerate(players)}

        wk = week_key(g.date)
        chunks.setdefault(wk, {})[gid] = {
            "p": players, "s": shot_block(sh, idx), "f": ft_block(fts, idx),
            "r": officials([g.ref1, g.ref2, g.ref3], shown),
        }
        rows.append({
            "id": gid, "d": None if league in INTERPOLATED_DATES else g.date.isoformat(), "wk": wk,
            "h": home, "a": away,
            "hl": abbrev.get(home, home), "al": abbrev.get(away, away),
            "s": [hs, as_],
            "x": [round(tot[0]["xfg"] + tot[0]["xft"], 1), round(tot[1]["xfg"] + tot[1]["xft"], 1)],
            "share": share,
        })

    if not rows:
        raise ValueError(f"{league} {season}: none of its {len(games)} games rebuilds its final ({out})")
    excluded = sum(out.values())
    return {
        "season": season, "rows": rows, "chunks": chunks, "wins": wins, "home_won": home_won,
        "excluded": excluded, "why": out, "total": len(games), "forfeits": forfeits,
        "name_fallbacks": name_fallbacks,
        "unlocated": n_unlocated, "unattributed": n_unattributed,
        "anchor": (xpts_sum - act_sum) / act_sum,
    }


def dump(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def league_files(league: str, results: list, ok: bool, ece: float) -> dict:
    """File name -> text for one league: an index per season, a chunk per week."""
    files = {}
    for r in results:
        files[f"games-{league}-{r['season']}.json"] = dump({
            "meta": {"league": league, "season": r["season"], "winShare": ok,
                     "ece": round(ece, 4), "excluded": r["excluded"], "forfeits": r["forfeits"],
                     "games": len(r["rows"])},
            "games": [{**{k: v for k, v in g.items() if k != "share"},
                       "w": int(round(100 * g["share"])) if ok else None} for g in r["rows"]],
        })
        for wk, chunk in r["chunks"].items():
            name = f"gameweek-{league}-{wk}.json"
            if name in files:
                raise ValueError(f"{name} written by two seasons")
            files[name] = dump({"games": chunk})
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("leagues", nargs="*", help=f"any of {' '.join(config.LEAGUES)}; default: all")
    leagues = ap.parse_args(argv).leagues or config.LEAGUES
    unknown = [lg for lg in leagues if lg not in config.LEAGUES]
    if unknown:
        ap.error(f"unknown league {unknown}")

    files, failures = {}, []
    for league in leagues:
        seasons = json.loads((OUT_DIR / f"data-{league}.json").read_text())["meta"]["seasons"]
        shown = official_names(pd.read_sql("SELECT ref1, ref2, ref3 FROM games WHERE league = ?",
                                           conn, params=(league,)))
        abbrev = team_abbrev_map(league)

        results = []
        for s in seasons:
            r = export_season(conn, league, s, abbrev, shown)
            results.append(r)
            played = r["total"] - r["forfeits"]
            frac = r["excluded"] / played
            tol = anchor_tolerance(s)
            why = ", ".join(f"{k} {v}" for k, v in r["why"].items() if v)
            print(f"{league} {s}: {len(r['rows'])} games, {r['excluded']} excluded ({frac:.2%})"
                  f"{' [' + why + ']' if why else ''}"
                  f"{', forfeits ' + str(r['forfeits']) if r['forfeits'] else ''}"
                  f", FG anchor {r['anchor']:+.2%}"
                  f"{', unlocated shots ' + str(r['unlocated']) if r['unlocated'] else ''}"
                  f"{', no shooter on ' + str(r['unattributed']) + ' shots' if r['unattributed'] else ''}"
                  f"{', names from events ' + str(r['name_fallbacks']) if r['name_fallbacks'] else ''}",
                  flush=True)
            if r["excluded"] > left_out_limit(played, league):
                failures.append(f"{league} {s}: {r['excluded']} of {played} games left out "
                                f"(limit {left_out_limit(played, league)})")
            if abs(r["anchor"]) > tol:
                failures.append(f"{league} {s}: FG anchor {r['anchor']:+.2%} outside +-{tol:.1%}")

        win = np.array([w for r in results for w in r["wins"]])
        won = np.array([w for r in results for w in r["home_won"]])
        bins, ece = calibration(win, won)
        ok = ece < ECE_MAX and all(abs(b["mean"] - b["actual"]) <= DECILE_MAX_GAP
                                   for b in bins if b["n"] >= DECILE_MIN_N)
        print(f"{league} calibration: ECE {ece:.4f}, winShare {str(ok).lower()}; deciles (mean -> actual, n): "
              + ", ".join(f"{b['mean']:.3f}->{b['actual']:.3f} ({b['n']})" for b in bins))
        if not ok:
            print(f"WARNING: {league} win share failed calibration; its games ship without it "
                  f"(meta.winShare false)")
        files.update(league_files(league, results, ok, ece))

    if failures:
        print("GATE FAILED, nothing written:\n  " + "\n  ".join(failures))
        return 1
    for name, text in files.items():
        (OUT_DIR / name).write_text(text, encoding="utf-8")
    print(f"wrote {len(files)} files -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
