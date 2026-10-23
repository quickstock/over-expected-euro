"""Export euro_oe.db into static JSON for the website.

Mirrors the NBA export contract, one payload per league:
  site/public/data-{LG}.json           core: meta, leaderboard,
                                       distributions, leagueZones,
                                       calibration, referees, refProfiles,
                                       teams, shotValue
  site/public/players-{LG}-{season}.json  per-season player detail
                                          (games, zones, fouls)

Differences from the NBA export: player ids are strings (EL person codes,
ACB license ids), no style-adjusted baseline (sper100/spct stay null),
referees are keyed by name (no stable id in either feed), and fold lifts /
reliability are computed inline from predictions_poss instead of model
artifacts.
"""
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import config
import referees

OUT_DIR = Path(__file__).parent / "site" / "public"
QUALIFY = config.QUALIFY_POSS
QUALIFY_REF_GAMES = 15

# Greek to Latin, longest first, the way the GBL feed spells officials in Latin
# script ("Πέτρος Παπαπέτρου" / "Petros Papapetrou").
_GREEK = [("ου", "ou"), ("θ", "th"), ("χ", "ch"), ("ψ", "ps"), ("α", "a"),
          ("β", "v"), ("γ", "g"), ("δ", "d"), ("ε", "e"), ("ζ", "z"), ("η", "i"),
          ("ι", "i"), ("κ", "k"), ("λ", "l"), ("μ", "m"), ("ν", "n"), ("ξ", "x"),
          ("ο", "o"), ("π", "p"), ("ρ", "r"), ("σ", "s"), ("ς", "s"), ("τ", "t"),
          ("υ", "y"), ("φ", "f"), ("ω", "o")]


def ref_key(name: str) -> str:
    """An official's id: the slugged name, since no feed carries an id. GBL
    lists the same officials in Greek in some games and in Latin in others, and
    the slug kept only a-z and 0-9, so every Greek spelling became the same
    empty id: 139 of 222 GBL names merged into one "official". Greek is now
    transliterated first, so both spellings of a person land on one id. Names
    without Greek letters are untouched, so no other league's URLs move."""
    s = name.lower()
    if any("\u0370" <= c <= "\u03ff" for c in s):
        s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
        for g, latin in _GREEK:
            s = s.replace(g, latin)
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")

conn = sqlite3.connect(config.DB_PATH)


_LBA_GENERIC = {"basket", "basketball", "pallacanestro", "shark", "sharks",
                "olidata", "yachts"}


def _lba_city(name):
    """LBA team names are 'Sponsor(s) City [Basket]'; the city is the
    stable identity (team_id changes per season/edition). Strip generic
    tokens and take the city (the last remaining word)."""
    toks = [t for t in name.replace("Napolibasket", "Napoli").split()
            if t.lower().strip(".") not in _LBA_GENERIC]
    if len(toks) >= 2 and toks[-2] == "Reggio":
        return "Reggio Emilia"
    return toks[-1] if toks else name


_ABA_SPONSOR_SUFFIXES = {
    "mozzart", "bet", "bh", "telecom", "voli", "meridianbet", "m:tel",
    "superbet", "office", "shoes",
}


def _aba_club(name):
    """ABA names are 'Club [Sponsor words...]', unlike LBA's
    'Sponsor(s) City' — the club identity comes FIRST, so strip trailing
    sponsor tokens instead of taking the last word. Stops at the first
    trailing word that isn't a known sponsor token, so multi-word club
    identities ("Cedevita Olimpija", "U-BT Cluj-Napoca") pass through
    unchanged; an unrecognized future sponsor is a harmless no-op."""
    toks = name.split()
    while len(toks) > 1 and toks[-1].lower().strip(".:") \
            in _ABA_SPONSOR_SUFFIXES:
        toks.pop()
    return " ".join(toks) if toks else name


def team_abbrev_map(league):
    """team_id -> display label. EL, BBL and the Genius LiveStats leagues
    already carry short team codes as the id; ACB maps clubId via its
    calendars; LBA and ABA have only sponsored full names, so reduce to
    the stable club identity."""
    if league in ("LBA", "ABA"):
        rows = pd.read_sql(
            "SELECT DISTINCT home_team_id AS t, home_team_name AS n "
            "FROM games WHERE league = ? AND home_team_name IS NOT NULL",
            conn, params=(league,))
        fn = _lba_city if league == "LBA" else _aba_club
        return {r.t: fn(r.n) for r in rows.itertuples()}
    if league != "ACB":
        ids = pd.read_sql(
            "SELECT DISTINCT home_team_id AS t FROM games WHERE league = ?",
            conn, params=(league,))
        return {t: t for t in ids.t}
    m = {}
    for season in config.SEASONS:
        ed = config.ACB_EDITION[season]
        p = Path(config.CACHE_DIR) / "ACB" / f"calendar_{ed}.json"
        if not p.exists():
            continue
        for t in json.loads(p.read_text()).get("teams", []):
            m[str(t["clubId"])] = t.get("abbreviatedName") or \
                t["fullName"][:3].upper()
    return m


def poisson_deviance(y, mu):
    mu = np.clip(np.asarray(mu, float), 1e-9, None)
    y = np.asarray(y, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * np.sum(term - (y - mu))


def export_league(league):
    abbrev = team_abbrev_map(league)

    # ------------------------------------------------------- leaderboard
    lb = pd.read_sql(
        """SELECT player_id, season, possessions, actual_fta_from_fouls,
                  xfta_total, ftaoe, ftaoe_per_100, player_name, position
           FROM player_season_xfta_lb WHERE league = ?""",
        conn, params=(league,))

    lb["pct"] = np.nan
    q = lb["possessions"] >= QUALIFY
    for season, grp in lb[q].groupby("season"):
        ranks = grp["ftaoe_per_100"].rank(pct=True) * 100
        lb.loc[ranks.index, "pct"] = ranks

    team_rows = pd.read_sql(
        """SELECT p.finisher_player_id AS player_id, p.season,
                  p.offense_team_id AS team_id, MIN(g.date) AS first_game
           FROM possessions p
           JOIN games g ON g.league = p.league AND g.game_id = p.game_id
           WHERE p.league = ? AND p.finisher_player_id IS NOT NULL
             AND p.offense_team_id IS NOT NULL
           GROUP BY p.finisher_player_id, p.season, p.offense_team_id""",
        conn, params=(league,))
    team_rows = team_rows.sort_values("first_game")
    teams_by_key: dict = {}
    for r in team_rows.itertuples():
        teams_by_key.setdefault((r.player_id, r.season), []).append(
            abbrev.get(str(r.team_id), str(r.team_id)))

    leaderboard = [
        {
            "id": r.player_id,
            "name": r.player_name,
            "season": r.season,
            "pos": (r.position or "").split("-")[0] or None,
            "teams": teams_by_key.get((r.player_id, r.season), []),
            "poss": int(r.possessions),
            "fta": int(r.actual_fta_from_fouls),
            "xfta": round(r.xfta_total, 2),
            "ftaoe": round(r.actual_fta_from_fouls - round(r.xfta_total, 2),
                           2),
            "per100": round(r.ftaoe_per_100, 2),
            "pct": round(r.pct, 1) if pd.notna(r.pct) else None,
            "sper100": None,   # no tracking data in Europe
            "spct": None,
        }
        for r in lb.itertuples()
    ]

    distributions = {
        season: sorted(
            round(v, 2)
            for v in grp.loc[grp["possessions"] >= QUALIFY, "ftaoe_per_100"])
        for season, grp in lb.groupby("season")
    }

    # ------------------------------------------- per-game player series
    qual_keys = set(zip(lb.loc[q, "player_id"], lb.loc[q, "season"]))

    per_game = pd.read_sql(
        """SELECT pg.player_id, pg.season, pg.game_id, pg.actual,
                  COALESCE(d.xft, 0) AS xfta, pg.poss
           FROM (SELECT p.finisher_player_id AS player_id, p.season,
                        p.game_id, SUM(p.sfta) AS actual, COUNT(*) AS poss,
                        g.date
                 FROM possessions p
                 JOIN games g ON g.league = p.league AND g.game_id = p.game_id
                 WHERE p.league = ? AND p.finisher_player_id IS NOT NULL
                 GROUP BY p.finisher_player_id, p.season, p.game_id) pg
           -- expected FTs per game from the shot-diet baseline, so a
           -- player's games sum to his season
           LEFT JOIN diet_game d
             ON d.league = ? AND d.game_id = pg.game_id
            AND d.player_id = pg.player_id
           ORDER BY pg.date, pg.game_id""",
        conn, params=(league, league))

    pgfg = pd.read_sql(
        "SELECT player_id, game_id, season, act_fg_pts, exp_fg_pts "
        "FROM player_game_shot_value WHERE league = ?", conn,
        params=(league,))
    fg_by_key = {
        (r.player_id, r.game_id, r.season): (
            round(float(r.act_fg_pts), 2), round(float(r.exp_fg_pts), 2))
        for r in pgfg.itertuples()
    }

    # ------------------------------------------------- shot zones (FGA)
    zones = pd.read_sql(
        """SELECT player_id, season, zone_basic AS zone, zone_area AS area,
                  COUNT(*) AS n
           FROM shots WHERE league = ? AND zone_basic IS NOT NULL
           GROUP BY player_id, season, zone_basic, zone_area""",
        conn, params=(league,))

    league_zone_out = {}
    for season, grp in zones.groupby("season"):
        agg = grp.groupby(["zone", "area"])["n"].sum().reset_index()
        total = agg["n"].sum()
        league_zone_out[season] = [
            {"zone": r.zone, "area": r.area, "share": round(r.n / total, 4)}
            for r in agg.itertuples()
        ]

    # --------------------------------------------- shooting-foul detail
    fouls_agg = pd.read_sql(
        """SELECT finisher_player_id AS player_id, season,
                  SUM(ft_and1) AS and1, SUM(ft_sf2) AS sf2,
                  SUM(ft_sf3) AS sf3
           FROM possessions
           WHERE league = ? AND finisher_player_id IS NOT NULL
           GROUP BY finisher_player_id, season""",
        conn, params=(league,))
    fouls_by_key = {
        (r.player_id, r.season): (int(r.and1), int(r.sf2), int(r.sf3))
        for r in fouls_agg.itertuples()
    }

    and1_zones = pd.read_sql(
        """SELECT a.player_id, p.season, s.zone_basic AS zone,
                  s.zone_area AS area, COUNT(*) AS n
           FROM and1_shots a
           JOIN possessions p
             ON p.league = a.league AND p.game_id = a.game_id
            AND p.possession_number = a.possession_number
            AND p.finisher_player_id = a.player_id
           JOIN shots s ON s.league = a.league AND s.game_id = a.game_id
            AND s.event_id = a.event_id
           WHERE a.league = ? AND s.zone_basic IS NOT NULL
           GROUP BY a.player_id, p.season, s.zone_basic, s.zone_area""",
        conn, params=(league,))
    a1_grouped = and1_zones.groupby(["player_id", "season"])

    players_by_season: dict = {}
    pg_grouped = per_game.groupby(["player_id", "season"])
    zn_grouped = zones.groupby(["player_id", "season"])
    for (pid, season) in qual_keys:
        entry: dict = {}
        try:
            g = pg_grouped.get_group((pid, season))
            entry["games"] = [
                [int(a), round(float(x), 3), int(n),
                 *fg_by_key.get((pid, gid, season), (0.0, 0.0))]
                for a, x, n, gid in zip(g["actual"], g["xfta"], g["poss"],
                                        g["game_id"])
            ]
        except KeyError:
            entry["games"] = []
        try:
            z = zn_grouped.get_group((pid, season))
            total = z["n"].sum()
            entry["zones"] = [
                {"zone": zr.zone, "area": zr.area, "n": int(zr.n),
                 "share": round(zr.n / total, 4)}
                for zr in z.itertuples()
            ]
        except KeyError:
            entry["zones"] = []
        and1, sf2, sf3 = fouls_by_key.get((pid, season), (0, 0, 0))
        try:
            az = a1_grouped.get_group((pid, season))
            located = int(az["n"].sum())
            a1z = [
                {"zone": zr.zone, "area": zr.area, "n": int(zr.n),
                 "share": round(zr.n / located, 4)}
                for zr in az.itertuples()
            ]
        except KeyError:
            located, a1z = 0, []
        entry["fouls"] = {"and1": and1, "sf2": sf2, "sf3": sf3,
                          "located": located, "zones": a1z}
        players_by_season.setdefault(season, {})[str(pid)] = entry

    # --------------------------------------------------- calibration
    pred = pd.read_sql(
        """SELECT pr.sfta, pr.xfta, pr.season
           FROM predictions_poss pr
           JOIN possessions p
             ON p.league = pr.league AND p.game_id = pr.game_id
            AND p.possession_number = pr.possession_number
           WHERE pr.league = ? AND p.finisher_player_id IS NOT NULL""",
        conn, params=(league,))
    # calibration of the FTAOE baseline at the grain it is used: qualified
    # player-seasons in deciles of expected FTs per possession (shot diet)
    # against actual; `pred` (context model) still supplies the league rates
    cq_cal = lb[q].assign(
        exp_rate=lambda d: d.xfta_total / d.possessions,
        act_rate=lambda d: d.actual_fta_from_fouls / d.possessions)
    cq_cal["bin"] = pd.qcut(cq_cal["exp_rate"], 10, duplicates="drop")
    cal = (cq_cal.groupby("bin", observed=True)
           .agg(pred_mean=("exp_rate", "mean"),
                actual_mean=("act_rate", "mean"),
                n=("exp_rate", "size"))
           .reset_index(drop=True))
    calibration = [
        {"pred": round(r.pred_mean, 4), "actual": round(r.actual_mean, 4),
         "n": int(r.n)}
        for r in cal.itertuples()
    ]

    # fold lifts, recomputed from the OOF predictions (finisher universe)
    fold_lifts, lift_n, lift_sum = [], 0, 0.0
    for season, grp in pred.groupby("season"):
        d_pred = poisson_deviance(grp.sfta, grp.xfta)
        d_base = poisson_deviance(grp.sfta, np.full(len(grp),
                                                    grp.sfta.mean()))
        lift = (d_base - d_pred) / d_base * 100
        fold_lifts.append({"season": season, "liftPct": round(lift, 2)})
        lift_sum += lift * len(grp)
        lift_n += len(grp)

    # ------------------------------------------------ referees + teams
    # Officials over the expected whistle of the games they worked (shot
    # diet plus the players on the floor), shrunk to what the data supports:
    # referees.py, mirrored from the NBA pipeline. Keyed by name (no stable id
    # in these feeds), slugged.
    ref_poss = pd.read_sql(
        """WITH ls AS (
               SELECT game_id, ABS(score_margin) AS m,
                      ROW_NUMBER() OVER (
                          PARTITION BY game_id
                          ORDER BY period DESC, secs_remaining ASC
                      ) AS rn
               FROM shots WHERE league = ? AND score_margin != 0
           ),
           fm AS (
               SELECT game_id,
                      CASE WHEN m <= 5 THEN 'close'
                           WHEN m <= 12 THEN 'mid'
                           ELSE 'blowout' END AS bucket
               FROM ls WHERE rn = 1
           )
           SELECT p.game_id, p.period, p.sfta, p.season, fm.bucket,
                  p.finisher_player_id AS pid
           FROM possessions p
           JOIN fm ON fm.game_id = p.game_id
           WHERE p.league = ?""",
        conn, params=(league, league))
    ref_poss["q"] = np.where(ref_poss["period"] <= 4,
                             "Q" + ref_poss["period"].astype(int).astype(str),
                             "OT")
    ref_diet = pd.read_sql(
        "SELECT game_id, player_id AS pid, xft FROM diet_game WHERE league = ?",
        conn, params=(league,))
    _g = pd.read_sql("SELECT game_id, ref1, ref2, ref3 FROM games WHERE league = ?",
                     conn, params=(league,))
    ref_games = pd.concat(
        [_g[["game_id", f"ref{n}"]].set_axis(["game_id", "ref_name"], axis=1)
         for n in (1, 2, 3)], ignore_index=True)
    ref_games = ref_games[ref_games.ref_name.notna() & (ref_games.ref_name != "")]
    ref_games["ref_id"] = ref_games.ref_name.map(ref_key)
    # display a person's Latin spelling where the feed has one
    latin = ref_games[ref_games.ref_name.map(lambda n: n.isascii())]
    shown = latin.groupby("ref_id").ref_name.agg(lambda n: n.mode().iat[0])
    ref_games["ref_name"] = ref_games.ref_id.map(shown).fillna(ref_games.ref_name)
    referees_out, ref_profiles, referees_meta = referees.build(
        ref_poss, ref_diet, ref_games, qualify_games=QUALIFY_REF_GAMES)

    tm = pd.read_sql(
        """SELECT p.season, p.offense_team_id, p.sfta, pr.xfta,
                  g.home_team_id, g.away_team_id
           FROM possessions p
           JOIN games g ON g.league = p.league AND g.game_id = p.game_id
           JOIN predictions_poss pr
             ON pr.league = p.league AND pr.game_id = p.game_id
            AND pr.possession_number = p.possession_number
           WHERE p.league = ? AND p.offense_team_id IS NOT NULL
             AND p.finisher_player_id IS NOT NULL""",
        conn, params=(league,))
    tm["def_team_id"] = np.where(tm.offense_team_id == tm.home_team_id,
                                 tm.away_team_id, tm.home_team_id)

    def team_side(key):
        agg = (tm.groupby([key, "season"])
               .agg(poss=("sfta", "size"), fta=("sfta", "sum"),
                    xfta=("xfta", "sum")).reset_index())
        agg["per100"] = (agg["fta"] - agg["xfta"]) / agg["poss"] * 100
        return agg

    off, dfn = team_side("offense_team_id"), team_side("def_team_id")

    # Team-level shot-value suite: what the looks a team generated were worth
    # (selection) and FG points over expected (making), per 100 offensive
    # possessions, centred on the possession-weighted league mean of the
    # season so the columns read vs-league like drawn/conceded do. Sourced
    # from team_shot_value (written by shot_value.py, same xFG cross-fit as
    # the player suite) with the same possession denominator as the FT
    # columns above.
    tsv = pd.read_sql(
        """SELECT season, team_id, off_act_fg_pts, off_exp_fg_pts
           FROM team_shot_value WHERE league = ?""",
        conn, params=(league,))
    tsv_by_key = {(str(r.team_id), r.season): (float(r.off_exp_fg_pts),
                                               float(r.off_act_fg_pts))
                  for r in tsv.itertuples()}

    teams_out = {}
    for season in sorted(off.season.unique()):
        o = off[off.season == season].set_index("offense_team_id")
        d = dfn[dfn.season == season].set_index("def_team_id")
        valid = [tid for tid in o.index
                 if tid in d.index and (str(tid), season) in tsv_by_key]
        lg_poss = sum(int(o.loc[t, "poss"]) for t in valid)
        lg_exp = sum(tsv_by_key[(str(t), season)][0] for t in valid)
        lg_act = sum(tsv_by_key[(str(t), season)][1] for t in valid)
        lg_sv100 = lg_exp / lg_poss * 100 if lg_poss else 0.0
        lg_make100 = (lg_act - lg_exp) / lg_poss * 100 if lg_poss else 0.0
        rows = []
        for tid in o.index:
            if tid not in d.index:
                continue
            poss = int(o.loc[tid, "poss"])
            row = {"team": abbrev.get(str(tid), str(tid)),
                   "drawn": round(float(o.loc[tid, "per100"]), 2),
                   "conceded": round(float(d.loc[tid, "per100"]), 2),
                   "poss": poss}
            t = tsv_by_key.get((str(tid), season))
            if t and poss:
                exp_pts, act_pts = t
                row["sv100"] = round(exp_pts / poss * 100 - lg_sv100, 2)
                row["make100"] = round(
                    (act_pts - exp_pts) / poss * 100 - lg_make100, 2)
            else:
                row["sv100"] = None
                row["make100"] = None
            rows.append(row)
        rows.sort(key=lambda r: -r["drawn"])
        teams_out[season] = rows

    # --------------------------------------------------- shot value
    sv = pd.read_sql(
        """SELECT player_id, player_name, position, season, possessions,
                  fga, fg_pct, xfg_pct, shot_making_oe, xpoints_per_shot,
                  fg_pts_oe_per100, ft_pct, ftaoe, points_oe_per100
           FROM shot_value WHERE league = ? AND possessions >= ?""",
        conn, params=(league, QUALIFY))
    # EuroLeague projection for domestic leagues (build_translation.py)
    tr = pd.read_sql(
        "SELECT season, player_id, el_proj, el_lo, el_hi, el_actual, el_fga "
        "FROM translation WHERE league = ?", conn, params=(league,))
    tr_by_key = {(r.player_id, r.season): r for r in tr.itertuples()}

    def el_fields(pid, season):
        t = tr_by_key.get((pid, season))
        if t is None:
            return {}
        out = {"elProj": round(t.el_proj, 1), "elLo": round(t.el_lo, 1),
               "elHi": round(t.el_hi, 1)}
        if pd.notna(t.el_actual):
            out.update(elActual=round(t.el_actual, 1), elFga=int(t.el_fga))
        return out

    shot_value_out = {}
    for season, grp in sv.groupby("season"):
        rows = [
            {
                "id": r.player_id,
                "name": r.player_name,
                "pos": (r.position or "").split("-")[0] or None,
                "teams": teams_by_key.get((r.player_id, season), []),
                "fga": int(r.fga),
                "poss": int(r.possessions),
                "fgPct": round(r.fg_pct * 100, 1),
                "xfgPct": round(r.xfg_pct * 100, 1),
                "makeOE": round(r.shot_making_oe, 1),
                "xptsShot": round(r.xpoints_per_shot, 3),
                "fgPoe100": round(r.fg_pts_oe_per100, 2),
                "ftPct": round(r.ft_pct, 3),
                "ftaoe100": round(r.ftaoe / r.possessions * 100, 1),
                "poe100": round(r.points_oe_per100, 2),
                **el_fields(r.player_id, season),
            }
            for r in grp.itertuples()
        ]
        rows.sort(key=lambda x: -x["poe100"])
        shot_value_out[season] = rows

    # ---------------------------------------------------- reliability
    # split-half: odd/even game split of each qualified player-season's
    # per-100 rate, correlated across players
    halves = []
    for (pid, season), g in pg_grouped:
        if (pid, season) not in qual_keys or len(g) < 10:
            continue
        odd, even = g.iloc[::2], g.iloc[1::2]
        if odd.poss.sum() == 0 or even.poss.sum() == 0:
            continue
        halves.append(((odd.actual.sum() - odd.xfta.sum())
                       / odd.poss.sum() * 100,
                       (even.actual.sum() - even.xfta.sum())
                       / even.poss.sum() * 100))
    hdf = pd.DataFrame(halves, columns=["a", "b"])
    full_season_r = round(float(hdf.a.corr(hdf.b)), 3) if len(hdf) > 2 \
        else None

    # year-over-year: per-100 correlation for players qualified in both
    yoy_pairs = []
    seasons_sorted = sorted(lb["season"].unique())
    qlb = lb[lb["possessions"] >= QUALIFY]
    for s1, s2 in zip(seasons_sorted, seasons_sorted[1:]):
        a = qlb[qlb.season == s1].set_index("player_id")["ftaoe_per_100"]
        b = qlb[qlb.season == s2].set_index("player_id")["ftaoe_per_100"]
        common = a.index.intersection(b.index)
        if len(common) > 2:
            yoy_pairs.append({
                "pair": f"{s1[2:4]+s1[5:]}→{s2[2:4]+s2[5:]}",
                "r": round(float(a[common].corr(b[common])), 3),
                "n": int(len(common))})
    yoy_mean = round(float(np.mean([p["r"] for p in yoy_pairs])), 3) \
        if yoy_pairs else None

    # what the shot-diet baseline does and does not take out, on the
    # qualified rows as shipped (heights exist for EL and EuroCup only)
    cq = qlb.merge(pd.read_sql(
        "SELECT season, player_id, height_cm FROM player_season "
        "WHERE league = ?", conn, params=(league,)).drop_duplicates(
            ["season", "player_id"]), on=["season", "player_id"], how="left")
    cq["raw100"] = cq.actual_fta_from_fouls / cq.possessions * 100
    has_h = cq.height_cm.notna().sum() > 50
    confounds = {
        "rawRateCorr": round(float(cq.ftaoe_per_100.corr(cq.raw100)), 3),
        "heightCorr": round(float(cq.ftaoe_per_100.corr(cq.height_cm)), 3)
        if has_h else None,
        "rawHeightCorr": round(float(cq.raw100.corr(cq.height_cm)), 3)
        if has_h else None,
    }
    rates = pd.read_sql("SELECT season, zone, rate FROM diet_rates "
                        "WHERE league = ?", conn, params=(league,))
    diet_rates = {
        season: {r.zone: round(float(r.rate), 3) for r in grp.itertuples()}
        for season, grp in rates.groupby("season")}

    # league FT% (all free throws), FTA-weighted actual
    lf = pd.read_sql(
        "SELECT SUM(ftm) AS m, SUM(fta) AS a FROM player_season_ft "
        "WHERE league = ?", conn, params=(league,))
    league_ft = round(float(lf.m[0] / lf.a[0]), 3)

    # --------------------------------------------------------- meta
    all_seasons = sorted(lb["season"].unique().tolist())
    qual_counts = (lb[lb["possessions"] >= QUALIFY]
                   .groupby("season").size().to_dict())
    default_season = next(
        (s2 for s2 in reversed(all_seasons) if qual_counts.get(s2, 0) >= 40),
        all_seasons[-1])
    meta = {
        "league": league,
        "seasons": all_seasons,
        "defaultSeason": default_season,
        "qualifyPossessions": QUALIFY,
        "nPossessions": int(len(pred)),
        "leagueRatePer100": round(float(pred["sfta"].mean()) * 100, 2),
        "leagueRateBySeason": {
            season: round(float(grp["sfta"].mean()) * 100, 2)
            for season, grp in pred.groupby("season")},
        "modelLiftPct": round(lift_sum / lift_n, 2),
        "foldLifts": fold_lifts,
        "leagueFt": league_ft,
        "reliability": {
            "fullSeasonR": full_season_r,
            "yoyMeanR": yoy_mean,
            "yoyPairs": yoy_pairs,
        },
        "confounds": confounds,
        "dietRates": diet_rates,
        "referees": referees_meta,
    }
    tm = conn.execute("SELECT meta FROM translation_meta WHERE league = ?",
                      (league,)).fetchone()
    if tm:
        meta["translation"] = json.loads(tm[0])

    out = {
        "meta": meta,
        "leaderboard": leaderboard,
        "distributions": distributions,
        "leagueZones": league_zone_out,
        "calibration": calibration,
        "referees": referees_out,
        "refProfiles": ref_profiles,
        "teams": teams_out,
        "shotValue": shot_value_out,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core = OUT_DIR / f"data-{league}.json"
    core.write_text(json.dumps(out, separators=(",", ":")))
    print(f"Wrote {core} ({core.stat().st_size / 1e6:.2f} MB)")
    for season, payload in sorted(players_by_season.items()):
        chunk = OUT_DIR / f"players-{league}-{season}.json"
        chunk.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"  {chunk.name}: {len(payload)} players "
              f"({chunk.stat().st_size / 1e6:.2f} MB)")
    print(f"  leaderboard rows: {len(leaderboard)}, "
          f"league rate/100: {meta['leagueRatePer100']}, "
          f"lift: {meta['modelLiftPct']}%")


if __name__ == "__main__":
    for old in OUT_DIR.glob("players-*.json"):
        old.unlink()
    for league in config.LEAGUES:
        export_league(league)
