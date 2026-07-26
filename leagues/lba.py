"""Lega Basket Serie A (Italy) adapter.

Source: legabasket.it's own open JSON API (no auth, no key). fibalivestats
hosts LBA too, but its gameIds are not publicly enumerable; legabasket.it
re-serves the same feed under integer ids that ARE enumerable per
season/matchday, with play-by-play carrying shot x/y plus fouls and free
throws.

Coordinates are 0-100 percentages of the FULL court and are true-court:
a team attacks opposite baskets in the two halves, so the attacking hoop
is resolved per shot by which half of the court the shot sits in (x >= 50
-> right basket) and folded to the normalized frame (meters from hoop
center, x lateral, y toward midcourt). Player and team ids are stable
integers. The period clock comes from `print_time` (MM:SS remaining).
"""

import config
from leagues.common import (cache_exists, cache_read, cache_write,
                            fetch_json, titlecase_name)

LEAGUE = "LBA"
_UA = {"User-Agent": "Mozilla/5.0"}

# description -> (etype, shot_value, shot_made)
_DESC = {
    "2 punti segnato": ("fg", 2, 1), "2 punti sbagliato": ("fg", 2, 0),
    "3 punti segnato": ("fg", 3, 1), "3 punti sbagliato": ("fg", 3, 0),
    "Tiro libero segnato": ("ft", 1, 1), "Tiro libero sbagliato": ("ft", 1, 0),
    "Rimbalzo offensivo": ("reb_off", None, None),
    "Rimbalzi offensivi di squadra": ("reb_off", None, None),
    "Rimbalzo difensivo": ("reb_def", None, None),
    "Rimbalzi difensivi di squadra": ("reb_def", None, None),
    "Palla persa": ("turnover", None, None),
    "Palle perse di squadra": ("turnover", None, None),
    "Palla recuperata": ("steal", None, None),
    "Assist": ("assist", None, None),
    "Stoppata": ("block", None, None),
    "Stoppata subita": ("other", None, None),
    "Ingresso": ("sub", None, None), "Uscita": ("sub", None, None),
    "Time Out": ("timeout", None, None),
    "Palla contesa": ("jumpball", None, None),
    "Inizio Tempo": ("period_start", None, None),
    "Fine Tempo": ("period_end", None, None),
    "Fallo Subito": ("foul_drawn", None, None),
}

COURT_LEN_M = 28.0
COURT_WID_M = 15.0
HOOP_FROM_BASELINE_M = config.FIBA_HOOP_TO_BASELINE


def _api(path, **params):
    return fetch_json(f"{config.LBA_API}/api/{path}", headers=_UA,
                      params=params, sleep=config.LBA_SLEEP)


def _foul_kind(qualifier):
    q = (qualifier or "").lower()
    if "offensiv" in q:
        return "offensive"
    if "tecnico" in q:
        return "technical"
    if "antisportivo" in q:
        return "unsportsmanlike"
    return "personal"  # personale, tiro (shooting), and untyped


def _clock(a):
    """Seconds remaining in the period. `print_time` (MM:SS remaining) is
    the clean source when present; older seasons (e.g. 2020-21) omit it, so
    fall back to `minute`/`seconds` (elapsed) against the period length."""
    pt = a.get("print_time")
    if pt and ":" in pt:
        mm, ss = pt.split(":")[:2]
        try:
            return int(mm) * 60 + int(ss)
        except ValueError:
            pass
    mn, sc, per = a.get("minute"), a.get("seconds"), a.get("period")
    if mn is not None and sc is not None:
        plen = 600 if (per or 1) <= 4 else 300  # FIBA: 10' regulation, 5' OT
        return max(0, plen - (int(mn) * 60 + int(sc)))
    return None


# ------------------------------------------------------------------ pull

def _calendar(season):
    champ = config.LBA_CHAMP[season]
    part = ("LBA", f"calendar_{champ}.json")
    # the current season keeps adding games; never trust a frozen list
    if season == config.SEASONS[-1] or not cache_exists(*part):
        days = _api("championships/get-championships-calendar-by-id",
                    id=champ).get("filters", {}).get("days", [])
        matches = []
        for d in days:
            code = d.get("code")
            if code is None:
                continue
            day = _api("championships/get-championships-calendar-by-id",
                       id=champ, d=code)
            matches.extend(day.get("matches") or [])
        cache_write(matches, *part)
    return cache_read(*part)


def list_games(season):
    """Finished regular-season games for the season."""
    games = []
    for m in _calendar(season):
        if str(m.get("game_status")) != "2":  # 2 = finished
            continue
        games.append({
            "game_id": str(m["id"]), "game_code": m["id"],
            "league": LEAGUE, "season": season,
            "date": (m.get("match_datetime") or "")[:10],
            "home_team_id": str(m["h_team_id"]),
            "away_team_id": str(m["v_team_id"]),
            "home_team_name": m.get("h_team_name"),
            "away_team_name": m.get("v_team_name"),
            "home_score": m.get("home_final_score"),
            "away_score": m.get("visitor_final_score"),
            "ref1": None, "ref2": None, "ref3": None,  # from match header
        })
    return sorted(games, key=lambda g: (g["date"], g["game_code"]))


def pull_game(season, game_code):
    for kind, path in (
            ("pbp", "championships/get-championships-matches-play-by-play"),
            ("match", "championships/get-championships-matches-by-id")):
        part = ("LBA", kind, f"{game_code}.json")
        if not cache_exists(*part):
            cache_write(_api(path, id=game_code), *part)
    return True


def referees(game):
    try:
        m = cache_read("LBA", "match", f"{game['game_id']}.json")
    except FileNotFoundError:
        return [None, None, None]
    out = []
    for r in (m.get("referees") or [])[:3]:
        nm = f"{r.get('name', '')} {titlecase_name(r.get('surname', ''))}"
        out.append(nm.strip() or None)
    return (out + [None, None, None])[:3]


# ------------------------------------------------------------- normalize

def _to_meters(x, y, attacks_right):
    xm_court = x / 100.0 * COURT_LEN_M
    ym_court = y / 100.0 * COURT_WID_M
    if attacks_right:
        y_m = (COURT_LEN_M - HOOP_FROM_BASELINE_M) - xm_court
        x_m = ym_court - COURT_WID_M / 2
    else:
        y_m = xm_court - HOOP_FROM_BASELINE_M
        x_m = COURT_WID_M / 2 - ym_court
    return x_m, y_m


def normalize_game(season, game):
    gid = game["game_id"]
    pbp = cache_read("LBA", "pbp", f"{gid}.json")["pbp"]
    actions = sorted(pbp.get("actions") or [], key=lambda a: a["order"])
    home_id = str(pbp.get("ht_id"))

    game["ref1"], game["ref2"], game["ref3"] = referees(game)

    events, shots = [], []
    eid = 0
    sh = sa = 0  # running score, reconstructed (the pbp `score` string is
    #             unreliable — it undercounts free throws)
    for a in actions:
        desc = a.get("description")
        if desc == "Fallo commesso":
            etype, val, made = "foul", None, None
            foul_kind = _foul_kind(a.get("action_1_qualifier_description"))
        else:
            mapped = _DESC.get(desc)
            etype, val, made = mapped if mapped else ("other", None, None)
            foul_kind = None

        tid = a.get("team_id")
        team_id = str(tid) if tid is not None else None
        pid = a.get("player_id")

        if etype in ("fg", "ft") and made:
            pts = val
            if team_id == home_id:
                sh += pts
            else:
                sa += pts

        eid += 1
        secs = _clock(a)
        events.append({
            "game_id": gid, "event_id": eid, "src_num": a.get("order"),
            "period": a.get("period"), "secs_remaining": secs,
            "team_id": team_id,
            "player_id": str(pid) if pid else None,
            "player_name": " ".join(
                x for x in (a.get("player_name"), a.get("player_surname")) if x
            ) or None,
            "etype": etype, "shot_value": val, "shot_made": made,
            "foul_kind": foul_kind, "foul_fts": 0,
            "score_home": sh, "score_away": sa,
        })

        if etype == "fg" and a.get("x") is not None:
            # LBA coordinates are true-court and flip at halftime, so the
            # attacking hoop is resolved PER SHOT by which half it's in
            # (x >= 50 -> right basket) rather than a static per-team side.
            x_m, y_m = _to_meters(a["x"], a["y"], a["x"] >= 50)
            shots.append({
                "game_id": gid, "event_id": eid,
                "player_id": str(pid) if pid else None, "team_id": team_id,
                "period": a.get("period"), "secs_remaining": secs,
                "shot_made": made, "shot_value": val,
                "x_m": x_m, "y_m": y_m,
                "second_chance": 0,   # derived in build_possessions
                "off_turnover": 0,    # derived in build_possessions
                "dunk": 1 if a.get("dunk") else 0,
            })
    return events, shots
