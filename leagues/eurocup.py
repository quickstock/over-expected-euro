"""EuroCup adapter.

Same platform, same endpoints, same schema as EuroLeague (`euroleague.py`)
— Euroleague Basketball SA runs both competitions on api-live.euroleague.net
and live.euroleague.net, distinguished only by the competition letter ("U"
vs "E") baked into the season code (e.g. "U2024" vs "E2024") and API path.
Verified identical response shapes for games/people/PlayByPlay/Points, so
this is a straight copy of euroleague.py with EL -> EUC, E -> U everywhere
that matters, kept as a separate file (not a shared factory) to match this
project's one-file-per-adapter convention and avoid touching the
already-proven EL adapter.

Gotchas handled here: live.euroleague.net 403s python UAs (send browser
UA); every string field is space-padded; PBP quarter arrays are the true
event order (NUMBEROFPLAY is not globally monotone) but NUMBEROFPLAY is
the join key to Points.NUM_ANOT.
"""

import config
from leagues.common import (cache_exists, cache_read, cache_write,
                            fetch_json, titlecase_name)

LEAGUE = "EUC"

_PLAYTYPE_MAP = {
    "2FGM": ("fg", 2, 1), "2FGA": ("fg", 2, 0),
    "3FGM": ("fg", 3, 1), "3FGA": ("fg", 3, 0),
    "FTM": ("ft", 1, 1), "FTA": ("ft", 1, 0),
    "O": ("reb_off", None, None), "D": ("reb_def", None, None),
    "TO": ("turnover", None, None), "ST": ("steal", None, None),
    "AS": ("assist", None, None), "FV": ("block", None, None),
    "IN": ("sub", None, None), "OUT": ("sub", None, None),
    "TOUT": ("timeout", None, None), "TOUT_TV": ("timeout", None, None),
    "JB": ("jumpball", None, None), "TPOFF": ("jumpball", None, None),
    "BP": ("period_start", None, None), "EP": ("period_end", None, None),
    "EG": ("period_end", None, None),
}

_FOUL_KIND = {
    "CM": "personal", "OF": "offensive", "CMT": "technical",
    "CMU": "unsportsmanlike", "C": "bench", "CMD": "unsportsmanlike",
}

_QUARTER_KEYS = ["FirstQuarter", "SecondQuarter", "ThirdQuarter",
                 "ForthQuarter", "ExtraTime"]  # 'Forth' is theirs


def _hdr():
    return {"User-Agent": config.EL_UA, "Accept": "application/json"}


def _s(v):
    return v.strip() if isinstance(v, str) else v


def _clock_secs(markertime, period):
    """MARKERTIME '09:36' counts down; quarters 600s, OT 300s."""
    if not markertime or not str(markertime).strip():
        return None
    try:
        m, s = str(markertime).strip().split(":")
        return int(m) * 60 + int(s)
    except ValueError:
        return None


# ------------------------------------------------------------------ pull

def list_games(season):
    """All played regular-season games with referees. Cached."""
    code = config.EUC_SEASON_CODE[season]
    part = ("EUC", f"games_{code}.json")
    if season == config.SEASONS[-1] or not cache_exists(*part):
        items, page = [], 0
        while True:
            data = fetch_json(
                f"{config.EL_API}/v2/competitions/U/seasons/{code}/games",
                headers=_hdr(),
                params={"phaseTypeCode": "RS", "pageSize": 100,
                        "pageNumber": page},
                sleep=config.EL_SLEEP)
            items.extend(data.get("data", []))
            if len(items) >= data.get("total", 0) or not data.get("data"):
                break
            page += 1
        cache_write(items, *part)
    items = cache_read(*part)

    games = []
    for g in items:
        if not g.get("played"):
            continue
        refs = [(titlecase_name(_s(((g.get(f"referee{i}") or {}).get("name"))))
                 or None) for i in (1, 2, 3)]
        games.append({
            "game_id": f"{code}_{g['gameCode']}",
            "game_code": g["gameCode"],
            "league": LEAGUE,
            "season": season,
            "date": (g.get("utcDate") or g.get("date") or "")[:10],
            "home_team_id": _s(g["local"]["club"]["code"]),
            "away_team_id": _s(g["road"]["club"]["code"]),
            "home_team_name": _s(g["local"]["club"]["name"]),
            "away_team_name": _s(g["road"]["club"]["name"]),
            "home_score": g["local"].get("score"),
            "away_score": g["road"].get("score"),
            "ref1": refs[0], "ref2": refs[1], "ref3": refs[2],
        })
    return sorted(games, key=lambda g: (g["date"], g["game_code"]))


def pull_game(season, game_code):
    """Fetch PBP + Points into cache. Returns True if fetched or cached."""
    code = config.EUC_SEASON_CODE[season]
    pbp_part = ("EUC", "pbp", f"{code}_{game_code}.json")
    pts_part = ("EUC", "pts", f"{code}_{game_code}.json")
    params = {"gamecode": game_code, "seasoncode": code}
    if not cache_exists(*pbp_part):
        cache_write(fetch_json(f"{config.EL_LIVE}/PlayByPlay",
                               headers=_hdr(), params=params,
                               sleep=config.EL_SLEEP), *pbp_part)
    if not cache_exists(*pts_part):
        cache_write(fetch_json(f"{config.EL_LIVE}/Points",
                               headers=_hdr(), params=params,
                               sleep=config.EL_SLEEP), *pts_part)
    return True


def pull_people(season):
    """Player bios (height cm, position) for a season. Cached."""
    code = config.EUC_SEASON_CODE[season]
    part = ("EUC", f"people_{code}.json")
    if season == config.SEASONS[-1] or not cache_exists(*part):
        items, page = [], 0
        while True:
            data = fetch_json(
                f"{config.EL_API}/v2/competitions/U/seasons/{code}/people",
                headers=_hdr(),
                params={"personType": "J", "pageSize": 200,
                        "pageNumber": page},
                sleep=config.EL_SLEEP)
            items.extend(data.get("data", []))
            if len(items) >= data.get("total", 0) or not data.get("data"):
                break
            page += 1
        cache_write(items, *part)
    return cache_read(*part)


# ------------------------------------------------------------- normalize

def _pid(raw):
    """PBP ids look like 'P004194'; people codes like '004194'."""
    p = _s(raw) or ""
    if p and not p.startswith("P"):
        p = "P" + p
    return p or None


def normalize_game(season, game):
    """Cached raw JSON -> (events, shots) in the normalized schema."""
    code = config.EUC_SEASON_CODE[season]
    gc = game["game_code"]
    pbp = cache_read("EUC", "pbp", f"{code}_{gc}.json")
    pts = cache_read("EUC", "pts", f"{code}_{gc}.json")
    gid = game["game_id"]

    code_a = _s(pbp.get("CodeTeamA"))
    a_is_home = (code_a == game["home_team_id"])

    events, by_src = [], {}
    eid = 0
    for q_idx, qkey in enumerate(_QUARTER_KEYS):
        period = q_idx + 1  # ExtraTime lumped as period 5
        for e in (pbp.get(qkey) or []):
            pt = _s(e.get("PLAYTYPE")) or ""
            if pt in _FOUL_KIND:
                etype, shot_value, shot_made = "foul", None, None
                foul_kind = _FOUL_KIND[pt]
            elif pt == "RV":
                etype, shot_value, shot_made, foul_kind = ("foul_drawn",
                                                           None, None, None)
            elif pt in _PLAYTYPE_MAP:
                etype, shot_value, shot_made = _PLAYTYPE_MAP[pt]
                foul_kind = None
                if etype == "ft":
                    shot_value = 1
            else:
                etype, shot_value, shot_made, foul_kind = ("other", None,
                                                           None, None)
            pa, pb = e.get("POINTS_A"), e.get("POINTS_B")
            if pa is not None or pb is not None:
                score_home = pa if a_is_home else pb
                score_away = pb if a_is_home else pa
            else:
                score_home = score_away = None
            eid += 1
            row = {
                "game_id": gid, "event_id": eid,
                "src_num": e.get("NUMBEROFPLAY"),
                "period": period,
                "secs_remaining": _clock_secs(e.get("MARKERTIME"), period),
                "team_id": _s(e.get("CODETEAM")) or None,
                "player_id": _pid(e.get("PLAYER_ID")),
                "player_name": titlecase_name(_s(e.get("PLAYER"))) or None,
                "etype": etype, "shot_value": shot_value,
                "shot_made": shot_made, "foul_kind": foul_kind,
                "foul_fts": None,
                "score_home": score_home, "score_away": score_away,
            }
            events.append(row)
            if row["src_num"] is not None:
                by_src[row["src_num"]] = row

    shots = []
    for r in pts.get("Rows") or []:
        act = _s(r.get("ID_ACTION")) or ""
        if act not in ("2FGM", "2FGA", "3FGM", "3FGA"):
            continue  # FTM rows carry no real coords
        ev = by_src.get(r.get("NUM_ANOT"))
        if ev is None or ev["etype"] != "fg":
            continue  # orphan Points row; possession side still counts it
        cx, cy = r.get("COORD_X"), r.get("COORD_Y")
        if cx in (None, 0) and cy in (None, 0):
            continue  # sentinel (0,0): no real location for this attempt
        # NOTE: the Points feed's FASTBREAK / SECOND_CHANCE /
        # POINTS_OFF_TURNOVER flags are stamped only on MADE shots, so
        # they leak the outcome — deliberately dropped. second_chance
        # and off_turnover are derived from possession structure in
        # build_possessions.
        shots.append({
            "game_id": gid, "event_id": ev["event_id"],
            "player_id": ev["player_id"] or _pid(r.get("ID_PLAYER")),
            "team_id": ev["team_id"] or _s(r.get("TEAM")),
            "period": ev["period"],
            "secs_remaining": ev["secs_remaining"],
            "shot_made": 1 if act.endswith("M") else 0,
            "shot_value": 3 if act.startswith("3") else 2,
            "x_m": cx / 100.0,
            "y_m": cy / 100.0,
            "second_chance": 0,   # filled by build_possessions
            "off_turnover": 0,    # filled by build_possessions
            "dunk": 0,            # EuroLeague-family PBP has no dunk marker
        })
    return events, shots


def player_seasons(season):
    """people cache -> player_season rows (display data)."""
    rows = []
    for p in pull_people(season):
        person = p.get("person") or {}
        pid = _pid(person.get("code"))
        if not pid:
            continue
        rows.append({
            "league": LEAGUE, "season": season, "player_id": pid,
            "player_name": titlecase_name(_s(person.get("name"))),
            "height_cm": person.get("height") or None,
            "position": _s(p.get("positionName")) or None,
            "team_id": _s((p.get("club") or {}).get("code")),
        })
    return rows
