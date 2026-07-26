"""Liga ACB adapter.

Data source: api2.acb.com (the JSON backend of live.acb.com / acb.com),
authenticated with a static X-APIKEY from their public JS bundle.

Regular season = calendar rounds with subphaseId == null. Team identity
across seasons is clubId (the per-edition team id changes every year).
Shot rows carry no event id that joins PBP reliably; the tuple
(quarter, minute, second, playerLicenseId, playType) joins 1:1.
"""

import config
from leagues.common import (cache_exists, cache_read, cache_write,
                            fetch_json)

LEAGUE = "ACB"

# playType -> (etype, shot_value, shot_made, dunk)
_SHOT_TYPES = {
    93: ("fg", 2, 1, 0), 97: ("fg", 2, 0, 0),
    94: ("fg", 3, 1, 0), 98: ("fg", 3, 0, 0),
    100: ("fg", 2, 1, 1), 533: ("fg", 2, 0, 1),
    92: ("ft", 1, 1, 0), 96: ("ft", 1, 0, 0),
}

# personal fouls typed by awarded FTs
_PERSONAL_FOUL_FTS = {159: 0, 160: 1, 161: 2, 162: 3}
_OFFENSIVE_FOUL = {109}
_UNSPORTSMANLIKE = set(range(165, 170))
_TECHNICAL_BENCH = set(range(170, 178)) | set(range(537, 564))

_SIMPLE = {
    101: "reb_off", 104: "reb_def", 106: "turnover", 103: "steal",
    107: "assist", 108: "assist", 102: "block", 105: "other",
    110: "foul_drawn", 119: "other",
    112: "sub", 115: "sub", 113: "timeout", 118: "timeout",
    121: "period_start", 116: "period_end", 122: "period_start",
    123: "period_end",
}
_VIOLATION_TURNOVERS = set(range(1, 16))


def _hdr():
    return {"X-APIKEY": config.ACB_API_KEY, "Accept": "application/json"}


# ------------------------------------------------------------------ pull

def _calendar(season):
    ed = config.ACB_EDITION[season]
    part = ("ACB", f"calendar_{ed}.json")
    # the current season keeps adding games; never trust a frozen list
    if season == config.SEASONS[-1] or not cache_exists(*part):
        cache_write(fetch_json(
            f"{config.ACB_API}/api/seasondata/Calendar/season-calendar",
            headers=_hdr(),
            params={"competitionId": 1, "editionId": ed},
            sleep=config.ACB_SLEEP), *part)
    return cache_read(*part)


def list_games(season):
    """Finished regular-season games (rounds with subphaseId == null)."""
    cal = _calendar(season)
    club = {t["id"]: t for t in cal.get("teams", [])}
    games = []
    for rnd in cal.get("rounds", []):
        if rnd.get("subphaseId") is not None:
            continue  # playoffs
        for m in rnd.get("matches", []):
            if m.get("matchStatus") != "FINALIZED":
                continue
            ht, at = club.get(m["homeTeamId"]), club.get(m["awayTeamId"])
            if not ht or not at:
                continue
            games.append({
                "game_id": str(m["id"]), "game_code": m["id"],
                "league": LEAGUE, "season": season,
                "date": (m.get("startDateTime") or "")[:10],
                "home_team_id": str(ht["clubId"]),
                "away_team_id": str(at["clubId"]),
                "home_team_name": ht["fullName"],
                "away_team_name": at["fullName"],
                "home_score": m.get("homeScore"),
                "away_score": m.get("awayScore"),
                "ref1": None, "ref2": None, "ref3": None,  # from boxscore
            })
    return sorted(games, key=lambda g: (g["date"], g["game_code"]))


def pull_game(season, match_id):
    """Fetch PBP + shots + boxscore (referees) into cache."""
    for kind, endpoint in (
            ("pbp", "/api/matchdata/PlayByPlay/play-by-play"),
            ("shots", "/api/matchdata/MatchShots/match-shots"),
            ("box", "/api/matchdata/Result/boxscores")):
        part = ("ACB", kind, f"{match_id}.json")
        if not cache_exists(*part):
            cache_write(fetch_json(f"{config.ACB_API}{endpoint}",
                                   headers=_hdr(),
                                   params={"matchId": match_id},
                                   sleep=config.ACB_SLEEP), *part)
    return True


def referees(game):
    """Boxscore referee names for a pulled game."""
    try:
        box = cache_read("ACB", "box", f"{game['game_id']}.json")
    except FileNotFoundError:
        return [None, None, None]
    refs = [r for r in (box.get("referees") or []) if r]
    return (refs + [None, None, None])[:3]


# ------------------------------------------------------------- normalize

def _classify(play_type):
    """playType -> (etype, shot_value, shot_made, foul_kind, foul_fts, dunk)"""
    if play_type in _SHOT_TYPES:
        etype, val, made, dunk = _SHOT_TYPES[play_type]
        return etype, val, made, None, None, dunk
    if play_type in _PERSONAL_FOUL_FTS:
        return "foul", None, None, "personal", _PERSONAL_FOUL_FTS[play_type], 0
    if play_type in _OFFENSIVE_FOUL:
        return "foul", None, None, "offensive", 0, 0
    if play_type in _UNSPORTSMANLIKE:
        return "foul", None, None, "unsportsmanlike", None, 0
    if play_type in _TECHNICAL_BENCH:
        return "foul", None, None, "technical", None, 0
    if play_type in _SIMPLE:
        return _SIMPLE[play_type], None, None, None, None, 0
    if play_type in _VIOLATION_TURNOVERS:
        return "turnover", None, None, None, None, 0
    return "other", None, None, None, None, 0


def normalize_game(season, game):
    """Cached raw JSON -> (events, shots) in the normalized schema."""
    gid = game["game_id"]
    pbp = cache_read("ACB", "pbp", f"{gid}.json")
    shots_raw = cache_read("ACB", "shots", f"{gid}.json")

    home_id, away_id = game["home_team_id"], game["away_team_id"]
    plays = sorted(pbp.get("plays", []), key=lambda p: p["order"])

    events, by_key = [], {}
    eid = 0
    for p in plays:
        etype, val, made, foul_kind, foul_fts, dunk = _classify(
            p.get("playType"))
        eid += 1
        lic = p.get("playerLicenseId")
        row = {
            "game_id": gid, "event_id": eid, "src_num": p.get("order"),
            "period": p.get("quarter"),
            "secs_remaining": (p.get("minute") or 0) * 60
                              + (p.get("second") or 0),
            "team_id": home_id if p.get("local") else away_id,
            "player_id": str(lic) if lic else None,
            "player_name": p.get("playerName") or None,
            "etype": etype, "shot_value": val, "shot_made": made,
            "foul_kind": foul_kind, "foul_fts": foul_fts,
            "score_home": p.get("scoreHome"), "score_away": p.get("scoreAway"),
        }
        events.append(row)
        if etype == "fg":
            key = (p.get("quarter"), p.get("minute"), p.get("second"),
                   lic, p.get("playType"))
            by_key.setdefault(key, []).append((row, dunk))  # FIFO on collision

    shots = []
    for s in shots_raw.get("shotPoints", []):
        if s.get("playType") not in (93, 94, 97, 98, 100, 533):
            continue  # FT rows carry sentinel (0,0) coords
        key = (s.get("quarter"), s.get("minute"), s.get("second"),
               s.get("playerLicenseId"), s.get("playType"))
        bucket = by_key.get(key)
        if not bucket:
            continue  # orphan shotPoint; possession side still counts the FGA
        ev, dunk = bucket.pop(0)  # consume in order; distinct events, distinct rows
        shots.append({
            "game_id": gid, "event_id": ev["event_id"],
            "player_id": ev["player_id"], "team_id": ev["team_id"],
            "period": ev["period"], "secs_remaining": ev["secs_remaining"],
            "shot_made": ev["shot_made"], "shot_value": ev["shot_value"],
            # posX = toward midcourt, posY = lateral, both mm from hoop
            "x_m": (s.get("posY") or 0) / 1000.0,
            "y_m": (s.get("posX") or 0) / 1000.0,
            "second_chance": 0,       # derived later in build_possessions
            "off_turnover": 0,        # derived later in build_possessions
            "dunk": dunk,
        })
    return events, shots
