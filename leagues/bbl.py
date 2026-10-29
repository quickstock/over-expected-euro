"""Basketball Bundesliga (Germany) adapter.

Source: api.basketball-bundesliga.de — the league's own API. Not Genius;
its own "SCB" stats engine. Open with a static public header pair (key
must match; secret unvalidated). A clean season->games list endpoint
(no id-guessing) plus per-game coordinate play-by-play.

Play-by-play `actions` carry `seasonPlayerId`/`seasonTeamId`, `type`,
`isSuccessful`, `qualifiers`, `gameTime` (HH:MM:SS remaining), and shot
`coordinates` on a half-court-folded grid (single attacking basket, no
period flip). The per-game roster (`playerStats[].seasonPlayer`) maps the
season player id to a stable cross-season `realPlayerId`, name and
position; team `tlc` codes are the stable display ids. Running score is on
scoring actions only (homeTeamPoints/guestTeamPoints) and forward-filled
downstream.
"""

import config
from leagues.common import cache_exists, cache_read, cache_write, fetch_json

LEAGUE = "BBL"

# Half-court grid -> meters from hoop. x runs sideline to sideline over
# 0-280 (the league site draws shots in a 460x280 full-court frame), so
# 15 m = 280 units. The same scale holds lengthwise, with the baseline at
# y = 0 and the hoop 1.575 m in front of it. Checked on the court's own
# lines, every season 2020-21 to 2026-27 (the frame never changed): threes
# stop at the 6.60 m corner line and the 6.75 m arc as closely as
# EuroLeague's do, and corner shooters stand as deep as EuroLeague's, GBL's
# and the WNBA's. The old 17.5 units/m from one 2pt/3pt fit stretched the
# court by 7%: corner threes at |x| 7.45 m, 3% of shots past the sideline.
_K = 280 / 15.0
_X0, _Y0 = 140.0, 1.575 * _K
_X_MAX = 280  # the far sideline; a shot tapped outside 0.._X_MAX is off court

_SHOT_TYPE = {"TWO_POINT_THROW": 2, "THREE_POINT_THROW": 3}
_SIMPLE = {
    "TURN_OVER": "turnover", "TEAM_TURN_OVER": "turnover",
    "STEAL": "steal", "BLOCK": "block", "SUBSTITUTION": "sub",
    "TIME_OUT": "timeout", "JUMP_BALL_DURING_GAME": "jumpball",
    "START": "period_start", "END": "period_end", "RECEIVED_FOUL": "foul_drawn",
}


def _api(path, **params):
    return fetch_json(f"{config.BBL_API}/{path}", headers=config.BBL_HEADERS,
                      params=params, sleep=config.BBL_SLEEP)


def _foul_kind(quals):
    q = " ".join(quals or []).upper()
    if "OFFENSIVE" in q:
        return "offensive"
    if "UNSPORTSMANLIKE" in q or "DISQUALIF" in q:
        return "unsportsmanlike"
    if "TECHNICAL" in q or "BENCH" in q:
        return "technical"
    return "personal"


def _secs_remaining(gt):
    if not gt or ":" not in gt:
        return None
    parts = gt.split(":")
    try:
        return int(parts[-2]) * 60 + int(parts[-1])
    except (ValueError, IndexError):
        return None


# ------------------------------------------------------------------ pull

def list_games(season):
    # Skip seasons whose feed lacks shooter attribution on misses/FTs.
    if season < config.BBL_MIN_SEASON:
        return []
    data = _api("games", **{
        "stages[]": "MAIN_ROUND", "competition": "BBL",
        "seasonId": config.BBL_SEASON[season],
        "currentPage": 1, "pageSize": 500,
    })
    games = []
    for m in data.get("items", []):
        if m.get("status") != "OFFICIAL":
            continue
        ht, at = m.get("homeTeam") or {}, m.get("guestTeam") or {}
        res = m.get("result") or {}
        games.append({
            "game_id": str(m["id"]), "game_code": m["id"],
            "league": LEAGUE, "season": season,
            "date": (m.get("scheduledTime") or "")[:10],
            "home_team_id": ht.get("tlc") or str(ht.get("teamId")),
            "away_team_id": at.get("tlc") or str(at.get("teamId")),
            "home_team_name": ht.get("name"),
            "away_team_name": at.get("name"),
            "home_score": res.get("homeTeamFinalScore"),
            "away_score": res.get("guestTeamFinalScore"),
            "ref1": None, "ref2": None, "ref3": None,  # from meta
        })
    return sorted(games, key=lambda g: (g["date"], g["game_code"]))


def pull_game(season, game_code):
    for kind, path in (("stats", f"games/{game_code}/stats"),
                       ("meta", f"games/{game_code}")):
        part = ("BBL", kind, f"{game_code}.json")
        if not cache_exists(*part):
            cache_write(_api(path), *part)
    return True


def _ref_name(r):
    if not r:
        return None
    nm = r.get("name") or " ".join(
        x for x in (r.get("firstName"), r.get("lastName")) if x)
    return nm.strip() or None


def referees(game):
    try:
        m = cache_read("BBL", "meta", f"{game['game_id']}.json")
    except FileNotFoundError:
        return [None, None, None]
    return [_ref_name(m.get(f"referee{i}")) for i in (1, 2, 3)]


# ------------------------------------------------------------- normalize

def _rosters(stats):
    """seasonPlayerId -> (realPlayerId, name, position);
       seasonTeamId -> tlc."""
    players, teams = {}, {}
    meta_teams = {}
    for side in ("homeTeam", "guestTeam"):
        t = stats.get(side) or {}
        for pstat in t.get("playerStats") or []:
            sp = pstat.get("seasonPlayer") or {}
            spid = str(sp.get("id"))
            real = sp.get("realPlayerId")
            name = " ".join(x for x in (sp.get("firstName"),
                                        sp.get("lastName")) if x) or None
            players[spid] = (
                str(real) if real else spid, name,
                (sp.get("position") or "").replace("_", " ").title() or None)
            st = sp.get("seasonTeam")
            if st:
                meta_teams.setdefault(str(st), side)
    return players, meta_teams


def normalize_game(season, game):
    gid = game["game_id"]
    stats = cache_read("BBL", "stats", f"{gid}.json")
    players, _ = _rosters(stats)

    # team_id -> tlc from the game row (list_games already resolved tlc);
    # map the numeric seasonTeamId on each action to home/away tlc.
    home_tlc, away_tlc = game["home_team_id"], game["away_team_id"]
    home_stid = str((stats.get("homeTeam") or {}).get("id"))
    away_stid = str((stats.get("guestTeam") or {}).get("id"))
    tlc_of = {home_stid: home_tlc, away_stid: away_tlc}

    game["ref1"], game["ref2"], game["ref3"] = referees(game)

    actions = sorted(stats.get("actions") or [],
                     key=lambda a: a.get("orderId") or 0)

    # Early seasons omit seasonPlayerId on some free throws of a trip
    # (sometimes every attempt). A trip's FTs share (period, gameTime,
    # team), so pool the group and stamp the one shooter that appears on
    # any attempt onto all of them. Groups where no attempt names a
    # shooter (rare) stay unattributed.
    ft_shooter = {}
    for a in actions:
        if a.get("type") == "FREE_THROW" and a.get("seasonPlayerId"):
            key = (a.get("period"), a.get("gameTime"),
                   str(a.get("seasonTeamId")))
            ft_shooter.setdefault(key, a["seasonPlayerId"])
    for a in actions:
        if a.get("type") == "FREE_THROW" and not a.get("seasonPlayerId"):
            key = (a.get("period"), a.get("gameTime"),
                   str(a.get("seasonTeamId")))
            if key in ft_shooter:
                a["seasonPlayerId"] = ft_shooter[key]

    events, shots = [], []
    eid = 0
    for a in actions:
        atype = a.get("type")
        stid = str(a.get("seasonTeamId"))
        team_id = tlc_of.get(stid)
        pid, pname, _pos = players.get(str(a.get("seasonPlayerId")),
                                       (None, None, None))

        val = shot_made = foul_kind = None
        if atype in _SHOT_TYPE:
            etype = "fg"
            val = _SHOT_TYPE[atype]
            shot_made = 1 if a.get("isSuccessful") else 0
        elif atype == "FREE_THROW":
            etype, val = "ft", 1
            shot_made = 1 if a.get("isSuccessful") else 0
        elif atype == "FOUL":
            etype = "foul"
            foul_kind = _foul_kind(a.get("qualifiers"))
        elif atype == "REBOUND" or atype == "TEAM_REBOUND":
            etype = "reb_off" if "OFFENSIVE" in (a.get("qualifiers") or []) \
                else "reb_def"
        else:
            etype = _SIMPLE.get(atype, "other")

        eid += 1
        secs = _secs_remaining(a.get("gameTime"))
        events.append({
            "game_id": gid, "event_id": eid, "src_num": a.get("orderId"),
            "period": a.get("period"), "secs_remaining": secs,
            "team_id": team_id, "player_id": pid, "player_name": pname,
            "etype": etype, "shot_value": val, "shot_made": shot_made,
            "foul_kind": foul_kind, "foul_fts": 0,
            "score_home": a.get("homeTeamPoints"),
            "score_away": a.get("guestTeamPoints"),
        })

        # (0,0) is the feed's no-location sentinel, and a tap past the
        # sideline (15 in 120k shots) is not a position either: both keep
        # the FGA on the possession side but get no shot row.
        c = a.get("coordinates") or {}
        if (etype == "fg" and (c.get("x") or c.get("y"))
                and 0 <= c["x"] <= _X_MAX):
            shots.append({
                "game_id": gid, "event_id": eid,
                "player_id": pid, "team_id": team_id,
                "period": a.get("period"), "secs_remaining": secs,
                "shot_made": shot_made, "shot_value": val,
                "x_m": (c["x"] - _X0) / _K, "y_m": (c["y"] - _Y0) / _K,
                "second_chance": 0, "off_turnover": 0,
                "dunk": 1 if "DUNK" in (a.get("qualifiers") or []) else 0,
            })
    return events, shots


def player_positions(season):
    """BBL exposes positions; build_tables merges these into player_season.
    Returns [{season, player_id, position}] from cached game rosters."""
    import glob
    import json
    import os
    seen, rows = set(), []
    gids = list_games(season)
    for g in gids:
        p = os.path.join(config.CACHE_DIR, "BBL", "stats",
                         f"{g['game_code']}.json")
        if not os.path.exists(p):
            continue
        stats = json.load(open(p))
        players, _ = _rosters(stats)
        for real, name, pos in players.values():
            if real not in seen:
                seen.add(real)
                rows.append({"season": season, "player_id": real,
                             "position": pos})
    return rows
