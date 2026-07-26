"""WNBA adapter.

Source: `nba_api` (the same official stats.nba.com package family NBA_OE
consumes for the NBA), with league_id="10" for WNBA. This is the
highest-quality free source in Euro_OE — proper official league data, not
a scrape: explicit free-throw trip position ("Free Throw N of M"),
explicit foul subtypes (Shooting/Personal/Offensive), and real shot
coordinates on every field-goal attempt. Endpoints used:
`leaguegamefinder` (schedule/scores), `playbyplayv3` (event stream + shot
coordinates), `boxscoresummaryv2` (officials).

WNBA plays a single-calendar-year season (May-Oct), unlike every other
league here's cross-year "2020-21" label. To keep every downstream date
utility (which slices season strings assuming "YYYY-YY", e.g.
`seasonShort()` in the front end) working unmodified, WNBA is keyed by the
SAME "2020-21"-style season strings as everyone else; `config.WNBA_SEASON`
maps each to just the first year, which is the actual season nba_api
needs — the second half of the label is cosmetic for this league.

Offensive vs defensive rebounds aren't labeled in the feed (`subType` is
almost always "Unknown"); resolved by comparing the rebounding team to the
team of the immediately preceding missed shot/free throw (the only events
a rebound follows) — same team = offensive, opposing team = defensive.
Team identity is resolved via the `location` field ('h'/'v'), which is
populated on every row that has a real team association (more robust than
`teamTricode`, which is blank on team-level rows like team rebounds).

Coordinates (`xLegacy`/`yLegacy`) are tenths of a foot from the hoop,
confirmed empirically against real 2PT/3PT distances (clean separation:
2PT max ~20.7 ft, 3PT min ~22.8 ft, matching the WNBA/NBA arc) — converted
to meters (x0.03048) in the same (x lateral, y toward midcourt) frame the
rest of the pipeline uses. Zone classification (`build_tables.zone_basic`)
still uses the shared FIBA-shaped constants; NBA/WNBA restricted-area and
paint dimensions are within a few centimeters of FIBA's, so this is a
defensible v1 approximation rather than a new geometry, and it never
affects shot value (2 vs 3) itself, which comes straight from the feed.
"""

import re
import time

import config
from leagues.common import cache_exists, cache_read, cache_write

LEAGUE = "WNBA"
_LEAGUE_ID = "10"
_TIMEOUT = 60

_FOUL_KIND = {
    "shooting": "personal", "personal": "personal", "loose ball": "personal",
    "offensive": "offensive",
    "technical": "technical",
    "flagrant": "personal", "flagrant type 1": "personal",
    "flagrant type 2": "personal",
}

_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")
_FT_TO_M = 0.03048  # tenths-of-a-foot -> meters


def _clock_secs(clock):
    m = _CLOCK_RE.match(clock or "")
    if not m:
        return None
    return int(m.group(1)) * 60 + int(float(m.group(2)))


def _foul_kind(subtype):
    return _FOUL_KIND.get((subtype or "").strip().lower(), "personal")


# ------------------------------------------------------------------ pull

def _season_games_raw(season):
    part = (LEAGUE, f"games_{config.WNBA_SEASON[season]}.json")
    if season == config.SEASONS[-1] or not cache_exists(*part):
        from nba_api.stats.endpoints import leaguegamefinder
        df = leaguegamefinder.LeagueGameFinder(
            league_id_nullable=_LEAGUE_ID,
            season_nullable=config.WNBA_SEASON[season],
            season_type_nullable="Regular Season",
            timeout=_TIMEOUT,
        ).get_data_frames()[0]
        cache_write(df.to_dict("records"), *part)
        time.sleep(config.WNBA_SLEEP)
    return cache_read(*part)


def list_games(season):
    if season not in config.WNBA_SEASON:
        return []
    rows = _season_games_raw(season)
    by_game = {}
    for r in rows:
        by_game.setdefault(r["GAME_ID"], []).append(r)
    games = []
    for gid, pair in by_game.items():
        if len(pair) != 2:
            continue  # incomplete pairing (all-star/exhibition rows etc.)
        home = next((r for r in pair if "vs." in (r.get("MATCHUP") or "")), None)
        away = next((r for r in pair if "@" in (r.get("MATCHUP") or "")), None)
        if not home or not away:
            continue
        games.append({
            "game_id": gid, "game_code": gid,
            "league": LEAGUE, "season": season,
            "date": home["GAME_DATE"],
            "home_team_id": home["TEAM_ABBREVIATION"],
            "away_team_id": away["TEAM_ABBREVIATION"],
            "home_team_name": home["TEAM_NAME"],
            "away_team_name": away["TEAM_NAME"],
            "home_score": home["PTS"], "away_score": away["PTS"],
            "ref1": None, "ref2": None, "ref3": None,
        })
    return sorted(games, key=lambda g: (g["date"], g["game_code"]))


def pull_game(season, game_code):
    pbp_part = (LEAGUE, "pbp", f"{game_code}.json")
    if not cache_exists(*pbp_part):
        from nba_api.stats.endpoints import playbyplayv3
        df = playbyplayv3.PlayByPlayV3(
            game_id=game_code, timeout=_TIMEOUT).get_data_frames()[0]
        cache_write(df.to_dict("records"), *pbp_part)
        time.sleep(config.WNBA_SLEEP)
    off_part = (LEAGUE, "officials", f"{game_code}.json")
    if not cache_exists(*off_part):
        from nba_api.stats.endpoints import boxscoresummaryv2
        df = boxscoresummaryv2.BoxScoreSummaryV2(
            game_id=game_code, timeout=_TIMEOUT).get_data_frames()[2]
        cache_write(df.to_dict("records"), *off_part)
        time.sleep(config.WNBA_SLEEP)
    return True


def referees(game):
    try:
        rows = cache_read(LEAGUE, "officials", f"{game['game_code']}.json")
    except FileNotFoundError:
        return [None, None, None]
    names = []
    for r in rows:
        nm = f"{(r.get('FIRST_NAME') or '').strip()} {(r.get('LAST_NAME') or '').strip()}".strip()
        names.append(nm or None)
    return (names + [None, None, None])[:3]


# ------------------------------------------------------------- normalize

def normalize_game(season, game):
    gid = game["game_id"]
    rows = cache_read(LEAGUE, "pbp", f"{gid}.json")
    game["ref1"], game["ref2"], game["ref3"] = referees(game)

    home_id, away_id = game["home_team_id"], game["away_team_id"]
    events, shots = [], []
    eid = 0
    last_shot_team = None  # team of the last missed FG/FT, for reb classification
    for r in rows:
        at = r.get("actionType") or ""
        subtype = (r.get("subType") or "").strip()
        loc = r.get("location")
        team_tri = (r.get("teamTricode") or "").strip()
        team_id = home_id if loc == "h" else away_id if loc == "v" else None
        person_id = r.get("personId")
        player_id = str(person_id) if (person_id and team_tri) else None
        player_name = (r.get("playerNameI") or r.get("playerName") or "").strip() \
            or None
        desc = (r.get("description") or "").upper()
        missed = desc.startswith("MISS")

        etype, shot_value, shot_made, foul_kind = "other", None, None, None
        if at == "Made Shot":
            etype, shot_made = "fg", 1
            shot_value = 3 if r.get("shotValue") == 3 else 2
        elif at == "Missed Shot":
            etype, shot_made = "fg", 0
            shot_value = 3 if r.get("shotValue") == 3 else 2
            last_shot_team = team_id
        elif at == "Free Throw":
            etype, shot_value = "ft", 1
            shot_made = 0 if missed else 1
            if missed:
                last_shot_team = team_id
        elif at == "Foul":
            etype = "foul"
            foul_kind = _foul_kind(subtype)
        elif at == "Rebound":
            etype = "reb_off" if team_id == last_shot_team else "reb_def"
        elif at == "Turnover":
            etype = "turnover"
        elif at == "Steal":
            etype = "steal"
        elif at == "Block":
            etype = "block"
        elif at == "Substitution":
            etype = "sub"
        elif at == "Timeout":
            etype = "timeout"
        elif at == "Jump Ball":
            etype = "jumpball"
        elif at == "period":
            etype = "period_start" if subtype.lower() == "start" else "period_end"

        eid += 1
        secs = _clock_secs(r.get("clock"))
        # "Instant Replay" review rows carry a stale/unrelated scoreHome
        # /scoreAway pair (confirmed against real games: e.g. 8-9 stamped
        # mid-game while the actual score was 15-20) — never trust them.
        if at == "Instant Replay":
            sh = sa = None
        else:
            sh, sa = r.get("scoreHome"), r.get("scoreAway")
            sh = int(sh) if sh not in (None, "") else None
            sa = int(sa) if sa not in (None, "") else None
        events.append({
            "game_id": gid, "event_id": eid, "src_num": r.get("actionNumber"),
            "period": r.get("period"), "secs_remaining": secs,
            "team_id": team_id, "player_id": player_id,
            "player_name": player_name,
            "etype": etype, "shot_value": shot_value, "shot_made": shot_made,
            "foul_kind": foul_kind, "foul_fts": 0,
            "score_home": sh, "score_away": sa,
        })

        if etype == "fg":
            x, y = r.get("xLegacy"), r.get("yLegacy")
            if x or y:
                shots.append({
                    "game_id": gid, "event_id": eid,
                    "player_id": player_id, "team_id": team_id,
                    "period": r.get("period"), "secs_remaining": secs,
                    "shot_made": shot_made, "shot_value": shot_value,
                    "x_m": x * _FT_TO_M, "y_m": y * _FT_TO_M,
                    "second_chance": 0,   # filled by build_possessions
                    "off_turnover": 0,    # filled by build_possessions
                    "dunk": 1 if "dunk" in subtype.lower() else 0,
                })
    return events, shots
