"""Genius Sports FIBA LiveStats adapter — shared by every league on the
platform (LBA, BSL, GBL, ABA, BBL, PROA...).

One file per game has everything:
`https://fibalivestats.dcd.shared.geniussports.com/data/{gameId}/data.json`
carries `pbp` (made/miss via `success`, fouls typed by FT-count qualifier,
running scores `s1`/`s2`, officials) and per-team `shot` arrays with x/y
that join `pbp` on `actionNumber`. No auth.

Per-league differences are ONLY (a) the league code and (b) game-id
enumeration; both are injected via `make_adapter(code, enumerate_fn)`.
Everything else — pull, parse, coordinate normalization, referees — is
identical, because the feed is identical.

Coordinates: shot x/y are 0-100 percentages of the FULL court (28 x 15 m).
Teams switch baskets at halftime, so the attacking hoop is self-calibrated
per game from each team's first-half shot cluster (no hardcoded
orientation), then folded to the normalized frame (meters from hoop
center, x lateral, y toward midcourt) the rest of the pipeline expects.

Player identity: the feed has no stable player id, only names. Identity is
a normalized "FamilyName, FirstName" slug, resolved via the game roster
(`pl`) so every event — not just shots — gets the full name. Stable within
a league across games and seasons; rare same-name collisions are a known
v1 simplification.
"""

import re
import unicodedata

import config
from leagues.common import (cache_exists, cache_read, cache_write,
                            fetch_json)

DATA_URL = ("https://fibalivestats.dcd.shared.geniussports.com"
            "/data/{gid}/data.json")

# FIBA full court, meters.
COURT_LEN_M = 28.0
COURT_WID_M = 15.0
HOOP_FROM_BASELINE_M = config.FIBA_HOOP_TO_BASELINE  # 1.575

_SHOT_VALUE = {"2pt": 2, "3pt": 3}
_FOUL_KIND = {
    "personal": "personal",
    "offensive": "offensive",
    "unsportsmanlike": "unsportsmanlike",
    "technical": "technical",
    "bench": "bench",
    "coach": "technical",
    "": "personal",  # untyped personal foul
}
_FT_QUALIFIER = {"1freethrow": 1, "2freethrow": 2, "3freethrow": 3}


def _score_int(v):
    """Genius stamps s1/s2 (running score) as int in most feeds, but some
    leagues (confirmed: GBL) mix in numeric strings within the same game —
    cast defensively rather than let a str/int subtraction blow up later."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _drop_phantoms(pbp):
    """Some Genius feeds (confirmed on GBL) splice in stray events from a
    duplicate/partial second submission of the same game; their
    actionNumbers fall inside the real sequence's range, so sorting
    interleaves them, and they carry a stale (lower) running score. A real
    basketball score never decreases, so an event whose s1/s2 drops below
    the running max is a phantom — drop it. Clean feeds never regress, so
    this is a no-op there. Assumes `pbp` is already actionNumber-sorted."""
    out, sh, sa = [], 0, 0
    for e in pbp:
        s1, s2 = _score_int(e.get("s1")), _score_int(e.get("s2"))
        if (s1 is not None and s1 < sh) or (s2 is not None and s2 < sa):
            continue
        if s1 is not None:
            sh = s1
        if s2 is not None:
            sa = s2
        out.append(e)
    return out


def _slug(family, first):
    """Stable ASCII player key from name parts."""
    s = f"{family or ''}_{first or ''}".strip("_")
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s or None


def _secs_remaining(gt):
    """Genius `gt` is "MM:SS" remaining in the period."""
    if not gt or ":" not in gt:
        return None
    mm, ss = gt.split(":")[:2]
    try:
        return int(mm) * 60 + int(ss)
    except ValueError:
        return None


def _team_id(tm_entry):
    return (tm_entry.get("code") or tm_entry.get("codeInternational")
            or tm_entry.get("shortName") or tm_entry.get("name") or "").strip()


def _roster_name(tm_entry, pno):
    """(family, first) from the game roster, international spelling."""
    pl = (tm_entry.get("pl") or {}).get(str(pno)) or {}
    fam = pl.get("internationalFamilyName") or pl.get("familyName")
    fir = pl.get("internationalFirstName") or pl.get("firstName")
    return fam, fir


def _basket_side(tm):
    """Per team (tno "1"/"2"), which side its first-half baskets are on:
    'right' (attacking x>50) or 'left'. Second half flips."""
    side = {}
    for tno in ("1", "2"):
        shots = tm[tno].get("shot", [])
        xs = [s["x"] for s in shots if s.get("per") in (1, 2)] \
            or [s["x"] for s in shots]
        med = sorted(xs)[len(xs) // 2] if xs else 50.0
        side[tno] = "right" if med > 50 else "left"
    return side


def _to_meters(x, y, period, tno, side):
    """Genius court-% (x,y) -> (x_m lateral, y_m toward midcourt)."""
    half = side.get(tno, "right")
    if period and period >= 3:
        half = "left" if half == "right" else "right"
    xm_court = x / 100.0 * COURT_LEN_M
    ym_court = y / 100.0 * COURT_WID_M
    if half == "right":
        hoop_x = COURT_LEN_M - HOOP_FROM_BASELINE_M
        y_m = hoop_x - xm_court
        x_m = ym_court - COURT_WID_M / 2
    else:
        hoop_x = HOOP_FROM_BASELINE_M
        y_m = xm_court - hoop_x
        x_m = COURT_WID_M / 2 - ym_court
    return x_m, y_m


class GeniusAdapter:
    """One instance per league. `enumerate_fn(season) -> [game dict]` must
    yield rows carrying at least `game_id` (str) and `game_code` (the
    fibalivestats gameId used to fetch); any team/score/date fields it can
    fill from the league schedule are advisory — `normalize_game`
    overwrites them from the authoritative feed."""

    def __init__(self, code, enumerate_fn):
        self.LEAGUE = code
        self._enumerate = enumerate_fn

    # ---------------------------------------------------------- pull
    def list_games(self, season):
        return self._enumerate(season)

    def pull_game(self, season, game_code):
        part = (self.LEAGUE, "data", f"{game_code}.json")
        if not cache_exists(*part):
            cache_write(fetch_json(DATA_URL.format(gid=game_code),
                                   sleep=config.GENIUS_SLEEP), *part)
        return True

    def _raw(self, game):
        return cache_read(self.LEAGUE, "data", f"{game['game_code']}.json")

    def referees(self, game):
        try:
            d = self._raw(game)
        except FileNotFoundError:
            return [None, None, None]
        offs = d.get("officials") or {}
        names = [(offs.get(f"referee{i}") or {}).get("name") for i in (1, 2, 3)]
        return [(n.strip() if n else None) for n in names]

    # ----------------------------------------------------- normalize
    def normalize_game(self, season, game):
        d = self._raw(game)
        tm = d["tm"]
        home, away = tm["1"], tm["2"]
        home_id, away_id = _team_id(home), _team_id(away)

        # authoritative game metadata (mutate the row build_tables stores)
        game["home_team_id"], game["away_team_id"] = home_id, away_id
        game["home_team_name"] = home.get("name")
        game["away_team_name"] = away.get("name")
        game["home_score"] = home.get("score")
        game["away_score"] = away.get("score")
        game["ref1"], game["ref2"], game["ref3"] = self.referees(game)

        side = _basket_side(tm)
        gid = game["game_id"]

        # coordinates live in the per-team shot arrays, joined on actionNumber
        shot_xy = {}
        for tno in ("1", "2"):
            for s in tm[tno].get("shot", []):
                an = s.get("actionNumber")
                if an is not None:
                    shot_xy[an] = (s, tno)

        pbp = sorted((e for e in d.get("pbp", [])
                      if e.get("actionNumber") is not None),
                     key=lambda e: e["actionNumber"])
        pbp = _drop_phantoms(pbp)

        events, shots = [], []
        eid = 0
        for e in pbp:
            at = e.get("actionType")
            tno = str(e.get("tno") or "")
            team_id = home_id if tno == "1" else away_id if tno == "2" else None
            fam, fir = (None, None)
            if e.get("pno") and tno in ("1", "2"):
                fam, fir = _roster_name(tm[tno], e.get("pno"))
            fam = fam or e.get("internationalFamilyName") or e.get("familyName")
            fir = fir or e.get("internationalFirstName") or e.get("firstName")
            pid = _slug(fam, fir) if fam else None
            pname = f"{fam}, {fir}" if (fam and fir) else (fam or None)

            etype = "other"
            shot_value = shot_made = foul_kind = foul_fts = None
            quals = e.get("qualifier") or []

            if at in _SHOT_VALUE:
                etype = "fg"
                shot_value = _SHOT_VALUE[at]
                shot_made = 1 if e.get("success") else 0
            elif at == "freethrow":
                etype = "ft"
                shot_value = 1
                shot_made = 1 if e.get("success") else 0
            elif at == "foul":
                etype = "foul"
                foul_kind = _FOUL_KIND.get(e.get("subType"), "personal")
                foul_fts = next((_FT_QUALIFIER[q] for q in quals
                                 if q in _FT_QUALIFIER), 0)
            elif at == "foulon":
                etype = "foul_drawn"
            elif at == "rebound":
                etype = "reb_off" if e.get("subType") == "offensive" \
                    else "reb_def"
            elif at == "turnover":
                etype = "turnover"
            elif at == "steal":
                etype = "steal"
            elif at == "assist":
                etype = "assist"
            elif at == "block":
                etype = "block"
            elif at == "substitution":
                etype = "sub"
            elif at == "timeout":
                etype = "timeout"
            elif at == "jumpball":
                etype = "jumpball"
            elif at == "period":
                etype = "period_end" if e.get("subType") == "end" \
                    else "period_start"

            eid += 1
            secs = _secs_remaining(e.get("gt"))
            row = {
                "game_id": gid, "event_id": eid,
                "src_num": e.get("actionNumber"),
                "period": e.get("period"), "secs_remaining": secs,
                "team_id": team_id, "player_id": pid, "player_name": pname,
                "etype": etype, "shot_value": shot_value,
                "shot_made": shot_made, "foul_kind": foul_kind,
                "foul_fts": foul_fts,
                # Genius stamps s1/s2 on every event (tm1 = home).
                "score_home": _score_int(e.get("s1")),
                "score_away": _score_int(e.get("s2")),
            }
            events.append(row)

            if etype == "fg":
                sxy = shot_xy.get(e["actionNumber"])
                if sxy is not None:
                    s, stno = sxy
                    x_m, y_m = _to_meters(s.get("x"), s.get("y"),
                                          e.get("period"), stno, side)
                    shots.append({
                        "game_id": gid, "event_id": eid,
                        "player_id": pid, "team_id": team_id,
                        "period": e.get("period"), "secs_remaining": secs,
                        "shot_made": shot_made, "shot_value": shot_value,
                        "x_m": x_m, "y_m": y_m,
                        "second_chance": 0,   # derived in build_possessions
                        "off_turnover": 0,    # derived in build_possessions
                        "dunk": 1 if e.get("subType") == "dunk" else 0,
                    })
        return events, shots


def make_adapter(code, enumerate_fn):
    return GeniusAdapter(code, enumerate_fn)
