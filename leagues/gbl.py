"""Greek Basket League (GBL / Stoiximan GBL, ESAKE) enumeration.

GBL runs on Genius Sports FIBA LiveStats (org code ESAKE, leagueId 115),
so per-game data comes from the shared `leagues.genius` adapter — same
`data.json` shape already validated for LBA-adjacent and other Genius
leagues. The only GBL-specific problem is enumeration: the open Genius
competition endpoint (`/data/competition/{compId}.json`) returns only the
season's LAST game, not a schedule, and esake.gr's own fixture list uses
unrelated ids with no bridge to Genius matchIds.

The fix: Wayback Machine has organically crawled the public Genius viewer
pages (`/u/ESAKE/{matchId}/*.html`) over the years, which exposes matchIds
a schedule endpoint would otherwise give us. A one-time CDX query
(`http://web.archive.org/cdx/search/cdx?url=.../u/ESAKE&matchType=prefix`)
recovered 770 distinct matchIds; bucketing by NUMERIC ID RANGE (anchored to
each season's real last-game id, cross-checked against the live
competition endpoint) is far more reliable than bucketing by Wayback
capture timestamp — old, rarely-crawled pages surface with recent capture
dates and would otherwise contaminate the wrong season.

Coverage: 2020-21 and 2021-22 recovered too few ids to trust (2% and 38%
of a full season) and are excluded. 2022-23 through 2025-26 recovered
128-204 games each (spot-checked: 100% legitimate GBL teams, no
cross-league contamination) and are shipped.

The manifest (`data/gbl_matchids.json`) is a frozen, one-time research
artifact, not something to re-derive at pull time — Wayback's crawl of
these pages won't change, and re-querying the CDX API on every pull would
be slow and add a fragile external dependency for no benefit.
"""

import json
import os

_DATA = os.path.join(os.path.dirname(__file__), "data", "gbl_matchids.json")

# Approximate season window (GBL runs Oct -> mid-June) for the display-only
# `date` field; no per-game date exists in the Genius feed itself, and nothing
# in the model depends on it (event order within a game comes from the pbp).
_SEASON_START = {  # (month, day) the season roughly begins
    "2022-23": (10, 1), "2023-24": (10, 1),
    "2024-25": (10, 1), "2025-26": (10, 1),
}
_SEASON_END = {
    "2022-23": (6, 15), "2023-24": (6, 15),
    "2024-25": (6, 15), "2025-26": (6, 15),
}


def _interpolated_date(season, rank, n):
    import datetime
    y = int(season[:4])
    start = datetime.date(y, *_SEASON_START[season])
    end = datetime.date(y + 1, *_SEASON_END[season])
    span = (end - start).days
    frac = rank / max(n - 1, 1)
    return (start + datetime.timedelta(days=round(span * frac))).isoformat()


def enumerate_season(season):
    manifest = json.load(open(_DATA))
    ids = manifest.get(season, [])
    n = len(ids)
    games = []
    for rank, mid in enumerate(ids):
        games.append({
            "game_id": str(mid), "game_code": mid,
            "league": "GBL", "season": season,
            "date": _interpolated_date(season, rank, n),
            "home_team_id": None, "away_team_id": None,
            "home_team_name": None, "away_team_name": None,
            "home_score": None, "away_score": None,
            "ref1": None, "ref2": None, "ref3": None,
        })
    return games
