"""Per-league game-id enumeration for the Genius LiveStats leagues.

Every one of these leagues serves identical per-game data via
`leagues.genius`; the ONLY per-league code is how to list a season's
fibalivestats gameIds. Each enumerator returns rows carrying at least
`game_id` (str) and `game_code` (the fibalivestats gameId to fetch).
`genius.normalize_game` fills teams/scores/date/refs from the feed, so
enumerators only need the ids (plus a date when the schedule gives one,
for ordering).

A code appears in ENUMERATORS only once its enumeration is verified; that
gates whether `leagues` builds an adapter for it. Wiring a league live
still also requires adding its code to `config.LEAGUES`.
"""

from leagues import gbl as _gbl

# code -> callable(season) -> [ {game_id, game_code, league, season, date?} ]
ENUMERATORS = {
    "GBL": _gbl.enumerate_season,
}
