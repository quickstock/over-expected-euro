"""Adapter registry. ADAPTERS[code] exposes list_games, pull_game,
normalize_game, referees — the EuroLeague and ACB modules, plus a Genius
LiveStats adapter for every league whose enumeration is wired in
genius_leagues.ENUMERATORS. Consumers (pull.py, build_tables.py) dispatch on
this; config.LEAGUES controls which of them a run actually processes.
"""
from leagues import (aba, acb, bbl, eurocup, euroleague, genius,
                     genius_leagues, lba, wnba)

ADAPTERS = {
    "EL": euroleague, "EUC": eurocup, "ACB": acb,
    "LBA": lba, "BBL": bbl, "ABA": aba, "WNBA": wnba,
}

for _code, _enum in genius_leagues.ENUMERATORS.items():
    ADAPTERS[_code] = genius.make_adapter(_code, _enum)
