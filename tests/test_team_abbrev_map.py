"""team_abbrev_map labels every team a league has played, home or away."""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import export_site_data


def games_conn(rows):
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE games (league TEXT, home_team_id TEXT, away_team_id TEXT,
                                        home_team_name TEXT, away_team_name TEXT)""")
    conn.executemany("INSERT INTO games VALUES (?,?,?,?,?)", rows)
    return conn


def label_map(monkeypatch, league, rows):
    monkeypatch.setattr(export_site_data, "conn", games_conn(rows))
    return export_site_data.team_abbrev_map(league)


def test_aba_labels_a_team_that_has_only_played_away(monkeypatch):
    m = label_map(monkeypatch, "ABA", [
        ("ABA", "22", "110", "Partizan Mozzart Bet", "Slovan Bratislava"),     # 110: away so far
        ("ABA", None, None, "Koper Primorska", "Zadar")])                       # a forfeit carries no ids
    assert m == {"22": "Partizan", "110": "Slovan Bratislava"}


def test_lba_labels_an_away_team_and_a_home_name_wins_a_conflict(monkeypatch):
    m = label_map(monkeypatch, "LBA", [
        ("LBA", "1", "2", "Nutribullet Treviso Basket", "Sponsor Roma"),
        ("LBA", "3", "1", "Sponsor Napoli", "Old Sponsor Venezia")])    # id 1 was Treviso at home, Venezia away
    assert m == {"1": "Treviso", "2": "Roma", "3": "Napoli"}


def test_code_leagues_label_every_id_they_have_played(monkeypatch):
    m = label_map(monkeypatch, "EL", [("EL", "BAR", "ASV", None, None), ("EL", "ASV", "MAD", None, None),
                                      ("ELSE", "XXX", "YYY", None, None)])
    assert m == {"BAR": "BAR", "ASV": "ASV", "MAD": "MAD"}
