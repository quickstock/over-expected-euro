"""Unit tests for export_games.py (European game-page exporter). Run directly or via pytest."""
import json
import os
import sqlite3
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import export_site_data
from export_games import (FG_TOL, anchor_tolerance, cm, export_season, ft_block, league_files, left_out_limit,
                          make_rates, official_names, officials, place_unlocated, shooter_sides, shot_block)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cm_nulls_unlocated():
    assert cm([float("nan"), 12.0, -3.0, 0.5]) == [None, 12, -3, 0]


def test_make_rates_by_value():
    shots = pd.DataFrame({"value": [2, 2, 2, 2, 3, 3], "made": [1, 1, 0, 0, 1, 0]})
    assert make_rates(shots) == {2: 0.5, 3: 0.5}
    shots = pd.DataFrame({"value": [2, 3, 3, 3, 3], "made": [1, 1, 0, 0, 0]})
    assert make_rates(shots) == {2: 1.0, 3: 0.25}


def test_place_unlocated_gets_the_league_rate_and_no_coordinates():
    un = pd.DataFrame({"value": [3, 2], "made": [1, 0]})
    placed = place_unlocated(un, {2: 0.5, 3: 0.36})
    assert placed["xfg"].tolist() == [0.36, 0.5]
    assert placed["xpts"].tolist() == pytest.approx([1.08, 1.0])
    assert placed["x"].isna().all() and placed["y"].isna().all()
    assert "xfg" not in un.columns  # the input is not modified


def test_shot_block_is_columnar_with_a_player_index():
    sh = pd.DataFrame({"player_id": ["P20", "P10", "P20"], "period": [1, 1, 5], "sec": [590, 540, 12],
                       "x": [-546.0, float("nan"), 30.0], "y": [float("nan"), float("nan"), 0.0],
                       "made": [0, 1, 1], "value": [3, 2, 3], "xfg": [0.3584, 0.5, 0.3604]})
    blk = shot_block(sh, {"P10": 0, "P20": 1})
    # a shot with a real x but no depth (ACB below the hoop line) keeps its x
    assert blk == {"p": [1, 0, 1], "q": [1, 1, 5], "c": [590, 540, 12],
                   "x": [-546, None, 30], "y": [None, None, 0],
                   "m": "011", "v": "323", "e": [358, 500, 360]}


def test_ft_block_is_columnar():
    fts = pd.DataFrame({"pid": ["P10", "P10", "P20"], "period": [2, 2, 4], "sec": [90, 90, 3],
                        "made": [True, False, True], "pct": [0.7634, 0.7634, 0.5]})
    blk = ft_block(fts, {"P10": 0, "P20": 1})
    assert blk == {"p": [0, 0, 1], "q": [2, 2, 4], "c": [90, 90, 3], "m": "101", "e": [763, 763, 500]}
    assert ft_block(fts.iloc[:0], {}) == {"p": [], "q": [], "c": [], "m": "", "e": []}


# --- gates -------------------------------------------------------------------

def test_left_out_limit_is_one_percent_with_a_floor_of_three():
    assert left_out_limit(380, "EL") == 3      # a European season is a few hundred games
    assert left_out_limit(192, "ABA") == 3
    assert left_out_limit(300, "WNBA") == 3
    assert left_out_limit(400, "ACB") == 4
    assert left_out_limit(1189, "EL") == 11


def test_the_gbl_may_leave_out_a_quarter_of_its_games():
    # its feed drops scoring rows: 6-24% of a season's games cannot rebuild their final
    assert left_out_limit(141, "GBL") == 35
    assert left_out_limit(194, "GBL") == 48
    assert left_out_limit(8, "GBL") == 3
    assert left_out_limit(141, "ABA") == 3


def test_anchor_tolerance_only_loosens_for_the_in_progress_season(monkeypatch):
    monkeypatch.setattr(config, "SEASONS", ["2024-25", "2025-26", "2026-27"])
    assert anchor_tolerance("2026-27") == 0.02
    assert anchor_tolerance("2025-26") == 0.005
    assert anchor_tolerance("2024-25") == 0.005


# --- helpers ------------------------------------------------------------------

def test_shooter_sides_joins_ft_only_players_and_flags_a_conflict():
    sh = pd.DataFrame({"player_id": ["101", "201"], "side": [0, 1]})
    fts = pd.DataFrame({"pid": ["103", "101"], "side": [0, 0]})
    assert shooter_sides(sh, fts) == {"101": 0, "201": 1, "103": 0}
    assert shooter_sides(sh, pd.DataFrame({"pid": ["101"], "side": [1]})) is None


GREEK, LATIN = "Πέτρος Παπαπέτρου", "Petros Papapetrou"


def test_officials_get_the_referee_pages_ids_and_display_names():
    games = pd.DataFrame({"ref1": [GREEK, LATIN, "Γιώργος Νικολάου"],
                          "ref2": [LATIN, "Ref C", None],
                          "ref3": [None, "", None]})
    shown = official_names(games)
    # both spellings of one person are one id, shown in the Latin spelling the feed also has
    assert shown == {"petros-papapetrou": LATIN, "ref-c": "Ref C"}
    assert officials([GREEK, LATIN, None], shown) == [["petros-papapetrou", LATIN], ["petros-papapetrou", LATIN]]
    # a Greek-only official keeps the feed's own name, under the id ref_key gives it
    assert officials(["Γιώργος Νικολάου", "Ref C", "", None], shown) == [["giorgos-nikolaou", "Γιώργος Νικολάου"],
                                                                        ["ref-c", "Ref C"]]
    assert officials([None, None, None], shown) == []


@pytest.mark.parametrize("league", ["GBL", "EL", "ACB"])
def test_official_names_match_the_referee_export(league):
    """The ids and names a game page links are the ones the league's referee pages carry."""
    path = export_site_data.OUT_DIR / f"data-{league}.json"
    if not path.exists() or not os.path.exists(config.DB_PATH):
        pytest.skip("needs the built database and the league's data export")
    profiles = json.loads(path.read_text())["refProfiles"]
    games = pd.read_sql("SELECT ref1, ref2, ref3 FROM games WHERE league = ?", export_site_data.conn,
                        params=(league,))
    shown = official_names(games)
    names = {}
    for col in ("ref1", "ref2", "ref3"):
        for n in games[col].dropna().unique():
            if n:
                names.setdefault(export_site_data.ref_key(n), shown.get(export_site_data.ref_key(n), n))
    assert set(profiles) <= set(names)
    assert {i: names[i] for i in profiles} == {i: p["name"] for i, p in profiles.items()}


# --- export_season on a synthetic game ----------------------------------------
# Home AAA (labelled "Alpha BC" by the abbreviation map), away BBB (not in the map: the id is the
# label). Eight field goals: six in shots (player 202's has a real x and no depth, as ACB's below the
# hoop line; event 6 has no shooter), two only in events (102's, no shot-chart row: unlocated).
# Player 103 only shoots free throws, one free throw has no shooter, 202 has no player_season row,
# 102 has no player_game_shot_value row. Final 6-8.

LG, SEASON, GID = "GBL", "2025-26", "g1"
ABBREV = {"AAA": "Alpha BC"}


def synthetic_conn(exp_101=2.2, league=LG):
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE games (game_id TEXT, league TEXT, season TEXT, date TEXT, home_team_id TEXT, away_team_id TEXT,
                            home_score INTEGER, away_score INTEGER, ref1 TEXT, ref2 TEXT, ref3 TEXT);
        CREATE TABLE shots (game_id TEXT, event_id INTEGER, player_id TEXT, team_id TEXT, period INTEGER,
                            secs_remaining INTEGER, shot_made INTEGER, shot_value INTEGER, x_m REAL, y_m REAL,
                            league TEXT, season TEXT, home_or_away TEXT);
        CREATE TABLE shots_xfg (league TEXT, season TEXT, game_id TEXT, event_id INTEGER, xfg REAL);
        CREATE TABLE events (game_id TEXT, event_id INTEGER, period INTEGER, secs_remaining REAL, team_id TEXT,
                             player_id TEXT, player_name TEXT, etype TEXT, shot_value REAL, shot_made REAL,
                             league TEXT, season TEXT);
        CREATE TABLE player_season (season TEXT, player_id TEXT, player_name TEXT, league TEXT);
        CREATE TABLE player_season_ft (season TEXT, player_id TEXT, ftm REAL, fta INTEGER, league TEXT);
        CREATE TABLE player_game_shot_value (season TEXT, player_id TEXT, game_id TEXT, fga INTEGER, fgm INTEGER,
                                             act_fg_pts INTEGER, exp_fg_pts REAL, league TEXT);
    """)
    L, S = league, SEASON
    conn.execute("INSERT INTO games VALUES (?, ?, ?, '2025-10-29', 'AAA', 'BBB', 6, 8, 'Ref A', 'Ref B', NULL)",
                 (GID, L, S))
    conn.execute("INSERT INTO games VALUES (?, 'ELSE', ?, '2025-10-30', 'XXX', 'YYY', 70, 60, NULL, NULL, NULL)",
                 (GID, S))  # another league's game with the same id: never read
    conn.executemany("INSERT INTO shots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        (GID, 1, "101", "AAA", 1, 590, 1, 2, -1.0, 0.5, L, S, "home"),
        (GID, 2, "101", "AAA", 1, 560, 0, 3, 0.0, 6.75, L, S, "home"),
        (GID, 3, "201", "BBB", 1, 520, 1, 3, 2.0, 1.0, L, S, "away"),
        (GID, 4, "202", "BBB", 2, 300, 1, 2, 0.1, None, L, S, "away"),
        (GID, 5, "202", "BBB", 2, 200, 0, 2, -0.1, 0.3, L, S, "away"),
        (GID, 6, None, "BBB", 2, 100, 1, 2, 0.5, 1.0, L, S, "away")])
    conn.executemany("INSERT INTO shots_xfg VALUES (?,?,?,?,?)", [
        (L, S, GID, 1, 0.5), (L, S, GID, 2, 0.4), (L, S, GID, 3, 0.36), (L, S, GID, 4, 0.6), (L, S, GID, 5, 0.4),
        (L, S, GID, 6, 0.5)])
    conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        (GID, 1, 1, 590.0, "AAA", "101", "Alpha", "fg", 2.0, 1.0, L, S),
        (GID, 2, 1, 560.0, "AAA", "101", "Alpha", "fg", 3.0, 0.0, L, S),
        (GID, 3, 1, 520.0, "BBB", "201", "Delta", "fg", 3.0, 1.0, L, S),
        (GID, 4, 2, 300.0, "BBB", "202", "Zed", "fg", 2.0, 1.0, L, S),
        (GID, 5, 2, 200.0, "BBB", "202", "Zed", "fg", 2.0, 0.0, L, S),
        (GID, 6, 2, 100.0, "BBB", None, None, "fg", 2.0, 1.0, L, S),
        (GID, 7, 3, 500.0, "AAA", "102", "Bravo", "fg", 3.0, 1.0, L, S),
        (GID, 8, 3, 100.0, "AAA", "102", "Bravo", "fg", 2.0, 0.0, L, S),
        (GID, 9, 2, 330.0, "AAA", "103", "Charlie", "ft", 1.0, 1.0, L, S),
        (GID, 10, 2, 330.0, "AAA", "103", "Charlie", "ft", 1.0, 0.0, L, S),
        (GID, 11, 4, 5.0, "BBB", None, None, "ft", 1.0, 1.0, L, S),
        (GID, 12, 4, 4.0, "BBB", "201", "Delta", "sub", None, None, L, S)])
    conn.executemany("INSERT INTO player_season VALUES (?,?,?,?)", [
        (S, "101", "Alpha One", L), (S, "102", "Bravo Two", L), (S, "103", "Charlie Three", L),
        (S, "201", "Delta One", L)])
    conn.executemany("INSERT INTO player_season_ft VALUES (?,?,?,?,?)", [
        (S, "103", 80, 100, L), (S, "999", 70, 100, L), (S, "103", 0, 100, "ELSE")])
    conn.executemany("INSERT INTO player_game_shot_value VALUES (?,?,?,?,?,?,?,?)", [
        (S, "101", GID, 2, 1, 2, exp_101, L), (S, "201", GID, 1, 1, 3, 1.08, L), (S, "202", GID, 2, 1, 2, 2.0, L)])
    return conn


def run(conn, league=LG):
    return export_season(conn, league, SEASON, ABBREV, {})


def with_second_game(conn):
    """A copy of the game as g2, so that a season whose first game is left out still publishes one."""
    for table in ("games", "shots", "shots_xfg", "events", "player_game_shot_value"):
        df = pd.read_sql(f"SELECT * FROM {table} WHERE game_id = ? AND league = ?", conn, params=(GID, LG))
        df["game_id"] = "g2"
        df.to_sql(table, conn, if_exists="append", index=False)
    return conn


def test_export_season_assembles_the_contract():
    r = run(synthetic_conn())
    assert r["excluded"] == 0 and r["total"] == 1 and r["unlocated"] == 2 and r["unattributed"] == 1
    assert r["name_fallbacks"] == 1
    (row,) = r["rows"]
    # index row: ids as strings, the abbreviation map's label with the id as the fallback, x to 0.1; the
    # GBL's dates are interpolated, so no date, but the week is still keyed from it
    assert {k: v for k, v in row.items() if k != "share"} == {
        "id": GID, "d": None, "wk": "2025-10-27", "h": "AAA", "a": "BBB",
        "hl": "Alpha BC", "al": "BBB", "s": [6, 8], "x": [6.8, 4.8]}
    assert 0 < row["share"] < 1
    # the anchor is the six shots with a model xFG: 6.28 expected against 9 scored; the two imputed
    # unlocated shots are in every other total
    assert r["anchor"] == pytest.approx((6.28 - 9) / 9)
    assert r["forfeits"] == 0
    game = r["chunks"]["2025-10-27"][GID]
    assert set(game) == {"p", "s", "f", "r"}
    # p rows: [pid, name, side, fga, fgPts, xFgPts, fta, ftm], by side then points. 102's two unlocated shots
    # are added at the league make rate x value (0.5 x 3, 0.75 x 2); the shot with no shooter and the free
    # throw with none are one "Unattributed" row per side; 103 only shot free throws; 202 has the events' name.
    assert game["p"] == [
        ["102", "Bravo Two", 0, 2, 3, 3.0, 0, 0],
        ["101", "Alpha One", 0, 2, 2, 2.2, 0, 0],
        ["103", "Charlie Three", 0, 0, 0, 0.0, 2, 1],
        ["201", "Delta One", 1, 1, 3, 1.08, 0, 0],
        ["?away", "Unattributed", 1, 1, 2, 1.0, 1, 1],
        ["202", "Zed", 1, 2, 2, 2.0, 0, 0]]
    # shots in play order; players are indices into p; 202's has x and no y, 102's two have neither
    assert game["s"] == {
        "p": [1, 1, 3, 5, 5, 4, 0, 0], "q": [1, 1, 1, 2, 2, 2, 3, 3],
        "c": [590, 560, 520, 300, 200, 100, 500, 100],
        "x": [-100, 0, 200, 10, -10, 50, None, None], "y": [50, 675, 100, None, 30, 100, None, None],
        "m": "10110110", "v": "23322232", "e": [500, 400, 360, 600, 400, 500, 500, 750]}
    assert game["f"] == {"p": [2, 2, 4], "q": [2, 2, 4], "c": [330, 330, 5], "m": "101", "e": [783, 783, 750]}
    assert game["r"] == [["ref-a", "Ref A"], ["ref-b", "Ref B"]]


def test_league_files_name_an_index_per_season_and_a_chunk_per_week():
    r = run(synthetic_conn())
    files = league_files(LG, [r], True, 0.0412)
    assert set(files) == {f"gameweek-{LG}-2025-10-27.json", f"games-{LG}-{SEASON}.json"}
    index = json.loads(files[f"games-{LG}-{SEASON}.json"])
    assert index["meta"] == {"league": LG, "season": SEASON, "winShare": True, "ece": 0.0412,
                             "excluded": 0, "forfeits": 0, "games": 1}
    (row,) = index["games"]
    assert row["w"] == int(round(100 * r["rows"][0]["share"])) and "share" not in row
    assert index["meta"]["forfeits"] == 0
    assert set(json.loads(files[f"gameweek-{LG}-2025-10-27.json"])["games"]) == {GID}
    # a league whose win share failed calibration ships none
    withheld = json.loads(league_files(LG, [r], False, 0.1)[f"games-{LG}-{SEASON}.json"])
    assert withheld["meta"]["winShare"] is False and withheld["games"][0]["w"] is None


def test_the_anchor_leaves_out_imputed_shots():
    # Re-price the six located shots so their expected points equal the 9 they scored exactly
    # (1.8 + 1.8 + 2.1 + 1.2 + 0.6 + 1.5); players' game lines follow. Both unlocated shots now go in
    # (final 8-8), and, imputed at the league rate (1.5 + 1.5 expected), 5 scored: with them the season
    # would sit 14% under its anchor.
    conn = synthetic_conn()
    for event, xfg in ((1, 0.9), (2, 0.6), (3, 0.7), (4, 0.6), (5, 0.3), (6, 0.75)):
        conn.execute("UPDATE shots_xfg SET xfg = ? WHERE event_id = ?", (xfg, event))
    for player, exp in (("101", 3.6), ("201", 2.1), ("202", 1.8)):
        conn.execute("UPDATE player_game_shot_value SET exp_fg_pts = ? WHERE player_id = ?", (exp, player))
    conn.execute("UPDATE events SET shot_made = 1 WHERE event_id = 8")
    conn.execute("UPDATE games SET home_score = 8 WHERE league = 'GBL'")
    r = run(conn)
    assert r["excluded"] == 0 and r["unlocated"] == 2
    assert r["anchor"] == pytest.approx(0, abs=1e-9)
    assert abs(r["anchor"]) <= anchor_tolerance(SEASON)
    # the imputed shots are still in every player line: their points exceed their expected points
    lines = r["chunks"]["2025-10-27"][GID]["p"]
    all_in = (sum(p[5] for p in lines) - sum(p[4] for p in lines)) / sum(p[4] for p in lines)
    assert all_in < -FG_TOL


def test_export_season_dates_every_league_but_the_gbl():
    (row,) = run(synthetic_conn(league="EL"), "EL")["rows"]
    assert row["d"] == "2025-10-29" and row["wk"] == "2025-10-27"


def test_export_season_xfg_points_of_players_tie_to_the_side():
    # player 101's expected FG points are 1.0 off what his shots in shots_xfg are worth
    with pytest.raises(ValueError, match="expected FG points"):
        run(synthetic_conn(exp_101=3.2))


# --- reconstruction against the official final --------------------------------

def test_a_game_that_misses_its_final_is_left_out_and_counted():
    conn = with_second_game(synthetic_conn())
    conn.execute("UPDATE games SET home_score = 7 WHERE game_id = 'g1' AND league = 'GBL'")  # a point the events never saw
    r = run(conn)
    assert [g["id"] for g in r["rows"]] == ["g2"] and list(r["chunks"]["2025-10-27"]) == ["g2"]
    assert r["excluded"] == 1 and r["why"]["mismatch"] == 1


def test_a_forfeit_has_no_events_and_no_date():
    # ABA 2020-21: awarded 20-0, no team ids, no date, nothing played. Not a game left out: a forfeit.
    conn = synthetic_conn()
    conn.execute("INSERT INTO games VALUES ('g0', 'GBL', ?, '', NULL, NULL, 0, 20, NULL, NULL, NULL)", (SEASON,))
    r = run(conn)
    assert r["total"] == 2 and r["forfeits"] == 1 and r["excluded"] == 0 and len(r["rows"]) == 1
    assert r["why"]["no_shots"] == 0


def test_a_game_with_free_throws_and_no_shots_is_left_out_not_a_forfeit():
    conn = with_second_game(synthetic_conn())
    conn.execute("DELETE FROM events WHERE game_id = 'g1' AND etype = 'fg'")
    conn.execute("DELETE FROM shots WHERE game_id = 'g1'")
    r = run(conn)
    assert r["forfeits"] == 0 and r["excluded"] == 1 and r["why"]["no_shots"] == 1


def test_events_by_a_team_outside_the_game_leave_it_out():
    for event in (9, 7):    # a free throw, an unlocated shot
        conn = with_second_game(synthetic_conn())
        conn.execute("UPDATE events SET team_id = 'ZZZ' WHERE game_id = 'g1' AND event_id = ?", (event,))
        r = run(conn)
        assert r["excluded"] == 1 and r["why"]["no_team"] == 1 and len(r["rows"]) == 1


def test_a_player_on_both_sides_leaves_the_game_out():
    conn = with_second_game(synthetic_conn())  # 201 shoots for BBB; here he is also shooting free throws for AAA
    conn.execute("UPDATE events SET player_id = '201' WHERE game_id = 'g1' AND event_id IN (9, 10)")
    r = run(conn)
    assert r["excluded"] == 1 and r["why"]["side"] == 1


def test_a_shot_with_no_side_is_a_data_error():
    conn = synthetic_conn()
    conn.execute("UPDATE shots SET home_or_away = NULL WHERE event_id = 1")
    with pytest.raises(ValueError, match="side"):
        run(conn)


# --- real data ------------------------------------------------------------------

def test_a_completed_season_rebuilds_its_finals_and_sits_on_the_anchor():
    """EuroLeague 2020-21 on the built database: the play-by-play field goals the shot chart lacks are
    what let its games rebuild the official finals, so both gates hold."""
    if not os.path.exists(config.DB_PATH):
        pytest.skip("needs the built database")
    season = "2020-21"
    shown = official_names(pd.read_sql("SELECT ref1, ref2, ref3 FROM games WHERE league = 'EL'",
                                       export_site_data.conn))
    r = export_season(export_site_data.conn, "EL", season, export_site_data.team_abbrev_map("EL"), shown)
    assert len(r["rows"]) + r["excluded"] + r["forfeits"] == r["total"]
    assert r["excluded"] <= left_out_limit(r["total"] - r["forfeits"], "EL")
    assert abs(r["anchor"]) <= anchor_tolerance(season)
    assert r["unlocated"] > 0
