"""ABA Liga (Adriatic) adapter.

Source: aba-liga.com's own public pages, no auth. Three fetches:
  - the season calendar page (`calendar/{sez}/1/`), once per season, for
    match ids + date + official final score;
  - the Boxscore page per game, which embeds BOTH the box score AND the
    full play-by-play (one `<table class="match_table_play_by_play">` per
    period; rows run reverse-chronological within each table — end of
    period first, tip-off last — so each table is reversed before use)
    plus the referee names;
  - the shooting-chart JSON (`create_shooting_chart.php`), which carries
    shot x/y but no period/clock/shot-type/FT flag, so it's zipped onto
    the PBP's field-goal events in per-(team, player) chronological order
    (verified 1:1 against a real game: identical made/miss sequences).

Unlike EuroLeague/ACB/LBA/BBL, foul-drawn <-> foul-committed linkage and
free-throw trip position are both explicit in the PBP text (the fouling
and fouled-on events share a clock; free throws say "(N of M)"), so no
extra adapter-side linking is needed — `build_possessions.py`'s generic
(player, period, clock) trip grouping already does the right thing here.

Coordinates: `koordinata_x`/`koordinata_y` are 0-100 court percentages, one
row per team's OWN attacking half for the whole game (no period flip —
same convention LBA uses), confirmed by median-x separation by team.
"""

import re
import urllib.parse

import config
from leagues.common import (cache_exists, cache_read, cache_write,
                            fetch_json, titlecase_name)

try:
    from bs4 import BeautifulSoup
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ABA adapter needs beautifulsoup4 (pip install beautifulsoup4)"
    ) from e

LEAGUE = "ABA"
_BASE = "https://www.aba-liga.com"
_UA = {"User-Agent": "Mozilla/5.0"}

COURT_LEN_M = 28.0
COURT_WID_M = 15.0
HOOP_FROM_BASELINE_M = config.FIBA_HOOP_TO_BASELINE

_QUARTER_RE = re.compile(r"^(start|end) of (\d+)\. quarter$")
_FG_RE = re.compile(r"(made|missed) (\d) points")
_FT_RE = re.compile(r"(made|missed) free throw \((\d)\s*of\s*(\d)\)")
_FOUL_RE = re.compile(r"foul \((\w+)\)")


def _fetch_text(url):
    import requests
    r = requests.get(url, headers=_UA, timeout=config.TIMEOUT)
    r.raise_for_status()
    return r.text


def _foul_kind(kind):
    k = (kind or "").lower()
    if "offensiv" in k:
        return "offensive"
    if "technical" in k:
        return "technical"
    if "unsportsmanlike" in k or "disqualif" in k:
        return "unsportsmanlike"
    return "personal"


# ------------------------------------------------------------------ pull

def _calendar(season):
    part = (LEAGUE, f"calendar_{season}.json")
    if season == config.SEASONS[-1] or not cache_exists(*part):
        sez = config.ABA_SEASON[season]
        html = _fetch_text(f"{_BASE}/calendar/{sez}/1/")
        soup = BeautifulSoup(html, "html.parser")
        rows = []
        for tr in soup.select("table.league_calendar_table tbody tr"):
            a = tr.find("a", href=re.compile(r"/match/\d+/"))
            if not a:
                continue
            m = re.search(r"/match/(\d+)/", a["href"])
            if not m:
                continue
            gid = int(m.group(1))
            score_td = tr.find("td", class_="scoretable")
            score_txt = score_td.get_text(strip=True) if score_td else ""
            sm = re.match(r"(\d+)\s*:\s*(\d+)", score_txt)
            home_score = int(sm.group(1)) if sm else None
            away_score = int(sm.group(2)) if sm else None
            date_td = tr.find("td", class_="locationtable")
            date_txt = date_td.get_text(" ", strip=True) if date_td else ""
            dm = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", date_txt)
            date = f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}" if dm else ""
            rows.append({
                # ABA match ids reset each season, but the rest of the
                # pipeline assumes game_id is unique per league — so the
                # DB key carries the season, while game_code stays the raw
                # id used for fetch URLs and cache filenames.
                "game_id": f"{sez}-{gid}", "game_code": gid,
                "league": LEAGUE, "season": season, "date": date,
                "home_team_id": None, "away_team_id": None,
                "home_team_name": None, "away_team_name": None,
                "home_score": home_score, "away_score": away_score,
                "ref1": None, "ref2": None, "ref3": None,
            })
        cache_write(rows, *part)
    return cache_read(*part)


def list_games(season):
    if season not in config.ABA_SEASON:
        return []
    sez = config.ABA_SEASON[season]
    games = _calendar(season)
    # recompute game_id from game_code so cached calendars from an earlier
    # id scheme still resolve to the current unique key
    for g in games:
        g["game_id"] = f"{sez}-{g['game_code']}"
    # only games with a final score are finished
    return [g for g in games
            if g["home_score"] is not None and g["away_score"] is not None]


def pull_game(season, game_code):
    # Match ids RESET each season (season 1 restarts near id 1), so the
    # cache key must include the season — otherwise a later season's fetch
    # for a numerically-overlapping id would silently reuse an earlier
    # season's cached page for `cache_exists` and never re-fetch.
    sez = config.ABA_SEASON[season]
    box_part = (LEAGUE, "box", f"{season}_{game_code}.html")
    if not cache_exists(*box_part):
        html = _fetch_text(
            f"{_BASE}/match/{game_code}/{sez}/1/Boxscore/q1/1/home/x/")
        cache_write(html, *box_part)
    shots_part = (LEAGUE, "shots", f"{season}_{game_code}.json")
    if not cache_exists(*shots_part):
        url = (f"{_BASE}/live-match/rezultati-1718/create_shooting_chart.php"
               f"?id={game_code}&sez={sez}&lea=1")
        cache_write(fetch_json(url, headers=_UA, sleep=config.ABA_SLEEP),
                    *shots_part)
    return True


def _box_soup(game):
    return BeautifulSoup(
        cache_read(LEAGUE, "box", f"{game['season']}_{game['game_code']}.html"),
        "html.parser")


def referees(game):
    try:
        soup = _box_soup(game)
    except FileNotFoundError:
        return [None, None, None]
    span = soup.find("span", class_="smallCaps", string=re.compile("Referee"))
    if not span or not span.next_sibling:
        return [None, None, None]
    text = str(span.next_sibling).split("<br")[0]
    names = [n.strip() for n in text.split(",") if n.strip()]
    return (names + [None, None, None])[:3]


# ------------------------------------------------------------- normalize

def _cell_info(td, sez, lea):
    """(club_id, player_id, display_name, action_text) for a populated
    event cell, or None if the cell has no player link at all."""
    img = td.find("img")
    club_id = None
    if img and img.get("src"):
        m = re.search(r"/club/50x50/(\d+)\.png", img["src"])
        club_id = m.group(1) if m else None
    a = td.find("a", href=re.compile(r"^/player/\d+/"))
    if not a:
        return None
    pm = re.search(r"/player/(\d+)/", a["href"])
    player_id = pm.group(1)
    display_name = a.get_text(strip=True)
    # text after the anchor within this cell (the action), minus a
    # leading ". " separator when present
    tail = "".join(str(s) for s in a.next_siblings)
    tail = BeautifulSoup(tail, "html.parser").get_text(" ", strip=True)
    action = tail[2:] if tail.startswith(". ") else tail.lstrip(". ").strip()
    return club_id, player_id, display_name, action


def normalize_game(season, game):
    gid = game["game_id"]            # unique DB key (season-prefixed)
    game_code = game["game_code"]    # raw id, for cache filenames
    soup = _box_soup(game)
    sez = config.ABA_SEASON[season]

    tables = soup.select("table.match_table_play_by_play")
    # home = left column, away = right column (matches the calendar's
    # "home-slug-away-slug" URL order and the header row team order)
    home_id = away_id = None
    for t in tables:
        for tr in t.select("tbody tr"):
            tds = tr.find_all("td")
            if len(tds) != 3:
                continue
            hi = _cell_info(tds[0], sez, 1)
            ai = _cell_info(tds[2], sez, 1)
            if hi and home_id is None:
                home_id = hi[0]
            if ai and away_id is None:
                away_id = ai[0]
        if home_id and away_id:
            break
    game["home_team_id"], game["away_team_id"] = home_id, away_id
    if tables:
        hdr = tables[0].select("thead td")
        if len(hdr) == 3:
            game["home_team_name"] = hdr[0].get_text(strip=True) or None
            game["away_team_name"] = hdr[2].get_text(strip=True) or None
    game["ref1"], game["ref2"], game["ref3"] = referees(game)

    # shots JSON: coordinates only, no period/clock/value/FT flag. Build a
    # per-(team, player) queue in file order (== chronological, verified).
    try:
        shots_raw = cache_read(LEAGUE, "shots", f"{season}_{game_code}.json")
    except FileNotFoundError:
        shots_raw = []
    shot_name = {}
    shot_queue = {}
    by_team_x = {"1": [], "2": []}
    for s in shots_raw:
        tid = home_id if s.get("ekipa") == "1" else away_id
        key = (tid, str(s.get("player_id")))
        shot_queue.setdefault(key, []).append(s)
        if s.get("player_name"):
            shot_name[str(s.get("player_id"))] = s["player_name"]
        if s.get("koordinata_x") is not None:
            by_team_x[s.get("ekipa")].append(float(s["koordinata_x"]))
    # Each team's shots sit on ITS OWN attacking half for the whole game
    # (no period flip — same convention as lba.py); resolve per team_id.
    import statistics as _stats
    sides = {}
    for ek, tid in (("1", home_id), ("2", away_id)):
        if by_team_x[ek]:
            sides[tid] = _stats.median(by_team_x[ek]) > 50

    events, shots = [], []
    eid = 0
    sh = sa = 0  # running score, reconstructed from makes (see lba.py)
    for period_idx, table in enumerate(tables, start=1):
        rows = list(reversed(table.select("tbody tr")))  # -> chronological
        plen = 600 if period_idx <= 4 else 300
        for tr in rows:
            tds = tr.find_all("td")
            if len(tds) != 3:
                continue
            home_cell, clock_cell, away_cell = tds
            clock_text = clock_cell.get_text(" ", strip=True)
            clock_line = clock_text.split("\n")[0].strip()

            qm = _QUARTER_RE.match(clock_line)
            if qm:
                eid += 1
                events.append({
                    "game_id": gid, "event_id": eid, "src_num": eid,
                    "period": period_idx, "secs_remaining":
                        plen if qm.group(1) == "start" else 0,
                    "team_id": None, "player_id": None, "player_name": None,
                    "etype": "period_start" if qm.group(1) == "start"
                             else "period_end",
                    "shot_value": None, "shot_made": None,
                    "foul_kind": None, "foul_fts": 0,
                    "score_home": sh, "score_away": sa,
                })
                continue
            if clock_line in ("game started", "game finished", "jumpball"):
                continue  # redundant with the quarter markers / a side event

            is_home = bool(home_cell.find("a"))
            cell = home_cell if is_home else away_cell
            info = _cell_info(cell, sez, 1)
            if info is None:
                continue
            club_id, player_id, disp_name, action = info
            if "in starting lineup" in action:
                continue
            team_id = home_id if is_home else away_id

            mclock = re.match(r"(\d{1,2}):(\d{2})", clock_line)
            elapsed = int(mclock.group(1)) * 60 + int(mclock.group(2)) \
                if mclock else 0
            secs = max(0, plen - elapsed)

            name = shot_name.get(player_id) or titlecase_name(disp_name)

            etype, val, made, foul_kind = "other", None, None, None
            fg = _FG_RE.search(action)
            ft = _FT_RE.search(action)
            if fg:
                etype = "fg"
                made = 1 if fg.group(1) == "made" else 0
                val = int(fg.group(2))
            elif ft:
                etype = "ft"
                made = 1 if ft.group(1) == "made" else 0
                val = 1
            elif action == "foul on":
                etype = "foul_drawn"
            elif action.startswith("foul"):
                etype = "foul"
                fm = _FOUL_RE.search(action)
                foul_kind = _foul_kind(fm.group(1) if fm else None)
            elif action.startswith("rebound (defensive"):
                etype = "reb_def"
            elif action.startswith("rebound (offensive"):
                etype = "reb_off"
            elif action.startswith("turnover"):
                etype = "turnover"
            elif action.startswith("assist"):
                etype = "assist"
            elif action.startswith("steal"):
                etype = "steal"
            elif action.startswith("block"):
                etype = "block"
            elif action.startswith("substitution"):
                etype = "sub"
            elif action.startswith("jumpball"):
                etype = "jumpball"

            if etype in ("fg", "ft") and made:
                if is_home:
                    sh += val
                else:
                    sa += val

            eid += 1
            events.append({
                "game_id": gid, "event_id": eid, "src_num": eid,
                "period": period_idx, "secs_remaining": secs,
                "team_id": team_id, "player_id": player_id,
                "player_name": name,
                "etype": etype, "shot_value": val, "shot_made": made,
                "foul_kind": foul_kind, "foul_fts": 0,
                "score_home": sh, "score_away": sa,
            })

            if etype == "fg":
                queue = shot_queue.get((team_id, player_id))
                s = queue.pop(0) if queue else None
                if s is not None and team_id in sides:
                    x_m, y_m = _to_meters(s["koordinata_x"], s["koordinata_y"],
                                          sides[team_id])
                    shots.append({
                        "game_id": gid, "event_id": eid,
                        "player_id": player_id, "team_id": team_id,
                        "period": period_idx, "secs_remaining": secs,
                        "shot_made": made, "shot_value": val,
                        "x_m": x_m, "y_m": y_m,
                        "second_chance": 0, "off_turnover": 0, "dunk": 0,
                    })
    return events, shots


def _to_meters(x, y, attacks_right):
    xm_court = float(x) / 100.0 * COURT_LEN_M
    ym_court = float(y) / 100.0 * COURT_WID_M
    if attacks_right:
        y_m = (COURT_LEN_M - HOOP_FROM_BASELINE_M) - xm_court
        x_m = ym_court - COURT_WID_M / 2
    else:
        y_m = xm_court - HOOP_FROM_BASELINE_M
        x_m = COURT_WID_M / 2 - ym_court
    return x_m, y_m
