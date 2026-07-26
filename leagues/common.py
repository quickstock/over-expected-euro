"""Shared HTTP + normalized-event vocabulary for the league adapters.

Every adapter emits the same normalized shapes:

game row     {game_id, league, season, date, home_team_id, away_team_id,
              home_score, away_score, ref1, ref2, ref3}
event row    {game_id, event_id, src_num, period, secs_remaining, team_id,
              player_id, player_name, etype, shot_value, shot_made,
              foul_kind, foul_fts, score_home, score_away}
shot row     {game_id, event_id, player_id, team_id, period, secs_remaining,
              shot_made, shot_value, x_m, y_m, fastbreak, second_chance,
              off_turnover, dunk}

etype vocabulary:
  fg, ft, foul, foul_drawn, reb_off, reb_def, turnover, steal, assist,
  block, sub, timeout, jumpball, period_start, period_end, other
foul_kind vocabulary:
  personal, offensive, technical, unsportsmanlike, bench
(shots carry x/y in meters from hoop center: x lateral, y toward midcourt)
"""

import json
import os
import re
import time

import requests

import config

COUNTED_FOUL_KINDS = {"personal", "offensive", "unsportsmanlike"}  # team fouls
EXCLUDED_TRIP_FOUL_KINDS = {"technical", "unsportsmanlike", "bench"}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # Unicode letters only


def titlecase_name(s):
    """ALL-CAPS "SURNAME, FIRST" -> "Surname, First". Hyphens, apostrophes,
    and the comma split into separate runs, each capitalized on its own
    (Garcia-Lopez, O'Neal), so punctuation and word order are untouched —
    only casing changes. EuroLeague's PBP/bio/referee feeds are shouting
    caps; ACB's are already properly cased and never passed through this."""
    if not s:
        return s
    s = re.sub(r",(?=\S)", ", ", s)  # a handful of EL rows omit the space
    return _WORD_RE.sub(lambda m: m.group(0)[:1].upper() + m.group(0)[1:].lower(), s)


def fetch_json(url, headers=None, params=None, sleep=0.5):
    """GET with retries; returns parsed JSON or raises.

    429s back off hard (the EuroLeague live API rate-limits sustained
    pulls) — wait out the window instead of burning retries.
    """
    last_err = None
    for attempt in range(config.RETRIES + 3):
        try:
            r = requests.get(url, headers=headers, params=params,
                             timeout=config.TIMEOUT)
            if r.status_code == 200:
                time.sleep(sleep)
                return r.json()
            last_err = f"HTTP {r.status_code}"
            if r.status_code in (401, 403, 404):
                break  # not transient
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = int(retry_after) if (retry_after or "").isdigit() \
                    else 45 * (attempt + 1)
                time.sleep(wait)
                continue
        except (requests.RequestException, json.JSONDecodeError) as e:
            last_err = repr(e)
        time.sleep(sleep * (2 ** (attempt + 1)))
    raise RuntimeError(f"GET {url} failed: {last_err}")


def cache_path(*parts):
    p = os.path.join(config.CACHE_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def cache_write(obj, *parts):
    # atomic: a reader never sees a half-written file
    p = cache_path(*parts)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, p)


def cache_read(*parts):
    with open(os.path.join(config.CACHE_DIR, *parts)) as f:
        return json.load(f)


def cache_exists(*parts):
    return os.path.exists(os.path.join(config.CACHE_DIR, *parts))
