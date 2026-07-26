"""Euro OE configuration — leagues, seasons, features, paths.

Port of the NBA Over Expected system to EuroLeague and Liga ACB.
Feature *selection* happens at model time; tables are a superset.
"""

import datetime as _dt
import os

# ---------------------------------------------------------------- seasons

FIRST_SEASON_START = 2020


def _season_str(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _current_season_start(today: "_dt.date | None" = None) -> int:
    d = today or _dt.date.today()
    # European seasons tip off in late September (EL 2025-26 started
    # Sep 30, ACB 2023-24 on Sep 23); treat September onward as the new
    # season, July-August as the one that just finished.
    return d.year if d.month >= 9 else d.year - 1


SEASONS = [
    _season_str(y)
    for y in range(FIRST_SEASON_START, _current_season_start() + 1)
]

PRIOR_SEASON = {
    _season_str(y): _season_str(y - 1)
    for y in range(FIRST_SEASON_START, _current_season_start() + 1)
}

# ---------------------------------------------------------------- leagues

LEAGUES = ["EL", "EUC", "ACB", "LBA", "BBL", "ABA", "GBL", "WNBA"]

# EuroLeague season code: "2020-21" -> "E2020"
EL_SEASON_CODE = {s: f"E{s[:4]}" for s in SEASONS}
# EuroCup runs on the same platform as EuroLeague, competition code "U".
EUC_SEASON_CODE = {s: f"U{s[:4]}" for s in SEASONS}

# Liga ACB editionId: 85 = 2020-21 ... rolls forward one per season
ACB_EDITION = {s: 85 + (int(s[:4]) - 2020) for s in SEASONS}

EL_API = "https://api-live.euroleague.net"
EL_LIVE = "https://live.euroleague.net/api"
EL_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

LBA_API = "https://www.legabasket.it"
# Lega Basket Serie A regular-season championship id per season (from
# /api/championships/get-championships?cs_id=1, ctype_code == "RS").
LBA_CHAMP = {
    "2020-21": 221, "2021-22": 429, "2022-23": 466,
    "2023-24": 530, "2024-25": 567, "2025-26": 596,
}

# ABA Liga (Adriatic): aba-liga.com season code = 2-digit start year.
ABA_SEASON = {s: int(s[:4]) - 2000 for s in SEASONS}

# WNBA plays a single calendar-year season; map each shared "2020-21"-style
# season string to just its first year (the actual nba_api season value).
WNBA_SEASON = {s: s[:4] for s in SEASONS}

BBL_API = "https://api.basketball-bundesliga.de"
# Static public credentials from the easycredit-bbl.de JS bundle. The key
# must be exactly this; the secret is not validated (any non-empty string).
BBL_HEADERS = {
    "x-api-key": "publicWebUser",
    "x-api-secret": "publicWebUser",
    "User-Agent": "Mozilla/5.0",
}
# Basketball Bundesliga seasonId = the season's start year.
BBL_SEASON = {s: int(s[:4]) for s in SEASONS}
# BBL's SCB feed did not tag the shooter on MISSED shots (or on 2nd+ free
# throws) before 2023-24, which makes per-player shot value impossible for
# the earlier seasons. Ship only the seasons with complete attribution.
BBL_MIN_SEASON = "2023-24"

ACB_API = "https://api2.acb.com"
# Static key embedded in live.acb.com's public JS bundle. If it ever
# rotates, grep a live.acb.com game-page chunk for `API_KEY:"`.
ACB_API_KEY = "0dd94928-6f57-4c08-a3bd-b1b2f092976e"

EL_SLEEP = 1.2    # seconds between calls (live API 429s faster pulls)
ACB_SLEEP = 0.8
LBA_SLEEP = 0.3   # legabasket.it native API; no rate limit observed
BBL_SLEEP = 0.3   # api.basketball-bundesliga.de; no rate limit observed
ABA_SLEEP = 0.5   # aba-liga.com; be polite, no documented limit
# Genius Sports FIBA LiveStats CDN (data.json per game). Static files on a
# CDN; polite but no known hard rate limit.
GENIUS_SLEEP = 0.4
# stats.nba.com via nba_api; same courtesy delay NBA_OE uses for the NBA.
WNBA_SLEEP = 0.6
RETRIES = 3
TIMEOUT = 45

# ---------------------------------------------------------------- geometry
# Normalized shot frame for both leagues: meters from the center of the
# hoop; x = lateral (+/- toward sidelines), y = toward midcourt.
# FIBA court: arc 6.75 m, corner-3 line 6.60 m; the straight corner
# segments end 2.99 m from the baseline. Hoop center is 1.575 m from the
# baseline. Key (paint) is 4.9 m wide and 5.8 m deep from the baseline.

FIBA_HOOP_TO_BASELINE = 1.575
FIBA_RA_RADIUS = 1.25
FIBA_PAINT_HALF_WIDTH = 2.45
FIBA_PAINT_DEPTH_FROM_HOOP = 5.8 - FIBA_HOOP_TO_BASELINE   # 4.225
FIBA_CORNER_Y_MAX = 2.99 - FIBA_HOOP_TO_BASELINE           # 1.415

# ---------------------------------------------------------------- features

# Possession-level xFTA model (Poisson GLM), per league
XFTA_FEATURES = [
    "period",
    "seconds_remaining_in_period",
    "score_margin",        # at possession start (shifted at train time)
    "in_bonus",            # real FIBA team-foul penalty state (>=4 fouls)
    "offense_is_home",
    "opp_rate_logo",       # defense season foul rate, leave-one-game-out
    "crew_rate_logo",      # officiating crew season rate, leave-one-game-out
]

# xFG model (LightGBM), shared numeric features
XFG_NUM_FEATURES = [
    "distance_m",
    "angle_deg",
    "x_m",
    "y_m",
    "period",
    "seconds_remaining_in_period",
    "run_margin",
]
XFG_CAT_FEATURES = ["zone_basic", "zone_area", "shot_type"]

# League-specific binary context flags appended to XFG_NUM_FEATURES.
# second_chance / off_turnover are derived from possession structure for
# BOTH leagues (build_possessions), so they never leak the make/miss
# outcome. The EuroLeague Points feed's native fastbreak/second_chance/
# off_turnover flags are stamped only on MADE shots (points-off-X stats)
# and would leak shot_made — they are deliberately NOT used. ACB adds
# `dunk` (from playType 100/533); EuroLeague PBP has no dunk marker.
XFG_FLAGS = {
    "EL": ["second_chance", "off_turnover"],
    "EUC": ["second_chance", "off_turnover"],
    "ACB": ["second_chance", "off_turnover", "dunk"],
    # LBA (dunk flag in pbp) and BBL (DUNK qualifier) both mark dunks;
    # second_chance / off_turnover are possession-derived for both.
    "LBA": ["second_chance", "off_turnover", "dunk"],
    "BBL": ["second_chance", "off_turnover", "dunk"],
    "GBL": ["second_chance", "off_turnover", "dunk"],
    # ABA's shooting-chart feed carries no dunk marker.
    "ABA": ["second_chance", "off_turnover"],
    # WNBA has a real dunk marker (subType); second_chance/off_turnover
    # possession-derived like every other league.
    "WNBA": ["second_chance", "off_turnover", "dunk"],
}

# LightGBM hyperparameters — identical to the NBA system
LGBM_PARAMS = dict(
    objective="binary",
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=300,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
)

# Leaderboard qualification (finisher-possessions). NBA used 300 over an
# 82-game schedule; European seasons are 34-38 rounds, so scale to ~40%.
QUALIFY_POSS = 125

# ---------------------------------------------------------------- paths

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_REPO_ROOT, "cache")
DB_PATH = os.path.join(_REPO_ROOT, "euro_oe.db")
