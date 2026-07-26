# Euro Over Expected

A shot-value system for **EuroLeague** and **Liga ACB**, ported from the
NBA [Over Expected](https://overexpected.com) build. It asks one question
three ways: how much does a player add over what an average player would do
with the same looks?

- **Shot value** — points over expected per 100 possessions, fusing the
  shot and the fouls it draws (headline: `points_oe_per100`).
- **Shot-making** — field-goal points over expected: actual conversion
  against the difficulty of the looks taken (`FG% − xFG%`).
- **Foul-drawing (FTAOE)** — free throws drawn over the league rate per 100
  possessions.

Six seasons, 2020-21 to 2025-26, regular season only. 3,754 games, 2.1M
play-by-play events, 474k located field-goal attempts, 548k possessions.
Two leagues in one SQLite DB; models trained per league, never pooled.

No website — results are browsed via `search.py` (CLI) and `app.py`
(Streamlit).

## Quick start

```bash
pip install -r requirements.txt

python3 pull.py --league EL          # network -> cache/ (idempotent)
python3 pull.py --league ACB
python3 build_tables.py              # cache -> euro_oe.db base tables
python3 build_possessions.py         # events -> possessions + sfta target
python3 train_xfta.py                # Poisson GLM cross-fit -> FTAOE board
python3 shot_value.py                # LightGBM xFG + shot-value suite
python3 validate.py                  # the gate — must pass before trusting

python3 search.py "shermadini"                       # find a player
python3 search.py --league EL --season 2024-25 --top 20
python3 search.py --league ACB --season 2024-25 --sort ftaoe --top 15
streamlit run app.py                                 # browsable leaderboards
```

`pull.py` is idempotent — per-game caches are permanent, the current
season's fixture list is re-fetched each run so new games are picked up. A
full cold pull is roughly 20 min (ACB) + longer (EuroLeague, which
rate-limits with 429s and paces at ~1 call/s).

## How it works

```
league API  ->  cache/ (json)  ->  euro_oe.db  ->  models  ->  leaderboards
  pull.py         raw            build_tables    LightGBM /     search.py
                                 build_poss.     statsmodels    app.py
```

The two leagues expose different feeds; `leagues/euroleague.py` and
`leagues/acb.py` normalize both into one event/shot vocabulary
(`leagues/common.py`), and everything downstream is league-agnostic.

**Leak-free by construction.** For each season the models train on the
other five and predict the held-out one, so a shot's or possession's
expected value never comes from a model that saw it. Within-season
residuals are anchored to sum to ~0, so seasons are directly comparable
rather than drifting with each league's foul environment.

**The possession is the unit for foul-drawing.** A fouled miss has no shot
location and is not a charged FGA, so any per-shot foul rate silently drops
the exact plays the stat is about. FTAOE is per 100 possessions.

## What's modeled

| model | file | grain | method |
|---|---|---|---|
| expected FTA (FTAOE) | `train_xfta.py` | possession | Poisson GLM, 7 context features, season cross-fit + per-season anchor |
| expected FG% (xFG) | `shot_value.py` | field-goal attempt | LightGBM on shot context only (never the shooter), season cross-fit + anchor |
| shot value | `shot_value.py` | player-season | xFG (make value) + xFTA (foul value) → points over expected /100 |

Qualification is 125 finisher-possessions (European seasons are 34-38
rounds vs the NBA's 82 games). 1,898 qualified player-seasons across both
leagues.

## Data sources

- **EuroLeague** (`EL`): `api-live.euroleague.net/v2` for season game lists
  (with referees) and player bios; `live.euroleague.net/api` for
  PlayByPlay + Points (shot chart). No auth; a browser User-Agent is
  required and sustained pulls get 429s (handled by backoff).
- **Liga ACB** (`ACB`): `api2.acb.com` — the JSON backend of live.acb.com —
  with a static `X-APIKEY` from their public JS bundle (in `config.py`).
  Unofficial access; be polite and cache aggressively.

See [CLAUDE.md](CLAUDE.md) for the full data contract, schema, and every
deviation from the NBA build.

## Honesty and scope

The numbers are descriptive. They blend playstyle, contact-seeking skill,
and officiating, and do not isolate or prove referee bias. Models are
leak-free (season cross-fit) and anchored per season. `validate.py` is a
hard gate: FT-ledger identities, anchoring, xFG calibration (ECE < 0.02),
possession sanity, and final-score reconstruction all have to pass.

Built by Kevin Krajnc. `cache/` and `euro_oe.db` are local artifacts
(gitignored).
