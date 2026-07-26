# Euro Over Expected

Port of the NBA Over Expected system (`../NBA_OE`, live at
overexpected.com) to a registry of leagues beyond the NBA — despite the
name, now eight of them: **EuroLeague, EuroCup, Liga ACB, LBA, BBL, ABA,
GBL, and the WNBA** (the last one American, routed here anyway — see
`../CLAUDE.md`'s Layout section for why). Same three lenses: shot value
(points over expected /100), shot-making (FG points over expected vs
xFG%), and foul-drawing (FTA over expected). Six seasons, 2020-21 to
2025-26 (regular season only), except where a league's own data floor is
shorter (noted per league below).

**Live: https://euro-overexpected.vercel.app** (Vercel project
`euro-overexpected` under quickstock-s-projects). The site in `site/` is
the NBA front end ported: React + Vite + TS, league toggle in the nav
(EL / ACB, both datasets load on boot; player league inferred from id —
EL ids start with "P", ACB ids are numeric), EuroLeague-orange warm pole
(oklch 0.62 0.195 45), FIBA court geometry in CourtZones, crackdown
story removed, per-league data files. Results are also browsable via
`search.py` (CLI) and `app.py` (Streamlit).

## How to run

```bash
python3 pull.py --league EL          # network -> cache/ (idempotent)
python3 pull.py --league ACB
python3 build_tables.py              # cache -> euro_oe.db base tables
python3 build_possessions.py         # events -> possessions + sfta
python3 train_xfta.py                # Poisson GLM cross-fit -> FTAOE board
python3 shot_value.py                # LightGBM xFG + shot-value suite
python3 validate.py                  # the gate — run before trusting results

python3 search.py "player name"      # CLI search
streamlit run app.py                 # browsable leaderboards

python3 export_site_data.py          # euro_oe.db -> site/public/*.json
cd site && npm run build && node scripts/generate-static.mjs \
  --base https://euro-overexpected.vercel.app   # + OG cards & shells
vercel --prod --yes                  # deploy (run from site/)
```

## Data sources

- **EuroLeague** (league code `EL`): `api-live.euroleague.net/v2` for
  season game lists (with referees) and player bios;
  `live.euroleague.net/api` for PlayByPlay + Points (shot chart). No
  auth; browser User-Agent required; sustained pulls get 429s, handled
  by hard backoff in `leagues/common.py` (keep `EL_SLEEP >= 1.2`).
- **EuroCup** (league code `EUC`): the identical Euroleague Basketball SA
  platform as EL, just competition code `"U"` instead of `"E"`
  (`leagues/eurocup.py` is a near-verbatim copy of `euroleague.py`). Same
  rate-limit behavior — a full 6-season pull runs several hours under
  sustained `EL_SLEEP`-paced backoff, budget for it.
- **WNBA** (league code `WNBA`): official stats.nba.com via `nba_api`,
  `league_id="10"` (`leagues/wnba.py`) — the same infrastructure NBA_OE
  uses for the NBA, routed through this pipeline instead because it's the
  registry built for onboarding a league from an adapter, not NBA_OE's
  bespoke one-league system. `leaguegamefinder` for schedule/scores,
  `playbyplayv3` for events + shot coordinates (`xLegacy`/`yLegacy`,
  tenths of a foot — converted to meters same as every other league),
  `boxscoresummaryv2` for officials. WNBA plays a single calendar-year
  season; `config.WNBA_SEASON` maps the shared "2020-21"-style strings to
  just the first year for the actual API calls, so every date utility
  built around the cross-year format keeps working unmodified. Offensive
  vs. defensive rebounds aren't labeled in the feed (`subType` is almost
  always "Unknown") — resolved by comparing the rebounding team to the
  team of the immediately preceding missed shot/FT. **"Instant Replay"
  review-ruling rows stamp a stale/garbage `scoreHome`/`scoreAway`** (e.g.
  8-9 mid-game while the real score was 15-20, confirmed the sole
  offender via a full-cache scan across every non-scoring `actionType`) —
  those two fields are nulled specifically for that type; before the fix
  this corrupted-score-skipped 52 games, after it only 7 (heterogeneous
  one-off feed glitches, in line with every other league's accepted
  mismatch rate).
- **Liga ACB** (league code `ACB`): `api2.acb.com` — the JSON backend of
  live.acb.com — with a static `X-APIKEY` from their public JS bundle
  (in `config.py`; if it rotates, grep a live.acb.com game-page chunk
  for `API_KEY:"`). Regular season = calendar rounds with
  `subphaseId == null`. Team identity across seasons is `clubId`.
  Editions: 85 = 2020-21 ... 90 = 2025-26. Unofficial access — be
  polite (`ACB_SLEEP >= 0.8`), cache aggressively.
- **Lega Basket Serie A** (league code `LBA`, Italy): legabasket.it's own
  open JSON API (`leagues/lba.py`). `championships/get-championships` →
  regular-season championship id per season (`config.LBA_CHAMP`);
  `...-calendar-by-id?id={champ}[&d={day}]` → fixtures;
  `...-matches-play-by-play?id={gid}` → pbp with x/y (0-100 court %,
  half-folded per team, no period flip), fouls, FTs;
  `...-matches-by-id?id={gid}` → referees. Clock from `print_time`
  (older seasons: `minute`/`seconds` elapsed). No auth. All 6 seasons.
- **Basketball Bundesliga** (league code `BBL`, Germany):
  `api.basketball-bundesliga.de` (`leagues/bbl.py`), static public headers
  (`config.BBL_HEADERS`). `/games?stages[]=MAIN_ROUND&competition=BBL&
  seasonId={startYear}` → fixtures; `/games/{id}/stats` → coordinate
  actions + rosters (`realPlayerId` stable id, name, position);
  `/games/{id}` → refs. **Only 2023-24+ (`config.BBL_MIN_SEASON`)**: the
  SCB feed didn't tag the shooter on missed shots / 2nd+ free throws
  before then, which breaks per-player shot value.
- **Genius FIBA LiveStats** (`leagues/genius.py`, uniform for any league on
  the platform): `fibalivestats.dcd.shared.geniussports.com/data/{id}/
  data.json` — pbp + per-team `shot` arrays (x/y join on `actionNumber`).
  Drops phantom duplicate-feed events (score-regression signal) and casts
  string-typed `s1`/`s2` scores. Powers **GBL** (Greece, `leagues/gbl.py`):
  its game-ids aren't in any open Genius schedule, but the Wayback Machine
  crawl of the viewer pages (`/u/ESAKE/{id}/`, CDX API) recovers them —
  bucketed by numeric id range into 4 clean seasons (2022-23 → 2025-26),
  frozen in `leagues/data/gbl_matchids.json`. BSL (Turkey) also parses with
  `genius.py` but its enumerator (TBF site) is Cloudflare-gated and only
  maps 2 of 6 seasons, so it's not wired. PROA/France has no free
  coordinate source at all.
- **ABA Liga** (league code `ABA`, Adriatic): a bespoke scrape of
  aba-liga.com (`leagues/aba.py`, needs `beautifulsoup4`). Shot x/y from
  `create_shooting_chart.php?id={id}&sez={YY}&lea=1`; fouls, free throws
  (explicit "N of M" trips + paired foul-on/foul events), referees, and
  team names all from the Boxscore page's embedded per-period play-by-play
  tables. `sez` = 2-digit start year. **Match ids reset each season**, so
  the DB `game_id` is season-prefixed (`{sez}-{raw}`) while `game_code`
  stays the raw id for URLs/cache. Coordinates are 0-100 court %,
  half-folded per team. All 6 seasons.

## Decisions and deviations from the NBA build

- **sfta = FTs drawn from personal fouls, bonus included.** FIBA PBP
  types fouls by awarded FTs, not shooting vs penalty, and a 2-FT trip
  in the bonus is genuinely ambiguous. The NBA build excluded bonus FTs;
  here they are in. Technical / unsportsmanlike / bench FTs are
  excluded; and-1s detected via made FG by the same player at the same
  clock (±2 s). The stat reads "free throws drawn over expected".
- **`in_bonus` is real** (defense ≥ 4 counted team fouls in the quarter,
  OT extends Q4, derived from the foul stream) — an upgrade over the
  NBA's period≥4 proxy.
- **Shot zones are FIBA-cut** from coordinates (arc 6.75 m, corner
  6.60 m to y = 1.415 m above the hoop line, RA 1.25 m). Both leagues
  normalized to meters from hoop center, x lateral / y toward midcourt
  (ACB's native axes are swapped and in mm; EuroLeague's are in cm).
- **No NBA-style `action_type`.** Context flags instead: EL has native
  fastbreak / second-chance / off-turnover per shot; ACB gets dunk from
  playType and second-chance / off-turnover derived from possessions.
- **Style-adjusted baseline dropped** (no tracking data in Europe).
  Height / position / prior-season rates were display-only in the NBA
  build and stay display-only here (EL bios from v2 `people`; ACB has
  no bio API — names come from PBP, position/height NULL).
- **Qualification 125 finisher-possessions** (NBA used 300 over 82
  games; European seasons are 34-38 rounds).
- **One DB, two leagues**: every table carries a `league` column
  (`EL` / `ACB`); models are trained per league, never pooled.
- EuroLeague PBP quarter arrays are the true event order
  (`NUMBEROFPLAY` is not monotone but joins Points.`NUM_ANOT`); ACB
  shots join PBP on (quarter, minute, second, playerLicenseId,
  playType) — verified 1:1, the `id` field does NOT join.

- **Context flags are leak-free and possession-derived for BOTH
  leagues.** EuroLeague's native Points-feed fastbreak / second-chance /
  off-turnover flags are stamped only on *made* shots (they are
  points-off-X stats), so they leak the make/miss label — they are NOT
  used. `second_chance` (OREB earlier in the possession) and
  `off_turnover` (prior possession ended in a turnover by the other
  team) are computed in `build_possessions` and stored in
  `possession_shot_flags`, which `shot_value` joins. ACB adds `dunk`
  (playType 100/533); EuroLeague PBP has no dunk marker.

## Schema (euro_oe.db)

`games, events, shots, player_season, player_season_ft` (build_tables) →
`possessions, and1_shots, possession_shot_flags` (build_possessions) →
`predictions_poss, player_season_xfta_lb` (train_xfta) →
`shots_xfg, shot_value, player_game_shot_value, team_shot_value`
(shot_value). The headline table is `shot_value`; `qualified = 1` rows
are the leaderboard.

## Site (site/)

- Port of `../NBA_OE/site`. Data contract: `data-{EL|ACB}.json` +
  `players-{LG}-{season}.json` from `export_site_data.py` (same shapes
  as the NBA export; player ids are strings; `meta.leagueFt` replaces
  the hardcoded 0.77; reliability computed inline, no paddingK).
- EuroLeague names are "SURNAME, FIRST" — `lastName()` in
  `src/lib/format.ts` handles both that and ACB's "First Last".
- ACB has no public positions → the boards hide the position filter
  when no row carries one. Color scale `SCALE_MAX = 18` (drawn-FT
  spreads run ~1.5x the NBA's shooting-foul-only stat).
- Deploy: `vercel --prod --yes` from `site/` (linked to
  `euro-overexpected`). `vercel.json` runs build + generate-static on
  Vercel; base URL comes from `VERCEL_PROJECT_PRODUCTION_URL`.

## Known defects

- **RETRACTED 2026-07-24: "WNBA `run_margin` over-corrects" was not a defect.**
  A previous entry here claimed WNBA's `run_margin` was biased and that
  published WNBA numbers were affected. **That was wrong, and the error was in
  the audit, not in this pipeline.** Recorded rather than deleted so the
  reasoning is not repeated.
  - *What the audit claimed:* the make-minus-miss separation goes from +0.084
    raw to −0.078 after subtracting the shot's own points, so the correction
    "overshoots" — and the supposed mechanism was that
    `playbyplayv3` omits `scoreHome`/`scoreAway` on some made-shot rows, letting
    `build_tables`' forward-fill supply a pre-shot score that then gets
    double-subtracted.
  - *Why it was wrong, measured two ways.* First, the stated mechanism does not
    exist: across 400 cached WNBA games, **zero** made field goals and **zero**
    made free throws are missing a score. Second, and decisively, WNBA stamps
    scores **post-event on 100.00% of made field goals** (77,723 of 77,726 —
    the cleanest rate of any league, alongside ABA and LBA), verified with a
    `LAG` window over the event stream in
    `xpts_calibration/wo7_margin_structural.py`. So subtracting the shot's own
    points does recover the pre-shot state, exactly as intended.
  - *What the −0.078 actually is.* The separation in a pre-shot feature
    decomposes into a **mechanical** part (the shot's own points) and a
    **behavioural** part (does the pre-shot margin genuinely differ between
    makes and misses?). Subtracting removes the mechanical part; whatever
    remains is a real predictive correlation — in the WNBA, trailing teams
    convert slightly better. That is a legitimate feature-outcome relationship,
    not leakage. The audit's pass criterion was a ratio test
    (`|corrected| < 0.5 × |raw|`) which only works when the behavioural
    component is negligible; in the WNBA it is not.
  - *Resolution:* the audit's post-shot check is now **structural** — it asks
    whether the stored score is stamped after the event, which makes the
    subtraction valid by construction — with the statistical figures kept as
    information rather than as the gate. G2 now passes all eight leagues, and
    WNBA is back in every calibration figure. No pipeline change was needed and
    none was made; **no published WNBA number was ever affected.**

## State

- **2026-07-13: EuroCup + WNBA added**, both full 6-season pull → build →
  model → validate chains clean (`validate.py` gate passes all eight
  leagues). See the Data sources section above for both adapters and the
  WNBA Instant Replay score-corruption fix. EuroCup's pull ran under the
  same rate-limit backoff as EuroLeague — several hours for 6 seasons,
  not a bug, budget for it on any future re-pull.
- Built 2026-07-08, hardened 2026-07-10, site shipped 2026-07-10.
  Full pulls + models run end-to-end; `validate.py` is the gate (all
  checks pass both leagues, six seasons). `cache/` and `euro_oe.db` are
  local artifacts (gitignored).
- Weekly refresh: `pull.py` both leagues (current-season lists are
  never cached), rebuild chain, `validate.py`, `export_site_data.py`,
  rebuild + redeploy site.
- An adversarial code review (3 lenses × verify) caught and fixed: the
  EuroLeague xFG leak above; an ACB context-flag backfill that never
  completed (correlated UPDATE → replaced by the `possession_shot_flags`
  join); an `atan2` clamp that collapsed behind-the-hoop shot angles; a
  LOGO-rate zero-denominator that skipped the season-mean fallback; an
  ACB shot↔PBP join that dropped rows on same-second collisions (now
  FIFO); (0,0)-sentinel EuroLeague shots (now excluded); score-margin
  re-signing (aligned to the raw home−away contract); plus pull
  robustness, current-season cache refresh, September season rollover,
  and a final-score-reconstruction gate.
- Known v1 simplifications: FT trips are grouped on
  (player, period, clock-second) rather than the explicit "N of M" /
  typed-foul structure (FT accounting still balances exactly); EuroLeague
  lumps all OT into period 5, so a same-clock and-1 match could in rare
  cases cross OT segments; no jump-ball offense lookahead. None affect
  the gate or the leaderboards materially.
