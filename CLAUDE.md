# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SuperNovaBets is a sports betting prediction pipeline with two leagues:
- **`src/nba_pipeline/`** — NBA spread/total/player-prop predictions (production)
- **`src/mlb_pipeline/`** — MLB run line/total/player-prop predictions (new 2026-03-27)

Both pipelines share the same PostgreSQL database (`nba` db), the same `raw.api_responses` table, and the NBA `fetcher.py` / `raw_store.py` utilities (the MLB pipeline imports them directly).

## Running the Pipelines

### Environment
```bash
.venv/Scripts/activate        # Windows
source .venv/bin/activate     # Unix/Mac
pip install -e .              # installs both nba_pipeline and mlb_pipeline
```

### NBA — Full daily run
```bash
python -m nba_pipeline.run_daily
python -m nba_pipeline.run_daily --date 2026-02-24
python -m nba_pipeline.run_daily --close-only   # ~6:30 PM ET: capture closing lines for CLV
```

Skip flags: `--skip-crawl`, `--skip-parse`, `--skip-train`, `--skip-predict`, `--skip-scan`, `--backfill-odds`

### NBA — Individual steps
```bash
python -m nba_pipeline.crawler                          # MySportsFeeds (games/boxscores/gamelogs)
python -m nba_pipeline.crawler_oddsapi                  # Odds API (spreads/totals/props)
python -m nba_pipeline.crawler_oddsapi_backfill         # Historical odds backfill
python -m nba_pipeline.parse_all                        # All parsers + apply SQL views
python -m nba_pipeline.compute_elo                      # Elo ratings (run after parse_all)
python -m nba_pipeline.modeling.train_game_models       # Spread/total models
python -m nba_pipeline.modeling.train_player_prop_models
python -m nba_pipeline.modeling.predict_today
python -m nba_pipeline.modeling.predict_player_props
python -m nba_pipeline.modeling.scan_alt_lines_grid
python -m nba_pipeline.modeling.update_outcomes         # Grade completed games
python -m nba_pipeline.modeling.paper_trading_report [--days 90]
```

### MLB — Full daily run (new)
```bash
python -m mlb_pipeline.run_daily
python -m mlb_pipeline.run_daily --date 2025-04-01
```

Skip flags: `--skip-crawl`, `--skip-parse`, `--skip-train`, `--skip-predict`

### MLB — First-time setup
```bash
# 1. Bootstrap DB tables
psql postgresql://josh:password@localhost:5432/nba -f sql/MLB000_schema_bootstrap.sql

# 2. Backfill 2024 season
python -m mlb_pipeline.crawler --start-date 2024-03-20 --end-date 2024-09-29
python -m mlb_pipeline.crawler_oddsapi   # catches up from last saved date automatically

# 3. Parse and apply views
python -m mlb_pipeline.parse_all
```

## Architecture

### Shared infrastructure
- **`raw.api_responses`** — single table storing every raw API payload from all providers/leagues. Key: `(provider, endpoint, url)`. Idempotent via `ON CONFLICT DO UPDATE`. The `payload_sha256` enables dedup.
- **`raw_store.py`** (`save_api_response`) and **`fetcher.py`** (`MySportsFeedsClient`) live in `nba_pipeline/` but are imported directly by `mlb_pipeline/` — not copied.
- **DB**: `postgresql://josh:password@localhost:5432/nba` (hardcoded throughout; no env var needed for local dev)

### NBA data flow
```
MySportsFeeds API                 The Odds API
       │                                │
       ▼                                ▼
raw.api_responses (all payloads, sha256 dedup)
       │
       ▼  parse_* scripts (dimensions before facts)
raw.nba_games / nba_player_gamelogs / nba_boxscore_* / nba_lineups / ...
       │
       ▼  sql/V001–V026 + parse_all._apply_view_fixes()
features.game_training_features_mat   ← MATERIALIZED VIEW (~90s to build)
features.player_training_features     ← VIEW (depends on matview)
       │
       ▼  train_game_models.py / train_player_prop_models.py (XGBoost + LightGBM + Optuna)
src/nba_pipeline/modeling/models/*.json
       │
       ▼  predict_today.py / predict_player_props.py
bets.game_predictions / bets.prop_predictions
```

### MLB data flow (same pattern, no matviews in V1)
```
MySportsFeeds (mlb/ seasons)      The Odds API (baseball_mlb)
       │                                │
       ▼                                ▼
raw.api_responses (same table, different endpoint names)
       │
       ▼  mlb_pipeline/parse_* scripts
raw.mlb_games / mlb_player_gamelogs / mlb_boxscore_* / mlb_starting_pitchers / ...
       │
       ▼  sql/MLB001–MLB006 (applied by parse_all._apply_sql_views)
features.mlb_game_training_features / mlb_game_prediction_features
       │
       ▼  mlb_pipeline/modeling/train_game_models.py
src/mlb_pipeline/modeling/models/*.json
       │
       ▼  mlb_pipeline/modeling/predict_today.py
bets.mlb_game_predictions
```

### SQL view system
- **NBA**: `sql/V001_*` through `sql/V026_*` — applied via `parse_all._apply_view_fixes()`. No migration runner; apply manually when adding new views. Views up to ~V015 build on top of earlier ones; order matters.
- **MLB**: `sql/MLB000_schema_bootstrap.sql` (DDL, run once) + `sql/MLB001–006_*.sql` (views, applied each run by `mlb_pipeline.parse_all`).
- **`game_training_features` and `game_prediction_features` are MATERIALIZED VIEWS** (NBA only) backed by thin wrapper views of the same name. `parse_all._materialize_game_features()` creates/refreshes them. This collapses an ~80s query to <1s for predictions.

### NBA parse pipeline order (order matters)
```
parse_meta        → venues, teams, standings, injuries
parse_games       → nba_games (sync scores from boxscores after this)
parse_player_gamelogs → player stat backbone
parse_lineup      → availability / starters
parse_boxscore    → game + player box scores  ← then re-sync scores
parse_pbp         → play-by-play advanced features
parse_referees    → referee assignments
parse_game_odds / parse_game_odds_historical → odds.nba_game_lines
parse_prop_odds / parse_prop_odds_alt → odds.nba_player_prop_lines
```

### MLB parse pipeline order
```
parse_meta → parse_games → parse_player_gamelogs → parse_boxscore
→ parse_starting_pitchers → parse_game_odds → sync_scores_from_boxscores
→ _apply_sql_views (MLB001–006)
```

### Modeling — shared patterns (both pipelines)

**Two model families for game lines:**
- *Direct*: XGBoost predicting `run_diff` / `total_runs` directly (or `margin` / `total` for NBA)
- *Residual*: predicts deviation from market line; reconstruction = `market_line + residual`. Quality gate: residual MAE must be ≤ direct MAE × 1.02.

**Walk-forward CV**: no leakage. NBA: `min_train_days=60`, `test_window=7d`, `step=7d`. MLB: `min_train_days=120`, `test_window=14d`, `step=14d`.

**Median imputation**: fit on train split only, applied to val/test. Stored in `feature_medians.json`.

**Feature engineering lives in two places** that must stay in sync:
- `make_xy_raw()` in `train_game_models.py` (training)
- `_prep_features()` in `predict_today.py` (inference)
- Both call `add_game_derived_features()` from `modeling/features.py`

**Edge formula** (critical — sign convention):
- `edge_spread = pred_margin_home + market_spread_home` (NBA; positive = home covers)
- `edge_run_line = pred_run_diff + market_run_line` (MLB; same convention)
- `edge_total = pred_total - market_total` (positive = bet OVER)

**Bet thresholds**: NBA spread ≥ 7.0 pts, NBA total ≥ 5.0 pts. MLB run line ≥ 1.5 runs, MLB total ≥ 1.0 runs.

### Model artifacts
NBA: `src/nba_pipeline/modeling/models/`
- `spread_direct_xgb.json`, `total_direct_xgb.json`, `spread_resid_xgb.json`, `total_resid_xgb.json`
- `spread_direct_lgb.txt`, `total_direct_lgb.txt` (LightGBM; 50/50 ensemble with XGB)
- `feature_columns.json`, `feature_medians.json`, `calibration.json`, `optuna_best_params.json`, `feature_importance.json`
- `player_props/`: `points_xgb.json`, `rebounds_xgb.json`, `assists_xgb.json`, `lgb_multi.pkl` (joblib)

MLB: `src/mlb_pipeline/modeling/models/`
- `run_line_direct_xgb.json`, `total_direct_xgb.json`, `run_line_resid_xgb.json`, `total_resid_xgb.json`
- Same calibration/feature JSON files

### Grading and paper trading (NBA)
- `update_outcomes.py` fills `bets.game_predictions.actual_*` and computes `clv_spread`/`clv_total` (Closing Line Value). Run nightly after games complete.
- `paper_trading_report.py` prints W/L records segmented by edge bucket, sport, and prop stat.
- `bets.game_predictions` tracks: predictions, kelly fractions, actual scores, ATS coverage, CLV.
- `bets.prop_predictions` tracks: prop predictions, book lines, edges, kelly fractions, `*_over_hit` boolean.

### External APIs
- **MySportsFeeds v2.1**: `https://api.mysportsfeeds.com/v2.1/pull/{league}/{season_slug}/...`
  - NBA seasons: `2024-2025-regular`, `2025-2026-regular`
  - MLB seasons: `2024-regular`, `2025-regular`
  - API key in `CrawlerConfig.api_key` (same key for both leagues)
- **The Odds API**: sport keys `basketball_nba` / `baseball_mlb`. Key in `OddsCrawlerConfig.oddsapi_key`. Credits reset April 1. Floor: 200 credits (daily), 500 (backfill).

### Notifications
`run_daily_and_notify.py` posts to Discord after the NBA pipeline completes. Webhook URL via `DISCORD_WEBHOOK_URL` env var. `DISCORD_FORMAT=1` env var switches predict scripts to compact Discord output.

## Key Gotchas

- **MSF team abbrs**: NBA uses `BRO` (not BKN) and `OKL` (not OKC). MLB uses `SFG→SF`, `SDP→SD`, `KCR→KC`, `TBR→TB`, `WSN→WAS`, `CHW→CWS`. See `TEAM_ABBR_NORMALIZE` in each crawler.
- **Per-game endpoint dedup**: boxscore uses `_boxscore_is_completed()` (checks `raw.nba_boxscore_games`/`raw.mlb_boxscore_games`) instead of `already_fetched()` — prevents caching partial mid-game scores. All other per-game endpoints use `already_fetched()` with `as_of_date=None`.
- **JSONB player IDs** from MySportsFeeds may serialize as floats — always cast via `::numeric::int` in SQL.
- **`game_training_features` matview schema**: `CREATE OR REPLACE VIEW` rejects column reordering — new columns must be appended. If column count changes, `_matview_needs_recreate()` auto-detects and drops/recreates the matview.
- **Incremental parsers**: `parse_prop_odds`, `parse_game_odds_historical` default to `MAX(as_of_date) - 1`. After a backfill, pass an explicit `since_date` to force reprocessing.
- **Score staleness**: `raw.nba_games` / `raw.mlb_games` are the training spine but lag behind reality. `sync_scores_from_boxscores()` is the fix — called twice in `parse_all` (before and after `parse_boxscore`).
