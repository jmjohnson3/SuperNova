# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SuperNovaBets is an NBA sports betting prediction pipeline. It crawls odds and game data, loads it into PostgreSQL, engineers features via SQL views, trains XGBoost models, and produces spread/total/player-prop predictions.

## Running the Pipeline

### Full daily run
```bash
python -m nba_pipeline.run_daily
python -m nba_pipeline.run_daily --date 2026-02-24   # specific ET date
```

### Skip flags for the orchestrator
```
--skip-crawl     Skip Odds API crawler
--skip-parse     Skip parse/load step
--skip-train     Skip model training
--skip-predict   Skip prediction output
--skip-scan      Skip alt-line grid scan
--backfill-odds  Also run historical odds backfill
--report PATH    Write markdown report to PATH (default: reports/daily_YYYY-MM-DD.md)
```

### Run individual steps
```bash
python -m nba_pipeline.crawler_oddsapi          # Fetch today's NBA odds
python -m nba_pipeline.parse_all                # Parse all raw data into structured tables
python -m nba_pipeline.modeling.train_game_models       # Train spread/total models
python -m nba_pipeline.modeling.train_player_prop_models
python -m nba_pipeline.modeling.predict_today           # Predict today's games
python -m nba_pipeline.modeling.predict_player_props
python -m nba_pipeline.modeling.scan_alt_lines_grid     # Alt-line hit-rate scan
```

### Environment setup
```bash
# Activate the venv (already in .venv/)
.venv/Scripts/activate      # Windows
source .venv/bin/activate   # Unix

# Install package in editable mode
pip install -e .
```

## Architecture

### Data flow

```
Odds API / MySportsFeeds API
         │
         ▼
  raw.api_responses        ← all raw JSON payloads stored here (idempotent, sha256 dedup)
         │
         ▼ parse_* scripts
  Structured tables        ← nba_games, nba_player_gamelogs, nba_injuries, nba_lineups, etc.
         │
         ▼ SQL views (features schema)
  features.game_training_features      ← rolling-window stats for model training
  features.game_prediction_features   ← today's games for inference
  features.player_prop_training_features
         │
         ▼ train_*.py (XGBoost + Optuna)
  models/*.json            ← spread_direct_xgb.json, total_direct_xgb.json, etc.
         │
         ▼ predict_*.py
  Console output / Discord notification
```

### Database
- **DSN**: `postgresql://josh:password@localhost:5432/nba`
- **`raw` schema**: `api_responses` (all API payloads, keyed by provider/endpoint/game_slug), plus `nba_injuries`, `nba_player_gamelogs`
- **`features` schema**: PostgreSQL views that compute rolling stats. SQL files in `sql/` are numbered `V001`–`V015` and applied manually (no migration runner).

### Parse pipeline order (matters — dimensions before facts)
```
parse_meta        → venues, teams, standings, injuries
parse_games       → nba_games
parse_player_gamelogs → player stat backbone
parse_lineup      → availability, starters
parse_boxscore    → game + player box scores
parse_pbp         → play-by-play advanced features
parse_referees    → referee assignments
```

### Modeling approach
- **Two model families** for spread/total:
  - *Direct*: pure XGBoost predicting margin/total directly
  - *Residual*: predicts how much reality deviates from the market line; final prediction = market line + residual (used when market data is available)
- **Walk-forward cross-validation**: no future data leaks into training. `min_train_days=60`, `test_window_days=7`, `step_days=7`.
- **Median imputation**: fit on train rows only, applied to test rows, to prevent leakage.
- **Optuna tuning**: enabled by default (`run_optuna=True`, `optuna_n_trials=40`, `optuna_n_folds=5`) inside `train_game_models.py`.
- **Player props**: XGBoost for PTS, REB, AST via `train_player_prop_models.py`.

### Derived interaction features
Feature engineering happens in Python (not SQL) inside `make_xy_raw()` in `train_game_models.py` and must be mirrored in `_prep_features()` in `predict_today.py`. Key derived features include rest advantage, net rating diff, pace diff, efficiency diffs, injury pts diff, clutch net diff, standings diffs. When adding new derived features, update **both** files.

### Model artifacts
Stored in `src/nba_pipeline/modeling/models/`:
- `spread_direct_xgb.json`, `total_direct_xgb.json`
- `spread_resid_xgb.json`, `total_resid_xgb.json` (optional, needs market data)
- `feature_columns.json`, `feature_medians.json` (schema required at inference)
- `player_props/points_xgb.json`, `rebounds_xgb.json`, `assists_xgb.json`
- `optuna_best_params.json` (written after Optuna tuning)

### External data sources
- **MySportsFeeds** (`fetcher.py` → `MySportsFeedsClient`): game schedules, box scores, play-by-play, player gamelogs, injuries
- **The Odds API** (`crawler_oddsapi.py`): FanDuel/DraftKings spreads and totals

### Notifications
`run_daily_and_notify.py` wraps the pipeline and posts results to Discord. Webhook URL is hardcoded but can be overridden via `DISCORD_WEBHOOK_URL` env var.
