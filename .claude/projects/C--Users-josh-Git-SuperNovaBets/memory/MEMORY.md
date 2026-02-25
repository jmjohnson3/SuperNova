# SuperNovaBets Project Memory

## Project
NBA sports betting prediction pipeline (`nba_pipeline` package in `src/`).

## Key Facts
- **DB**: `postgresql://josh:password@localhost:5432/nba`
- **Python venv**: `.venv/` (Windows: `.venv/Scripts/activate`)
- **Models dir**: `src/nba_pipeline/modeling/models/`
- **SQL migrations**: `sql/V001–V015` applied manually (no runner)

## Architecture
- All raw API payloads → `raw.api_responses` (idempotent, sha256 dedup)
- Parse scripts read from `raw.api_responses` → structured tables
- `features` schema = PostgreSQL views with rolling stats
- Walk-forward CV: `train_game_models.py` / `train_player_prop_models.py`
- Two game model types: direct XGBoost + residual (market line correction)
- Optuna tuning enabled by default in `train_game_models.py`

## Critical Pattern: Feature Engineering Mirror
Derived features in `make_xy_raw()` (train_game_models.py) **must** be mirrored in `_prep_features()` (predict_today.py). Always update both.

## Pipeline Run Order
```
crawler_oddsapi → parse_all → train_game_models → train_player_prop_models
→ predict_today → predict_player_props → scan_alt_lines_grid
```

## Common Commands
```bash
python -m nba_pipeline.run_daily                    # full run
python -m nba_pipeline.run_daily --skip-crawl       # skip odds fetch
python -m nba_pipeline.modeling.train_game_models   # retrain game models
python -m nba_pipeline.modeling.predict_today       # predict today
```
