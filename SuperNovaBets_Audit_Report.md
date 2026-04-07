# SuperNovaBets Full Codebase Audit Report

**Date:** April 6, 2026
**Scope:** All 55+ Python source files, 40+ SQL views, shared infrastructure
**Pipelines:** NBA (production) and MLB (new, 2026-03-27)

---

## Executive Summary

A comprehensive audit of every script and SQL file in the SuperNovaBets project was performed across four parallel workstreams: NBA pipeline, MLB pipeline, SQL/schema, and syntax/import validation.

**The NBA pipeline is mature and largely production-ready.** All Python files compile, edge formulas follow correct sign conventions, walk-forward CV prevents data leakage, and the shared infrastructure is clean.

**The MLB pipeline has 5 critical bugs that will prevent it from running.** The MSF crawler uses wrong season slug formats, the main feature SQL views reference non-existent columns, team abbreviation normalization is missing from parsers, and a lineup quality view references a column that doesn't exist in the schema.

**Totals: 5 Critical, 8 Warnings, 7 Minor issues found.**

---

## Critical Issues (5) - MLB Pipeline Blockers

### #1 & #2: MLB006 SQL View Uses Wrong Column Names
**File:** `sql/MLB006_mlb_game_features.sql` (lines 29-66 and 456-482)

Both `features.mlb_game_training_features` and `features.mlb_game_prediction_features` contain a `market_lines` CTE that references column names from the NBA odds schema rather than the MLB odds schema:

| What the view uses (WRONG) | What the schema has (CORRECT) |
|---|---|
| `spread_home_points` | `run_line_home` |
| `spread_home_price` | `run_line_home_price` |
| `spread_away_price` | `run_line_away_price` |
| `total_points` | `total_line` |
| `total_over_price` | `over_price` |
| `total_under_price` | `under_price` |
| `home_team` / `away_team` | `home_team_abbr` / `away_team_abbr` |

**Impact:** Both views will fail with "column does not exist" errors, completely blocking MLB model training and predictions.

---

### #3: MLB011 References Non-Existent `is_home` Column
**File:** `sql/MLB011_mlb_lineup_quality.sql` (line 20)

The lineup quality view references `bps.is_home` from `raw.mlb_boxscore_player_stats`, but this column was never created in the `MLB000_schema_bootstrap.sql` schema. The table only has: `game_slug`, `player_id`, `team_abbr`, `primary_position`, `batting_order`, `stats`.

**Impact:** View creation will fail. Lineup quality features won't be available.

---

### #4: Wrong MSF Season Slugs in MLB Crawler
**File:** `src/mlb_pipeline/crawler.py` (lines 500-501)

```python
Season(league="mlb", season_slug="2024-2025-regular"),  # WRONG
Season(league="mlb", season_slug="2025-2026-regular"),  # WRONG
```

MySportsFeeds MLB uses single-year format: `2024-regular` and `2025-regular`. The NBA pipeline correctly uses two-year format because NBA seasons span calendar years, but MLB seasons don't.

**Impact:** All MSF API calls for MLB will return 404 errors. No game data will be fetched.

---

### #5: Missing Team Abbreviation Normalization in MLB Parsers
**Files:** `src/mlb_pipeline/parse_games.py`, `parse_boxscore.py`, `parse_player_gamelogs.py`

The MLB crawler defines `TEAM_ABBR_NORMALIZE` and a `_norm_abbr()` helper, but **none of the parsers import or use it**. Raw MSF abbreviations (SFG, SDP, KCR, TBR, WSN, CHW) will be stored as-is, while the odds parser normalizes to (SF, SD, KC, TB, WAS, CWS).

**Impact:** Mismatched team keys break JOINs across tables. Games with these 6 teams will have orphaned or duplicate records.

---

## Warning Issues (8)

| # | Location | Description |
|---|---|---|
| 6 | `crawler_oddsapi.py` (both pipelines) | Hardcoded OddsAPI key in dataclass defaults. Should use environment variables. |
| 7 | `run_daily_and_notify.py` (both pipelines) | Hardcoded Discord webhook URLs in source code. Security risk if repo is shared. |
| 8 | `pyproject.toml` | Only declares `requests` as dependency. Missing 9+ required packages: numpy, pandas, psycopg2, xgboost, lightgbm, scikit-learn, scipy, aiohttp, httpx. |
| 9 | `mlb_pipeline/run_daily.py` (L161) | Passes `--season 2026-regular` to crawler_statsapi. Should verify this is the correct current season. |
| 10 | `sql/MLB004_mlb_ballpark_factors.sql` | `venue_id` declared as TEXT with string IDs but `raw.mlb_venues` uses INTEGER. Type mismatch prevents direct venue_id joins. |
| 11 | `train_game_models.py` vs `predict_today.py` | Feature prep split across two files with different column alignment strategies. No integration test ensures they stay in sync. |
| 12 | `sql/MLB006` (L720-722) | SP rest calculation for predictions uses simple date subtraction. Training view correctly uses LAG window function. |
| 13 | `raw_store.py` (L80-81) | No explicit transaction management in `save_api_response()`. Race condition possible under concurrent crawlers. |

---

## Minor Issues (7)

| # | Location | Description |
|---|---|---|
| 14 | `sql/V016_*.sql` | Two files share V016 prefix (h2h_features and opponent_position_defense). Ambiguous load order. |
| 15 | `example_fetch.py` (L10) | MSF API key hardcoded in example code. |
| 16 | `src/nba_pipeline/` | Missing `__init__.py` files (NBA pipeline and src/). MLB pipeline has them. |
| 17 | Multiple SQL files | Some division operations lack explicit zero-check guards. |
| 18 | `sql/MLB000_schema_bootstrap.sql` | Missing indexes on frequently-joined columns (venue_id, sp_id, team_abbr). |
| 19 | `compute_elo.py` | Season regression timing logic lacks explicit season start date constant. |
| 20 | `crawler_statsapi.py` vs `crawler.py` | Two separate team abbreviation normalization dicts not in sync. |

---

## Recommended Fix Priority

### Immediate (Blocks MLB Pipeline)
1. Fix MLB006 column names in both training and prediction view CTEs
2. Fix MLB crawler season slugs from `2024-2025-regular` to `2024-regular` (and `2025-2026` to `2025-regular`)
3. Add team abbreviation normalization to all MLB parsers
4. Fix MLB011 `is_home` reference (add to schema or derive via JOIN)

### High Priority (Security & Reliability)
5. Move all API keys and webhook URLs to environment variables
6. Update `pyproject.toml` dependencies to include all required packages

### Medium Priority (Data Quality)
7. Add integration test for feature column alignment between training and inference
8. Verify `--season 2026-regular` argument is appropriate for current date
9. Add missing indexes to MLB schema

### Low Priority (Cleanup)
10. Resolve duplicate V016 SQL file naming
11. Add `__init__.py` to NBA pipeline directories
12. Unify team abbreviation normalization dicts into shared module

---

## What's Working Well

- **All 55+ Python files compile without syntax errors.** Zero TODO/FIXME/HACK comments.
- **Walk-forward cross-validation prevents data leakage.** Proper temporal splits in both pipelines.
- **Edge formulas follow correct sign conventions.** Both pipelines compute edge correctly.
- **Clean one-way dependency.** MLB imports from NBA for shared utilities, no circular imports.
- **Consistent database connections.** All 35+ references use the same PostgreSQL connection string.
- **Robust SHA256 dedup.** The `raw.api_responses` table handles idempotent upserts correctly.
- **Proper logging.** 48 of 56 files use standard Python logging.
- **Dual model architecture with quality gates.** Residual MAE must be within 2% of direct MAE.
- **Proper median imputation.** Fit on train split only, stored in `feature_medians.json`.
