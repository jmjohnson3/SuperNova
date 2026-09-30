# SuperNovaBets MLB Daily Run (2026-07-17 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 7.5s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 7.4s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 7.8s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 9.1s)
- **Refresh prop replay CLV**: OK (rc=0, 88.8s)
- **Build prop market training table**: OK (rc=0, 51.0s)
- **Prop walk-forward accuracy report**: OK (rc=0, 7.5s)
- **Prop shadow selector report**: OK (rc=0, 6.6s)
- **Prop miss diagnostic report**: OK (rc=0, 101.7s)
- **Prop bucket repair report**: OK (rc=0, 31.0s)
- **TB prop repair report**: OK (rc=0, 26.6s)
- **Prop target quality report**: OK (rc=0, 61.2s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 14.9s)
- **Grade outcomes + ledgers**: OK (rc=0, 58.2s)
- **Grade daily forecast ledger**: OK (rc=0, 6.8s)
- **Daily forecast projection audit**: OK (rc=0, 16.3s)
- **Hitter live-vs-legacy forecast diff**: OK (rc=0, 6.7s)
- **Prop micro promotion evaluation**: OK (rc=0, 0.5s)
- **Prop micro gate sensitivity report**: OK (rc=0, 1.1s)
- **Prop drift guard diagnostic**: OK (rc=0, 39.6s)
- **Prop layer promotion control**: OK (rc=0, 0.4s)
- **Forecast repair error decomposition**: OK (rc=0, 12.5s)
- **TB tail repair challenger**: OK (rc=0, 35.0s)
- **Prop snapshot coverage report**: OK (rc=0, 86.7s)
- **Real-money operational prop reports**: OK (rc=0, 20.2s)
- **Pitcher K-rate challenger diagnostic**: OK (rc=0, 87.3s)
- **End-of-slate prop close diagnostic**: OK (rc=0, 6.0s)
- **Frozen-release five-date checkpoint**: OK (rc=0, 81.1s)
- **Post-checkpoint micro promotion evaluation**: OK (rc=0, 0.8s)
- **Post-checkpoint micro gate sensitivity report**: OK (rc=0, 0.3s)
- **Post-checkpoint prop drift guard diagnostic**: OK (rc=0, 25.0s)
- **Post-checkpoint prop layer promotion control**: OK (rc=0, 0.4s)
- **Post-checkpoint real-money operational prop reports**: OK (rc=0, 6.3s)
- **Daily slate trust monitor**: OK (rc=0, 80.0s)
- **Grade shadow prop replay**: OK (rc=0, 48.8s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-17 21:45:17,724 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 30 teams today (2026-07-17)
2026-07-17 21:45:18,107 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 56 rows for 2026-07-17
2026-07-17 21:45:18,107 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 56 assignments for 2026-07-17
2026-07-17 21:45:18,122 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-17 21:45:18,465 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 15 unique games for season=2026-regular
2026-07-17 21:45:18,547 | INFO | mlb_pipeline.crawler_statsapi | Upserted 15 rows into raw.mlb_games
2026-07-17 21:45:18,594 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6301 completed games, 6287 already done, 5 to fetch
2026-07-17 21:45:21,490 | INFO | mlb_pipeline.crawler_statsapi |   Progress: 5 / 5 games fetched
2026-07-17 21:45:21,647 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=5, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-17 21:45:26,978 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-18. Catching up from 2026-07-19 to 2026-07-16
2026-07-17 21:45:26,996 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-17 window=2026-07-17T04:00:00Z..2026-07-18T04:00:00Z
2026-07-17 21:45:27,664 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-17 | events=4 | credits_remaining=66882
2026-07-17 21:45:27,870 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-18 window=2026-07-18T04:00:00Z..2026-07-19T04:00:00Z
2026-07-17 21:45:29,039 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-18 | events=16 | credits_remaining=66880
2026-07-17 21:45:29,049 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=66880
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-17 21:45:31,732 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-18. Catching up from 2026-07-19 to 2026-07-16
2026-07-17 21:45:32,262 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 4 events (game_date=2026-07-17, as_of=2026-07-17)
2026-07-17 21:45:32,884 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-17 event=e047c3dd3d688fead07d77b8b32852af (Detroit Tigers@Los Angeles Angels) | credits=66877
2026-07-17 21:45:33,934 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-17 event=2e309c8e4682895ea0516d055b78f53a (St. Louis Cardinals@Arizona Diamondbacks) | credits=66872
2026-07-17 21:45:34,997 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-17 event=e0989f5a2163779f2b0c456d7f844ce9 (Washington Nationals@Athletics) | credits=66867
2026-07-17 21:45:35,789 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-17 event=25f714905d486c907ee2e0c1377097ea (San Francisco Giants@Seattle Mariners) | credits=66861
2026-07-17 21:45:36,732 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 16 events (game_date=2026-07-18, as_of=2026-07-17)
2026-07-17 21:45:36,810 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=66861
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-17 21:45:40,325 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1577 rows into odds.mlb_game_lines (live odds).
2026-07-17 21:45:40,341 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-17
2026-07-17 21:45:40,341 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-17 21:45:40,341 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-18T03:15:40.341490+00:00.
2026-07-17 21:45:40,450 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-16
2026-07-17 21:45:45,919 | INFO | mlb_pipeline.parse_oddsapi | Upserted 323 rows into odds.mlb_player_prop_lines.
2026-07-17 21:45:45,919 | INFO | mlb_pipeline.parse_oddsapi | Processed 323 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-17 21:45:45,919 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 7705,
  "run_ids": "all",
  "date_from": "2026-07-17",
  "date_to": "2026-07-17",
  "include_graded": true,
  "only_missing": false
}
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 0,
  "replay_rows": 8268,
  "examples": 8268
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "reused_fresh_artifact",
  "artifact": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_walk_forward_accuracy_report.json",
  "age_minutes": 56.1,
  "rows": 245202,
  "graded_rows": 216309,
  "valid_clv_rows": 206735,
  "generated_at_utc": "2026-07-18T02:51:55+00:00"
}
```

### Prop shadow selector report

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "active_rows": 2605,
  "real_candidate_rows": 0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_shadow_selector_latest.md"
}
```

### Prop miss diagnostic report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 23190,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 221308,
  "bucket_count": 145,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=91701
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 248976,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### FanDuel one-sided diagnostic

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "root_causes": {
    "raw_api_one_sided": 15053
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 226-150 (60.1%) ROI: +14.7% | Total: 71-67 (51.4%) ROI: -1.8%
MLB CLV Run Line: beat close 3/84 (4%) avg CLV=+0.11 runs | CLV Total avg=+0.09 runs
MLB Price CLV Run Line: 81 bets  avg=+0.40%
```

**stderr (tail)**
```
2026-07-17 21:52:27,481 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 5 rows
2026-07-17 21:52:27,481 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 5 MLB game outcome rows
2026-07-17 21:52:27,481 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-17 21:53:11,856 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 882 prop rows
2026-07-17 21:53:11,887 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 882 MLB prop outcome rows
2026-07-17 21:53:12,029 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-17 21:53:12,029 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-17 21:53:12,388 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 10 game model-pick ledger rows
2026-07-17 21:53:13,250 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 36 prop model-pick ledger rows
2026-07-17 21:53:13,262 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 46 MLB model-pick ledger rows
```

### Grade daily forecast ledger

- rc: 0

**stdout (tail)**
```
{
  "schema_ready": true,
  "graded": 2139
}
```

### Daily forecast projection audit

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 14963,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_daily_forecast_projection_audit_latest.md"
}
```

### Hitter live-vs-legacy forecast diff

- rc: 0

**stdout (tail)**
```
{
  "rows": 3417,
  "phase": "day_pregame",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_hitter_live_vs_legacy_forecast_diff_latest.md"
}
```

### Prop micro promotion evaluation

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "micro_ready_count": 0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_promotion_evaluation_latest.md"
}
```

### Prop micro gate sensitivity report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "profiles": {
    "watch_only": 0,
    "micro_test": 0,
    "strict_micro": 0
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
}
```

### Prop drift guard diagnostic

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-17', 'micro_rows': 99, 'micro_bettable_now': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

### Prop layer promotion control

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "auto_integrations": {
    "hitter_rate_challenger_production": {
      "enabled": false,
      "layer": "hitter_hits_hr_shadow",
      "target_mode": "production_scoring",
      "blockers": [
        "batter_hits_projection_not_better_than_baseline",
        "completed_dates<5"
      ]
    },
    "hitter_pa_v3_production": {
      "enabled": true,
      "layer": "hitter_pa_v3",
      "target_mode": "production_scoring",
      "blockers": []
    },
    "exact_bucket_micro_ladder": {
      "enabled": false,
      "layer": "exact_bucket_micro",
      "target_mode": "micro",
      "blockers": [
        "micro_ready_exact_buckets=0",
        "real_money_kill_switch_active"
      ]
    }
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_layer_promotion_latest.md"
}
```

### Forecast repair error decomposition

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "active_source": "prospective_ledger",
  "prospective_rows": 9613,
  "historical_rows": 24110,
  "tb_repair": "train_gated_direct_player_game_tb_head",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_forecast_repair_error_latest.md"
}
```

### TB tail repair challenger

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 8022,
  "accepted": true,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb_tail_repair_challenger_latest.md"
}
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-07-17T23:56:38-04:00
Range: 2026-07-04 to 2026-07-17

## Collection Status

**COLLECTING**

- Clean shadow slates: 1 / 10
- Additional clean slates needed: 9
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 90.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 2.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-07-17 | no | 4447 | 2593 | 7705 | 7705 | 195834 | 6471 | 84.0% | 0.0% | 1.0% | 3 | 83 | valid_close_coverage<0.90 |
| 2026-07-16 | yes | 280 | 177 | 563 | 563 | 5265 | 551 | 97.9% | 0.0% | 0.0% | 3 | 29 |  |
| 2026-07-15 | no | 0 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, captured_side_locks<100, close_capture_coverage<0.90, valid_close_coverage<0.90, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.02, no_close_snapshot_time |
| 2026-07-14 | no | 42 | 42 | 0 | 0 | 990 | 0 | - | - | - | 0 | 24 | side_locks<100, captured_side_locks<100, close_capture_coverage<0.90, valid_close_coverage<0.90, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.02 |
| 2026-07-13 | no | 42 | 42 | 0 | 0 | 42 | 0 | - | - | - | 0 | 1 | side_locks<100, captured_side_locks<100, close_capture_coverage<0.90, valid_close_coverage<0.90, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.02 |
| 2026-07-12 | no | 4577 | 2643 | 7667 | 7667 | 76486 | 4599 | 60.0% | 0.0% | 27.4% | 3 | 33 | close_capture_coverage<0.90, valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-07-11 | no | 4850 | 2754 | 6643 | 6643 | 123974 | 5923 | 89.2% | 0.0% | 0.0% | 3 | 58 | valid_close_coverage<0.90 |
| 2026-07-10 | no | 4688 | 2855 | 8100 | 8100 | 97641 | 7024 | 86.7% | 0.0% | 0.0% | 3 | 44 | valid_close_coverage<0.90 |
| 2026-07-09 | no | 3913 | 2435 | 5964 | 5964 | 62333 | 5292 | 88.7% | 0.0% | 0.0% | 3 | 40 | valid_close_coverage<0.90 |
| 2026-07-08 | no | 4609 | 2727 | 7951 | 7951 | 73124 | 6833 | 85.9% | 0.0% | 0.0% | 3 | 32 | valid_close_coverage<0.90 |
| 2026-07-07 | no | 4877 | 2689 | 8289 | 8289 | 75276 | 7151 | 86.3% | 0.0% | 0.6% | 3 | 31 | valid_close_coverage<0.90 |
| 2026-07-06 | no | 2495 | 1469 | 4100 | 4100 | 38682 | 3580 | 87.3% | 0.0% | 0.0% | 3 | 31 | valid_close_coverage<0.90 |
| 2026-07-05 | no | 4596 | 2708 | 5782 | 5782 | 34519 | 4954 | 85.7% | 0.0% | 0.0% | 3 | 20 | valid_close_coverage<0.90 |
| 2026-07-04 | no | 4640 | 2783 | 7530 | 7530 | 43937 | 6333 | 84.1% | 0.0% | 0.0% | 3 | 20 | valid_close_coverage<0.90 |
```

### Real-money operational prop reports

- rc: 0

**stdout (tail)**
```
{
  "generated_at_utc": "2026-07-18T03:56:40Z",
  "slate_date": "2026-07-17",
  "lookback_days": 45,
  "reports": {
    "close_coverage_dashboard": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_close_coverage_dashboard_latest.md"
    },
    "player_game_projection_ledger_v2": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_player_game_projection_ledger_v2_latest.md"
    },
    "tb_error_decomposition": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_tb_error_decomposition_latest.md"
    },
    "true_pair_coverage_repair": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_true_pair_coverage_repair_latest.md"
    },
    "shadow_challenger_promotion": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_shadow_challenger_promotion_latest.md"
    },
    "micro_bucket_almost_there": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_micro_bucket_almost_there_latest.md"
    },
    "micro_gate_sensitivity": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
    }
  }
}
```

### Pitcher K-rate challenger diagnostic

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 840,
  "mae_gain": 0.0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_pitcher_k_rate_challenger_latest.md"
}
```

### End-of-slate prop close diagnostic

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-17",
  "evaluation_status": "provisional",
  "strict_clean_slate": false,
  "valid_close_coverage": 0.833273316528628,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_end_of_slate_close_latest.md"
}
```

### Frozen-release five-date checkpoint

- rc: 0

**stdout (tail)**
```
{
  "status": "evaluation_ready_no_micro_buckets",
  "hitter_completed_dates": 8,
  "hitter_rate_shadow_completed_dates": 4,
  "pitcher_completed_dates": 10,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_five_date_checkpoint_latest.md"
}
```

### Post-checkpoint micro promotion evaluation

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "micro_ready_count": 0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_promotion_evaluation_latest.md"
}
```

### Post-checkpoint micro gate sensitivity report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "profiles": {
    "watch_only": 0,
    "micro_test": 0,
    "strict_micro": 0
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
}
```

### Post-checkpoint prop drift guard diagnostic

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-18', 'micro_rows': 0, 'micro_bettable_now': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

**stderr (tail)**
```
No normalized prop offers available for 2026-07-18
C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\modeling\predict_player_props.py:4402: UserWarning: pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy.
  df = pd.read_sql(SQL_PROP_LINES, conn, params={"game_date": game_date})
```

### Post-checkpoint prop layer promotion control

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "auto_integrations": {
    "hitter_rate_challenger_production": {
      "enabled": false,
      "layer": "hitter_hits_hr_shadow",
      "target_mode": "production_scoring",
      "blockers": [
        "batter_hits_projection_not_better_than_baseline",
        "completed_dates<5"
      ]
    },
    "hitter_pa_v3_production": {
      "enabled": true,
      "layer": "hitter_pa_v3",
      "target_mode": "production_scoring",
      "blockers": []
    },
    "exact_bucket_micro_ladder": {
      "enabled": false,
      "layer": "exact_bucket_micro",
      "target_mode": "micro",
      "blockers": [
        "micro_ready_exact_buckets=0",
        "real_money_kill_switch_active"
      ]
    }
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_layer_promotion_latest.md"
}
```

### Post-checkpoint real-money operational prop reports

- rc: 0

**stdout (tail)**
```
{
  "generated_at_utc": "2026-07-18T04:00:20Z",
  "slate_date": "2026-07-18",
  "lookback_days": 45,
  "reports": {
    "close_coverage_dashboard": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_close_coverage_dashboard_latest.md"
    },
    "player_game_projection_ledger_v2": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_player_game_projection_ledger_v2_latest.md"
    },
    "tb_error_decomposition": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_tb_error_decomposition_latest.md"
    },
    "true_pair_coverage_repair": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_true_pair_coverage_repair_latest.md"
    },
    "shadow_challenger_promotion": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_shadow_challenger_promotion_latest.md"
    },
    "micro_bucket_almost_there": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_micro_bucket_almost_there_latest.md"
    },
    "micro_gate_sensitivity": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
    }
  }
}
```

### Daily slate trust monitor

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-17",
  "status": "provisional",
  "decision": "wait_for_final_results_or_repair_failed_checks",
  "failures": [
    "all_games_finalized",
    "valid_close_coverage",
    "targeted_close_captures",
    "hitter_anchor_settled_or_voided",
    "pitcher_anchor_settled_or_voided"
  ],
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_daily_slate_trust_latest.md"
}
```

### Grade shadow prop replay

- rc: 0

**stdout (tail)**
```
{
  "graded_rows": 2582,
  "run_ids": "all_pending",
  "regrade": false
}
```
