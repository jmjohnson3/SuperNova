# SuperNovaBets MLB Daily Run (2026-07-15 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 4.0s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 5.1s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 8.0s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 3.6s)
- **Refresh prop replay CLV**: OK (rc=0, 6.9s)
- **Build prop market training table**: OK (rc=0, 17.1s)
- **Prop walk-forward accuracy report**: OK (rc=0, 73.4s)
- **Prop shadow selector report**: OK (rc=0, 9.5s)
- **Prop miss diagnostic report**: OK (rc=0, 41.4s)
- **Prop bucket repair report**: OK (rc=0, 10.4s)
- **TB prop repair report**: OK (rc=0, 12.1s)
- **Prop target quality report**: OK (rc=0, 17.3s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 10.7s)
- **Grade outcomes + ledgers**: OK (rc=0, 7.9s)
- **Grade daily forecast ledger**: OK (rc=0, 3.6s)
- **Daily forecast projection audit**: OK (rc=0, 10.2s)
- **Hitter live-vs-legacy forecast diff**: OK (rc=0, 10.7s)
- **Prop micro promotion evaluation**: OK (rc=0, 0.3s)
- **Prop micro gate sensitivity report**: OK (rc=0, 0.2s)
- **Prop layer promotion control**: OK (rc=0, 0.4s)
- **Forecast repair error decomposition**: OK (rc=0, 6.1s)
- **TB tail repair challenger**: OK (rc=0, 18.8s)
- **Prop snapshot coverage report**: OK (rc=0, 61.1s)
- **Real-money operational prop reports**: OK (rc=0, 5.3s)
- **Pitcher K-rate challenger diagnostic**: OK (rc=0, 31.6s)
- **End-of-slate prop close diagnostic**: OK (rc=0, 4.3s)
- **Frozen-release five-date checkpoint**: OK (rc=0, 76.9s)
- **Post-checkpoint micro promotion evaluation**: OK (rc=0, 0.3s)
- **Post-checkpoint micro gate sensitivity report**: OK (rc=0, 0.2s)
- **Post-checkpoint prop layer promotion control**: OK (rc=0, 0.4s)
- **Post-checkpoint real-money operational prop reports**: OK (rc=0, 3.9s)
- **Daily slate trust monitor**: OK (rc=0, 85.0s)
- **Grade shadow prop replay**: OK (rc=0, 18.9s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-15 21:45:16,182 | INFO | mlb_pipeline.crawler_statsapi | fetch_pregame_umpires_for_date: no officials found for 2026-07-15 (may not be announced yet â€” typically released by 5-6 PM ET the day before)
2026-07-15 21:45:16,213 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-15 21:45:16,471 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 0 unique games for season=2026-regular
2026-07-15 21:45:16,576 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6290 completed games, 6281 already done, 0 to fetch
2026-07-15 21:45:16,789 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=0, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-15 21:45:20,083 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-16. Catching up from 2026-07-17 to 2026-07-14
2026-07-15 21:45:20,083 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-15 window=2026-07-15T04:00:00Z..2026-07-16T04:00:00Z
2026-07-15 21:45:20,896 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-15 | events=0 | credits_remaining=73642
2026-07-15 21:45:21,068 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-16 window=2026-07-16T04:00:00Z..2026-07-17T04:00:00Z
2026-07-15 21:45:21,911 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-16 | events=1 | credits_remaining=73640
2026-07-15 21:45:21,911 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=73640
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-15 21:45:28,458 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-16. Catching up from 2026-07-17 to 2026-07-14
2026-07-15 21:45:29,067 | INFO | mlb_pipeline.crawler_oddsapi | No events found for 2026-07-15 â€” skipping prop fetch
2026-07-15 21:45:29,911 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 1 events (game_date=2026-07-16, as_of=2026-07-15)
2026-07-15 21:45:29,958 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=unknown
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-15 21:45:33,442 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1542 rows into odds.mlb_game_lines (live odds).
2026-07-15 21:45:33,442 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-15
2026-07-15 21:45:33,442 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-15 21:45:33,442 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-16T03:15:33.442859+00:00.
2026-07-15 21:45:33,536 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-13
2026-07-15 21:45:33,536 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_prop_odds snapshots found (as_of_date=None).
2026-07-15 21:45:33,536 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 0,
  "run_ids": "all",
  "date_from": "2026-07-15",
  "date_to": "2026-07-15",
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
  "replay_rows": 0,
  "examples": 0
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 73315,
  "graded_rows": 67088,
  "valid_clv_rows": 61673,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_walk_forward_accuracy_latest.md",
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_walk_forward_accuracy_report.json"
}
```

### Prop shadow selector report

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "active_rows": 0,
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
  "rows": 4234,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 67088,
  "bucket_count": 127,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=28488
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 73315,
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
    "raw_api_one_sided": 15570
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 223-149 (59.9%) ROI: +14.4% | Total: 71-67 (51.4%) ROI: -1.8%
MLB CLV Run Line: beat close 3/80 (4%) avg CLV=+0.11 runs | CLV Total avg=+0.09 runs
MLB Price CLV Run Line: 77 bets  avg=+0.39%
```

**stderr (tail)**
```
2026-07-15 21:48:56,364 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 23 pending predictions.
2026-07-15 21:48:56,364 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-07-15 21:48:56,380 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-15 21:48:59,553 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-07-15 21:48:59,576 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-07-15 21:48:59,709 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-15 21:48:59,710 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-15 21:48:59,776 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-07-15 21:49:00,133 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-07-15 21:49:00,149 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Grade daily forecast ledger

- rc: 0

**stdout (tail)**
```
{
  "schema_ready": true,
  "graded": 0
}
```

### Daily forecast projection audit

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 13704,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_daily_forecast_projection_audit_latest.md"
}
```

### Hitter live-vs-legacy forecast diff

- rc: 0

**stdout (tail)**
```
{
  "rows": 3096,
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
  "prospective_rows": 8710,
  "historical_rows": 6444,
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
  "rows": 2144,
  "accepted": false,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb_tail_repair_challenger_latest.md"
}
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-07-15T23:50:51-04:00
Range: 2026-07-02 to 2026-07-15

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
| 2026-07-03 | no | 3879 | 2026 | 6784 | 6784 | 39279 | 5921 | 87.3% | 0.0% | 0.0% | 3 | 19 | valid_close_coverage<0.90 |
| 2026-07-02 | yes | 2683 | 1678 | 4505 | 4505 | 22054 | 4063 | 90.2% | 0.0% | 0.0% | 3 | 18 |  |
```

### Real-money operational prop reports

- rc: 0

**stdout (tail)**
```
{
  "generated_at_utc": "2026-07-16T03:50:52Z",
  "slate_date": "2026-07-15",
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
  "rows": 226,
  "mae_gain": 0.0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_pitcher_k_rate_challenger_latest.md"
}
```

### End-of-slate prop close diagnostic

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-15",
  "evaluation_status": "provisional",
  "strict_clean_slate": false,
  "valid_close_coverage": null,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_end_of_slate_close_latest.md"
}
```

### Frozen-release five-date checkpoint

- rc: 0

**stdout (tail)**
```
{
  "status": "evaluation_ready_no_micro_buckets",
  "hitter_completed_dates": 7,
  "hitter_rate_shadow_completed_dates": 3,
  "pitcher_completed_dates": 9,
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
  "generated_at_utc": "2026-07-16T03:52:51Z",
  "slate_date": "2026-07-15",
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
  "slate_date": "2026-07-15",
  "status": "provisional",
  "decision": "wait_for_final_results_or_repair_failed_checks",
  "failures": [
    "all_games_finalized",
    "valid_close_coverage",
    "stale_close_rate",
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
  "graded_rows": 0,
  "run_ids": "all_pending",
  "regrade": false
}
```
