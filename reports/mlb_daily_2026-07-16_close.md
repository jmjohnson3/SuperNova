# SuperNovaBets MLB Daily Run (2026-07-16 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 7.8s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 6.2s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 7.1s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 7.7s)
- **Refresh prop replay CLV**: OK (rc=0, 6.8s)
- **Build prop market training table**: FAIL (rc=124, 612.6s)
- **Prop walk-forward accuracy report**: OK (rc=0, 25.2s)
- **Prop shadow selector report**: OK (rc=0, 13.8s)
- **Prop miss diagnostic report**: FAIL (rc=124, 184.7s)
- **Prop bucket repair report**: OK (rc=0, 127.4s)
- **TB prop repair report**: OK (rc=0, 89.7s)
- **Prop target quality report**: OK (rc=0, 80.6s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 16.2s)
- **Grade outcomes + ledgers**: OK (rc=0, 17.4s)
- **Grade daily forecast ledger**: OK (rc=0, 6.9s)
- **Daily forecast projection audit**: OK (rc=0, 35.9s)
- **Hitter live-vs-legacy forecast diff**: OK (rc=0, 8.1s)
- **Prop micro promotion evaluation**: OK (rc=0, 0.5s)
- **Prop micro gate sensitivity report**: OK (rc=0, 0.5s)
- **Prop layer promotion control**: OK (rc=0, 0.3s)
- **Forecast repair error decomposition**: OK (rc=0, 7.0s)
- **TB tail repair challenger**: OK (rc=0, 82.0s)
- **Prop snapshot coverage report**: OK (rc=0, 76.2s)
- **Real-money operational prop reports**: OK (rc=0, 11.6s)
- **Pitcher K-rate challenger diagnostic**: OK (rc=0, 83.8s)
- **End-of-slate prop close diagnostic**: OK (rc=0, 12.0s)
- **Frozen-release five-date checkpoint**: OK (rc=0, 88.5s)
- **Post-checkpoint micro promotion evaluation**: OK (rc=0, 0.9s)
- **Post-checkpoint micro gate sensitivity report**: OK (rc=0, 0.2s)
- **Post-checkpoint prop layer promotion control**: OK (rc=0, 0.4s)
- **Post-checkpoint real-money operational prop reports**: OK (rc=0, 4.9s)
- **Daily slate trust monitor**: OK (rc=0, 101.8s)
- **Grade shadow prop replay**: OK (rc=0, 28.3s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-16 20:45:12,461 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 2 teams today (2026-07-16)
2026-07-16 20:45:12,792 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 4 rows for 2026-07-16
2026-07-16 20:45:12,792 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 4 assignments for 2026-07-16
2026-07-16 20:45:12,829 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-16 20:45:14,188 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 1 unique games for season=2026-regular
2026-07-16 20:45:14,410 | INFO | mlb_pipeline.crawler_statsapi | Upserted 1 rows into raw.mlb_games
2026-07-16 20:45:14,526 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6291 completed games, 6282 already done, 0 to fetch
2026-07-16 20:45:14,740 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=0, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-16 20:45:20,548 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-17. Catching up from 2026-07-18 to 2026-07-15
2026-07-16 20:45:20,560 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-16 window=2026-07-16T04:00:00Z..2026-07-17T04:00:00Z
2026-07-16 20:45:21,921 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-16 | events=0 | credits_remaining=73399
2026-07-16 20:45:22,481 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-17 window=2026-07-17T04:00:00Z..2026-07-18T04:00:00Z
2026-07-16 20:45:23,121 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-17 | events=14 | credits_remaining=73397
2026-07-16 20:45:23,215 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=73397
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-16 20:45:25,967 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-17. Catching up from 2026-07-18 to 2026-07-15
2026-07-16 20:45:29,590 | INFO | mlb_pipeline.crawler_oddsapi | No events found for 2026-07-16 â€” skipping prop fetch
2026-07-16 20:45:30,195 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 14 events (game_date=2026-07-17, as_of=2026-07-16)
2026-07-16 20:45:30,307 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=unknown
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-16 20:45:37,875 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1567 rows into odds.mlb_game_lines (live odds).
2026-07-16 20:45:37,885 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-16
2026-07-16 20:45:37,886 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-16 20:45:37,886 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-17T02:15:37.886534+00:00.
2026-07-16 20:45:37,999 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-15
2026-07-16 20:45:38,000 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_prop_odds snapshots found (as_of_date=None).
2026-07-16 20:45:38,000 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 563,
  "run_ids": "all",
  "date_from": "2026-07-16",
  "date_to": "2026-07-16",
  "include_graded": true,
  "only_missing": false
}
```

### Build prop market training table

- rc: 124

**stderr (tail)**
```
Timed out after 600s; killed process tree rooted at PID 12244
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "reused_fresh_artifact",
  "artifact": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_walk_forward_accuracy_report.json",
  "age_minutes": 331.4,
  "rows": 240856,
  "graded_rows": 218198,
  "valid_clv_rows": 199913,
  "generated_at_utc": "2026-07-16T21:24:33+00:00"
}
```

### Prop shadow selector report

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "active_rows": 207,
  "real_candidate_rows": 0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_shadow_selector_latest.md"
}
```

### Prop miss diagnostic report

- rc: 124

**stderr (tail)**
```
Timed out after 180s; killed process tree rooted at PID 5588
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 218392,
  "bucket_count": 144,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=90469
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 241064,
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
    "raw_api_one_sided": 14963
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
2026-07-16 21:05:04,458 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 23 pending predictions.
2026-07-16 21:05:04,458 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-07-16 21:05:04,482 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-16 21:05:08,758 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-07-16 21:05:08,781 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-07-16 21:05:08,927 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-16 21:05:08,930 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-16 21:05:09,028 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-07-16 21:05:13,309 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-07-16 21:05:13,331 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
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
  "rows": 13811,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_daily_forecast_projection_audit_latest.md"
}
```

### Hitter live-vs-legacy forecast diff

- rc: 0

**stdout (tail)**
```
{
  "rows": 3116,
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
  "prospective_rows": 8786,
  "historical_rows": 24065,
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
  "rows": 8007,
  "accepted": true,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb_tail_repair_challenger_latest.md"
}
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-07-16T23:08:48-04:00
Range: 2026-07-03 to 2026-07-16

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
| 2026-07-16 | yes | 257 | 177 | 563 | 563 | 5265 | 551 | 97.9% | 0.0% | 0.6% | 3 | 29 |  |
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
```

### Real-money operational prop reports

- rc: 0

**stdout (tail)**
```
{
  "generated_at_utc": "2026-07-17T03:08:52Z",
  "slate_date": "2026-07-16",
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
  "rows": 831,
  "mae_gain": 0.0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_pitcher_k_rate_challenger_latest.md"
}
```

### End-of-slate prop close diagnostic

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-16",
  "evaluation_status": "final",
  "strict_clean_slate": false,
  "valid_close_coverage": 0.0,
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
  "generated_at_utc": "2026-07-17T03:12:09Z",
  "slate_date": "2026-07-16",
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
  "slate_date": "2026-07-16",
  "status": "fail",
  "decision": "do_not_use_slate_for_clean_promotion_evidence",
  "failures": [
    "valid_close_coverage",
    "targeted_close_captures"
  ],
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_daily_slate_trust_latest.md"
}
```

### Grade shadow prop replay

- rc: 0

**stdout (tail)**
```
{
  "graded_rows": 563,
  "run_ids": "all_pending",
  "regrade": false
}
```
