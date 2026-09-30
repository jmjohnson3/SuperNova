# SuperNovaBets MLB Daily Run (2026-07-23 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 4.9s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 6.0s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 4.0s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 3.7s)
- **Refresh prop replay CLV**: OK (rc=0, 11.3s)
- **Build prop market training table**: OK (rc=0, 108.1s)
- **Prop walk-forward accuracy report**: OK (rc=0, 7.3s)
- **Prop shadow selector report**: OK (rc=0, 7.3s)
- **Prop miss diagnostic report**: OK (rc=0, 117.0s)
- **Prop bucket repair report**: OK (rc=0, 54.7s)
- **TB prop repair report**: OK (rc=0, 32.3s)
- **Prop target quality report**: OK (rc=0, 88.8s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 15.7s)
- **Grade outcomes + ledgers**: OK (rc=0, 15.8s)
- **Grade daily forecast ledger**: OK (rc=0, 4.5s)
- **Daily forecast projection audit**: OK (rc=0, 16.5s)
- **Hitter live-vs-legacy forecast diff**: OK (rc=0, 21.1s)
- **Player-game bankroll proof**: OK (rc=0, 25.5s)
- **TB 1.5 line calibration**: OK (rc=0, 21.2s)
- **TB 1.5 close repair report**: OK (rc=0, 4.5s)
- **K-under repair report**: OK (rc=0, 4.4s)
- **DK K 4.5-6.0 under repair diagnostic**: OK (rc=0, 4.2s)
- **Exact-bucket CLV priors**: OK (rc=0, 12.1s)
- **Prop micro promotion evaluation**: OK (rc=0, 1.1s)
- **Prop micro gate sensitivity report**: OK (rc=0, 2.5s)
- **Prop micro bucket repair report**: OK (rc=0, 1.9s)
- **Prop trial candidate queue report**: OK (rc=0, 2.4s)
- **Prop drift guard diagnostic**: OK (rc=0, 31.9s)
- **Prop bettable-now scan**: OK (rc=0, 5.1s)
- **Lock micro projection ledger**: OK (rc=0, 6.2s)
- **Prop micro ledger report**: OK (rc=0, 3.2s)
- **Prop micro loss diagnostic**: OK (rc=0, 3.3s)
- **Prop micro probability calibrator**: OK (rc=0, 2.5s)
- **Prop post-gate candidate report**: OK (rc=0, 5.1s)
- **Prop layer promotion control**: OK (rc=0, 0.6s)
- **Forecast repair error decomposition**: OK (rc=0, 9.5s)
- **TB tail repair challenger**: OK (rc=0, 39.7s)
- **Prop snapshot coverage report**: OK (rc=0, 92.4s)
- **Real-money operational prop reports**: OK (rc=0, 11.1s)
- **Pitcher K-rate challenger diagnostic**: OK (rc=0, 68.3s)
- **End-of-slate prop close diagnostic**: OK (rc=0, 8.5s)
- **Frozen-release five-date checkpoint**: FAIL (rc=124, 182.2s)
- **Post-checkpoint micro promotion evaluation**: OK (rc=0, 12.2s)
- **Post-checkpoint micro gate sensitivity report**: OK (rc=0, 1.8s)
- **Post-checkpoint micro bucket repair report**: OK (rc=0, 0.8s)
- **Post-checkpoint prop trial candidate queue report**: OK (rc=0, 1.6s)
- **Post-checkpoint prop drift guard diagnostic**: OK (rc=0, 94.3s)
- **Post-checkpoint lock micro projection ledger**: OK (rc=0, 27.6s)
- **Post-checkpoint prop micro ledger report**: OK (rc=0, 5.7s)
- **Post-checkpoint prop micro probability calibrator**: OK (rc=0, 2.9s)
- **Post-checkpoint prop layer promotion control**: OK (rc=0, 0.3s)
- **Post-checkpoint real-money operational prop reports**: OK (rc=0, 8.8s)
- **Daily slate trust monitor**: OK (rc=0, 409.2s)
- **Grade shadow prop replay**: OK (rc=0, 21.4s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-23 21:45:11,533 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 10 teams today (2026-07-23)
2026-07-23 21:45:11,885 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 20 rows for 2026-07-23
2026-07-23 21:45:11,885 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 20 assignments for 2026-07-23
2026-07-23 21:45:11,971 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-23 21:45:12,271 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 5 unique games for season=2026-regular
2026-07-23 21:45:12,512 | INFO | mlb_pipeline.crawler_statsapi | Upserted 5 rows into raw.mlb_games
2026-07-23 21:45:12,649 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6386 completed games, 6377 already done, 0 to fetch
2026-07-23 21:45:12,923 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=0, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-23 21:45:16,448 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-24. Catching up from 2026-07-25 to 2026-07-22
2026-07-23 21:45:16,449 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-23 window=2026-07-23T04:00:00Z..2026-07-24T04:00:00Z
2026-07-23 21:45:17,737 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-23 | events=0 | credits_remaining=41745
2026-07-23 21:45:17,909 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-24 window=2026-07-24T04:00:00Z..2026-07-25T04:00:00Z
2026-07-23 21:45:18,830 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-24 | events=15 | credits_remaining=41743
2026-07-23 21:45:18,909 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=41743
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-23 21:45:21,628 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-24. Catching up from 2026-07-25 to 2026-07-22
2026-07-23 21:45:22,268 | INFO | mlb_pipeline.crawler_oddsapi | No events found for 2026-07-23 â€” skipping prop fetch
2026-07-23 21:45:22,894 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-07-24, as_of=2026-07-23)
2026-07-23 21:45:22,955 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=unknown
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-23 21:45:26,564 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1597 rows into odds.mlb_game_lines (live odds).
2026-07-23 21:45:26,580 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-23
2026-07-23 21:45:26,580 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-23 21:45:26,580 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-24T03:15:26.580315+00:00.
2026-07-23 21:45:26,689 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-22
2026-07-23 21:45:26,689 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_prop_odds snapshots found (as_of_date=None).
2026-07-23 21:45:26,689 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 1569,
  "run_ids": "all",
  "markets": "all",
  "sides": "all",
  "bookmakers": "all",
  "line_buckets": "all",
  "limit": null,
  "batch_size": 100,
  "statement_timeout_ms": 30000,
  "lock_timeout_ms": 2000,
  "ensure_schema": false,
  "date_from": "2026-07-23",
  "date_to": "2026-07-23",
  "include_graded": true,
  "only_missing": false
}
```

**stderr (tail)**
```
2026-07-23 21:45:30,636 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=100 updated=100 skipped=0 last_id=302567
2026-07-23 21:45:31,190 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=200 updated=200 skipped=0 last_id=302667
2026-07-23 21:45:31,721 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=300 updated=300 skipped=0 last_id=302767
2026-07-23 21:45:32,035 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=400 updated=400 skipped=0 last_id=302867
2026-07-23 21:45:33,142 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=500 updated=500 skipped=0 last_id=302967
2026-07-23 21:45:34,242 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=600 updated=600 skipped=0 last_id=303067
2026-07-23 21:45:34,660 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=700 updated=700 skipped=0 last_id=303167
2026-07-23 21:45:35,111 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=800 updated=800 skipped=0 last_id=303267
2026-07-23 21:45:35,517 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=900 updated=900 skipped=0 last_id=303367
2026-07-23 21:45:35,876 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1000 updated=1000 skipped=0 last_id=303467
2026-07-23 21:45:36,146 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1100 updated=1100 skipped=0 last_id=303567
2026-07-23 21:45:36,425 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1200 updated=1200 skipped=0 last_id=303667
2026-07-23 21:45:36,939 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1300 updated=1300 skipped=0 last_id=303767
2026-07-23 21:45:37,409 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1400 updated=1400 skipped=0 last_id=303867
2026-07-23 21:45:37,799 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1500 updated=1500 skipped=0 last_id=303967
2026-07-23 21:45:38,012 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1569 updated=1569 skipped=0 last_id=304036
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 0,
  "replay_rows": 16710,
  "examples": 16710
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "reused_fresh_artifact",
  "artifact": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_walk_forward_accuracy_report.json",
  "age_minutes": 54.4,
  "rows": 250721,
  "graded_rows": 226184,
  "valid_clv_rows": 210771,
  "generated_at_utc": "2026-07-24T02:52:49+00:00"
}
```

### Prop shadow selector report

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "active_rows": 213,
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
  "rows": 23917,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 257073,
  "bucket_count": 155,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=107037
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 284780,
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
    "raw_api_one_sided": 13669
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 229-156 (59.5%) ROI: +13.6% | Total: 72-69 (51.1%) ROI: -2.5%
MLB CLV Run Line: beat close 3/93 (3%) avg CLV=+0.10 runs | CLV Total avg=+0.10 runs
MLB Price CLV Run Line: 90 bets  avg=+0.34%
```

**stderr (tail)**
```
2026-07-23 21:52:59,471 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 27 pending predictions.
2026-07-23 21:52:59,471 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-07-23 21:52:59,487 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-23 21:53:03,706 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-07-23 21:53:03,737 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-07-23 21:53:04,158 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-23 21:53:04,158 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-23 21:53:04,377 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-07-23 21:53:05,033 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-07-23 21:53:05,049 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
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
  "rows": 24346,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_daily_forecast_projection_audit_latest.md"
}
```

### Hitter live-vs-legacy forecast diff

- rc: 0

**stdout (tail)**
```
{
  "rows": 5043,
  "phase": "day_pregame",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_hitter_live_vs_legacy_forecast_diff_latest.md"
}
```

### Player-game bankroll proof

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "allowed_stats": [],
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_player_game_bankroll_model_proof_latest.md"
}
```

### TB 1.5 line calibration

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 10657,
  "enabled_count": 6,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb15_line_calibration_latest.md"
}
```

### TB 1.5 close repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "focus_rows": 4406,
  "valid_close_coverage": 0.7945982750794371,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb15_close_repair_latest.md"
}
```

### K-under repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 2204,
  "micro_allowed_count": 1,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_k_under_repair_latest.md"
}
```

### DK K 4.5-6.0 under repair diagnostic

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 761,
  "roi": -0.05525729303547963,
  "clv_beat_rate": 0.47909407665505227,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_k_under_46_repair_diagnostic_latest.md"
}
```

### Exact-bucket CLV priors

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 53930,
  "micro_clv_confirmed_buckets": 3,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_exact_bucket_clv_priors_latest.md"
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
    "watch_only": 1,
    "micro_test": 0,
    "strict_micro": 0
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
}
```

### Prop micro bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "trial_ready_count": 0,
  "near_miss_count": 98,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_bucket_repair_latest.md"
}
```

### Prop trial candidate queue report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "trial_ready_count": 0,
  "near_miss_1_2_gate_count": 3,
  "proof_refresh_markets": [
    "pitcher_strikeouts",
    "batter_total_bases",
    "batter_hits"
  ],
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_trial_candidate_queue_latest.md"
}
```

### Prop drift guard diagnostic

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-23', 'micro_rows': 1, 'micro_bettable_now': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

### Prop bettable-now scan

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-23', 'approved_rows': 6, 'near_approved_rows': 2, 'bettable_now_before_cap': 0, 'bettable_now_inside_cap': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_bettable_now_scan.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bettable_now_scan_latest.md'}
```

### Lock micro projection ledger

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "game_date": "2026-07-23",
  "active_prediction_rows": 213,
  "locked_rows_attempted": 0,
  "ledger_before": [
    {
      "model_tier": "micro_projection",
      "rows": 1,
      "stake_usd": 1.0
    },
    {
      "model_tier": "watch",
      "rows": 30,
      "stake_usd": 0.0
    }
  ],
  "ledger_after": [
    {
      "model_tier": "micro_projection",
      "rows": 1,
      "stake_usd": 1.0
    },
    {
      "model_tier": "watch",
      "rows": 30,
      "stake_usd": 0.0
    }
  ],
  "micro_lock_audit": {
    "target_buckets": [
      "batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings",
      "batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings",
      "pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings"
    ],
    "target_ledger_locked": 0,
    "micro_ledger_locked": 0,
    "status_counts": {
      "selector_not_micro_projection": 212,
      "stale_after_expired": 1
    },
    "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_lock_audit.json",
    "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_lock_audit_latest.md"
  }
}
```

### Prop micro ledger report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "graded": 24,
  "pending": 3,
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_ledger_report.json",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_ledger_latest.md"
}
```

### Prop micro loss diagnostic

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 27,
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_loss_diagnostic.json",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_loss_diagnostic_latest.md"
}
```

### Prop micro probability calibrator

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "enabled": true,
  "graded_rows": 24,
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_probability_calibrator.json",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_probability_calibrator_latest.md"
}
```

### Prop post-gate candidate report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "active_rows": 213,
  "micro_projection_rows": 1,
  "near_misses": 71,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_post_gate_candidate_latest.md"
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
        "batter_hits_projection_not_better_than_baseline"
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
  "prospective_rows": 16111,
  "historical_rows": 27016,
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
  "rows": 8986,
  "accepted": true,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb_tail_repair_challenger_latest.md"
}
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-07-23T23:58:26-04:00
Range: 2026-07-10 to 2026-07-23

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
| 2026-07-23 | no | 1546 | 864 | 1569 | 1569 | 44834 | 1335 | 85.1% | 0.0% | 0.0% | 3 | 68 | valid_close_coverage<0.90 |
| 2026-07-22 | no | 5316 | 2931 | 7248 | 7248 | 129373 | 6127 | 84.5% | 0.0% | 0.5% | 3 | 54 | valid_close_coverage<0.90 |
| 2026-07-21 | no | 4683 | 2771 | 7893 | 7893 | 127679 | 6225 | 78.9% | 0.0% | 1.9% | 3 | 55 | valid_close_coverage<0.90 |
| 2026-07-20 | no | 4685 | 2740 | 7638 | 7638 | 123069 | 6711 | 87.9% | 0.0% | 0.0% | 3 | 49 | valid_close_coverage<0.90 |
| 2026-07-19 | no | 4978 | 2676 | 5423 | 5423 | 81193 | 4648 | 85.7% | 0.0% | 0.4% | 3 | 42 | valid_close_coverage<0.90 |
| 2026-07-18 | no | 4910 | 2748 | 6033 | 6033 | 202052 | 4896 | 81.2% | 0.0% | 0.9% | 3 | 101 | valid_close_coverage<0.90 |
| 2026-07-17 | no | 4447 | 2687 | 7705 | 7705 | 195834 | 6471 | 84.0% | 0.0% | 1.0% | 3 | 83 | valid_close_coverage<0.90 |
| 2026-07-16 | yes | 280 | 177 | 563 | 563 | 5265 | 551 | 97.9% | 0.0% | 0.0% | 3 | 29 |  |
| 2026-07-15 | no | 0 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, captured_side_locks<100, close_capture_coverage<0.90, valid_close_coverage<0.90, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.02, no_close_snapshot_time |
| 2026-07-14 | no | 42 | 42 | 0 | 0 | 990 | 0 | - | - | - | 0 | 24 | side_locks<100, captured_side_locks<100, close_capture_coverage<0.90, valid_close_coverage<0.90, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.02 |
| 2026-07-13 | no | 42 | 42 | 0 | 0 | 42 | 0 | - | - | - | 0 | 1 | side_locks<100, captured_side_locks<100, close_capture_coverage<0.90, valid_close_coverage<0.90, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.02 |
| 2026-07-12 | no | 4577 | 2643 | 7667 | 7667 | 76486 | 4599 | 60.0% | 0.0% | 27.4% | 3 | 33 | close_capture_coverage<0.90, valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-07-11 | no | 4850 | 2754 | 6643 | 6643 | 123974 | 5923 | 89.2% | 0.0% | 0.0% | 3 | 58 | valid_close_coverage<0.90 |
| 2026-07-10 | no | 4688 | 2855 | 8100 | 8100 | 97641 | 7024 | 86.7% | 0.0% | 0.0% | 3 | 44 | valid_close_coverage<0.90 |
```

### Real-money operational prop reports

- rc: 0

**stdout (tail)**
```
{
  "generated_at_utc": "2026-07-24T03:58:27Z",
  "slate_date": "2026-07-23",
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
  "rows": 975,
  "mae_gain": 0.0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_pitcher_k_rate_challenger_latest.md"
}
```

### End-of-slate prop close diagnostic

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-23",
  "evaluation_status": "final",
  "strict_clean_slate": false,
  "valid_close_coverage": 0.8615179760319573,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_end_of_slate_close_latest.md"
}
```

### Frozen-release five-date checkpoint

- rc: 124

**stderr (tail)**
```
Timed out after 180s; killed process tree rooted at PID 12492
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
    "watch_only": 1,
    "micro_test": 0,
    "strict_micro": 0
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
}
```

### Post-checkpoint micro bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "trial_ready_count": 0,
  "near_miss_count": 98,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_bucket_repair_latest.md"
}
```

### Post-checkpoint prop trial candidate queue report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "trial_ready_count": 0,
  "near_miss_1_2_gate_count": 3,
  "proof_refresh_markets": [
    "pitcher_strikeouts",
    "batter_total_bases",
    "batter_hits"
  ],
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_trial_candidate_queue_latest.md"
}
```

### Post-checkpoint prop drift guard diagnostic

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-24', 'micro_rows': 0, 'micro_bettable_now': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

**stderr (tail)**
```
No normalized prop offers available for 2026-07-24
C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\modeling\predict_player_props.py:4950: UserWarning: pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy.
  df = pd.read_sql(SQL_PROP_LINES, conn, params={"game_date": game_date})
```

### Post-checkpoint lock micro projection ledger

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "game_date": "2026-07-24",
  "active_prediction_rows": 0,
  "locked_rows_attempted": 0,
  "ledger_before": [],
  "ledger_after": [],
  "micro_lock_audit": {
    "target_buckets": [
      "batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings",
      "batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings",
      "pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings"
    ],
    "target_ledger_locked": 0,
    "micro_ledger_locked": 0,
    "status_counts": {},
    "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_lock_audit.json",
    "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_lock_audit_latest.md"
  }
}
```

**stderr (tail)**
```
No normalized prop offers available for 2026-07-24
C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\modeling\predict_player_props.py:4950: UserWarning: pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy.
  df = pd.read_sql(SQL_PROP_LINES, conn, params={"game_date": game_date})
```

### Post-checkpoint prop micro ledger report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "graded": 24,
  "pending": 3,
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_ledger_report.json",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_ledger_latest.md"
}
```

### Post-checkpoint prop micro probability calibrator

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "enabled": true,
  "graded_rows": 24,
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_probability_calibrator.json",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_probability_calibrator_latest.md"
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
        "batter_hits_projection_not_better_than_baseline"
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
  "generated_at_utc": "2026-07-24T04:05:24Z",
  "slate_date": "2026-07-24",
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
  "slate_date": "2026-07-23",
  "status": "fail",
  "decision": "do_not_use_slate_for_clean_promotion_evidence",
  "failures": [
    "valid_close_coverage"
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
