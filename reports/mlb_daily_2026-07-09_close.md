# SuperNovaBets MLB Daily Run (2026-07-09 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 5.3s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 7.6s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 13.8s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 8.8s)
- **Refresh prop replay CLV**: OK (rc=0, 42.3s)
- **Build prop market training table**: OK (rc=0, 103.8s)
- **Prop walk-forward accuracy report**: OK (rc=0, 598.2s)
- **Prop shadow selector report**: OK (rc=0, 13.6s)
- **Prop miss diagnostic report**: OK (rc=0, 125.4s)
- **Prop bucket repair report**: OK (rc=0, 28.4s)
- **TB prop repair report**: OK (rc=0, 16.0s)
- **Prop target quality report**: OK (rc=0, 60.7s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 19.0s)
- **Grade outcomes + ledgers**: OK (rc=0, 27.1s)
- **Grade daily forecast ledger**: OK (rc=0, 15.8s)
- **Daily forecast projection audit**: OK (rc=0, 16.8s)
- **Prop micro promotion evaluation**: OK (rc=0, 0.4s)
- **Prop layer promotion control**: OK (rc=0, 0.6s)
- **Forecast repair error decomposition**: OK (rc=0, 6.6s)
- **Prop snapshot coverage report**: OK (rc=0, 53.6s)
- **End-of-slate prop close diagnostic**: OK (rc=0, 3.9s)
- **Frozen-release five-date checkpoint**: OK (rc=0, 36.6s)
- **Daily slate trust monitor**: OK (rc=0, 42.1s)
- **Grade shadow prop replay**: OK (rc=0, 8.1s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-09 21:45:07,113 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 26 teams today (2026-07-09)
2026-07-09 21:45:07,634 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 52 rows for 2026-07-09
2026-07-09 21:45:07,634 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 52 assignments for 2026-07-09
2026-07-09 21:45:07,650 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-09 21:45:07,996 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 13 unique games for season=2026-regular
2026-07-09 21:45:08,074 | INFO | mlb_pipeline.crawler_statsapi | Upserted 13 rows into raw.mlb_games
2026-07-09 21:45:08,121 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6243 completed games, 6233 already done, 1 to fetch
2026-07-09 21:45:08,969 | INFO | mlb_pipeline.crawler_statsapi |   Progress: 1 / 1 games fetched
2026-07-09 21:45:09,141 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=1, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-09 21:45:14,849 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-10. Catching up from 2026-07-11 to 2026-07-08
2026-07-09 21:45:14,849 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-09 window=2026-07-09T04:00:00Z..2026-07-10T04:00:00Z
2026-07-09 21:45:15,609 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-09 | events=2 | credits_remaining=84461
2026-07-09 21:45:15,822 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-10 window=2026-07-10T04:00:00Z..2026-07-11T04:00:00Z
2026-07-09 21:45:16,619 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-10 | events=15 | credits_remaining=84459
2026-07-09 21:45:16,681 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=84459
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-09 21:45:19,134 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-10. Catching up from 2026-07-11 to 2026-07-08
2026-07-09 21:45:19,828 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 2 events (game_date=2026-07-09, as_of=2026-07-09)
2026-07-09 21:45:24,434 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-09 event=ff7fcfaa73e919c6a69c375313853a91 (Arizona Diamondbacks@San Diego Padres) | credits=84454
2026-07-09 21:45:29,421 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-09 event=f6d5d25a3b26ed6f4b0aa584e75b872c (Colorado Rockies@San Francisco Giants) | credits=84449
2026-07-09 21:45:30,373 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-07-10, as_of=2026-07-09)
2026-07-09 21:45:30,467 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=84449
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-09 21:45:37,042 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1556 rows into odds.mlb_game_lines (live odds).
2026-07-09 21:45:37,057 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-09
2026-07-09 21:45:37,057 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-09 21:45:37,073 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-10T03:15:37.073280+00:00.
2026-07-09 21:45:37,104 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-08
2026-07-09 21:45:39,260 | INFO | mlb_pipeline.parse_oddsapi | Upserted 195 rows into odds.mlb_player_prop_lines.
2026-07-09 21:45:39,260 | INFO | mlb_pipeline.parse_oddsapi | Processed 195 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-09 21:45:39,260 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 5964,
  "run_ids": "all",
  "date_from": "2026-07-09",
  "date_to": "2026-07-09",
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
  "replay_rows": 22204,
  "examples": 22204
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 219413,
  "graded_rows": 197639,
  "valid_clv_rows": 182729,
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
  "active_rows": 1348,
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
  "rows": 22051,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 196813,
  "bucket_count": 144,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=81298
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 218512,
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
    "raw_api_one_sided": 17738
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 217-144 (60.1%) ROI: +14.8% | Total: 71-67 (51.4%) ROI: -1.8%
MLB CLV Run Line: beat close 2/69 (3%) avg CLV=+0.09 runs | CLV Total avg=+0.08 runs
MLB Price CLV Run Line: 67 bets  avg=+0.51%
```

**stderr (tail)**
```
2026-07-09 22:02:30,140 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 1 rows
2026-07-09 22:02:30,140 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1 MLB game outcome rows
2026-07-09 22:02:30,140 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-09 22:02:52,354 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 150 prop rows
2026-07-09 22:02:52,370 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 150 MLB prop outcome rows
2026-07-09 22:02:52,818 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-09 22:02:52,833 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-09 22:02:53,177 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 2 game model-pick ledger rows
2026-07-09 22:02:53,849 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 21 prop model-pick ledger rows
2026-07-09 22:02:53,865 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 23 MLB model-pick ledger rows
```

### Grade daily forecast ledger

- rc: 0

**stdout (tail)**
```
{
  "schema_ready": true,
  "graded": 267
}
```

### Daily forecast projection audit

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 8398,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_daily_forecast_projection_audit_latest.md"
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
        "batter_hits_rows<100",
        "batter_home_runs_projection_not_better_than_baseline",
        "batter_home_runs_rows<100",
        "completed_dates<5"
      ]
    },
    "hitter_pa_v3_production": {
      "enabled": false,
      "layer": "hitter_pa_v3",
      "target_mode": "production_scoring",
      "blockers": [
        "completed_dates<5",
        "hitter_plate_appearances_challenger_projection_gate_failed",
        "hitter_plate_appearances_challenger_projection_not_better_than_baseline",
        "pending_anchor_rows"
      ]
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
  "prospective_rows": 5061,
  "historical_rows": 21674,
  "tb_repair": "train_gated_direct_player_game_tb_head",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_forecast_repair_error_latest.md"
}
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-07-10T00:04:27-04:00
Range: 2026-06-27 to 2026-07-10

## Collection Status

**COLLECTING**

- Clean shadow slates: 0 / 10
- Additional clean slates needed: 10
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 90.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 2.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-07-10 | no | 0 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, captured_side_locks<100, close_capture_coverage<0.90, valid_close_coverage<0.90, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.02, no_close_snapshot_time |
| 2026-07-09 | no | 3481 | 2248 | 5964 | 5964 | 62333 | 5292 | 88.7% | 0.0% | 0.0% | 3 | 40 | valid_close_coverage<0.90 |
| 2026-07-08 | no | 4609 | 2727 | 7951 | 7951 | 73124 | 6833 | 85.9% | 0.0% | 0.0% | 3 | 32 | valid_close_coverage<0.90 |
| 2026-07-07 | no | 4877 | 2689 | 8289 | 8289 | 75276 | 7151 | 86.3% | 0.0% | 0.6% | 3 | 31 | valid_close_coverage<0.90 |
| 2026-07-06 | no | 2495 | 1469 | 4100 | 4100 | 38682 | 3580 | 87.3% | 0.0% | 0.0% | 3 | 31 | valid_close_coverage<0.90 |
| 2026-07-05 | no | 4596 | 2708 | 5782 | 5782 | 34519 | 4954 | 85.7% | 0.0% | 1.3% | 3 | 20 | valid_close_coverage<0.90 |
| 2026-07-04 | no | 4640 | 2783 | 7530 | 7530 | 43937 | 6333 | 84.1% | 0.0% | 4.2% | 3 | 20 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-07-03 | no | 3879 | 2026 | 6784 | 6784 | 39279 | 5921 | 87.3% | 0.0% | 2.0% | 3 | 19 | valid_close_coverage<0.90 |
| 2026-07-02 | no | 2683 | 1678 | 4505 | 4505 | 22054 | 4063 | 90.2% | 0.0% | 2.1% | 3 | 18 | stale_close_rate>0.02 |
| 2026-07-01 | no | 4174 | 2582 | 6538 | 6538 | 25183 | 5254 | 80.4% | 0.0% | 3.1% | 3 | 15 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-06-30 | no | 4519 | 2785 | 8060 | 8060 | 41133 | 6911 | 85.7% | 0.0% | 3.0% | 3 | 18 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-06-29 | no | 3883 | 2282 | 6912 | 6912 | 35281 | 5954 | 86.1% | 0.0% | 3.0% | 3 | 18 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-06-28 | no | 4600 | 2631 | 5935 | 5935 | 18826 | 4885 | 82.3% | 0.0% | 10.7% | 3 | 14 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-06-27 | no | 4389 | 2875 | 7032 | 7032 | 28903 | 6054 | 86.1% | 0.0% | 2.5% | 3 | 15 | valid_close_coverage<0.90, stale_close_rate>0.02 |
```

### End-of-slate prop close diagnostic

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-09",
  "evaluation_status": "provisional",
  "strict_clean_slate": false,
  "valid_close_coverage": 0.8959223300970873,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_end_of_slate_close_latest.md"
}
```

### Frozen-release five-date checkpoint

- rc: 0

**stdout (tail)**
```
{
  "status": "collecting",
  "hitter_completed_dates": 3,
  "hitter_rate_shadow_completed_dates": 0,
  "pitcher_completed_dates": 5,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_five_date_checkpoint_latest.md"
}
```

### Daily slate trust monitor

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-09",
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
  "graded_rows": 442,
  "run_ids": "all_pending",
  "regrade": false
}
```
