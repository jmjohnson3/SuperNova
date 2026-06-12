# SuperNovaBets MLB Daily Run (2026-06-09 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 6.9s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 10.6s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 9.5s)
- **Refresh prop replay CLV**: OK (rc=0, 16.3s)
- **Build prop market training table**: OK (rc=0, 105.3s)
- **Prop walk-forward accuracy report**: OK (rc=0, 69.0s)
- **Prop miss diagnostic report**: OK (rc=0, 25.2s)
- **Prop target quality report**: OK (rc=0, 24.1s)
- **Grade outcomes + ledgers**: OK (rc=0, 11.6s)
- **Prop snapshot coverage report**: OK (rc=0, 25.7s)
- **Grade shadow prop replay**: OK (rc=0, 10.9s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-09 21:45:48,067 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-10. Catching up from 2026-06-11 to 2026-06-08
2026-06-09 21:45:48,067 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-09 window=2026-06-09T04:00:00Z..2026-06-10T04:00:00Z
2026-06-09 21:45:48,953 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-09 | events=5 | credits_remaining=87434
2026-06-09 21:45:49,270 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-10 window=2026-06-10T04:00:00Z..2026-06-11T04:00:00Z
2026-06-09 21:45:49,926 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-10 | events=14 | credits_remaining=87432
2026-06-09 21:45:49,942 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=87432
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-09 21:45:53,920 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-10. Catching up from 2026-06-11 to 2026-06-08
2026-06-09 21:45:54,879 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 5 events (game_date=2026-06-09, as_of=2026-06-09)
2026-06-09 21:45:55,503 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-09 event=f03156629a6e1e4d346ffc42fb0c0bdf (Minnesota Twins@Detroit Tigers) | credits=87429
2026-06-09 21:45:56,647 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-09 event=2d3e8521a55eeeae51961cf0d8ebd6c6 (Houston Astros@Los Angeles Angels) | credits=87426
2026-06-09 21:45:57,583 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-09 event=e2d199125ede80ca2e35bc0c0669f294 (Cincinnati Reds@San Diego Padres) | credits=87423
2026-06-09 21:45:58,510 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-09 event=f2761470e60cb7a1fa947535ccba356f (Washington Nationals@San Francisco Giants) | credits=87420
2026-06-09 21:45:59,582 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-09 event=53115c407b5594a84cae2269c1005ab0 (Milwaukee Brewers@Athletics) | credits=87415
2026-06-09 21:46:00,470 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 14 events (game_date=2026-06-10, as_of=2026-06-09)
2026-06-09 21:46:00,604 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=87415
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-09 21:46:05,435 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1404 rows into odds.mlb_game_lines (live odds).
2026-06-09 21:46:05,463 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-09
2026-06-09 21:46:05,464 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-09 21:46:05,465 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-10T03:16:05.465149+00:00.
2026-06-09 21:46:05,571 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-08
2026-06-09 21:46:09,929 | INFO | mlb_pipeline.parse_oddsapi | Upserted 372 rows into odds.mlb_player_prop_lines.
2026-06-09 21:46:09,929 | INFO | mlb_pipeline.parse_oddsapi | Processed 372 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-09 21:46:09,929 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 6566,
  "run_ids": "all",
  "date_from": "2026-06-09",
  "date_to": "2026-06-09",
  "include_graded": true,
  "only_missing": false
}
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 34273,
  "replay_rows": 34273,
  "examples": 34273
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 35174,
  "graded_rows": 25800,
  "valid_clv_rows": 26468,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_walk_forward_accuracy_latest.md",
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_walk_forward_accuracy_report.json"
}
```

### Prop miss diagnostic report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 24974,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 34273,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 185-125 (59.7%) ROI: +13.9% | Total: 57-55 (50.9%) ROI: -2.8%
MLB CLV Run Line: beat close 2/19 (11%) avg CLV=+0.32 runs | CLV Total avg=-0.03 runs
MLB Price CLV Run Line: 17 bets  avg=+0.77%
```

**stderr (tail)**
```
2026-06-09 21:50:18,431 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 31 pending predictions.
2026-06-09 21:50:18,432 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-09 21:50:18,512 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-09 21:50:20,625 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-09 21:50:20,642 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-09 21:50:20,747 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-09 21:50:20,747 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-09 21:50:20,920 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-09 21:50:21,400 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-09 21:50:21,404 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-09T23:50:45-04:00
Range: 2026-05-27 to 2026-06-09

## Collection Status

**COLLECTING**

- Clean shadow slates: 6 / 10
- Additional clean slates needed: 4
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 25.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 5.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-09 | yes | 3821 | 3177 | 6566 | 6566 | 40669 | 5702 | 86.8% | 0.0% | 3.5% | 2 | 16 |  |
| 2026-06-08 | yes | 2447 | 1705 | 3589 | 3589 | 27069 | 3093 | 86.2% | 0.0% | 3.1% | 2 | 19 |  |
| 2026-06-07 | yes | 4496 | 3136 | 6521 | 6521 | 30276 | 5638 | 86.5% | 0.0% | 1.0% | 3 | 19 |  |
| 2026-06-06 | yes | 4345 | 3180 | 4562 | 4562 | 38898 | 4017 | 88.1% | 0.0% | 2.2% | 2 | 19 |  |
| 2026-06-05 | yes | 4520 | 3154 | 6699 | 6699 | 57725 | 5816 | 86.8% | 0.0% | 4.2% | 2 | 22 |  |
| 2026-06-04 | no | 2505 | 1945 | 2348 | 2348 | 7269 | 1640 | 69.8% | 0.0% | 13.7% | 2 | 10 | stale_close_rate>0.05 |
| 2026-06-03 | no | 3532 | 3089 | 3331 | 3331 | 3089 | 0 | 0.0% | 0.0% | 100.0% | 1 | 1 | valid_side_locks<100, valid_clv_coverage<0.25, stale_close_rate>0.05 |
| 2026-06-02 | yes | 4467 | 3104 | 235 | 235 | 6442 | 200 | 85.1% | 0.0% | 0.0% | 1 | 12 |  |
| 2026-06-01 | no | 2668 | 0 | 208 | 208 | 5851 | 183 | 88.0% | 0.0% | 9.6% | 1 | 13 | stale_close_rate>0.05 |
| 2026-05-31 | no | 4599 | 0 | 214 | 214 | 9800 | 179 | 83.6% | 0.0% | 15.0% | 1 | 20 | stale_close_rate>0.05 |
| 2026-05-30 | no | 3524 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-29 | no | 3124 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-28 | no | 1391 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-27 | no | 3481 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
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
