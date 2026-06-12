# SuperNovaBets MLB Daily Run (2026-06-04 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 6.6s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 9.4s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 6.5s)
- **Refresh prop replay CLV**: OK (rc=0, 9.7s)
- **Build prop market training table**: OK (rc=0, 69.0s)
- **Prop walk-forward accuracy report**: FAIL (rc=124, 180.0s)
- **Prop miss diagnostic report**: OK (rc=0, 42.4s)
- **Grade outcomes + ledgers**: OK (rc=0, 8.7s)
- **Prop snapshot coverage report**: OK (rc=0, 7.8s)
- **Grade shadow prop replay**: OK (rc=0, 3.1s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-04 21:45:13,277 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-05. Catching up from 2026-06-06 to 2026-06-03
2026-06-04 21:45:13,277 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-04 window=2026-06-04T04:00:00Z..2026-06-05T04:00:00Z
2026-06-04 21:45:14,337 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-04 | events=2 | credits_remaining=94338
2026-06-04 21:45:14,545 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-05 window=2026-06-05T04:00:00Z..2026-06-06T04:00:00Z
2026-06-04 21:45:15,442 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-05 | events=15 | credits_remaining=94336
2026-06-04 21:45:15,752 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=94336
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-04 21:45:20,934 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-05. Catching up from 2026-06-06 to 2026-06-03
2026-06-04 21:45:21,962 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 2 events (game_date=2026-06-04, as_of=2026-06-04)
2026-06-04 21:45:23,283 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-04 event=77284803e3bb753f121e4c2b4a20ea8e (Kansas City Royals@Minnesota Twins) | credits=94333
2026-06-04 21:45:24,211 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-04 event=2acc5473be6d726025c973a3d148e6b0 (Los Angeles Dodgers@Arizona Diamondbacks) | credits=94330
2026-06-04 21:45:25,324 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-06-05, as_of=2026-06-04)
2026-06-04 21:45:25,494 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=94330
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-04 21:45:29,720 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1370 rows into odds.mlb_game_lines (live odds).
2026-06-04 21:45:29,811 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-04
2026-06-04 21:45:29,812 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-04 21:45:29,812 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-05T03:15:29.812087+00:00.
2026-06-04 21:45:29,954 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-03
2026-06-04 21:45:32,088 | INFO | mlb_pipeline.parse_oddsapi | Upserted 27 rows into odds.mlb_player_prop_lines.
2026-06-04 21:45:32,088 | INFO | mlb_pipeline.parse_oddsapi | Processed 27 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-04 21:45:32,088 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 2348,
  "run_ids": "all",
  "date_from": "2026-06-04",
  "date_to": "2026-06-04",
  "include_graded": true,
  "only_missing": false
}
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 6336,
  "replay_rows": 6336,
  "examples": 6336
}
```

### Prop walk-forward accuracy report

- rc: 124

**stderr (tail)**
```
Timeout after 180s
```

### Prop miss diagnostic report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 3686,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 181-124 (59.3%) ROI: +13.3% | Total: 57-53 (51.8%) ROI: -1.1%
MLB CLV Run Line: beat close 2/15 (13%) avg CLV=+0.40 runs | CLV Total avg=-0.04 runs
MLB Price CLV Run Line: 13 bets  avg=+0.60%
```

**stderr (tail)**
```
2026-06-04 21:50:41,006 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 20 pending predictions.
2026-06-04 21:50:41,006 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-04 21:50:41,022 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-04 21:50:45,034 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-04 21:50:45,050 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-04 21:50:45,176 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-04 21:50:45,177 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-04 21:50:45,432 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-04 21:50:46,376 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-04 21:50:46,378 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-04T23:50:54-04:00
Range: 2026-05-22 to 2026-06-04

## Collection Status

**COLLECTING**

- Clean shadow slates: 1 / 10
- Additional clean slates needed: 9
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 25.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 5.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-04 | no | 2449 | 1945 | 2348 | 2348 | 7269 | 1640 | 69.8% | 0.0% | 13.7% | 2 | 10 | stale_close_rate>0.05 |
| 2026-06-03 | no | 3532 | 3089 | 3331 | 3331 | 3089 | 0 | 0.0% | 0.0% | 100.0% | 1 | 1 | valid_side_locks<100, valid_clv_coverage<0.25, stale_close_rate>0.05 |
| 2026-06-02 | yes | 3547 | 3104 | 235 | 235 | 6442 | 200 | 85.1% | 0.0% | 0.0% | 1 | 12 |  |
| 2026-06-01 | no | 2093 | 0 | 208 | 208 | 5851 | 183 | 88.0% | 0.0% | 9.6% | 1 | 13 | stale_close_rate>0.05 |
| 2026-05-31 | no | 3555 | 0 | 214 | 214 | 9800 | 179 | 83.6% | 0.0% | 15.0% | 1 | 20 | stale_close_rate>0.05 |
| 2026-05-30 | no | 3524 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-29 | no | 3124 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-28 | no | 1391 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-27 | no | 3481 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-26 | no | 3511 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-25 | no | 3051 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-24 | no | 5772 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-23 | no | 3507 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-22 | no | 3272 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
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
