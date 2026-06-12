# SuperNovaBets MLB Daily Run (2026-06-05 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 27.7s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 32.5s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 7.5s)
- **Refresh prop replay CLV**: OK (rc=0, 22.9s)
- **Build prop market training table**: OK (rc=0, 71.7s)
- **Prop walk-forward accuracy report**: OK (rc=0, 32.5s)
- **Prop miss diagnostic report**: OK (rc=0, 35.1s)
- **Prop target quality report**: OK (rc=0, 15.0s)
- **Grade outcomes + ledgers**: OK (rc=0, 14.6s)
- **Prop snapshot coverage report**: OK (rc=0, 12.4s)
- **Grade shadow prop replay**: OK (rc=0, 3.9s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-05 21:45:11,270 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-06. Catching up from 2026-06-07 to 2026-06-04
2026-06-05 21:45:11,271 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-05 window=2026-06-05T04:00:00Z..2026-06-06T04:00:00Z
2026-06-05 21:45:30,985 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-05 | events=5 | credits_remaining=92418
2026-06-05 21:45:32,022 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-06 window=2026-06-06T04:00:00Z..2026-06-07T04:00:00Z
2026-06-05 21:45:33,561 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-06 | events=15 | credits_remaining=92416
2026-06-05 21:45:33,648 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=92416
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-05 21:45:56,463 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-06. Catching up from 2026-06-07 to 2026-06-04
2026-06-05 21:45:57,294 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 5 events (game_date=2026-06-05, as_of=2026-06-05)
2026-06-05 21:45:57,956 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-05 event=eb17640885894253c1069e2e9233ea65 (Milwaukee Brewers@Colorado Rockies) | credits=92414
2026-06-05 21:45:58,957 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-05 event=be13ae41e6022b172e8564f73444cce6 (Kansas City Royals@Minnesota Twins) | credits=92409
2026-06-05 21:46:02,834 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-05 event=3154fee42c445f5f90ad1da53cfefedb (Washington Nationals@Arizona Diamondbacks) | credits=92406
2026-06-05 21:46:03,945 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-05 event=b84d12d6466a2f541ef74605bc5f53c6 (New York Mets@San Diego Padres) | credits=92403
2026-06-05 21:46:04,875 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-05 event=ae2f29fce3de0447b7b9642b09c76d58 (Los Angeles Angels@Los Angeles Dodgers) | credits=92397
2026-06-05 21:46:05,743 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-06-06, as_of=2026-06-05)
2026-06-05 21:46:05,905 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=92397
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-05 21:46:10,925 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1379 rows into odds.mlb_game_lines (live odds).
2026-06-05 21:46:10,947 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-05
2026-06-05 21:46:10,957 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-05 21:46:10,957 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-06T03:16:10.957707+00:00.
2026-06-05 21:46:11,101 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-04
2026-06-05 21:46:13,686 | INFO | mlb_pipeline.parse_oddsapi | Upserted 240 rows into odds.mlb_player_prop_lines.
2026-06-05 21:46:13,686 | INFO | mlb_pipeline.parse_oddsapi | Processed 240 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-05 21:46:13,686 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 6699,
  "run_ids": "all",
  "date_from": "2026-06-05",
  "date_to": "2026-06-05",
  "include_graded": true,
  "only_missing": false
}
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 13035,
  "replay_rows": 13035,
  "examples": 13035
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 13936,
  "graded_rows": 6744,
  "valid_clv_rows": 8018,
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
  "rows": 5918,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 13035,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 181-125 (59.2%) ROI: +12.9% | Total: 57-54 (51.4%) ROI: -2.0%
MLB CLV Run Line: beat close 2/15 (13%) avg CLV=+0.40 runs | CLV Total avg=-0.04 runs
MLB Price CLV Run Line: 13 bets  avg=+0.60%
```

**stderr (tail)**
```
2026-06-05 21:49:22,244 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 15 pending predictions.
2026-06-05 21:49:22,244 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-05 21:49:22,291 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-05 21:49:24,294 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-05 21:49:24,310 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-05 21:49:24,430 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-05 21:49:25,210 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-05 21:49:25,221 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-05T23:49:37-04:00
Range: 2026-05-23 to 2026-06-05

## Collection Status

**COLLECTING**

- Clean shadow slates: 2 / 10
- Additional clean slates needed: 8
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 25.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 5.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-05 | yes | 3956 | 3058 | 6699 | 6699 | 57725 | 5816 | 86.8% | 0.0% | 4.2% | 2 | 22 |  |
| 2026-06-04 | no | 2505 | 1945 | 2348 | 2348 | 7269 | 1640 | 69.8% | 0.0% | 13.7% | 2 | 10 | stale_close_rate>0.05 |
| 2026-06-03 | no | 3532 | 3089 | 3331 | 3331 | 3089 | 0 | 0.0% | 0.0% | 100.0% | 1 | 1 | valid_side_locks<100, valid_clv_coverage<0.25, stale_close_rate>0.05 |
| 2026-06-02 | yes | 4467 | 3104 | 235 | 235 | 6442 | 200 | 85.1% | 0.0% | 0.0% | 1 | 12 |  |
| 2026-06-01 | no | 2668 | 0 | 208 | 208 | 5851 | 183 | 88.0% | 0.0% | 9.6% | 1 | 13 | stale_close_rate>0.05 |
| 2026-05-31 | no | 4599 | 0 | 214 | 214 | 9800 | 179 | 83.6% | 0.0% | 15.0% | 1 | 20 | stale_close_rate>0.05 |
| 2026-05-30 | no | 3524 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-29 | no | 3124 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-28 | no | 1391 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-27 | no | 3481 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-26 | no | 3511 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-25 | no | 3051 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-24 | no | 5772 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-23 | no | 3507 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
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
