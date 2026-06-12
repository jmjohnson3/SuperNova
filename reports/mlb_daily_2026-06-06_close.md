# SuperNovaBets MLB Daily Run (2026-06-06 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 14.6s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 7.6s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 8.0s)
- **Refresh prop replay CLV**: OK (rc=0, 11.7s)
- **Build prop market training table**: OK (rc=0, 62.1s)
- **Prop walk-forward accuracy report**: OK (rc=0, 40.1s)
- **Prop miss diagnostic report**: OK (rc=0, 51.5s)
- **Prop target quality report**: OK (rc=0, 5.7s)
- **Grade outcomes + ledgers**: OK (rc=0, 6.3s)
- **Prop snapshot coverage report**: OK (rc=0, 5.2s)
- **Grade shadow prop replay**: OK (rc=0, 2.8s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-06 21:45:14,362 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-07. Catching up from 2026-06-08 to 2026-06-05
2026-06-06 21:45:14,366 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-06 window=2026-06-06T04:00:00Z..2026-06-07T04:00:00Z
2026-06-06 21:45:18,749 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-06 | events=3 | credits_remaining=91031
2026-06-06 21:45:19,063 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-07 window=2026-06-07T04:00:00Z..2026-06-08T04:00:00Z
2026-06-06 21:45:19,684 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-07 | events=15 | credits_remaining=91029
2026-06-06 21:45:19,713 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=91029
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-06 21:45:22,474 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-07. Catching up from 2026-06-08 to 2026-06-05
2026-06-06 21:45:23,085 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 3 events (game_date=2026-06-06, as_of=2026-06-06)
2026-06-06 21:45:23,692 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-06 event=86ccb607d9a44f660c9cf85eada0f55e (Milwaukee Brewers@Colorado Rockies) | credits=91026
2026-06-06 21:45:24,780 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-06 event=f931ee769aeeeeb55e2565cdf1fc908b (Los Angeles Angels@Los Angeles Dodgers) | credits=91020
2026-06-06 21:45:25,668 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-06 event=7603b440617339875384ac0c19a99a15 (New York Mets@San Diego Padres) | credits=91014
2026-06-06 21:45:26,841 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-06-07, as_of=2026-06-06)
2026-06-06 21:45:27,380 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=91014
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-06 21:45:33,402 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1386 rows into odds.mlb_game_lines (live odds).
2026-06-06 21:45:33,416 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-06
2026-06-06 21:45:33,417 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-06 21:45:33,417 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-07T03:15:33.417356+00:00.
2026-06-06 21:45:33,548 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-05
2026-06-06 21:45:35,370 | INFO | mlb_pipeline.parse_oddsapi | Upserted 303 rows into odds.mlb_player_prop_lines.
2026-06-06 21:45:35,370 | INFO | mlb_pipeline.parse_oddsapi | Processed 303 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-06 21:45:35,370 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 4562,
  "run_ids": "all",
  "date_from": "2026-06-06",
  "date_to": "2026-06-06",
  "include_graded": true,
  "only_missing": false
}
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 17597,
  "replay_rows": 17597,
  "examples": 17597
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 18498,
  "graded_rows": 12911,
  "valid_clv_rows": 12035,
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
  "rows": 12085,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 17597,
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
2026-06-06 21:48:30,855 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 30 pending predictions.
2026-06-06 21:48:30,855 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-06 21:48:30,867 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-06 21:48:32,282 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-06 21:48:32,307 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-06 21:48:32,425 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-06 21:48:32,425 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-06 21:48:32,517 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-06 21:48:32,621 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-06 21:48:32,623 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-06T23:48:37-04:00
Range: 2026-05-24 to 2026-06-06

## Collection Status

**COLLECTING**

- Clean shadow slates: 3 / 10
- Additional clean slates needed: 7
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 25.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 5.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-06 | yes | 4044 | 3086 | 4562 | 4562 | 38898 | 4017 | 88.1% | 0.0% | 2.2% | 2 | 19 |  |
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
| 2026-05-26 | no | 3511 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-25 | no | 3051 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-05-24 | no | 5772 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
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
