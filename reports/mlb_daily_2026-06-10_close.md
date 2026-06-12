# SuperNovaBets MLB Daily Run (2026-06-10 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 6.2s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 6.9s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 8.8s)
- **Refresh prop replay CLV**: OK (rc=0, 27.6s)
- **Build prop market training table**: OK (rc=0, 153.1s)
- **Prop walk-forward accuracy report**: OK (rc=0, 101.2s)
- **Prop shadow selector report**: OK (rc=0, 57.1s)
- **Prop miss diagnostic report**: OK (rc=0, 43.2s)
- **Prop target quality report**: OK (rc=0, 45.8s)
- **Grade outcomes + ledgers**: OK (rc=0, 26.0s)
- **Prop snapshot coverage report**: OK (rc=0, 16.0s)
- **Grade shadow prop replay**: OK (rc=0, 19.4s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-10 21:45:10,962 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-11. Catching up from 2026-06-12 to 2026-06-09
2026-06-10 21:45:10,962 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-10 window=2026-06-10T04:00:00Z..2026-06-11T04:00:00Z
2026-06-10 21:45:11,690 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-10 | events=2 | credits_remaining=85841
2026-06-10 21:45:11,981 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-11 window=2026-06-11T04:00:00Z..2026-06-12T04:00:00Z
2026-06-10 21:45:12,753 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-11 | events=8 | credits_remaining=85839
2026-06-10 21:45:12,818 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=85839
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-10 21:45:15,370 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-11. Catching up from 2026-06-12 to 2026-06-09
2026-06-10 21:45:15,987 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 2 events (game_date=2026-06-10, as_of=2026-06-10)
2026-06-10 21:45:16,818 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-10 event=715a197e2d41a434a760c22ec0a823c5 (Milwaukee Brewers@Athletics) | credits=85836
2026-06-10 21:45:18,144 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-10 event=325a216ed291897f85ab8c9b49ebd33d (Houston Astros@Los Angeles Angels) | credits=85833
2026-06-10 21:45:19,238 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 8 events (game_date=2026-06-11, as_of=2026-06-10)
2026-06-10 21:45:19,779 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=85833
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-10 21:45:27,033 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1396 rows into odds.mlb_game_lines (live odds).
2026-06-10 21:45:27,050 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-10
2026-06-10 21:45:27,051 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-10 21:45:27,051 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-11T03:15:27.051839+00:00.
2026-06-10 21:45:27,181 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-09
2026-06-10 21:45:28,552 | INFO | mlb_pipeline.parse_oddsapi | Upserted 32 rows into odds.mlb_player_prop_lines.
2026-06-10 21:45:28,552 | INFO | mlb_pipeline.parse_oddsapi | Processed 32 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-10 21:45:28,552 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 8414,
  "run_ids": "all",
  "date_from": "2026-06-10",
  "date_to": "2026-06-10",
  "include_graded": true,
  "only_missing": false
}
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 42687,
  "replay_rows": 42687,
  "examples": 42687
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 43588,
  "graded_rows": 31912,
  "valid_clv_rows": 33666,
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
  "active_rows": 2536,
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
  "rows": 31086,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 42687,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 187-125 (59.9%) ROI: +14.4% | Total: 59-55 (51.8%) ROI: -1.2%
MLB CLV Run Line: beat close 2/21 (10%) avg CLV=+0.29 runs | CLV Total avg=-0.03 runs
MLB Price CLV Run Line: 19 bets  avg=+0.72%
```

**stderr (tail)**
```
2026-06-10 21:53:00,202 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 31 pending predictions.
2026-06-10 21:53:00,202 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-10 21:53:00,272 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-10 21:53:01,469 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-10 21:53:01,486 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-10 21:53:01,565 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-10 21:53:01,565 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-10 21:53:01,749 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-10 21:53:02,302 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-10 21:53:02,306 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-10T23:53:18-04:00
Range: 2026-05-28 to 2026-06-10

## Collection Status

**COLLECTING**

- Clean shadow slates: 7 / 10
- Additional clean slates needed: 3
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 25.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 5.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-10 | yes | 4550 | 3317 | 8414 | 8414 | 40087 | 7198 | 85.5% | 0.0% | 4.5% | 3 | 18 |  |
| 2026-06-09 | yes | 4402 | 3237 | 6566 | 6566 | 40669 | 5702 | 86.8% | 0.0% | 3.5% | 2 | 16 |  |
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
