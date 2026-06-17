# SuperNovaBets MLB Daily Run (2026-06-12 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 6.3s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 8.4s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 8.7s)
- **Refresh prop replay CLV**: OK (rc=0, 20.7s)
- **Build prop market training table**: OK (rc=0, 146.3s)
- **Prop walk-forward accuracy report**: OK (rc=0, 98.7s)
- **Prop shadow selector report**: OK (rc=0, 70.9s)
- **Prop miss diagnostic report**: OK (rc=0, 31.6s)
- **Prop bucket repair report**: OK (rc=0, 5.8s)
- **TB prop repair report**: OK (rc=0, 10.5s)
- **Prop target quality report**: OK (rc=0, 11.2s)
- **Grade outcomes + ledgers**: OK (rc=0, 12.7s)
- **Prop snapshot coverage report**: OK (rc=0, 17.1s)
- **Grade shadow prop replay**: OK (rc=0, 4.5s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-12 21:45:09,372 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-13. Catching up from 2026-06-14 to 2026-06-11
2026-06-12 21:45:09,372 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-12 window=2026-06-12T04:00:00Z..2026-06-13T04:00:00Z
2026-06-12 21:45:10,267 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-12 | events=4 | credits_remaining=83613
2026-06-12 21:45:10,554 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-13 window=2026-06-13T04:00:00Z..2026-06-14T04:00:00Z
2026-06-12 21:45:11,183 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-13 | events=15 | credits_remaining=83611
2026-06-12 21:45:11,197 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=83611
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-12 21:45:13,697 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-13. Catching up from 2026-06-14 to 2026-06-11
2026-06-12 21:45:14,295 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 4 events (game_date=2026-06-12, as_of=2026-06-12)
2026-06-12 21:45:14,905 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-12 event=3f2c1cb7fb42f7b00783f896040bb847 (Atlanta Braves@New York Mets) | credits=83609
2026-06-12 21:45:15,875 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-12 event=1972be0f278db4fe870972f2d97254c2 (Tampa Bay Rays@Los Angeles Angels) | credits=83606
2026-06-12 21:45:16,818 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-12 event=6cf3e8040acf1a1d7d7e7f5b1c0ce4e8 (Colorado Rockies@Athletics) | credits=83600
2026-06-12 21:45:17,776 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-12 event=f22b44b04b70c9956c01233a8b94b5cd (Chicago Cubs@San Francisco Giants) | credits=83594
2026-06-12 21:45:19,038 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-06-13, as_of=2026-06-12)
2026-06-12 21:45:19,658 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=83594
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-12 21:45:23,674 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1416 rows into odds.mlb_game_lines (live odds).
2026-06-12 21:45:23,741 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-12
2026-06-12 21:45:23,742 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-12 21:45:23,742 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-13T03:15:23.742569+00:00.
2026-06-12 21:45:23,858 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-11
2026-06-12 21:45:28,343 | INFO | mlb_pipeline.parse_oddsapi | Upserted 322 rows into odds.mlb_player_prop_lines.
2026-06-12 21:45:28,343 | INFO | mlb_pipeline.parse_oddsapi | Processed 322 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-12 21:45:28,344 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 8682,
  "run_ids": "all",
  "date_from": "2026-06-12",
  "date_to": "2026-06-12",
  "include_graded": true,
  "only_missing": false
}
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 53571,
  "replay_rows": 53571,
  "examples": 53571
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 54472,
  "graded_rows": 41206,
  "valid_clv_rows": 43251,
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
  "active_rows": 3176,
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
  "rows": 40380,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 40380,
  "bucket_count": 123,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=15270
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 53571,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 188-125 (60.1%) ROI: +14.7% | Total: 61-55 (52.6%) ROI: +0.4%
MLB CLV Run Line: beat close 2/22 (9%) avg CLV=+0.27 runs | CLV Total avg=-0.03 runs
MLB Price CLV Run Line: 20 bets  avg=+0.66%
```

**stderr (tail)**
```
2026-06-12 21:52:12,970 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 32 pending predictions.
2026-06-12 21:52:12,970 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-12 21:52:12,992 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-12 21:52:15,529 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-12 21:52:15,547 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-12 21:52:15,656 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-12 21:52:15,658 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-12 21:52:15,894 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-12 21:52:16,647 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-12 21:52:16,656 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-12T23:52:33-04:00
Range: 2026-05-30 to 2026-06-12

## Collection Status

**COLLECTING**

- Clean shadow slates: 8 / 10
- Additional clean slates needed: 2
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 25.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 5.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-12 | yes | 3623 | 2620 | 8682 | 8682 | 40945 | 7706 | 88.8% | 0.0% | 2.5% | 3 | 17 |  |
| 2026-06-11 | no | 2452 | 1703 | 2202 | 2202 | 14587 | 1879 | 85.3% | 0.0% | 6.7% | 2 | 15 | stale_close_rate>0.05 |
| 2026-06-10 | yes | 4550 | 3350 | 8414 | 8414 | 40087 | 7198 | 85.5% | 0.0% | 4.5% | 3 | 18 |  |
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
