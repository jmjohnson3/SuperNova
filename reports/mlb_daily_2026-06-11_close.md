# SuperNovaBets MLB Daily Run (2026-06-11 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 5.5s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 3.8s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 7.5s)
- **Refresh prop replay CLV**: OK (rc=0, 9.6s)
- **Build prop market training table**: OK (rc=0, 133.4s)
- **Prop walk-forward accuracy report**: OK (rc=0, 93.5s)
- **Prop shadow selector report**: OK (rc=0, 15.6s)
- **Prop miss diagnostic report**: OK (rc=0, 34.5s)
- **Prop bucket repair report**: OK (rc=0, 5.9s)
- **TB prop repair report**: OK (rc=0, 10.2s)
- **Prop target quality report**: OK (rc=0, 10.2s)
- **Grade outcomes + ledgers**: OK (rc=0, 9.3s)
- **Prop snapshot coverage report**: OK (rc=0, 27.3s)
- **Grade shadow prop replay**: OK (rc=0, 4.9s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-11 21:45:07,471 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-12. Catching up from 2026-06-13 to 2026-06-10
2026-06-11 21:45:07,471 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-11 window=2026-06-11T04:00:00Z..2026-06-12T04:00:00Z
2026-06-11 21:45:08,312 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-11 | events=0 | credits_remaining=85194
2026-06-11 21:45:08,620 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-12 window=2026-06-12T04:00:00Z..2026-06-13T04:00:00Z
2026-06-11 21:45:10,091 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-12 | events=15 | credits_remaining=85192
2026-06-11 21:45:10,166 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=85192
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-11 21:45:12,597 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-12. Catching up from 2026-06-13 to 2026-06-10
2026-06-11 21:45:13,322 | INFO | mlb_pipeline.crawler_oddsapi | No events found for 2026-06-11 â€” skipping prop fetch
2026-06-11 21:45:13,935 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-06-12, as_of=2026-06-11)
2026-06-11 21:45:14,036 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=unknown
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-11 21:45:21,400 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1410 rows into odds.mlb_game_lines (live odds).
2026-06-11 21:45:21,427 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-11
2026-06-11 21:45:21,428 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-11 21:45:21,428 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-12T03:15:21.428238+00:00.
2026-06-11 21:45:21,547 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-10
2026-06-11 21:45:21,548 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_prop_odds snapshots found (as_of_date=None).
2026-06-11 21:45:21,548 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 2202,
  "run_ids": "all",
  "date_from": "2026-06-11",
  "date_to": "2026-06-11",
  "include_graded": true,
  "only_missing": false
}
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 44889,
  "replay_rows": 44889,
  "examples": 44889
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 45790,
  "graded_rows": 39536,
  "valid_clv_rows": 35545,
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
  "active_rows": 622,
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
  "rows": 38710,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 38710,
  "bucket_count": 123,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=14570
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 44889,
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
2026-06-11 21:50:41,120 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 24 pending predictions.
2026-06-11 21:50:41,120 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-11 21:50:41,145 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-11 21:50:42,605 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-11 21:50:42,621 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-11 21:50:42,720 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-11 21:50:42,940 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-11 21:50:43,634 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-11 21:50:43,637 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-11T23:51:10-04:00
Range: 2026-05-29 to 2026-06-11

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
| 2026-06-11 | no | 2179 | 1623 | 2202 | 2202 | 14587 | 1879 | 85.3% | 0.0% | 6.7% | 2 | 15 | stale_close_rate>0.05 |
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
| 2026-05-29 | no | 3124 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
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
