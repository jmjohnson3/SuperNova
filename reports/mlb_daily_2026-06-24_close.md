# SuperNovaBets MLB Daily Run (2026-06-24 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 10.9s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 7.7s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 9.6s)
- **Refresh prop replay CLV**: OK (rc=0, 16.7s)
- **Build prop market training table**: OK (rc=0, 79.1s)
- **Prop walk-forward accuracy report**: OK (rc=0, 447.3s)
- **Prop shadow selector report**: FAIL (rc=124, 300.8s)
- **Prop miss diagnostic report**: OK (rc=0, 120.0s)
- **Prop bucket repair report**: OK (rc=0, 39.7s)
- **TB prop repair report**: OK (rc=0, 69.3s)
- **Prop target quality report**: OK (rc=0, 58.2s)
- **Grade outcomes + ledgers**: OK (rc=0, 15.6s)
- **Prop snapshot coverage report**: OK (rc=0, 26.7s)
- **Grade shadow prop replay**: OK (rc=0, 6.9s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-24 21:49:55,273 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-25. Catching up from 2026-06-26 to 2026-06-23
2026-06-24 21:49:55,273 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-24 window=2026-06-24T04:00:00Z..2026-06-25T04:00:00Z
2026-06-24 21:49:56,189 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-24 | events=1 | credits_remaining=69767
2026-06-24 21:49:56,864 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-25 window=2026-06-25T04:00:00Z..2026-06-26T04:00:00Z
2026-06-24 21:49:57,506 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-25 | events=9 | credits_remaining=69765
2026-06-24 21:49:57,577 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=69765
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-24 21:50:00,095 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-25. Catching up from 2026-06-26 to 2026-06-23
2026-06-24 21:50:00,777 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 1 events (game_date=2026-06-24, as_of=2026-06-24)
2026-06-24 21:50:01,401 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=74b391c5a12588e6bdf6f98be072b920 (Athletics@San Francisco Giants) | credits=69762
2026-06-24 21:50:02,420 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 9 events (game_date=2026-06-25, as_of=2026-06-24)
2026-06-24 21:50:05,285 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=69762
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-24 21:50:12,939 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1464 rows into odds.mlb_game_lines (live odds).
2026-06-24 21:50:13,009 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-24
2026-06-24 21:50:13,010 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-24 21:50:13,010 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-25T03:20:13.010498+00:00.
2026-06-24 21:50:13,417 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-23
2026-06-24 21:50:14,867 | INFO | mlb_pipeline.parse_oddsapi | Upserted 31 rows into odds.mlb_player_prop_lines.
2026-06-24 21:50:14,867 | INFO | mlb_pipeline.parse_oddsapi | Processed 31 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-24 21:50:14,867 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 4697,
  "run_ids": "all",
  "date_from": "2026-06-24",
  "date_to": "2026-06-24",
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
  "replay_rows": 16356,
  "examples": 16356
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 123733,
  "graded_rows": 107159,
  "valid_clv_rows": 108582,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_walk_forward_accuracy_latest.md",
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_walk_forward_accuracy_report.json"
}
```

### Prop shadow selector report

- rc: 124

**stderr (tail)**
```
Timed out after 300s; killed process tree rooted at PID 15608
```

### Prop miss diagnostic report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 106333,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 106333,
  "bucket_count": 135,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=43148
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 122832,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 197-129 (60.4%) ROI: +15.4% | Total: 66-61 (52.0%) ROI: -0.8%
MLB CLV Run Line: beat close 2/35 (6%) avg CLV=+0.17 runs | CLV Total avg=+0.03 runs
MLB Price CLV Run Line: 33 bets  avg=+0.63%
```

**stderr (tail)**
```
2026-06-24 22:09:16,683 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 37 pending predictions.
2026-06-24 22:09:16,683 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-24 22:09:16,857 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-24 22:09:20,044 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-24 22:09:20,064 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-24 22:09:20,287 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-24 22:09:20,288 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-24 22:09:20,533 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-24 22:09:21,454 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-24 22:09:21,463 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-25T00:09:47-04:00
Range: 2026-06-12 to 2026-06-25

## Collection Status

**TARGET MET**

- Clean shadow slates: 10 / 10
- Additional clean slates needed: 0
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 25.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 5.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-25 | no | 0 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-06-24 | no | 3971 | 2418 | 4697 | 4697 | 21191 | 3644 | 77.6% | 0.0% | 5.2% | 2 | 11 | stale_close_rate>0.05 |
| 2026-06-23 | no | 4342 | 2721 | 4119 | 4119 | 28669 | 2350 | 57.1% | 0.0% | 5.4% | 2 | 14 | stale_close_rate>0.05 |
| 2026-06-22 | yes | 3807 | 2394 | 7540 | 7540 | 32994 | 6610 | 87.7% | 0.0% | 3.5% | 3 | 17 |  |
| 2026-06-21 | yes | 4514 | 2574 | 5500 | 5500 | 24573 | 4484 | 81.5% | 0.0% | 3.9% | 3 | 16 |  |
| 2026-06-20 | yes | 4153 | 2551 | 6337 | 6337 | 31532 | 5322 | 84.0% | 0.0% | 2.8% | 3 | 17 |  |
| 2026-06-19 | yes | 4181 | 2501 | 7929 | 7929 | 35937 | 7008 | 88.4% | 0.0% | 2.3% | 3 | 18 |  |
| 2026-06-18 | no | 2569 | 1592 | 2660 | 2660 | 14372 | 1994 | 75.0% | 0.0% | 12.2% | 2 | 15 | stale_close_rate>0.05 |
| 2026-06-17 | yes | 4267 | 380 | 3707 | 3707 | 31248 | 3328 | 89.8% | 0.0% | 3.4% | 2 | 18 |  |
| 2026-06-16 | yes | 4276 | 2600 | 8262 | 8262 | 38949 | 7125 | 86.2% | 0.0% | 1.7% | 3 | 17 |  |
| 2026-06-15 | yes | 2927 | 1891 | 5873 | 5873 | 28077 | 5221 | 88.9% | 0.0% | 2.6% | 3 | 18 |  |
| 2026-06-14 | yes | 4648 | 2741 | 6056 | 6056 | 24201 | 5004 | 82.6% | 0.0% | 3.1% | 3 | 16 |  |
| 2026-06-13 | yes | 4564 | 2780 | 6581 | 6581 | 37746 | 5656 | 85.9% | 0.0% | 1.0% | 3 | 19 |  |
| 2026-06-12 | yes | 4438 | 2804 | 8682 | 8682 | 40945 | 7706 | 88.8% | 0.0% | 2.5% | 3 | 17 |  |
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
