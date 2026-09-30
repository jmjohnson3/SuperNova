# SuperNovaBets MLB Daily Run (2026-06-25 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 9.7s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 5.0s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 5.5s)
- **Refresh prop replay CLV**: OK (rc=0, 8.6s)
- **Build prop market training table**: OK (rc=0, 49.0s)
- **Prop walk-forward accuracy report**: OK (rc=0, 233.2s)
- **Prop shadow selector report**: OK (rc=0, 136.4s)
- **Prop miss diagnostic report**: OK (rc=0, 156.8s)
- **Prop bucket repair report**: OK (rc=0, 19.7s)
- **TB prop repair report**: OK (rc=0, 15.7s)
- **Prop target quality report**: OK (rc=0, 19.6s)
- **Grade outcomes + ledgers**: OK (rc=0, 11.9s)
- **Prop snapshot coverage report**: OK (rc=0, 17.0s)
- **Grade shadow prop replay**: OK (rc=0, 5.8s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-25 21:49:04,034 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-26. Catching up from 2026-06-27 to 2026-06-24
2026-06-25 21:49:04,048 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-25 window=2026-06-25T04:00:00Z..2026-06-26T04:00:00Z
2026-06-25 21:49:04,899 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-25 | events=0 | credits_remaining=69222
2026-06-25 21:49:05,077 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-26 window=2026-06-26T04:00:00Z..2026-06-27T04:00:00Z
2026-06-25 21:49:05,698 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-26 | events=14 | credits_remaining=69220
2026-06-25 21:49:05,805 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=69220
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-25 21:49:09,226 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-26. Catching up from 2026-06-27 to 2026-06-24
2026-06-25 21:49:10,140 | INFO | mlb_pipeline.crawler_oddsapi | No events found for 2026-06-25 â€” skipping prop fetch
2026-06-25 21:49:10,693 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 14 events (game_date=2026-06-26, as_of=2026-06-25)
2026-06-25 21:49:10,867 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=unknown
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-25 21:49:15,751 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1473 rows into odds.mlb_game_lines (live odds).
2026-06-25 21:49:15,787 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-25
2026-06-25 21:49:15,795 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-25 21:49:15,795 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-26T03:19:15.795463+00:00.
2026-06-25 21:49:15,905 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-24
2026-06-25 21:49:15,920 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_prop_odds snapshots found (as_of_date=None).
2026-06-25 21:49:15,920 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 2285,
  "run_ids": "all",
  "date_from": "2026-06-25",
  "date_to": "2026-06-25",
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
  "replay_rows": 11101,
  "examples": 11101
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 126018,
  "graded_rows": 111446,
  "valid_clv_rows": 102678,
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
  "active_rows": 1064,
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
  "rows": 110620,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 110620,
  "bucket_count": 135,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=44918
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 125117,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 200-130 (60.6%) ROI: +15.7% | Total: 67-61 (52.3%) ROI: -0.1%
MLB CLV Run Line: beat close 2/39 (5%) avg CLV=+0.15 runs | CLV Total avg=+0.03 runs
MLB Price CLV Run Line: 37 bets  avg=+0.68%
```

**stderr (tail)**
```
2026-06-25 21:59:58,407 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 26 pending predictions.
2026-06-25 21:59:58,407 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-25 21:59:58,433 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-25 22:00:04,958 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-25 22:00:04,978 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-25 22:00:05,176 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-25 22:00:05,177 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-25 22:00:05,583 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-25 22:00:06,845 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-25 22:00:06,852 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-26T00:00:24-04:00
Range: 2026-06-13 to 2026-06-26

## Collection Status

**COLLECTING**

- Clean shadow slates: 9 / 10
- Additional clean slates needed: 1
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 25.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 5.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-26 | no | 0 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-06-25 | no | 2471 | 1656 | 2285 | 2285 | 10693 | 1810 | 79.2% | 0.0% | 9.4% | 2 | 10 | stale_close_rate>0.05 |
| 2026-06-24 | no | 4567 | 2646 | 4697 | 4697 | 21191 | 3644 | 77.6% | 0.0% | 5.2% | 2 | 11 | stale_close_rate>0.05 |
| 2026-06-23 | no | 4342 | 2721 | 4119 | 4119 | 28669 | 2350 | 57.1% | 0.0% | 5.5% | 2 | 14 | stale_close_rate>0.05 |
| 2026-06-22 | yes | 3807 | 2394 | 7540 | 7540 | 32994 | 6610 | 87.7% | 0.0% | 3.6% | 3 | 17 |  |
| 2026-06-21 | yes | 4514 | 2574 | 5500 | 5500 | 24573 | 4484 | 81.5% | 0.0% | 4.1% | 3 | 16 |  |
| 2026-06-20 | yes | 4153 | 2551 | 6337 | 6337 | 31532 | 5322 | 84.0% | 0.0% | 2.9% | 3 | 17 |  |
| 2026-06-19 | yes | 4181 | 2501 | 7929 | 7929 | 35937 | 7008 | 88.4% | 0.0% | 2.3% | 3 | 18 |  |
| 2026-06-18 | no | 2569 | 1592 | 2660 | 2660 | 14372 | 1994 | 75.0% | 0.0% | 12.2% | 2 | 15 | stale_close_rate>0.05 |
| 2026-06-17 | yes | 4267 | 380 | 3707 | 3707 | 31248 | 3328 | 89.8% | 0.0% | 3.4% | 2 | 18 |  |
| 2026-06-16 | yes | 4276 | 2600 | 8262 | 8262 | 38949 | 7125 | 86.2% | 0.0% | 1.7% | 3 | 17 |  |
| 2026-06-15 | yes | 2927 | 1891 | 5873 | 5873 | 28077 | 5221 | 88.9% | 0.0% | 2.6% | 3 | 18 |  |
| 2026-06-14 | yes | 4648 | 2741 | 6056 | 6056 | 24201 | 5004 | 82.6% | 0.0% | 3.2% | 3 | 16 |  |
| 2026-06-13 | yes | 4564 | 2780 | 6581 | 6581 | 37746 | 5656 | 85.9% | 0.0% | 1.0% | 3 | 19 |  |
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
