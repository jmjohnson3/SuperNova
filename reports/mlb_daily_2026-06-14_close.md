# SuperNovaBets MLB Daily Run (2026-06-14 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 7.1s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 28.7s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 6.3s)
- **Refresh prop replay CLV**: OK (rc=0, 15.2s)
- **Build prop market training table**: OK (rc=0, 228.6s)
- **Prop walk-forward accuracy report**: OK (rc=0, 123.6s)
- **Prop shadow selector report**: OK (rc=0, 15.3s)
- **Prop miss diagnostic report**: OK (rc=0, 80.8s)
- **Prop bucket repair report**: OK (rc=0, 8.6s)
- **TB prop repair report**: OK (rc=0, 7.8s)
- **Prop target quality report**: OK (rc=0, 13.9s)
- **Grade outcomes + ledgers**: OK (rc=0, 6.8s)
- **Prop snapshot coverage report**: OK (rc=0, 14.0s)
- **Grade shadow prop replay**: OK (rc=0, 5.1s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-14 21:45:10,839 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-15. Catching up from 2026-06-16 to 2026-06-13
2026-06-14 21:45:10,844 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-14 window=2026-06-14T04:00:00Z..2026-06-15T04:00:00Z
2026-06-14 21:45:11,728 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-14 | events=0 | credits_remaining=81126
2026-06-14 21:45:11,929 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-15 window=2026-06-15T04:00:00Z..2026-06-16T04:00:00Z
2026-06-14 21:45:12,555 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-15 | events=8 | credits_remaining=81124
2026-06-14 21:45:12,644 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=81124
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-14 21:45:16,495 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-15. Catching up from 2026-06-16 to 2026-06-13
2026-06-14 21:45:25,351 | INFO | mlb_pipeline.crawler_oddsapi | No events found for 2026-06-14 â€” skipping prop fetch
2026-06-14 21:45:40,566 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 8 events (game_date=2026-06-15, as_of=2026-06-14)
2026-06-14 21:45:41,139 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=unknown
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-14 21:45:47,586 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1412 rows into odds.mlb_game_lines (live odds).
2026-06-14 21:45:47,602 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-14
2026-06-14 21:45:47,607 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-14 21:45:47,607 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-15T03:15:47.607550+00:00.
2026-06-14 21:45:47,749 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-13
2026-06-14 21:45:47,765 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_prop_odds snapshots found (as_of_date=None).
2026-06-14 21:45:47,765 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 6056,
  "run_ids": "all",
  "date_from": "2026-06-14",
  "date_to": "2026-06-14",
  "include_graded": true,
  "only_missing": false
}
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 65994,
  "replay_rows": 65994,
  "examples": 65994
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 67109,
  "graded_rows": 55231,
  "valid_clv_rows": 53911,
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
  "active_rows": 250,
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
  "rows": 54405,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 54405,
  "bucket_count": 131,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=21286
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 66208,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 190-126 (60.1%) ROI: +14.8% | Total: 61-55 (52.6%) ROI: +0.4%
MLB CLV Run Line: beat close 2/25 (8%) avg CLV=+0.24 runs | CLV Total avg=-0.03 runs
MLB Price CLV Run Line: 23 bets  avg=+0.64%
```

**stderr (tail)**
```
2026-06-14 21:54:05,317 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 32 pending predictions.
2026-06-14 21:54:05,318 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-14 21:54:05,441 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-14 21:54:07,534 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-14 21:54:07,551 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-14 21:54:07,700 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-14 21:54:07,708 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-14 21:54:07,826 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-14 21:54:08,231 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-14 21:54:08,236 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-14T23:54:22-04:00
Range: 2026-06-01 to 2026-06-14

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
| 2026-06-14 | yes | 4567 | 2621 | 6056 | 6056 | 24201 | 5004 | 82.6% | 0.0% | 3.4% | 3 | 16 |  |
| 2026-06-13 | yes | 4564 | 2780 | 6581 | 6581 | 37746 | 5656 | 85.9% | 0.0% | 1.2% | 3 | 19 |  |
| 2026-06-12 | yes | 4438 | 2804 | 8682 | 8682 | 40945 | 7706 | 88.8% | 0.0% | 2.5% | 3 | 17 |  |
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
