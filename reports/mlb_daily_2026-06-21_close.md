# SuperNovaBets MLB Daily Run (2026-06-21 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 6.6s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 3.8s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 7.0s)
- **Refresh prop replay CLV**: OK (rc=0, 14.2s)
- **Build prop market training table**: OK (rc=0, 80.4s)
- **Prop walk-forward accuracy report**: OK (rc=0, 197.9s)
- **Prop shadow selector report**: OK (rc=0, 27.8s)
- **Prop miss diagnostic report**: OK (rc=0, 89.3s)
- **Prop bucket repair report**: OK (rc=0, 24.6s)
- **TB prop repair report**: OK (rc=0, 18.6s)
- **Prop target quality report**: OK (rc=0, 27.6s)
- **Grade outcomes + ledgers**: OK (rc=0, 11.2s)
- **Prop snapshot coverage report**: OK (rc=0, 25.0s)
- **Grade shadow prop replay**: OK (rc=0, 53.4s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-21 21:46:10,347 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-22. Catching up from 2026-06-23 to 2026-06-20
2026-06-21 21:46:10,347 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-21 window=2026-06-21T04:00:00Z..2026-06-22T04:00:00Z
2026-06-21 21:46:11,107 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-21 | events=0 | credits_remaining=73110
2026-06-21 21:46:11,265 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-22 window=2026-06-22T04:00:00Z..2026-06-23T04:00:00Z
2026-06-21 21:46:11,915 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-22 | events=12 | credits_remaining=73108
2026-06-21 21:46:12,023 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=73108
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-21 21:46:14,474 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-22. Catching up from 2026-06-23 to 2026-06-20
2026-06-21 21:46:15,096 | INFO | mlb_pipeline.crawler_oddsapi | No events found for 2026-06-21 â€” skipping prop fetch
2026-06-21 21:46:15,745 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 12 events (game_date=2026-06-22, as_of=2026-06-21)
2026-06-21 21:46:15,833 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=unknown
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-21 21:46:22,671 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1458 rows into odds.mlb_game_lines (live odds).
2026-06-21 21:46:22,692 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-21
2026-06-21 21:46:22,693 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-21 21:46:22,693 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-22T03:16:22.693352+00:00.
2026-06-21 21:46:22,776 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-20
2026-06-21 21:46:22,792 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_prop_odds snapshots found (as_of_date=None).
2026-06-21 21:46:22,792 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 5500,
  "run_ids": "all",
  "date_from": "2026-06-21",
  "date_to": "2026-06-21",
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
  "replay_rows": 19766,
  "examples": 19766
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 107377,
  "graded_rows": 92143,
  "valid_clv_rows": 88309,
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
  "active_rows": 243,
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
  "rows": 91317,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 91317,
  "bucket_count": 134,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=36904
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 106476,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 195-129 (60.2%) ROI: +14.9% | Total: 66-59 (52.8%) ROI: +0.8%
MLB CLV Run Line: beat close 2/33 (6%) avg CLV=+0.18 runs | CLV Total avg=+0.03 runs
MLB Price CLV Run Line: 31 bets  avg=+0.58%
```

**stderr (tail)**
```
2026-06-21 21:54:30,722 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 34 pending predictions.
2026-06-21 21:54:30,722 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-21 21:54:30,804 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-21 21:54:33,616 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-21 21:54:33,635 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-21 21:54:33,772 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-21 21:54:33,774 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-21 21:54:33,964 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-21 21:54:34,333 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-21 21:54:34,340 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-21T23:54:59-04:00
Range: 2026-06-08 to 2026-06-21

## Collection Status

**TARGET MET**

- Clean shadow slates: 12 / 10
- Additional clean slates needed: 0
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 25.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 5.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-21 | yes | 4399 | 2394 | 5500 | 5500 | 24573 | 4484 | 81.5% | 0.0% | 4.1% | 3 | 16 |  |
| 2026-06-20 | yes | 4153 | 2551 | 6337 | 6337 | 31532 | 5322 | 84.0% | 0.0% | 2.9% | 3 | 17 |  |
| 2026-06-19 | yes | 4181 | 2501 | 7929 | 7929 | 35937 | 7008 | 88.4% | 0.0% | 2.3% | 3 | 18 |  |
| 2026-06-18 | no | 2569 | 1592 | 2660 | 2660 | 14372 | 1994 | 75.0% | 0.0% | 12.2% | 2 | 15 | stale_close_rate>0.05 |
| 2026-06-17 | yes | 4267 | 380 | 3707 | 3707 | 31248 | 3328 | 89.8% | 0.0% | 3.4% | 2 | 18 |  |
| 2026-06-16 | yes | 4276 | 2600 | 8262 | 8262 | 38949 | 7125 | 86.2% | 0.0% | 1.7% | 3 | 17 |  |
| 2026-06-15 | yes | 2927 | 1891 | 5873 | 5873 | 28077 | 5221 | 88.9% | 0.0% | 2.6% | 3 | 18 |  |
| 2026-06-14 | yes | 4648 | 2741 | 6056 | 6056 | 24201 | 5004 | 82.6% | 0.0% | 3.2% | 3 | 16 |  |
| 2026-06-13 | yes | 4564 | 2780 | 6581 | 6581 | 37746 | 5656 | 85.9% | 0.0% | 1.0% | 3 | 19 |  |
| 2026-06-12 | yes | 4438 | 2804 | 8682 | 8682 | 40945 | 7706 | 88.8% | 0.0% | 2.5% | 3 | 17 |  |
| 2026-06-11 | no | 2452 | 1703 | 2202 | 2202 | 14587 | 1879 | 85.3% | 0.0% | 6.4% | 2 | 15 | stale_close_rate>0.05 |
| 2026-06-10 | yes | 4550 | 3350 | 8414 | 8414 | 40087 | 7198 | 85.5% | 0.0% | 4.4% | 3 | 18 |  |
| 2026-06-09 | yes | 4402 | 3237 | 6566 | 6566 | 40669 | 5702 | 86.8% | 0.0% | 3.2% | 2 | 16 |  |
| 2026-06-08 | yes | 2447 | 1705 | 3589 | 3589 | 27069 | 3093 | 86.2% | 0.0% | 2.8% | 2 | 19 |  |
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
