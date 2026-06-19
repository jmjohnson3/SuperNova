# SuperNovaBets MLB Daily Run (2026-06-18 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 10.9s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 9.5s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 6.0s)
- **Refresh prop replay CLV**: OK (rc=0, 13.9s)
- **Build prop market training table**: OK (rc=0, 83.9s)
- **Prop walk-forward accuracy report**: OK (rc=0, 260.1s)
- **Prop shadow selector report**: OK (rc=0, 110.2s)
- **Prop miss diagnostic report**: OK (rc=0, 81.9s)
- **Prop bucket repair report**: OK (rc=0, 25.4s)
- **TB prop repair report**: OK (rc=0, 30.3s)
- **Prop target quality report**: OK (rc=0, 32.8s)
- **Grade outcomes + ledgers**: OK (rc=0, 25.6s)
- **Prop snapshot coverage report**: OK (rc=0, 23.1s)
- **Grade shadow prop replay**: OK (rc=0, 11.9s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-18 21:45:14,478 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-19. Catching up from 2026-06-20 to 2026-06-17
2026-06-18 21:45:14,478 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-18 window=2026-06-18T04:00:00Z..2026-06-19T04:00:00Z
2026-06-18 21:45:16,375 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-18 | events=1 | credits_remaining=76687
2026-06-18 21:45:16,641 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-19 window=2026-06-19T04:00:00Z..2026-06-20T04:00:00Z
2026-06-18 21:45:17,770 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-19 | events=13 | credits_remaining=76685
2026-06-18 21:45:17,864 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=76685
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-18 21:45:21,127 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-19. Catching up from 2026-06-20 to 2026-06-17
2026-06-18 21:45:22,862 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 1 events (game_date=2026-06-18, as_of=2026-06-18)
2026-06-18 21:45:23,481 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-18 event=4d7345756b0575f6a757888c78a290f2 (Los Angeles Angels@Athletics) | credits=76682
2026-06-18 21:45:25,286 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 13 events (game_date=2026-06-19, as_of=2026-06-18)
2026-06-18 21:45:27,033 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-18 event=a8bc26eb19d767d6433749f35c2224ee (Los Angeles Angels@Athletics) | credits=76682
2026-06-18 21:45:27,347 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=76682
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-18 21:45:31,914 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1441 rows into odds.mlb_game_lines (live odds).
2026-06-18 21:45:31,946 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-18
2026-06-18 21:45:31,946 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-18 21:45:31,946 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-19T03:15:31.946003+00:00.
2026-06-18 21:45:32,133 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-17
2026-06-18 21:45:33,321 | INFO | mlb_pipeline.parse_oddsapi | Upserted 51 rows into odds.mlb_player_prop_lines.
2026-06-18 21:45:33,321 | INFO | mlb_pipeline.parse_oddsapi | Processed 51 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-18 21:45:33,321 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 2660,
  "run_ids": "all",
  "date_from": "2026-06-18",
  "date_to": "2026-06-18",
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
  "replay_rows": 14629,
  "examples": 14629
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 87611,
  "graded_rows": 76983,
  "valid_clv_rows": 71547,
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
  "active_rows": 1044,
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
  "rows": 76157,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 76157,
  "bucket_count": 133,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=30634
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 86710,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 193-129 (59.9%) ROI: +14.4% | Total: 62-58 (51.7%) ROI: -1.4%
MLB CLV Run Line: beat close 2/31 (6%) avg CLV=+0.19 runs | CLV Total avg=+0.02 runs
MLB Price CLV Run Line: 29 bets  avg=+0.58%
```

**stderr (tail)**
```
2026-06-18 21:56:31,921 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 27 pending predictions.
2026-06-18 21:56:31,921 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-18 21:56:32,124 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-18 21:56:35,206 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-18 21:56:35,222 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-18 21:56:35,394 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-18 21:56:35,409 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-18 21:56:35,628 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-18 21:56:36,947 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-18 21:56:36,947 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-18T23:56:59-04:00
Range: 2026-06-05 to 2026-06-18

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
| 2026-06-18 | no | 2288 | 1545 | 2660 | 2660 | 14372 | 1994 | 75.0% | 0.0% | 12.2% | 2 | 15 | stale_close_rate>0.05 |
| 2026-06-17 | yes | 4267 | 380 | 3707 | 3707 | 31248 | 3328 | 89.8% | 0.0% | 3.4% | 2 | 18 |  |
| 2026-06-16 | yes | 4276 | 2600 | 8262 | 8262 | 38949 | 7125 | 86.2% | 0.0% | 1.7% | 3 | 17 |  |
| 2026-06-15 | yes | 2927 | 1891 | 5873 | 5873 | 28077 | 5221 | 88.9% | 0.0% | 2.6% | 3 | 18 |  |
| 2026-06-14 | yes | 4648 | 2741 | 6056 | 6056 | 24201 | 5004 | 82.6% | 0.0% | 3.4% | 3 | 16 |  |
| 2026-06-13 | yes | 4564 | 2780 | 6581 | 6581 | 37746 | 5656 | 85.9% | 0.0% | 1.2% | 3 | 19 |  |
| 2026-06-12 | yes | 4438 | 2804 | 8682 | 8682 | 40945 | 7706 | 88.8% | 0.0% | 2.5% | 3 | 17 |  |
| 2026-06-11 | no | 2452 | 1703 | 2202 | 2202 | 14587 | 1879 | 85.3% | 0.0% | 6.7% | 2 | 15 | stale_close_rate>0.05 |
| 2026-06-10 | yes | 4550 | 3350 | 8414 | 8414 | 40087 | 7198 | 85.5% | 0.0% | 4.5% | 3 | 18 |  |
| 2026-06-09 | yes | 4402 | 3237 | 6566 | 6566 | 40669 | 5702 | 86.8% | 0.0% | 3.5% | 2 | 16 |  |
| 2026-06-08 | yes | 2447 | 1705 | 3589 | 3589 | 27069 | 3093 | 86.2% | 0.0% | 3.1% | 2 | 19 |  |
| 2026-06-07 | yes | 4496 | 3136 | 6521 | 6521 | 30276 | 5638 | 86.5% | 0.0% | 1.0% | 3 | 19 |  |
| 2026-06-06 | yes | 4345 | 3180 | 4562 | 4562 | 38898 | 4017 | 88.1% | 0.0% | 2.2% | 2 | 19 |  |
| 2026-06-05 | yes | 4520 | 3154 | 6699 | 6699 | 57725 | 5816 | 86.8% | 0.0% | 4.2% | 2 | 22 |  |
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
