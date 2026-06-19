# SuperNovaBets MLB Daily Run (2026-06-19 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 10.7s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 26.4s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 16.9s)
- **Refresh prop replay CLV**: OK (rc=0, 3.4s)
- **Build prop market training table**: OK (rc=0, 87.6s)
- **Prop walk-forward accuracy report**: OK (rc=0, 267.3s)
- **Prop shadow selector report**: OK (rc=0, 5.4s)
- **Prop miss diagnostic report**: OK (rc=0, 103.4s)
- **Prop bucket repair report**: OK (rc=0, 33.8s)
- **TB prop repair report**: OK (rc=0, 36.7s)
- **Prop target quality report**: OK (rc=0, 40.3s)
- **Grade outcomes + ledgers**: OK (rc=0, 17.9s)
- **Prop snapshot coverage report**: OK (rc=0, 31.2s)
- **Grade shadow prop replay**: OK (rc=0, 12.0s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-18 22:45:19,912 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-19. Catching up from 2026-06-20 to 2026-06-18
2026-06-18 22:45:19,928 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-19 window=2026-06-19T04:00:00Z..2026-06-20T04:00:00Z
2026-06-18 22:45:21,615 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-19 | events=13 | credits_remaining=76680
2026-06-18 22:45:22,084 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-20 window=2026-06-20T04:00:00Z..2026-06-21T04:00:00Z
2026-06-18 22:45:24,056 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-20 | events=0 | credits_remaining=76680
2026-06-18 22:45:24,087 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=76680
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-18 22:45:28,191 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-20. Catching up from 2026-06-21 to 2026-06-18
2026-06-18 22:45:30,317 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 13 events (game_date=2026-06-19, as_of=2026-06-19)
2026-06-18 22:45:31,962 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=3530063d3d9cd6eb5811962deca7c6ed (Toronto Blue Jays@Chicago Cubs) | credits=76676
2026-06-18 22:45:32,966 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=2969ea2e73c33f58b97b82fe562cf617 (Chicago White Sox@Detroit Tigers) | credits=76670
2026-06-18 22:45:34,853 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=7ef2ffad2e845a48d3df860e1a91b43d (Cincinnati Reds@New York Yankees) | credits=76664
2026-06-18 22:45:36,799 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=bdebd82757453b3cdc93ee168dadf2bb (Washington Nationals@Tampa Bay Rays) | credits=76658
2026-06-18 22:45:37,861 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=2bbd4e284d32261d81b8ebd0263116c3 (Milwaukee Brewers@Atlanta Braves) | credits=76652
2026-06-18 22:45:39,632 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=a072cb316f0d887f1c0251b417b34317 (San Diego Padres@Texas Rangers) | credits=76646
2026-06-18 22:45:41,193 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=d06fe764bea53051487b94fc022806d1 (Cleveland Guardians@Houston Astros) | credits=76641
2026-06-18 22:45:42,096 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=828d9387e45dac20bd4336888bc32025 (St. Louis Cardinals@Kansas City Royals) | credits=76636
2026-06-18 22:45:44,120 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=07f4bbde58b19584048fd125a63efb85 (Pittsburgh Pirates@Colorado Rockies) | credits=76630
2026-06-18 22:45:45,111 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=a8bc26eb19d767d6433749f35c2224ee (Los Angeles Angels@Athletics) | credits=76629
2026-06-18 22:45:46,353 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=68137f4094d6c3979398139c92d61aca (Minnesota Twins@Arizona Diamondbacks) | credits=76623
2026-06-18 22:45:47,600 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=150da5f2fa6db9e963e0b56fc2863769 (Baltimore Orioles@Los Angeles Dodgers) | credits=76617
2026-06-18 22:45:49,131 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=5bc4e94f9d24204a3f0dd424ced4a8be (Boston Red Sox@Seattle Mariners) | credits=76611
2026-06-18 22:45:50,505 | INFO | mlb_pipeline.crawler_oddsapi | No events found for 2026-06-20 â€” skipping prop fetch
2026-06-18 22:45:50,505 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=76611
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-18 22:45:56,070 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1442 rows into odds.mlb_game_lines (live odds).
2026-06-18 22:45:56,086 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-18
2026-06-18 22:45:56,086 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-18 22:45:56,086 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-19T04:15:56.086012+00:00.
2026-06-18 22:45:56,179 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-17
2026-06-18 22:46:06,688 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1742 rows into odds.mlb_player_prop_lines.
2026-06-18 22:46:06,688 | INFO | mlb_pipeline.parse_oddsapi | Processed 1742 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-18 22:46:06,688 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 0,
  "run_ids": "all",
  "date_from": "2026-06-19",
  "date_to": "2026-06-19",
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
  "active_rows": 0,
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
2026-06-18 22:55:57,510 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 27 pending predictions.
2026-06-18 22:55:57,510 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-18 22:55:57,542 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-18 22:56:00,745 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-18 22:56:00,776 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-18 22:56:00,917 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-18 22:56:00,932 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-18 22:56:01,151 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-18 22:56:02,495 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-18 22:56:02,510 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-19T00:56:34-04:00
Range: 2026-06-06 to 2026-06-19

## Collection Status

**TARGET MET**

- Clean shadow slates: 11 / 10
- Additional clean slates needed: 0
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 25.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 5.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-19 | no | 0 | 0 | 0 | 0 | 1742 | 0 | - | - | - | 0 | 1 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05 |
| 2026-06-18 | no | 2569 | 1545 | 2660 | 2660 | 14372 | 1994 | 75.0% | 0.0% | 12.2% | 2 | 15 | stale_close_rate>0.05 |
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
