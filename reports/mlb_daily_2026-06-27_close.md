# SuperNovaBets MLB Daily Run (2026-06-27 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 10.1s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 11.2s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 9.7s)
- **Refresh prop replay CLV**: OK (rc=0, 22.6s)
- **Build prop market training table**: OK (rc=0, 73.1s)
- **Prop walk-forward accuracy report**: OK (rc=0, 244.3s)
- **Prop shadow selector report**: OK (rc=0, 6.7s)
- **Prop miss diagnostic report**: OK (rc=0, 108.4s)
- **Prop bucket repair report**: OK (rc=0, 12.5s)
- **TB prop repair report**: OK (rc=0, 8.2s)
- **Prop target quality report**: OK (rc=0, 21.0s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 8.8s)
- **Grade outcomes + ledgers**: OK (rc=0, 6.3s)
- **Prop snapshot coverage report**: OK (rc=0, 11.9s)
- **Grade shadow prop replay**: OK (rc=0, 4.1s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-27 20:45:15,622 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-28. Catching up from 2026-06-29 to 2026-06-26
2026-06-27 20:45:15,622 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-27 window=2026-06-27T04:00:00Z..2026-06-28T04:00:00Z
2026-06-27 20:45:16,663 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-27 | events=3 | credits_remaining=66667
2026-06-27 20:45:17,362 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-28 window=2026-06-28T04:00:00Z..2026-06-29T04:00:00Z
2026-06-27 20:45:17,911 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-28 | events=15 | credits_remaining=66665
2026-06-27 20:45:17,928 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=66665
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-27 20:45:20,407 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-28. Catching up from 2026-06-29 to 2026-06-26
2026-06-27 20:45:20,967 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 3 events (game_date=2026-06-27, as_of=2026-06-27)
2026-06-27 20:45:21,610 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-27 event=27ffbf76753088bb73149d66116d30b5 (Los Angeles Dodgers@San Diego Padres) | credits=66662
2026-06-27 20:45:22,443 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-27 event=2a9b760c7cc231ff3ce0741965acc5c2 (Atlanta Braves@San Francisco Giants) | credits=66657
2026-06-27 20:45:23,346 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-27 event=338a013fc05917b67f3e883fac4d23ab (Athletics@Los Angeles Angels) | credits=66651
2026-06-27 20:45:24,143 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-06-28, as_of=2026-06-27)
2026-06-27 20:45:25,371 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-27 event=f156eda99a47c2d2e05bc9c5061b9fd1 (Philadelphia Phillies@New York Mets) | credits=66647
2026-06-27 20:45:26,346 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-27 event=9b4d594f5f7a6f6897b17090857664fb (Colorado Rockies@Minnesota Twins) | credits=66647
2026-06-27 20:45:27,232 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-27 event=08db99654eb5ae0d8d8655ed641188f0 (Chicago Cubs@Milwaukee Brewers) | credits=66647
2026-06-27 20:45:28,036 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-27 event=a0d8dffceba6456f69d9dbb4a78415a3 (Miami Marlins@St. Louis Cardinals) | credits=66647
2026-06-27 20:45:28,883 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-27 event=ae4c2503863105794c22aa36724c8ae8 (Athletics@Los Angeles Angels) | credits=66647
2026-06-27 20:45:29,149 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=66647
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-27 20:45:33,477 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1485 rows into odds.mlb_game_lines (live odds).
2026-06-27 20:45:33,514 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-27
2026-06-27 20:45:33,515 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-27 20:45:33,515 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-28T02:15:33.515847+00:00.
2026-06-27 20:45:33,709 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-26
2026-06-27 20:45:38,883 | INFO | mlb_pipeline.parse_oddsapi | Upserted 434 rows into odds.mlb_player_prop_lines.
2026-06-27 20:45:38,883 | INFO | mlb_pipeline.parse_oddsapi | Processed 434 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-27 20:45:38,883 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 7032,
  "run_ids": "all",
  "date_from": "2026-06-27",
  "date_to": "2026-06-27",
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
  "replay_rows": 17330,
  "examples": 17330
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 141063,
  "graded_rows": 120776,
  "valid_clv_rows": 115598,
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
  "active_rows": 1864,
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
  "rows": 15333,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 119950,
  "bucket_count": 136,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=48730
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 140162,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### FanDuel one-sided diagnostic

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "root_causes": {
    "raw_api_one_sided": 31328
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 201-131 (60.5%) ROI: +15.6% | Total: 68-62 (52.3%) ROI: -0.1%
MLB CLV Run Line: beat close 2/41 (5%) avg CLV=+0.15 runs | CLV Total avg=+0.06 runs
MLB Price CLV Run Line: 39 bets  avg=+0.71%
```

**stderr (tail)**
```
2026-06-27 20:54:07,775 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 37 pending predictions.
2026-06-27 20:54:07,776 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-27 20:54:07,793 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-27 20:54:10,242 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-27 20:54:10,262 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-27 20:54:10,446 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-27 20:54:10,448 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-27 20:54:10,517 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-27 20:54:10,832 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-27 20:54:10,842 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-27T22:54:22-04:00
Range: 2026-06-14 to 2026-06-27

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
| 2026-06-27 | yes | 3843 | 2528 | 7032 | 7032 | 28851 | 6054 | 86.1% | 0.0% | 2.5% | 3 | 14 |  |
| 2026-06-26 | yes | 4099 | 2969 | 8013 | 8013 | 34632 | 6866 | 85.7% | 0.0% | 2.8% | 3 | 14 |  |
| 2026-06-25 | no | 2473 | 1660 | 2285 | 2285 | 10693 | 1810 | 79.2% | 0.0% | 9.4% | 2 | 10 | stale_close_rate>0.05 |
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
