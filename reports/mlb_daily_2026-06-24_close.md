# SuperNovaBets MLB Daily Run (2026-06-24 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 14.4s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 20.8s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 27.2s)
- **Refresh prop replay CLV**: OK (rc=0, 9.0s)
- **Build prop market training table**: OK (rc=0, 93.2s)
- **Prop walk-forward accuracy report**: OK (rc=0, 229.1s)
- **Prop shadow selector report**: OK (rc=0, 284.1s)
- **Prop miss diagnostic report**: OK (rc=0, 158.0s)
- **Prop bucket repair report**: OK (rc=0, 29.5s)
- **TB prop repair report**: OK (rc=0, 38.0s)
- **Prop target quality report**: OK (rc=0, 26.7s)
- **Grade outcomes + ledgers**: OK (rc=0, 21.7s)
- **Prop snapshot coverage report**: OK (rc=0, 29.2s)
- **Grade shadow prop replay**: OK (rc=0, 10.3s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-24 08:56:22,607 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-25. Catching up from 2026-06-26 to 2026-06-23
2026-06-24 08:56:22,646 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-24 window=2026-06-24T04:00:00Z..2026-06-25T04:00:00Z
2026-06-24 08:56:23,881 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-24 | events=16 | credits_remaining=70445
2026-06-24 08:56:24,797 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-25 window=2026-06-25T04:00:00Z..2026-06-26T04:00:00Z
2026-06-24 08:56:25,428 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-25 | events=0 | credits_remaining=70445
2026-06-24 08:56:25,429 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=70445
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-24 08:56:30,237 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-25. Catching up from 2026-06-26 to 2026-06-23
2026-06-24 08:56:30,958 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 16 events (game_date=2026-06-24, as_of=2026-06-24)
2026-06-24 08:56:31,506 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=6493d60c5df2964d50009d111534865e (Texas Rangers@Miami Marlins) | credits=70439
2026-06-24 08:56:32,459 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=b03e189e275017dfd8549bdf781bd2cc (Chicago Cubs@New York Mets) | credits=70435
2026-06-24 08:56:33,403 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=1f144a9853388d9cfff4f46b16fb544f (Cleveland Guardians@Chicago White Sox) | credits=70429
2026-06-24 08:56:34,236 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=74a9d15be6f7e75b97c7d220010253bd (Boston Red Sox@Colorado Rockies) | credits=70423
2026-06-24 08:56:35,072 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=6dcbe8dd9b8e932306789135de65db46 (Baltimore Orioles@Los Angeles Angels) | credits=70417
2026-06-24 08:56:35,892 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=d4ba9e7cff72c10fef86a3d3e71b80cd (New York Yankees@Detroit Tigers) | credits=70411
2026-06-24 08:56:36,746 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=6f91332c1c3d9d7efc30b8b5e5596da8 (Kansas City Royals@Tampa Bay Rays) | credits=70405
2026-06-24 08:56:37,637 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=6bd1412ad6a83bd72dfccfd3b52aba6b (Seattle Mariners@Pittsburgh Pirates) | credits=70399
2026-06-24 08:56:38,467 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=af718a14d01fafa34e924ea752814134 (Philadelphia Phillies@Washington Nationals) | credits=70393
2026-06-24 08:56:39,481 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=e5eb0d0e2884807a04a7a39cb730639d (Houston Astros@Toronto Blue Jays) | credits=70387
2026-06-24 08:56:40,387 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=0af952c1f3ce17d011acf2b042d9a497 (Chicago Cubs@New York Mets) | credits=70387
2026-06-24 08:56:41,370 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=2dcef2b58c37ee8ffd48e1d662c5ebda (Milwaukee Brewers@Cincinnati Reds) | credits=70381
2026-06-24 08:56:42,396 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=f7efadc6e4a428969fb12a38b1f5c343 (Los Angeles Dodgers@Minnesota Twins) | credits=70375
2026-06-24 08:56:43,254 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=3f6bd3e4a5b58ebe56c5f3bd89470591 (Arizona Diamondbacks@St. Louis Cardinals) | credits=70369
2026-06-24 08:56:44,192 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=0c4f44574b86346807453d2bfde5280e (Atlanta Braves@San Diego Padres) | credits=70363
2026-06-24 08:56:45,114 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-24 event=74b391c5a12588e6bdf6f98be072b920 (Athletics@San Francisco Giants) | credits=70357
2026-06-24 08:56:45,993 | INFO | mlb_pipeline.crawler_oddsapi | No events found for 2026-06-25 â€” skipping prop fetch
2026-06-24 08:56:45,993 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=70357
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-24 08:56:53,756 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1476 rows into odds.mlb_game_lines (live odds).
2026-06-24 08:56:53,834 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-23
2026-06-24 08:56:53,835 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-24 08:56:53,835 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-24T14:26:53.835305+00:00.
2026-06-24 08:56:53,993 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-23
2026-06-24 08:57:13,308 | INFO | mlb_pipeline.parse_oddsapi | Upserted 2595 rows into odds.mlb_player_prop_lines.
2026-06-24 08:57:13,308 | INFO | mlb_pipeline.parse_oddsapi | Processed 2595 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-24 08:57:13,308 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 2446,
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
  "replay_rows": 19605,
  "examples": 19605
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 121482,
  "graded_rows": 107159,
  "valid_clv_rows": 104823,
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
  "active_rows": 2446,
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
  "rows": 120581,
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
2026-06-24 09:11:50,528 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 37 pending predictions.
2026-06-24 09:11:50,528 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-24 09:11:50,560 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-24 09:12:00,522 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-24 09:12:00,541 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-24 09:12:01,828 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-24 09:12:01,830 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-24 09:12:02,057 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-24 09:12:02,800 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-24 09:12:02,807 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-24T11:12:31-04:00
Range: 2026-06-11 to 2026-06-24

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
| 2026-06-24 | no | 2900 | 2418 | 2446 | 2446 | 4948 | 142 | 5.8% | 0.0% | 3.5% | 1 | 2 | valid_clv_coverage<0.25 |
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
| 2026-06-11 | no | 2452 | 1703 | 2202 | 2202 | 14587 | 1879 | 85.3% | 0.0% | 6.1% | 2 | 15 | stale_close_rate>0.05 |
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
