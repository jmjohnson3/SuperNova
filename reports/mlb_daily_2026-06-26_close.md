# SuperNovaBets MLB Daily Run (2026-06-26 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 10.4s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 19.5s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 32.6s)
- **Refresh prop replay CLV**: OK (rc=0, 23.8s)
- **Build prop market training table**: OK (rc=0, 83.2s)
- **Prop walk-forward accuracy report**: OK (rc=0, 327.1s)
- **Prop shadow selector report**: OK (rc=0, 9.5s)
- **Prop miss diagnostic report**: OK (rc=0, 63.7s)
- **Prop bucket repair report**: OK (rc=0, 23.1s)
- **TB prop repair report**: OK (rc=0, 19.8s)
- **Prop target quality report**: OK (rc=0, 22.5s)
- **Grade outcomes + ledgers**: OK (rc=0, 14.0s)
- **Prop snapshot coverage report**: OK (rc=0, 27.3s)
- **Grade shadow prop replay**: OK (rc=0, 7.2s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-26 16:45:26,821 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-27. Catching up from 2026-06-28 to 2026-06-25
2026-06-26 16:45:26,827 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-26 window=2026-06-26T04:00:00Z..2026-06-27T04:00:00Z
2026-06-26 16:45:27,650 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-26 | events=15 | credits_remaining=67987
2026-06-26 16:45:28,031 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-27 window=2026-06-27T04:00:00Z..2026-06-28T04:00:00Z
2026-06-26 16:45:28,573 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-27 | events=14 | credits_remaining=67985
2026-06-26 16:45:28,586 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=67985
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-26 16:45:32,535 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-27. Catching up from 2026-06-28 to 2026-06-25
2026-06-26 16:45:33,104 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-06-26, as_of=2026-06-26)
2026-06-26 16:45:33,702 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=fadbcf06dc34a35aa764217449eed0bc (Cincinnati Reds@Pittsburgh Pirates) | credits=67979
2026-06-26 16:45:34,667 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=a46f3826966d75bbba418ffa315da7a1 (Houston Astros@Detroit Tigers) | credits=67973
2026-06-26 16:45:35,540 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=b15cd27ee7dc82b2fdc7120117c7573d (Washington Nationals@Baltimore Orioles) | credits=67967
2026-06-26 16:45:36,414 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=08d5e50d378787386a6a9d0b98f7d662 (Texas Rangers@Toronto Blue Jays) | credits=67961
2026-06-26 16:45:37,238 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=cf96e2e47e38bb6769a2000cd23b871b (Arizona Diamondbacks@Tampa Bay Rays) | credits=67955
2026-06-26 16:45:38,045 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=6eb5de1ab0595da964b22b0fc847fbc1 (New York Yankees@Boston Red Sox) | credits=67949
2026-06-26 16:45:38,913 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=9637bef3da89f22340fc6ea44bfe91b4 (Seattle Mariners@Cleveland Guardians) | credits=67943
2026-06-26 16:45:39,779 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=f950765d2ab77ff856b7643ec5cd880c (Philadelphia Phillies@New York Mets) | credits=67937
2026-06-26 16:45:40,595 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=fbe3805ebed0daf92e6e43ccfc287b96 (Kansas City Royals@Chicago White Sox) | credits=67931
2026-06-26 16:45:41,451 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=f2bbab66ff28ba66910103ce8596b5bc (Chicago Cubs@Milwaukee Brewers) | credits=67925
2026-06-26 16:45:42,284 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=dc2d29e6b7b6fd78c897d0818ef01208 (Colorado Rockies@Minnesota Twins) | credits=67919
2026-06-26 16:45:43,099 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=10965e1fdc600d49ff4fbcc534115980 (Miami Marlins@St. Louis Cardinals) | credits=67913
2026-06-26 16:45:43,928 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=c2d68e4640b7927c37fc5cda2352ee1f (Athletics@Los Angeles Angels) | credits=67907
2026-06-26 16:45:44,744 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=136805c629e76ccb3b3b770196f07915 (Los Angeles Dodgers@San Diego Padres) | credits=67901
2026-06-26 16:45:45,698 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=ffc9895ee406538679c2fdd5097dbbf6 (Atlanta Braves@San Francisco Giants) | credits=67895
2026-06-26 16:45:46,619 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 14 events (game_date=2026-06-27, as_of=2026-06-26)
2026-06-26 16:45:47,615 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-26 event=cad93485b644bd4b3adfacfc58e6cb0c (Chicago Cubs@Milwaukee Brewers) | credits=67895
2026-06-26 16:45:48,057 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=67895
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-26 16:45:53,020 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1498 rows into odds.mlb_game_lines (live odds).
2026-06-26 16:45:53,056 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-26
2026-06-26 16:45:53,057 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-26 16:45:53,057 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-26T22:15:53.057568+00:00.
2026-06-26 16:45:53,218 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-25
2026-06-26 16:46:20,745 | INFO | mlb_pipeline.parse_oddsapi | Upserted 2978 rows into odds.mlb_player_prop_lines.
2026-06-26 16:46:20,745 | INFO | mlb_pipeline.parse_oddsapi | Processed 2978 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-26 16:46:20,746 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 8013,
  "run_ids": "all",
  "date_from": "2026-06-26",
  "date_to": "2026-06-26",
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
  "replay_rows": 19114,
  "examples": 19114
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 134031,
  "graded_rows": 113211,
  "valid_clv_rows": 108586,
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
  "active_rows": 3015,
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
  "rows": 112385,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 112385,
  "bucket_count": 135,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=45574
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 133130,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 201-130 (60.7%) ROI: +15.9% | Total: 67-61 (52.3%) ROI: -0.1%
MLB CLV Run Line: beat close 2/40 (5%) avg CLV=+0.15 runs | CLV Total avg=+0.03 runs
MLB Price CLV Run Line: 38 bets  avg=+0.67%
```

**stderr (tail)**
```
2026-06-26 16:56:01,481 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 37 pending predictions.
2026-06-26 16:56:01,482 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-26 16:56:01,513 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-26 16:56:06,152 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-26 16:56:06,171 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-26 16:56:06,365 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-26 16:56:06,366 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-26 16:56:06,649 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-26 16:56:07,359 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-26 16:56:07,368 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-26T18:56:34-04:00
Range: 2026-06-13 to 2026-06-26

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
| 2026-06-26 | yes | 3667 | 2405 | 8013 | 8013 | 31706 | 5908 | 73.7% | 0.0% | 3.0% | 3 | 12 |  |
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
