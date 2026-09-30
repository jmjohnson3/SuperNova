# SuperNovaBets MLB Daily Run (2026-06-29 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 6.3s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 7.4s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 6.0s)
- **Refresh prop replay CLV**: OK (rc=0, 17.0s)
- **Build prop market training table**: OK (rc=0, 66.4s)
- **Prop walk-forward accuracy report**: OK (rc=0, 275.8s)
- **Prop shadow selector report**: OK (rc=0, 5.8s)
- **Prop miss diagnostic report**: OK (rc=0, 68.1s)
- **Prop bucket repair report**: OK (rc=0, 13.9s)
- **TB prop repair report**: OK (rc=0, 8.8s)
- **Prop target quality report**: OK (rc=0, 22.9s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 9.7s)
- **Grade outcomes + ledgers**: OK (rc=0, 8.9s)
- **Prop snapshot coverage report**: OK (rc=0, 14.6s)
- **Grade shadow prop replay**: OK (rc=0, 4.3s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-29 21:45:14,706 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-30. Catching up from 2026-07-01 to 2026-06-28
2026-06-29 21:45:14,706 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-29 window=2026-06-29T04:00:00Z..2026-06-30T04:00:00Z
2026-06-29 21:45:15,402 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-29 | events=4 | credits_remaining=64431
2026-06-29 21:45:15,723 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-30 window=2026-06-30T04:00:00Z..2026-07-01T04:00:00Z
2026-06-29 21:45:16,347 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-30 | events=14 | credits_remaining=64429
2026-06-29 21:45:16,357 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=64429
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-29 21:45:19,137 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-30. Catching up from 2026-07-01 to 2026-06-28
2026-06-29 21:45:19,737 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 4 events (game_date=2026-06-29, as_of=2026-06-29)
2026-06-29 21:45:20,273 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-29 event=08c10b4a78eb4168d07c1e53f345d13f (Miami Marlins@Colorado Rockies) | credits=64426
2026-06-29 21:45:21,191 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-29 event=bc5ac70107417075ffd9dda7565caff3 (San Francisco Giants@Arizona Diamondbacks) | credits=64422
2026-06-29 21:45:22,061 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-29 event=a898264b051f27a33274cbdacc554c46 (Los Angeles Dodgers@Athletics) | credits=64417
2026-06-29 21:45:22,959 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-29 event=0ce57b4f423d4bee56a8b4b941856426 (Los Angeles Angels@Seattle Mariners) | credits=64414
2026-06-29 21:45:23,745 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 14 events (game_date=2026-06-30, as_of=2026-06-29)
2026-06-29 21:45:23,818 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=64414
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-29 21:45:27,275 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1492 rows into odds.mlb_game_lines (live odds).
2026-06-29 21:45:27,278 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-29
2026-06-29 21:45:27,278 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-29 21:45:27,279 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-30T03:15:27.278869+00:00.
2026-06-29 21:45:27,318 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-28
2026-06-29 21:45:29,817 | INFO | mlb_pipeline.parse_oddsapi | Upserted 174 rows into odds.mlb_player_prop_lines.
2026-06-29 21:45:29,817 | INFO | mlb_pipeline.parse_oddsapi | Processed 174 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-29 21:45:29,817 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 6912,
  "run_ids": "all",
  "date_from": "2026-06-29",
  "date_to": "2026-06-29",
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
  "replay_rows": 19879,
  "examples": 19879
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 153910,
  "graded_rows": 132791,
  "valid_clv_rows": 126437,
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
  "active_rows": 2494,
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
  "rows": 17292,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 131965,
  "bucket_count": 136,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=53712
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 153009,
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
    "raw_api_one_sided": 26439
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 205-132 (60.8%) ROI: +16.1% | Total: 68-64 (51.5%) ROI: -1.7%
MLB CLV Run Line: beat close 2/46 (4%) avg CLV=+0.13 runs | CLV Total avg=+0.05 runs
MLB Price CLV Run Line: 44 bets  avg=+0.61%
```

**stderr (tail)**
```
2026-06-29 21:53:41,163 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 35 pending predictions.
2026-06-29 21:53:41,164 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-29 21:53:41,194 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-29 21:53:46,437 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-29 21:53:46,456 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-29 21:53:46,576 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-29 21:53:46,577 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-29 21:53:46,671 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-29 21:53:46,957 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-29 21:53:46,967 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-29T23:54:01-04:00
Range: 2026-06-16 to 2026-06-29

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
| 2026-06-29 | yes | 3034 | 2217 | 6912 | 6912 | 35281 | 5954 | 86.1% | 0.0% | 3.0% | 3 | 18 |  |
| 2026-06-28 | no | 4600 | 2631 | 5935 | 5935 | 18826 | 4885 | 82.3% | 0.0% | 10.7% | 3 | 14 | stale_close_rate>0.05 |
| 2026-06-27 | yes | 4389 | 2875 | 7032 | 7032 | 28903 | 6054 | 86.1% | 0.0% | 2.5% | 3 | 15 |  |
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
