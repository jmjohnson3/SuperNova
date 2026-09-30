# SuperNovaBets MLB Daily Run (2026-06-28 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 6.0s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 3.5s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 3.3s)
- **Refresh prop replay CLV**: OK (rc=0, 11.2s)
- **Build prop market training table**: OK (rc=0, 67.7s)
- **Prop walk-forward accuracy report**: OK (rc=0, 260.9s)
- **Prop shadow selector report**: OK (rc=0, 5.2s)
- **Prop miss diagnostic report**: OK (rc=0, 47.3s)
- **Prop bucket repair report**: OK (rc=0, 13.0s)
- **TB prop repair report**: OK (rc=0, 8.5s)
- **Prop target quality report**: OK (rc=0, 21.8s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 9.2s)
- **Grade outcomes + ledgers**: OK (rc=0, 7.6s)
- **Prop snapshot coverage report**: OK (rc=0, 13.4s)
- **Grade shadow prop replay**: OK (rc=0, 4.1s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-28 21:45:10,020 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-29. Catching up from 2026-06-30 to 2026-06-27
2026-06-28 21:45:10,020 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-28 window=2026-06-28T04:00:00Z..2026-06-29T04:00:00Z
2026-06-28 21:45:10,876 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-28 | events=0 | credits_remaining=65798
2026-06-28 21:45:11,060 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-29 window=2026-06-29T04:00:00Z..2026-06-30T04:00:00Z
2026-06-28 21:45:11,709 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-29 | events=13 | credits_remaining=65796
2026-06-28 21:45:11,725 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=65796
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-28 21:45:14,139 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-29. Catching up from 2026-06-30 to 2026-06-27
2026-06-28 21:45:14,663 | INFO | mlb_pipeline.crawler_oddsapi | No events found for 2026-06-28 â€” skipping prop fetch
2026-06-28 21:45:15,185 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 13 events (game_date=2026-06-29, as_of=2026-06-28)
2026-06-28 21:45:15,255 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=unknown
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-28 21:45:18,560 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1482 rows into odds.mlb_game_lines (live odds).
2026-06-28 21:45:18,572 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-28
2026-06-28 21:45:18,573 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-28 21:45:18,573 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-29T03:15:18.573097+00:00.
2026-06-28 21:45:18,598 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-27
2026-06-28 21:45:18,599 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_prop_odds snapshots found (as_of_date=None).
2026-06-28 21:45:18,599 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 5935,
  "run_ids": "all",
  "date_from": "2026-06-28",
  "date_to": "2026-06-28",
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
  "replay_rows": 20980,
  "examples": 20980
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 146998,
  "graded_rows": 127328,
  "valid_clv_rows": 120483,
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
  "active_rows": 230,
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
  "rows": 16322,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 126502,
  "bucket_count": 136,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=51458
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 146097,
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
    "raw_api_one_sided": 28895
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 202-132 (60.5%) ROI: +15.5% | Total: 68-63 (51.9%) ROI: -0.9%
MLB CLV Run Line: beat close 2/43 (5%) avg CLV=+0.14 runs | CLV Total avg=+0.06 runs
MLB Price CLV Run Line: 41 bets  avg=+0.70%
```

**stderr (tail)**
```
2026-06-28 21:52:46,302 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 37 pending predictions.
2026-06-28 21:52:46,303 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-28 21:52:46,334 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-28 21:52:50,262 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-28 21:52:50,283 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-28 21:52:50,416 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-28 21:52:50,418 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-28 21:52:50,521 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-28 21:52:50,839 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-28 21:52:50,850 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-28T23:53:04-04:00
Range: 2026-06-15 to 2026-06-28

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
| 2026-06-28 | no | 4549 | 2561 | 5935 | 5935 | 18826 | 4885 | 82.3% | 0.0% | 10.7% | 3 | 14 | stale_close_rate>0.05 |
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
| 2026-06-15 | yes | 2927 | 1891 | 5873 | 5873 | 28077 | 5221 | 88.9% | 0.0% | 2.6% | 3 | 18 |  |
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
