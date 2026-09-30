# SuperNovaBets MLB Daily Run (2026-06-30 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 5.4s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 6.8s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 3.8s)
- **Refresh prop replay CLV**: OK (rc=0, 18.8s)
- **Build prop market training table**: OK (rc=0, 60.3s)
- **Prop walk-forward accuracy report**: OK (rc=0, 280.5s)
- **Prop shadow selector report**: OK (rc=0, 6.3s)
- **Prop miss diagnostic report**: OK (rc=0, 67.1s)
- **Prop bucket repair report**: OK (rc=0, 13.6s)
- **TB prop repair report**: OK (rc=0, 8.6s)
- **Prop target quality report**: OK (rc=0, 23.5s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 9.7s)
- **Grade outcomes + ledgers**: OK (rc=0, 8.7s)
- **Prop snapshot coverage report**: OK (rc=0, 15.3s)
- **Grade shadow prop replay**: OK (rc=0, 4.4s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-30 21:45:08,935 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-01. Catching up from 2026-07-02 to 2026-06-29
2026-06-30 21:45:08,935 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-30 window=2026-06-30T04:00:00Z..2026-07-01T04:00:00Z
2026-06-30 21:45:09,654 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-30 | events=4 | credits_remaining=99813
2026-06-30 21:45:09,685 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-01 window=2026-07-01T04:00:00Z..2026-07-02T04:00:00Z
2026-06-30 21:45:10,279 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-01 | events=14 | credits_remaining=99811
2026-06-30 21:45:10,279 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99811
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-30 21:45:12,671 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-01. Catching up from 2026-07-02 to 2026-06-29
2026-06-30 21:45:13,233 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 4 events (game_date=2026-06-30, as_of=2026-06-30)
2026-06-30 21:45:13,837 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-30 event=c9a7f8f74b4ae9a65c80d47993dfeb86 (Miami Marlins@Colorado Rockies) | credits=99810
2026-06-30 21:45:14,624 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-30 event=1c4be9313fc412b34cc158daa4b7fdac (Los Angeles Dodgers@Athletics) | credits=99805
2026-06-30 21:45:15,456 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-30 event=2b3a5c8c1587ea527558ec8e6fb514ee (San Francisco Giants@Arizona Diamondbacks) | credits=99802
2026-06-30 21:45:16,279 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-30 event=835944b5efc736fb06b73068a981fde5 (Los Angeles Angels@Seattle Mariners) | credits=99797
2026-06-30 21:45:17,061 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 14 events (game_date=2026-07-01, as_of=2026-06-30)
2026-06-30 21:45:17,061 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=99797
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-30 21:45:19,622 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1498 rows into odds.mlb_game_lines (live odds).
2026-06-30 21:45:19,622 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-30
2026-06-30 21:45:19,622 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-30 21:45:19,622 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-01T03:15:19.622111+00:00.
2026-06-30 21:45:19,622 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-29
2026-06-30 21:45:20,895 | INFO | mlb_pipeline.parse_oddsapi | Upserted 210 rows into odds.mlb_player_prop_lines.
2026-06-30 21:45:20,895 | INFO | mlb_pipeline.parse_oddsapi | Processed 210 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-30 21:45:20,895 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 8060,
  "run_ids": "all",
  "date_from": "2026-06-30",
  "date_to": "2026-06-30",
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
  "replay_rows": 20907,
  "examples": 20907
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 161970,
  "graded_rows": 139199,
  "valid_clv_rows": 133348,
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
  "active_rows": 2962,
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
  "rows": 17876,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 138373,
  "bucket_count": 136,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=56450
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 161069,
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
    "raw_api_one_sided": 24082
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 206-134 (60.6%) ROI: +15.7% | Total: 68-64 (51.5%) ROI: -1.7%
MLB CLV Run Line: beat close 2/49 (4%) avg CLV=+0.12 runs | CLV Total avg=+0.05 runs
MLB Price CLV Run Line: 47 bets  avg=+0.60%
```

**stderr (tail)**
```
2026-06-30 21:53:32,455 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 37 pending predictions.
2026-06-30 21:53:32,456 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-30 21:53:32,468 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-30 21:53:37,326 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-30 21:53:37,342 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-30 21:53:37,489 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-30 21:53:37,491 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-30 21:53:37,561 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-30 21:53:37,842 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-30 21:53:37,842 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-30T23:53:53-04:00
Range: 2026-06-17 to 2026-06-30

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
| 2026-06-30 | yes | 3695 | 2581 | 8060 | 8060 | 41133 | 6911 | 85.7% | 0.0% | 3.0% | 3 | 18 |  |
| 2026-06-29 | yes | 3883 | 2282 | 6912 | 6912 | 35281 | 5954 | 86.1% | 0.0% | 3.0% | 3 | 18 |  |
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
