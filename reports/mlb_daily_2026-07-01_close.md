# SuperNovaBets MLB Daily Run (2026-07-01 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 4.5s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 6.0s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 3.6s)
- **Refresh prop replay CLV**: OK (rc=0, 16.0s)
- **Build prop market training table**: OK (rc=0, 69.1s)
- **Prop walk-forward accuracy report**: OK (rc=0, 377.9s)
- **Prop shadow selector report**: OK (rc=0, 5.3s)
- **Prop miss diagnostic report**: OK (rc=0, 53.3s)
- **Prop bucket repair report**: OK (rc=0, 20.3s)
- **TB prop repair report**: OK (rc=0, 12.5s)
- **Prop target quality report**: OK (rc=0, 32.1s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 11.4s)
- **Grade outcomes + ledgers**: OK (rc=0, 9.8s)
- **Grade daily forecast ledger**: OK (rc=0, 3.1s)
- **Daily forecast projection audit**: OK (rc=0, 8.4s)
- **Prop snapshot coverage report**: OK (rc=0, 28.3s)
- **Grade shadow prop replay**: OK (rc=0, 5.2s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-01 21:45:06,081 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-02. Catching up from 2026-07-03 to 2026-06-30
2026-07-01 21:45:06,081 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-01 window=2026-07-01T04:00:00Z..2026-07-02T04:00:00Z
2026-07-01 21:45:06,975 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-01 | events=2 | credits_remaining=98804
2026-07-01 21:45:06,984 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-02 window=2026-07-02T04:00:00Z..2026-07-03T04:00:00Z
2026-07-01 21:45:07,849 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-02 | events=9 | credits_remaining=98802
2026-07-01 21:45:07,866 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=98802
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-01 21:45:10,333 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-02. Catching up from 2026-07-03 to 2026-06-30
2026-07-01 21:45:11,036 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 2 events (game_date=2026-07-01, as_of=2026-07-01)
2026-07-01 21:45:11,909 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-01 event=4613f606f40dfab948c795bcde2caaa2 (San Francisco Giants@Arizona Diamondbacks) | credits=98797
2026-07-01 21:45:12,848 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-01 event=89e2407a67afd5e02ec12a372a53fa18 (Los Angeles Dodgers@Athletics) | credits=98794
2026-07-01 21:45:13,957 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 9 events (game_date=2026-07-02, as_of=2026-07-01)
2026-07-01 21:45:13,957 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=98794
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-01 21:45:16,546 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1494 rows into odds.mlb_game_lines (live odds).
2026-07-01 21:45:16,553 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-01
2026-07-01 21:45:16,558 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-01 21:45:16,558 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-02T03:15:16.558320+00:00.
2026-07-01 21:45:16,562 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-30
2026-07-01 21:45:17,556 | INFO | mlb_pipeline.parse_oddsapi | Upserted 134 rows into odds.mlb_player_prop_lines.
2026-07-01 21:45:17,556 | INFO | mlb_pipeline.parse_oddsapi | Processed 134 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-01 21:45:17,556 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 6538,
  "run_ids": "all",
  "date_from": "2026-07-01",
  "date_to": "2026-07-01",
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
  "replay_rows": 21510,
  "examples": 21510
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 168508,
  "graded_rows": 146427,
  "valid_clv_rows": 138602,
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
  "active_rows": 416,
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
  "rows": 18353,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 145601,
  "bucket_count": 136,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=59514
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 153947,
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
    "raw_api_one_sided": 23107
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 207-136 (60.3%) ROI: +15.2% | Total: 68-64 (51.5%) ROI: -1.7%
MLB CLV Run Line: beat close 2/52 (4%) avg CLV=+0.12 runs | CLV Total avg=+0.07 runs
MLB Price CLV Run Line: 50 bets  avg=+0.58%
```

**stderr (tail)**
```
2026-07-01 21:55:18,785 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 36 pending predictions.
2026-07-01 21:55:18,785 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-07-01 21:55:18,801 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-01 21:55:24,358 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-07-01 21:55:24,381 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-07-01 21:55:24,613 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-01 21:55:24,613 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-01 21:55:24,691 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-07-01 21:55:25,175 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-07-01 21:55:25,190 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Grade daily forecast ledger

- rc: 0

**stdout (tail)**
```
{
  "schema_ready": true,
  "graded": 0
}
```

### Daily forecast projection audit

- rc: 0

**stdout (tail)**
```
{
  "status": "collecting_prospective_forecasts",
  "rows": 0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_daily_forecast_projection_audit_latest.md"
}
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-07-01T23:56:05-04:00
Range: 2026-06-18 to 2026-07-01

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
| 2026-07-01 | yes | 4074 | 2335 | 6538 | 6538 | 25183 | 5254 | 80.4% | 0.0% | 3.1% | 3 | 15 |  |
| 2026-06-30 | yes | 4519 | 2785 | 8060 | 8060 | 41133 | 6911 | 85.7% | 0.0% | 3.0% | 3 | 18 |  |
| 2026-06-29 | yes | 3883 | 2282 | 6912 | 6912 | 35281 | 5954 | 86.1% | 0.0% | 3.0% | 3 | 18 |  |
| 2026-06-28 | no | 4600 | 2631 | 5935 | 5935 | 18826 | 4885 | 82.3% | 0.0% | 10.7% | 3 | 14 | stale_close_rate>0.05 |
| 2026-06-27 | yes | 4389 | 2875 | 7032 | 7032 | 28903 | 6054 | 86.1% | 0.0% | 1.5% | 3 | 15 |  |
| 2026-06-26 | yes | 4099 | 2969 | 8013 | 8013 | 34632 | 6866 | 85.7% | 0.0% | 1.2% | 3 | 14 |  |
| 2026-06-25 | no | 2473 | 1660 | 2285 | 2285 | 10693 | 1810 | 79.2% | 0.0% | 5.0% | 2 | 10 | stale_close_rate>0.05 |
| 2026-06-24 | yes | 4567 | 2646 | 4697 | 4697 | 21191 | 3644 | 77.6% | 0.0% | 2.5% | 2 | 11 |  |
| 2026-06-23 | yes | 4342 | 2721 | 4119 | 4119 | 28669 | 2350 | 57.1% | 0.0% | 1.0% | 2 | 14 |  |
| 2026-06-22 | yes | 3807 | 2394 | 7540 | 7540 | 32994 | 6610 | 87.7% | 0.0% | 1.9% | 3 | 17 |  |
| 2026-06-21 | yes | 4514 | 2574 | 5500 | 5500 | 24573 | 4484 | 81.5% | 0.0% | 1.4% | 3 | 16 |  |
| 2026-06-20 | yes | 4153 | 2551 | 6337 | 6337 | 31532 | 5322 | 84.0% | 0.0% | 1.0% | 3 | 17 |  |
| 2026-06-19 | yes | 4181 | 2501 | 7929 | 7929 | 35937 | 7008 | 88.4% | 0.0% | 0.6% | 3 | 18 |  |
| 2026-06-18 | yes | 2569 | 1592 | 2660 | 2660 | 14372 | 1994 | 75.0% | 0.0% | 2.3% | 2 | 15 |  |
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
