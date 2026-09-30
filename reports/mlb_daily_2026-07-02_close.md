# SuperNovaBets MLB Daily Run (2026-07-02 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 10.6s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 9.7s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 7.7s)
- **Refresh prop replay CLV**: OK (rc=0, 12.3s)
- **Build prop market training table**: OK (rc=0, 78.1s)
- **Prop walk-forward accuracy report**: OK (rc=0, 396.4s)
- **Prop shadow selector report**: OK (rc=0, 12.7s)
- **Prop miss diagnostic report**: OK (rc=0, 78.9s)
- **Prop bucket repair report**: OK (rc=0, 21.1s)
- **TB prop repair report**: OK (rc=0, 15.4s)
- **Prop target quality report**: OK (rc=0, 50.1s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 11.2s)
- **Grade outcomes + ledgers**: OK (rc=0, 10.3s)
- **Grade daily forecast ledger**: OK (rc=0, 3.7s)
- **Daily forecast projection audit**: OK (rc=0, 8.8s)
- **Forecast repair error decomposition**: OK (rc=0, 4.3s)
- **Prop snapshot coverage report**: OK (rc=0, 44.8s)
- **Grade shadow prop replay**: OK (rc=0, 5.2s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-02 21:45:07,402 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-03. Catching up from 2026-07-04 to 2026-07-01
2026-07-02 21:45:07,417 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-02 window=2026-07-02T04:00:00Z..2026-07-03T04:00:00Z
2026-07-02 21:45:13,067 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-02 | events=2 | credits_remaining=97932
2026-07-02 21:45:13,394 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-03 window=2026-07-03T04:00:00Z..2026-07-04T04:00:00Z
2026-07-02 21:45:14,229 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-03 | events=13 | credits_remaining=97930
2026-07-02 21:45:14,238 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=97930
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-02 21:45:20,114 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-03. Catching up from 2026-07-04 to 2026-07-01
2026-07-02 21:45:20,687 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 2 events (game_date=2026-07-02, as_of=2026-07-02)
2026-07-02 21:45:21,472 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-02 event=283444f773ee82cae9a5977e9266baac (Los Angeles Angels@Seattle Mariners) | credits=97927
2026-07-02 21:45:22,755 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-02 event=ae95cd98700fec55791774682d6b0b06 (San Diego Padres@Los Angeles Dodgers) | credits=97922
2026-07-02 21:45:23,863 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 13 events (game_date=2026-07-03, as_of=2026-07-02)
2026-07-02 21:45:23,973 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=97922
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-02 21:45:28,988 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1506 rows into odds.mlb_game_lines (live odds).
2026-07-02 21:45:28,990 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-02
2026-07-02 21:45:28,990 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-02 21:45:28,990 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-03T03:15:28.990200+00:00.
2026-07-02 21:45:29,021 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-01
2026-07-02 21:45:31,677 | INFO | mlb_pipeline.parse_oddsapi | Upserted 195 rows into odds.mlb_player_prop_lines.
2026-07-02 21:45:31,677 | INFO | mlb_pipeline.parse_oddsapi | Processed 195 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-02 21:45:31,677 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 4505,
  "run_ids": "all",
  "date_from": "2026-07-02",
  "date_to": "2026-07-02",
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
  "replay_rows": 19103,
  "examples": 19103
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 173013,
  "graded_rows": 152327,
  "valid_clv_rows": 142665,
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
  "active_rows": 1153,
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
  "rows": 18915,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 151501,
  "bucket_count": 137,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=62046
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 172112,
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
    "raw_api_one_sided": 20589
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 209-137 (60.4%) ROI: +15.3% | Total: 68-64 (51.5%) ROI: -1.7%
MLB CLV Run Line: beat close 2/54 (4%) avg CLV=+0.11 runs | CLV Total avg=+0.07 runs
MLB Price CLV Run Line: 52 bets  avg=+0.61%
```

**stderr (tail)**
```
2026-07-02 21:56:51,307 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 31 pending predictions.
2026-07-02 21:56:51,307 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-07-02 21:56:51,338 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-02 21:56:57,301 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-07-02 21:56:57,317 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-07-02 21:56:57,487 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-02 21:56:57,492 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-02 21:56:57,583 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-07-02 21:56:58,063 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-07-02 21:56:58,072 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
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
  "status": "ready",
  "rows": 126,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_daily_forecast_projection_audit_latest.md"
}
```

### Forecast repair error decomposition

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "active_source": "historical_player_game_fallback",
  "prospective_rows": 84,
  "historical_rows": 17197,
  "tb_repair": "train_gated_direct_player_game_tb_head",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_forecast_repair_error_latest.md"
}
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-07-02T23:57:59-04:00
Range: 2026-06-19 to 2026-07-02

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
| 2026-07-02 | yes | 2255 | 1566 | 4505 | 4505 | 22054 | 4063 | 90.2% | 0.0% | 2.1% | 3 | 18 |  |
| 2026-07-01 | yes | 4174 | 2582 | 6538 | 6538 | 25183 | 5254 | 80.4% | 0.0% | 3.1% | 3 | 15 |  |
| 2026-06-30 | yes | 4519 | 2785 | 8060 | 8060 | 41133 | 6911 | 85.7% | 0.0% | 3.0% | 3 | 18 |  |
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
