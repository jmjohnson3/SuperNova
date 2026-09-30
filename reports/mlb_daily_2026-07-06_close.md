# SuperNovaBets MLB Daily Run (2026-07-06 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 4.1s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 4.3s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 7.5s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.0s)
- **Refresh prop replay CLV**: OK (rc=0, 10.2s)
- **Build prop market training table**: OK (rc=0, 51.6s)
- **Prop walk-forward accuracy report**: OK (rc=0, 383.9s)
- **Prop shadow selector report**: OK (rc=0, 7.9s)
- **Prop miss diagnostic report**: OK (rc=0, 85.3s)
- **Prop bucket repair report**: OK (rc=0, 18.5s)
- **TB prop repair report**: OK (rc=0, 10.8s)
- **Prop target quality report**: OK (rc=0, 35.2s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 9.7s)
- **Grade outcomes + ledgers**: OK (rc=0, 7.6s)
- **Grade daily forecast ledger**: OK (rc=0, 4.5s)
- **Daily forecast projection audit**: OK (rc=0, 8.9s)
- **Prop micro promotion evaluation**: OK (rc=0, 0.3s)
- **Forecast repair error decomposition**: OK (rc=0, 4.1s)
- **Prop snapshot coverage report**: OK (rc=0, 36.5s)
- **End-of-slate prop close diagnostic**: OK (rc=0, 3.5s)
- **Frozen-release five-date checkpoint**: OK (rc=0, 7.3s)
- **Grade shadow prop replay**: OK (rc=0, 5.7s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-06 21:45:05,352 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 16 teams today (2026-07-06)
2026-07-06 21:45:05,677 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 32 rows for 2026-07-06
2026-07-06 21:45:05,677 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 32 assignments for 2026-07-06
2026-07-06 21:45:05,677 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-06 21:45:05,979 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 8 unique games for season=2026-regular
2026-07-06 21:45:05,979 | INFO | mlb_pipeline.crawler_statsapi | Upserted 8 rows into raw.mlb_games
2026-07-06 21:45:05,994 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6198 completed games, 6188 already done, 1 to fetch
2026-07-06 21:45:06,605 | INFO | mlb_pipeline.crawler_statsapi |   Progress: 1 / 1 games fetched
2026-07-06 21:45:06,667 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=1, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-06 21:45:09,509 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-07. Catching up from 2026-07-08 to 2026-07-05
2026-07-06 21:45:09,509 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-06 window=2026-07-06T04:00:00Z..2026-07-07T04:00:00Z
2026-07-06 21:45:10,306 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-06 | events=3 | credits_remaining=92014
2026-07-06 21:45:10,337 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-07 window=2026-07-07T04:00:00Z..2026-07-08T04:00:00Z
2026-07-06 21:45:10,946 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-07 | events=15 | credits_remaining=92012
2026-07-06 21:45:10,946 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=92012
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-06 21:45:13,352 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-07. Catching up from 2026-07-08 to 2026-07-05
2026-07-06 21:45:14,149 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 3 events (game_date=2026-07-06, as_of=2026-07-06)
2026-07-06 21:45:14,743 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-06 event=d1ba74caa5492c9e300650c9ba7bebab (Arizona Diamondbacks@San Diego Padres) | credits=92009
2026-07-06 21:45:15,730 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-06 event=92bad434cf32ec6dd5e2b856708a6596 (Toronto Blue Jays@San Francisco Giants) | credits=92005
2026-07-06 21:45:16,552 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-06 event=ce47831dfdb0f68ba311fd79cc19d68a (Colorado Rockies@Los Angeles Dodgers) | credits=92000
2026-07-06 21:45:17,424 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-07-07, as_of=2026-07-06)
2026-07-06 21:45:18,149 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-06 event=7b7ae2c411b9b8122144d502c0bd1757 (Philadelphia Phillies@Cincinnati Reds) | credits=92000
2026-07-06 21:45:18,425 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=92000
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-06 21:45:21,081 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1542 rows into odds.mlb_game_lines (live odds).
2026-07-06 21:45:21,081 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-06
2026-07-06 21:45:21,081 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-06 21:45:21,081 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-07T03:15:21.081277+00:00.
2026-07-06 21:45:21,081 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-05
2026-07-06 21:45:22,393 | INFO | mlb_pipeline.parse_oddsapi | Upserted 221 rows into odds.mlb_player_prop_lines.
2026-07-06 21:45:22,393 | INFO | mlb_pipeline.parse_oddsapi | Processed 221 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-06 21:45:22,393 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 4100,
  "run_ids": "all",
  "date_from": "2026-07-06",
  "date_to": "2026-07-06",
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
  "replay_rows": 17412,
  "examples": 17412
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 197209,
  "graded_rows": 176779,
  "valid_clv_rows": 163453,
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
  "active_rows": 1326,
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
  "rows": 20716,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 175953,
  "bucket_count": 140,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=72542
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 196308,
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
    "raw_api_one_sided": 17143
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 214-141 (60.3%) ROI: +15.1% | Total: 71-65 (52.2%) ROI: -0.3%
MLB CLV Run Line: beat close 2/63 (3%) avg CLV=+0.10 runs | CLV Total avg=+0.07 runs
MLB Price CLV Run Line: 61 bets  avg=+0.51%
```

**stderr (tail)**
```
2026-07-06 21:55:38,430 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 1 rows
2026-07-06 21:55:38,430 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1 MLB game outcome rows
2026-07-06 21:55:38,446 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-06 21:55:42,221 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 208 prop rows
2026-07-06 21:55:42,252 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 208 MLB prop outcome rows
2026-07-06 21:55:42,392 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-06 21:55:42,392 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-06 21:55:42,486 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 2 game model-pick ledger rows
2026-07-06 21:55:42,946 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 29 prop model-pick ledger rows
2026-07-06 21:55:42,961 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 31 MLB model-pick ledger rows
```

### Grade daily forecast ledger

- rc: 0

**stdout (tail)**
```
{
  "schema_ready": true,
  "graded": 282
}
```

### Daily forecast projection audit

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 4616,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_daily_forecast_projection_audit_latest.md"
}
```

### Prop micro promotion evaluation

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "micro_ready_count": 0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_promotion_evaluation_latest.md"
}
```

### Forecast repair error decomposition

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "active_source": "prospective_ledger",
  "prospective_rows": 2863,
  "historical_rows": 20015,
  "tb_repair": "train_gated_direct_player_game_tb_head",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_forecast_repair_error_latest.md"
}
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-07-06T23:56:37-04:00
Range: 2026-06-23 to 2026-07-06

## Collection Status

**COLLECTING**

- Clean shadow slates: 0 / 10
- Additional clean slates needed: 10
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 90.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 2.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-07-06 | no | 2020 | 1391 | 4100 | 4100 | 38682 | 3580 | 87.3% | 0.0% | 2.8% | 3 | 31 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-07-05 | no | 4596 | 2708 | 5782 | 5782 | 34519 | 4954 | 85.7% | 0.0% | 1.3% | 3 | 20 | valid_close_coverage<0.90 |
| 2026-07-04 | no | 4640 | 2783 | 7530 | 7530 | 43937 | 6333 | 84.1% | 0.0% | 4.2% | 3 | 20 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-07-03 | no | 3879 | 2026 | 6784 | 6784 | 39279 | 5921 | 87.3% | 0.0% | 2.0% | 3 | 19 | valid_close_coverage<0.90 |
| 2026-07-02 | no | 2683 | 1678 | 4505 | 4505 | 22054 | 4063 | 90.2% | 0.0% | 2.1% | 3 | 18 | stale_close_rate>0.02 |
| 2026-07-01 | no | 4174 | 2582 | 6538 | 6538 | 25183 | 5254 | 80.4% | 0.0% | 3.1% | 3 | 15 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-06-30 | no | 4519 | 2785 | 8060 | 8060 | 41133 | 6911 | 85.7% | 0.0% | 3.0% | 3 | 18 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-06-29 | no | 3883 | 2282 | 6912 | 6912 | 35281 | 5954 | 86.1% | 0.0% | 3.0% | 3 | 18 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-06-28 | no | 4600 | 2631 | 5935 | 5935 | 18826 | 4885 | 82.3% | 0.0% | 10.7% | 3 | 14 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-06-27 | no | 4389 | 2875 | 7032 | 7032 | 28903 | 6054 | 86.1% | 0.0% | 2.5% | 3 | 15 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-06-26 | no | 4099 | 2969 | 8013 | 8013 | 34632 | 6866 | 85.7% | 0.0% | 2.8% | 3 | 14 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-06-25 | no | 2473 | 1660 | 2285 | 2285 | 10693 | 1810 | 79.2% | 0.0% | 9.4% | 2 | 10 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-06-24 | no | 4567 | 2646 | 4697 | 4697 | 21191 | 3644 | 77.6% | 0.0% | 5.2% | 2 | 11 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-06-23 | no | 4342 | 2721 | 4119 | 4119 | 28669 | 2350 | 57.1% | 0.0% | 5.5% | 2 | 14 | close_capture_coverage<0.90, valid_close_coverage<0.90, stale_close_rate>0.02 |
```

### End-of-slate prop close diagnostic

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-06",
  "evaluation_status": "provisional",
  "strict_clean_slate": false,
  "valid_close_coverage": 0.8772845953002611,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_end_of_slate_close_latest.md"
}
```

### Frozen-release five-date checkpoint

- rc: 0

**stdout (tail)**
```
{
  "status": "collecting",
  "hitter_completed_dates": 0,
  "pitcher_completed_dates": 2,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_five_date_checkpoint_latest.md"
}
```

### Grade shadow prop replay

- rc: 0

**stdout (tail)**
```
{
  "graded_rows": 562,
  "run_ids": "all_pending",
  "regrade": false
}
```
