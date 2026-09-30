# SuperNovaBets MLB Daily Run (2026-07-05 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 4.2s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 4.1s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 4.6s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.7s)
- **Refresh prop replay CLV**: OK (rc=0, 13.5s)
- **Build prop market training table**: OK (rc=0, 66.0s)
- **Prop walk-forward accuracy report**: OK (rc=0, 369.8s)
- **Prop shadow selector report**: OK (rc=0, 3.9s)
- **Prop miss diagnostic report**: OK (rc=0, 46.4s)
- **Prop bucket repair report**: OK (rc=0, 17.3s)
- **TB prop repair report**: OK (rc=0, 10.0s)
- **Prop target quality report**: OK (rc=0, 30.2s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 8.9s)
- **Grade outcomes + ledgers**: OK (rc=0, 5.0s)
- **Grade daily forecast ledger**: OK (rc=0, 4.3s)
- **Daily forecast projection audit**: OK (rc=0, 9.4s)
- **Prop micro promotion evaluation**: OK (rc=0, 0.3s)
- **Forecast repair error decomposition**: OK (rc=0, 4.9s)
- **Prop snapshot coverage report**: OK (rc=0, 24.2s)
- **Grade shadow prop replay**: OK (rc=0, 4.3s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-05 21:45:05,711 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 30 teams today (2026-07-05)
2026-07-05 21:45:06,162 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 60 rows for 2026-07-05
2026-07-05 21:45:06,162 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 60 assignments for 2026-07-05
2026-07-05 21:45:06,177 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-05 21:45:06,523 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 15 unique games for season=2026-regular
2026-07-05 21:45:06,695 | INFO | mlb_pipeline.crawler_statsapi | Upserted 15 rows into raw.mlb_games
2026-07-05 21:45:06,820 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6192 completed games, 6183 already done, 0 to fetch
2026-07-05 21:45:06,961 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=0, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-05 21:45:09,591 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-06. Catching up from 2026-07-07 to 2026-07-04
2026-07-05 21:45:09,591 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-05 window=2026-07-05T04:00:00Z..2026-07-06T04:00:00Z
2026-07-05 21:45:10,278 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-05 | events=1 | credits_remaining=93419
2026-07-05 21:45:10,419 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-06 window=2026-07-06T04:00:00Z..2026-07-07T04:00:00Z
2026-07-05 21:45:10,997 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-06 | events=8 | credits_remaining=93417
2026-07-05 21:45:11,106 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=93417
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-05 21:45:13,572 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-06. Catching up from 2026-07-07 to 2026-07-04
2026-07-05 21:45:14,141 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 1 events (game_date=2026-07-05, as_of=2026-07-05)
2026-07-05 21:45:14,735 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-05 event=0351ac13f45c028104c1b860422817ef (Boston Red Sox@Los Angeles Angels) | credits=93414
2026-07-05 21:45:15,608 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 8 events (game_date=2026-07-06, as_of=2026-07-05)
2026-07-05 21:45:15,670 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=93414
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-05 21:45:19,008 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1522 rows into odds.mlb_game_lines (live odds).
2026-07-05 21:45:19,102 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-05
2026-07-05 21:45:19,102 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-05 21:45:19,102 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-06T03:15:19.102662+00:00.
2026-07-05 21:45:19,212 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-04
2026-07-05 21:45:20,337 | INFO | mlb_pipeline.parse_oddsapi | Upserted 89 rows into odds.mlb_player_prop_lines.
2026-07-05 21:45:20,337 | INFO | mlb_pipeline.parse_oddsapi | Processed 89 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-05 21:45:20,337 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 5782,
  "run_ids": "all",
  "date_from": "2026-07-05",
  "date_to": "2026-07-05",
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
  "replay_rows": 20096,
  "examples": 20096
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 193109,
  "graded_rows": 174329,
  "valid_clv_rows": 159873,
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
  "active_rows": 573,
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
  "rows": 20556,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 173503,
  "bucket_count": 140,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=71498
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 192208,
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
    "raw_api_one_sided": 17540
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 213-141 (60.2%) ROI: +14.9% | Total: 71-65 (52.2%) ROI: -0.3%
MLB CLV Run Line: beat close 2/62 (3%) avg CLV=+0.10 runs | CLV Total avg=+0.07 runs
MLB Price CLV Run Line: 60 bets  avg=+0.55%
```

**stderr (tail)**
```
2026-07-05 21:54:49,446 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 23 pending predictions.
2026-07-05 21:54:49,446 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-07-05 21:54:49,446 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-05 21:54:50,821 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-07-05 21:54:50,852 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-07-05 21:54:50,993 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-05 21:54:50,993 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-05 21:54:51,040 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-07-05 21:54:51,305 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-07-05 21:54:51,321 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Grade daily forecast ledger

- rc: 0

**stdout (tail)**
```
{
  "schema_ready": true,
  "graded": 12
}
```

### Daily forecast projection audit

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 4061,
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
  "prospective_rows": 2543,
  "historical_rows": 19182,
  "tb_repair": "train_gated_direct_player_game_tb_head",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_forecast_repair_error_latest.md"
}
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-07-05T23:55:34-04:00
Range: 2026-06-22 to 2026-07-05

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
| 2026-07-05 | no | 4337 | 2584 | 5782 | 5782 | 34519 | 4954 | 85.7% | 0.0% | 1.3% | 3 | 20 | valid_close_coverage<0.90 |
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
| 2026-06-22 | no | 3807 | 2394 | 7540 | 7540 | 32994 | 6610 | 87.7% | 0.0% | 3.6% | 3 | 17 | valid_close_coverage<0.90, stale_close_rate>0.02 |
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
