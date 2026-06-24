# SuperNovaBets MLB Daily Run (2026-06-20 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 9.6s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 12.7s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 50.3s)
- **Refresh prop replay CLV**: OK (rc=0, 28.3s)
- **Build prop market training table**: OK (rc=0, 100.9s)
- **Prop walk-forward accuracy report**: OK (rc=0, 260.4s)
- **Prop shadow selector report**: OK (rc=0, 150.2s)
- **Prop miss diagnostic report**: FAIL (rc=124, 185.2s)
- **Prop bucket repair report**: OK (rc=0, 20.0s)
- **TB prop repair report**: OK (rc=0, 29.8s)
- **Prop target quality report**: OK (rc=0, 51.7s)
- **Grade outcomes + ledgers**: OK (rc=0, 24.1s)
- **Prop snapshot coverage report**: OK (rc=0, 47.5s)
- **Grade shadow prop replay**: OK (rc=0, 16.7s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-20 21:45:30,036 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-21. Catching up from 2026-06-22 to 2026-06-19
2026-06-20 21:45:30,074 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-20 window=2026-06-20T04:00:00Z..2026-06-21T04:00:00Z
2026-06-20 21:45:31,403 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-20 | events=5 | credits_remaining=74108
2026-06-20 21:45:32,008 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-21 window=2026-06-21T04:00:00Z..2026-06-22T04:00:00Z
2026-06-20 21:45:32,599 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-21 | events=14 | credits_remaining=74106
2026-06-20 21:45:32,609 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=74106
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-20 21:45:36,385 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-21. Catching up from 2026-06-22 to 2026-06-19
2026-06-20 21:45:37,728 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 5 events (game_date=2026-06-20, as_of=2026-06-20)
2026-06-20 21:45:38,360 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-20 event=4bb2c510f37ac3221423f6fe96d81822 (Pittsburgh Pirates@Colorado Rockies) | credits=74104
2026-06-20 21:45:39,510 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-20 event=11f0aee64edb29086a8ca3368f0fe44f (Los Angeles Angels@Athletics) | credits=74098
2026-06-20 21:45:40,356 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-20 event=bc0eb3e2abd5d0ddb3e22dc71abf652a (Minnesota Twins@Arizona Diamondbacks) | credits=74092
2026-06-20 21:45:41,221 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-20 event=8e059160b850333f99e5414be7fede74 (Baltimore Orioles@Los Angeles Dodgers) | credits=74087
2026-06-20 21:45:42,149 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-20 event=c17cdf3ec5c9c493c782139a03e71bcc (Boston Red Sox@Seattle Mariners) | credits=74084
2026-06-20 21:45:43,017 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 14 events (game_date=2026-06-21, as_of=2026-06-20)
2026-06-20 21:45:44,094 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-20 event=af71fc71bb598645a13d40236b310307 (San Diego Padres@Texas Rangers) | credits=74084
2026-06-20 21:45:44,385 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=74084
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-20 21:45:52,357 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1462 rows into odds.mlb_game_lines (live odds).
2026-06-20 21:45:52,392 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-20
2026-06-20 21:45:52,397 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-20 21:45:52,397 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-21T03:15:52.397452+00:00.
2026-06-20 21:45:52,616 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-19
2026-06-20 21:46:10,763 | INFO | mlb_pipeline.parse_oddsapi | Upserted 473 rows into odds.mlb_player_prop_lines.
2026-06-20 21:46:10,764 | INFO | mlb_pipeline.parse_oddsapi | Processed 473 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-20 21:46:10,764 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 6337,
  "run_ids": "all",
  "date_from": "2026-06-20",
  "date_to": "2026-06-20",
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
  "replay_rows": 16926,
  "examples": 16926
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 101877,
  "graded_rows": 86454,
  "valid_clv_rows": 83825,
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
  "active_rows": 1324,
  "real_candidate_rows": 0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_shadow_selector_latest.md"
}
```

### Prop miss diagnostic report

- rc: 124

**stderr (tail)**
```
Timed out after 180s; killed process tree rooted at PID 288
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 85628,
  "bucket_count": 134,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=34508
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 100976,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 194-129 (60.1%) ROI: +14.7% | Total: 65-59 (52.4%) ROI: +0.1%
MLB CLV Run Line: beat close 2/32 (6%) avg CLV=+0.19 runs | CLV Total avg=+0.03 runs
MLB Price CLV Run Line: 30 bets  avg=+0.57%
```

**stderr (tail)**
```
2026-06-20 22:00:35,741 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 33 pending predictions.
2026-06-20 22:00:35,742 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-20 22:00:35,772 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-20 22:00:42,300 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-20 22:00:42,324 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-20 22:00:43,149 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-20 22:00:43,237 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-20 22:00:44,008 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-20 22:00:45,939 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-20 22:00:45,948 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-21T00:01:31-04:00
Range: 2026-06-08 to 2026-06-21

## Collection Status

**TARGET MET**

- Clean shadow slates: 11 / 10
- Additional clean slates needed: 0
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 25.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 5.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-21 | no | 0 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-06-20 | yes | 3657 | 2389 | 6337 | 6337 | 31532 | 5322 | 84.0% | 0.0% | 2.9% | 3 | 17 |  |
| 2026-06-19 | yes | 4181 | 2501 | 7929 | 7929 | 35937 | 7008 | 88.4% | 0.0% | 2.3% | 3 | 18 |  |
| 2026-06-18 | no | 2569 | 1592 | 2660 | 2660 | 14372 | 1994 | 75.0% | 0.0% | 12.2% | 2 | 15 | stale_close_rate>0.05 |
| 2026-06-17 | yes | 4267 | 380 | 3707 | 3707 | 31248 | 3328 | 89.8% | 0.0% | 3.4% | 2 | 18 |  |
| 2026-06-16 | yes | 4276 | 2600 | 8262 | 8262 | 38949 | 7125 | 86.2% | 0.0% | 1.7% | 3 | 17 |  |
| 2026-06-15 | yes | 2927 | 1891 | 5873 | 5873 | 28077 | 5221 | 88.9% | 0.0% | 2.6% | 3 | 18 |  |
| 2026-06-14 | yes | 4648 | 2741 | 6056 | 6056 | 24201 | 5004 | 82.6% | 0.0% | 3.2% | 3 | 16 |  |
| 2026-06-13 | yes | 4564 | 2780 | 6581 | 6581 | 37746 | 5656 | 85.9% | 0.0% | 1.0% | 3 | 19 |  |
| 2026-06-12 | yes | 4438 | 2804 | 8682 | 8682 | 40945 | 7706 | 88.8% | 0.0% | 2.5% | 3 | 17 |  |
| 2026-06-11 | no | 2452 | 1703 | 2202 | 2202 | 14587 | 1879 | 85.3% | 0.0% | 6.4% | 2 | 15 | stale_close_rate>0.05 |
| 2026-06-10 | yes | 4550 | 3350 | 8414 | 8414 | 40087 | 7198 | 85.5% | 0.0% | 4.4% | 3 | 18 |  |
| 2026-06-09 | yes | 4402 | 3237 | 6566 | 6566 | 40669 | 5702 | 86.8% | 0.0% | 3.2% | 2 | 16 |  |
| 2026-06-08 | yes | 2447 | 1705 | 3589 | 3589 | 27069 | 3093 | 86.2% | 0.0% | 2.8% | 2 | 19 |  |
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
