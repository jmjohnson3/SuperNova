# SuperNovaBets MLB Daily Run (2026-06-15 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 7.9s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 7.6s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 6.6s)
- **Refresh prop replay CLV**: OK (rc=0, 16.3s)
- **Build prop market training table**: OK (rc=0, 337.3s)
- **Prop walk-forward accuracy report**: OK (rc=0, 134.0s)
- **Prop shadow selector report**: OK (rc=0, 66.1s)
- **Prop miss diagnostic report**: FAIL (rc=124, 180.0s)
- **Prop bucket repair report**: OK (rc=0, 48.9s)
- **TB prop repair report**: OK (rc=0, 65.5s)
- **Prop target quality report**: OK (rc=0, 16.6s)
- **Grade outcomes + ledgers**: OK (rc=0, 11.0s)
- **Prop snapshot coverage report**: OK (rc=0, 46.9s)
- **Grade shadow prop replay**: OK (rc=0, 19.2s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-15 21:45:32,736 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-16. Catching up from 2026-06-17 to 2026-06-14
2026-06-15 21:45:32,736 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-15 window=2026-06-15T04:00:00Z..2026-06-16T04:00:00Z
2026-06-15 21:45:33,567 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-15 | events=3 | credits_remaining=80067
2026-06-15 21:45:33,842 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-16 window=2026-06-16T04:00:00Z..2026-06-17T04:00:00Z
2026-06-15 21:45:34,381 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-16 | events=15 | credits_remaining=80065
2026-06-15 21:45:34,444 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=80065
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-15 21:45:36,965 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-16. Catching up from 2026-06-17 to 2026-06-14
2026-06-15 21:45:37,550 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 3 events (game_date=2026-06-15, as_of=2026-06-15)
2026-06-15 21:45:38,220 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-15 event=38a1ef501107d0410c8595f25612567d (Los Angeles Angels@Arizona Diamondbacks) | credits=80062
2026-06-15 21:45:39,152 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-15 event=8e54fa4eee405d90f575942ad15b1d52 (Pittsburgh Pirates@Athletics) | credits=80057
2026-06-15 21:45:40,012 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-15 event=da125dfa2d7fe97d5836e50dd394db59 (Tampa Bay Rays@Los Angeles Dodgers) | credits=80052
2026-06-15 21:45:40,870 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-06-16, as_of=2026-06-15)
2026-06-15 21:45:41,741 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-15 event=91665afcba863c3c8347ba9ee65ac461 (Toronto Blue Jays@Boston Red Sox) | credits=80052
2026-06-15 21:45:42,024 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=80052
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-15 21:45:45,587 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1432 rows into odds.mlb_game_lines (live odds).
2026-06-15 21:45:45,598 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-15
2026-06-15 21:45:45,598 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-15 21:45:45,599 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-16T03:15:45.598928+00:00.
2026-06-15 21:45:45,710 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-14
2026-06-15 21:45:48,624 | INFO | mlb_pipeline.parse_oddsapi | Upserted 304 rows into odds.mlb_player_prop_lines.
2026-06-15 21:45:48,624 | INFO | mlb_pipeline.parse_oddsapi | Processed 304 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-15 21:45:48,625 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 5873,
  "run_ids": "all",
  "date_from": "2026-06-15",
  "date_to": "2026-06-15",
  "include_graded": true,
  "only_missing": false
}
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 71659,
  "replay_rows": 71659,
  "examples": 71659
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 72982,
  "graded_rows": 60407,
  "valid_clv_rows": 59132,
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
  "active_rows": 2137,
  "real_candidate_rows": 0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_shadow_selector_latest.md"
}
```

### Prop miss diagnostic report

- rc: 124

**stderr (tail)**
```
Timeout after 180s
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 59581,
  "bucket_count": 131,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=23540
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 72081,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 192-128 (60.0%) ROI: +14.5% | Total: 61-57 (51.7%) ROI: -1.3%
MLB CLV Run Line: beat close 2/29 (7%) avg CLV=+0.21 runs | CLV Total avg=+0.02 runs
MLB Price CLV Run Line: 27 bets  avg=+0.53%
```

**stderr (tail)**
```
2026-06-15 22:00:23,441 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 28 pending predictions.
2026-06-15 22:00:23,441 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-15 22:00:23,479 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-15 22:00:25,062 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-15 22:00:25,084 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-15 22:00:25,565 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-15 22:00:25,566 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-15 22:00:26,012 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-15 22:00:26,991 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-15 22:00:26,997 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-16T00:01:12-04:00
Range: 2026-06-03 to 2026-06-16

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
| 2026-06-16 | no | 0 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, valid_side_locks<100, valid_clv_coverage<0.25, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.05, no_close_snapshot_time |
| 2026-06-15 | yes | 2411 | 1736 | 5873 | 5873 | 28077 | 5221 | 88.9% | 0.0% | 2.9% | 3 | 18 |  |
| 2026-06-14 | yes | 4648 | 2741 | 6056 | 6056 | 24201 | 5004 | 82.6% | 0.0% | 3.4% | 3 | 16 |  |
| 2026-06-13 | yes | 4564 | 2780 | 6581 | 6581 | 37746 | 5656 | 85.9% | 0.0% | 1.2% | 3 | 19 |  |
| 2026-06-12 | yes | 4438 | 2804 | 8682 | 8682 | 40945 | 7706 | 88.8% | 0.0% | 2.5% | 3 | 17 |  |
| 2026-06-11 | no | 2452 | 1703 | 2202 | 2202 | 14587 | 1879 | 85.3% | 0.0% | 6.7% | 2 | 15 | stale_close_rate>0.05 |
| 2026-06-10 | yes | 4550 | 3350 | 8414 | 8414 | 40087 | 7198 | 85.5% | 0.0% | 4.5% | 3 | 18 |  |
| 2026-06-09 | yes | 4402 | 3237 | 6566 | 6566 | 40669 | 5702 | 86.8% | 0.0% | 3.5% | 2 | 16 |  |
| 2026-06-08 | yes | 2447 | 1705 | 3589 | 3589 | 27069 | 3093 | 86.2% | 0.0% | 3.1% | 2 | 19 |  |
| 2026-06-07 | yes | 4496 | 3136 | 6521 | 6521 | 30276 | 5638 | 86.5% | 0.0% | 1.0% | 3 | 19 |  |
| 2026-06-06 | yes | 4345 | 3180 | 4562 | 4562 | 38898 | 4017 | 88.1% | 0.0% | 2.2% | 2 | 19 |  |
| 2026-06-05 | yes | 4520 | 3154 | 6699 | 6699 | 57725 | 5816 | 86.8% | 0.0% | 4.2% | 2 | 22 |  |
| 2026-06-04 | no | 2505 | 1945 | 2348 | 2348 | 7269 | 1640 | 69.8% | 0.0% | 13.7% | 2 | 10 | stale_close_rate>0.05 |
| 2026-06-03 | no | 3532 | 3089 | 3331 | 3331 | 3089 | 0 | 0.0% | 0.0% | 100.0% | 1 | 1 | valid_side_locks<100, valid_clv_coverage<0.25, stale_close_rate>0.05 |
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
