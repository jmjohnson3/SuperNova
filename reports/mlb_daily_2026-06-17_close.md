# SuperNovaBets MLB Daily Run (2026-06-17 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 4.8s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 6.8s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 8.0s)
- **Refresh prop replay CLV**: OK (rc=0, 16.2s)
- **Build prop market training table**: OK (rc=0, 62.4s)
- **Prop walk-forward accuracy report**: OK (rc=0, 145.4s)
- **Prop shadow selector report**: OK (rc=0, 94.4s)
- **Prop miss diagnostic report**: OK (rc=0, 18.7s)
- **Prop bucket repair report**: OK (rc=0, 8.5s)
- **TB prop repair report**: OK (rc=0, 6.6s)
- **Prop target quality report**: OK (rc=0, 14.0s)
- **Grade outcomes + ledgers**: OK (rc=0, 5.5s)
- **Prop snapshot coverage report**: OK (rc=0, 23.8s)
- **Grade shadow prop replay**: OK (rc=0, 3.9s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-17 21:45:06,208 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-18. Catching up from 2026-06-19 to 2026-06-16
2026-06-17 21:45:06,208 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-17 window=2026-06-17T04:00:00Z..2026-06-18T04:00:00Z
2026-06-17 21:45:07,248 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-17 | events=2 | credits_remaining=77355
2026-06-17 21:45:07,508 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-18 window=2026-06-18T04:00:00Z..2026-06-19T04:00:00Z
2026-06-17 21:45:08,148 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-18 | events=9 | credits_remaining=77353
2026-06-17 21:45:08,269 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=77353
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-17 21:45:10,709 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-18. Catching up from 2026-06-19 to 2026-06-16
2026-06-17 21:45:11,349 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 2 events (game_date=2026-06-17, as_of=2026-06-17)
2026-06-17 21:45:12,325 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=ab9ae62c321b17a5d765118cc9bb3fbf (Pittsburgh Pirates@Athletics) | credits=77348
2026-06-17 21:45:13,329 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=1d049adfd45ba69776089a9846e60034 (Baltimore Orioles@Seattle Mariners) | credits=77345
2026-06-17 21:45:14,972 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 9 events (game_date=2026-06-18, as_of=2026-06-17)
2026-06-17 21:45:15,072 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=77345
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-17 21:45:19,012 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1432 rows into odds.mlb_game_lines (live odds).
2026-06-17 21:45:19,032 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-17
2026-06-17 21:45:19,032 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-17 21:45:19,032 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-18T03:15:19.032249+00:00.
2026-06-17 21:45:19,171 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-16
2026-06-17 21:45:23,052 | INFO | mlb_pipeline.parse_oddsapi | Upserted 184 rows into odds.mlb_player_prop_lines.
2026-06-17 21:45:23,052 | INFO | mlb_pipeline.parse_oddsapi | Processed 184 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-17 21:45:23,052 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 3707,
  "run_ids": "all",
  "date_from": "2026-06-17",
  "date_to": "2026-06-17",
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
  "replay_rows": 17842,
  "examples": 17842
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 84951,
  "graded_rows": 65898,
  "valid_clv_rows": 69553,
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
  "active_rows": 1305,
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
  "rows": 65072,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 65072,
  "bucket_count": 132,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=25882
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 84050,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 192-129 (59.8%) ROI: +14.2% | Total: 61-57 (51.7%) ROI: -1.3%
MLB CLV Run Line: beat close 2/30 (7%) avg CLV=+0.20 runs | CLV Total avg=+0.02 runs
MLB Price CLV Run Line: 28 bets  avg=+0.57%
```

**stderr (tail)**
```
2026-06-17 21:51:32,101 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 33 pending predictions.
2026-06-17 21:51:32,101 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-17 21:51:32,101 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-17 21:51:33,581 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-17 21:51:33,601 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-17 21:51:33,703 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-17 21:51:33,703 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-17 21:51:33,859 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-17 21:51:34,703 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-17 21:51:34,721 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-17T23:51:58-04:00
Range: 2026-06-04 to 2026-06-17

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
| 2026-06-17 | yes | 3761 | 0 | 3707 | 3707 | 31248 | 3328 | 89.8% | 0.0% | 3.4% | 2 | 18 |  |
| 2026-06-16 | yes | 4276 | 2600 | 8262 | 8262 | 38949 | 7125 | 86.2% | 0.0% | 1.8% | 3 | 17 |  |
| 2026-06-15 | yes | 2927 | 1891 | 5873 | 5873 | 28077 | 5221 | 88.9% | 0.0% | 2.9% | 3 | 18 |  |
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
