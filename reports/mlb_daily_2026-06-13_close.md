# SuperNovaBets MLB Daily Run (2026-06-13 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 6.5s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 11.5s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 7.2s)
- **Refresh prop replay CLV**: OK (rc=0, 15.7s)
- **Build prop market training table**: OK (rc=0, 159.7s)
- **Prop walk-forward accuracy report**: OK (rc=0, 115.1s)
- **Prop shadow selector report**: OK (rc=0, 31.4s)
- **Prop miss diagnostic report**: OK (rc=0, 28.6s)
- **Prop bucket repair report**: OK (rc=0, 7.4s)
- **TB prop repair report**: OK (rc=0, 6.6s)
- **Prop target quality report**: OK (rc=0, 11.8s)
- **Grade outcomes + ledgers**: OK (rc=0, 8.6s)
- **Prop snapshot coverage report**: OK (rc=0, 30.7s)
- **Grade shadow prop replay**: OK (rc=0, 14.1s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-13 21:45:09,746 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-14. Catching up from 2026-06-15 to 2026-06-12
2026-06-13 21:45:09,761 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-13 window=2026-06-13T04:00:00Z..2026-06-14T04:00:00Z
2026-06-13 21:45:10,604 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-13 | events=4 | credits_remaining=82161
2026-06-13 21:45:10,872 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-14 window=2026-06-14T04:00:00Z..2026-06-15T04:00:00Z
2026-06-13 21:45:11,515 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-14 | events=14 | credits_remaining=82159
2026-06-13 21:45:11,556 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=82159
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-13 21:45:15,603 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-14. Catching up from 2026-06-15 to 2026-06-12
2026-06-13 21:45:16,263 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 4 events (game_date=2026-06-13, as_of=2026-06-13)
2026-06-13 21:45:16,874 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-13 event=585fa0b6aa89841fa095b120643c7716 (Houston Astros@Kansas City Royals) | credits=82156
2026-06-13 21:45:17,860 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-13 event=197b2f8add16733591617503dd702d8f (Colorado Rockies@Athletics) | credits=82151
2026-06-13 21:45:18,792 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-13 event=d259bb716cd52554224d80732ac65232 (Chicago Cubs@San Francisco Giants) | credits=82146
2026-06-13 21:45:22,105 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-13 event=020b5bffb5a1d101be8c6d61b9e29572 (Tampa Bay Rays@Los Angeles Angels) | credits=82141
2026-06-13 21:45:22,937 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 14 events (game_date=2026-06-14, as_of=2026-06-13)
2026-06-13 21:45:23,048 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=82141
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-13 21:45:27,272 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1424 rows into odds.mlb_game_lines (live odds).
2026-06-13 21:45:27,284 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-13
2026-06-13 21:45:27,285 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-13 21:45:27,285 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-14T03:15:27.285362+00:00.
2026-06-13 21:45:27,392 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-12
2026-06-13 21:45:30,265 | INFO | mlb_pipeline.parse_oddsapi | Upserted 396 rows into odds.mlb_player_prop_lines.
2026-06-13 21:45:30,266 | INFO | mlb_pipeline.parse_oddsapi | Processed 396 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-13 21:45:30,266 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 6581,
  "run_ids": "all",
  "date_from": "2026-06-13",
  "date_to": "2026-06-13",
  "include_graded": true,
  "only_missing": false
}
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 60152,
  "replay_rows": 60152,
  "examples": 60152
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 61053,
  "graded_rows": 49224,
  "valid_clv_rows": 48907,
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
  "active_rows": 1041,
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
  "rows": 48398,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 48398,
  "bucket_count": 127,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=18674
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 60152,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 190-125 (60.3%) ROI: +15.2% | Total: 61-55 (52.6%) ROI: +0.4%
MLB CLV Run Line: beat close 2/24 (8%) avg CLV=+0.25 runs | CLV Total avg=-0.03 runs
MLB Price CLV Run Line: 22 bets  avg=+0.59%
```

**stderr (tail)**
```
2026-06-13 21:51:49,735 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 32 pending predictions.
2026-06-13 21:51:49,735 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-13 21:51:49,873 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-13 21:51:52,762 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-13 21:51:52,779 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-13 21:51:53,207 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-13 21:51:54,052 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-13 21:51:55,071 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-13 21:51:55,076 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-13T23:52:25-04:00
Range: 2026-05-31 to 2026-06-13

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
| 2026-06-13 | yes | 4061 | 2612 | 6581 | 6581 | 37746 | 5656 | 85.9% | 0.0% | 1.2% | 3 | 19 |  |
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
| 2026-06-02 | yes | 4467 | 3104 | 235 | 235 | 6442 | 200 | 85.1% | 0.0% | 0.0% | 1 | 12 |  |
| 2026-06-01 | no | 2668 | 0 | 208 | 208 | 5851 | 183 | 88.0% | 0.0% | 9.6% | 1 | 13 | stale_close_rate>0.05 |
| 2026-05-31 | no | 4599 | 0 | 214 | 214 | 9800 | 179 | 83.6% | 0.0% | 15.0% | 1 | 20 | stale_close_rate>0.05 |
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
