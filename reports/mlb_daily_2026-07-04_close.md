# SuperNovaBets MLB Daily Run (2026-07-04 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 4.8s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 5.6s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 11.2s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 6.2s)
- **Refresh prop replay CLV**: OK (rc=0, 24.6s)
- **Build prop market training table**: OK (rc=0, 53.3s)
- **Prop walk-forward accuracy report**: OK (rc=0, 380.8s)
- **Prop shadow selector report**: OK (rc=0, 6.0s)
- **Prop miss diagnostic report**: OK (rc=0, 79.1s)
- **Prop bucket repair report**: OK (rc=0, 17.2s)
- **TB prop repair report**: OK (rc=0, 11.3s)
- **Prop target quality report**: OK (rc=0, 30.7s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 10.5s)
- **Grade outcomes + ledgers**: OK (rc=0, 12.0s)
- **Grade daily forecast ledger**: OK (rc=0, 4.9s)
- **Daily forecast projection audit**: OK (rc=0, 8.2s)
- **Prop micro promotion evaluation**: OK (rc=0, 0.3s)
- **Forecast repair error decomposition**: OK (rc=0, 6.1s)
- **Prop snapshot coverage report**: OK (rc=0, 39.2s)
- **Grade shadow prop replay**: OK (rc=0, 7.2s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-04 21:45:05,871 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 30 teams today (2026-07-04)
2026-07-04 21:45:06,233 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 60 rows for 2026-07-04
2026-07-04 21:45:06,233 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 60 assignments for 2026-07-04
2026-07-04 21:45:06,233 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-04 21:45:06,602 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 15 unique games for season=2026-regular
2026-07-04 21:45:06,649 | INFO | mlb_pipeline.crawler_statsapi | Upserted 15 rows into raw.mlb_games
2026-07-04 21:45:06,664 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6172 completed games, 6161 already done, 2 to fetch
2026-07-04 21:45:08,034 | INFO | mlb_pipeline.crawler_statsapi |   Progress: 2 / 2 games fetched
2026-07-04 21:45:08,112 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=2, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-04 21:45:11,355 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-05. Catching up from 2026-07-06 to 2026-07-03
2026-07-04 21:45:11,355 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-04 window=2026-07-04T04:00:00Z..2026-07-05T04:00:00Z
2026-07-04 21:45:12,785 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-04 | events=6 | credits_remaining=94812
2026-07-04 21:45:12,800 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-05 window=2026-07-05T04:00:00Z..2026-07-06T04:00:00Z
2026-07-04 21:45:13,738 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-05 | events=14 | credits_remaining=94810
2026-07-04 21:45:13,738 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=94810
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-04 21:45:16,191 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-05. Catching up from 2026-07-06 to 2026-07-03
2026-07-04 21:45:17,556 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 6 events (game_date=2026-07-04, as_of=2026-07-04)
2026-07-04 21:45:18,150 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-04 event=01abcc9d9a9408b8be4a4acc7dc53ec5 (San Francisco Giants@Colorado Rockies) | credits=94807
2026-07-04 21:45:19,020 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-04 event=ee9972cacf647720c50225d0f8004a91 (St. Louis Cardinals@Chicago Cubs) | credits=94802
2026-07-04 21:45:20,347 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-04 event=56eb72104fb4612dbd0953df70201727 (Boston Red Sox@Los Angeles Angels) | credits=94796
2026-07-04 21:45:21,784 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-04 event=a523c1ba668649088416935d2bc9a573 (Milwaukee Brewers@Arizona Diamondbacks) | credits=94791
2026-07-04 21:45:22,673 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-04 event=33b4c5f526bcc56b10bb39d97a8b942d (Miami Marlins@Athletics) | credits=94786
2026-07-04 21:45:23,543 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-04 event=937c0d8ee53bf0a6f8f1e472832b86c5 (San Diego Padres@Los Angeles Dodgers) | credits=94780
2026-07-04 21:45:24,789 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 14 events (game_date=2026-07-05, as_of=2026-07-04)
2026-07-04 21:45:24,836 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=94780
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-04 21:45:28,111 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1532 rows into odds.mlb_game_lines (live odds).
2026-07-04 21:45:28,111 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-04
2026-07-04 21:45:28,111 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-04 21:45:28,111 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-05T03:15:28.111985+00:00.
2026-07-04 21:45:28,127 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-03
2026-07-04 21:45:31,169 | INFO | mlb_pipeline.parse_oddsapi | Upserted 433 rows into odds.mlb_player_prop_lines.
2026-07-04 21:45:31,169 | INFO | mlb_pipeline.parse_oddsapi | Processed 433 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-04 21:45:31,169 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 7530,
  "run_ids": "all",
  "date_from": "2026-07-04",
  "date_to": "2026-07-04",
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
  "replay_rows": 18819,
  "examples": 18819
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 187327,
  "graded_rows": 165557,
  "valid_clv_rows": 154919,
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
  "active_rows": 2139,
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
  "rows": 19849,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 164731,
  "bucket_count": 139,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=67628
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 186426,
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
    "raw_api_one_sided": 17516
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 212-139 (60.4%) ROI: +15.3% | Total: 69-64 (51.9%) ROI: -1.0%
MLB CLV Run Line: beat close 2/59 (3%) avg CLV=+0.10 runs | CLV Total avg=+0.06 runs
MLB Price CLV Run Line: 57 bets  avg=+0.58%
```

**stderr (tail)**
```
2026-07-04 21:55:47,986 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 2 rows
2026-07-04 21:55:47,986 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 2 MLB game outcome rows
2026-07-04 21:55:47,986 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-04 21:55:55,689 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 386 prop rows
2026-07-04 21:55:55,721 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 386 MLB prop outcome rows
2026-07-04 21:55:55,846 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-04 21:55:55,861 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-04 21:55:56,049 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 4 game model-pick ledger rows
2026-07-04 21:55:56,611 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 73 prop model-pick ledger rows
2026-07-04 21:55:56,611 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 77 MLB model-pick ledger rows
```

### Grade daily forecast ledger

- rc: 0

**stdout (tail)**
```
{
  "schema_ready": true,
  "graded": 564
}
```

### Daily forecast projection audit

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 2320,
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
  "prospective_rows": 1545,
  "historical_rows": 18213,
  "tb_repair": "train_gated_direct_player_game_tb_head",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_forecast_repair_error_latest.md"
}
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-07-04T23:56:55-04:00
Range: 2026-06-21 to 2026-07-04

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
| 2026-07-04 | yes | 3771 | 2627 | 7530 | 7530 | 43937 | 6333 | 84.1% | 0.0% | 4.2% | 3 | 20 |  |
| 2026-07-03 | yes | 3879 | 2026 | 6784 | 6784 | 39279 | 5921 | 87.3% | 0.0% | 2.0% | 3 | 19 |  |
| 2026-07-02 | yes | 2683 | 1678 | 4505 | 4505 | 22054 | 4063 | 90.2% | 0.0% | 2.1% | 3 | 18 |  |
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
```

### Grade shadow prop replay

- rc: 0

**stdout (tail)**
```
{
  "graded_rows": 1136,
  "run_ids": "all_pending",
  "regrade": false
}
```
