# SuperNovaBets MLB Daily Run (2026-07-08 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 4.9s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 4.5s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 6.6s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.1s)
- **Refresh prop replay CLV**: OK (rc=0, 25.0s)
- **Build prop market training table**: OK (rc=0, 67.6s)
- **Prop walk-forward accuracy report**: OK (rc=0, 411.7s)
- **Prop shadow selector report**: OK (rc=0, 11.1s)
- **Prop miss diagnostic report**: OK (rc=0, 78.7s)
- **Prop bucket repair report**: OK (rc=0, 19.0s)
- **TB prop repair report**: OK (rc=0, 11.6s)
- **Prop target quality report**: OK (rc=0, 34.9s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 9.7s)
- **Grade outcomes + ledgers**: OK (rc=0, 11.5s)
- **Grade daily forecast ledger**: OK (rc=0, 6.2s)
- **Daily forecast projection audit**: OK (rc=0, 9.4s)
- **Prop micro promotion evaluation**: OK (rc=0, 0.2s)
- **Forecast repair error decomposition**: OK (rc=0, 4.1s)
- **Prop snapshot coverage report**: OK (rc=0, 44.6s)
- **End-of-slate prop close diagnostic**: OK (rc=0, 4.3s)
- **Frozen-release five-date checkpoint**: OK (rc=0, 15.1s)
- **Grade shadow prop replay**: OK (rc=0, 7.0s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-08 21:45:05,047 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 30 teams today (2026-07-08)
2026-07-08 21:45:05,409 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 60 rows for 2026-07-08
2026-07-08 21:45:05,409 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 60 assignments for 2026-07-08
2026-07-08 21:45:05,409 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-08 21:45:05,763 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 15 unique games for season=2026-regular
2026-07-08 21:45:05,779 | INFO | mlb_pipeline.crawler_statsapi | Upserted 15 rows into raw.mlb_games
2026-07-08 21:45:05,795 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6230 completed games, 6219 already done, 2 to fetch
2026-07-08 21:45:06,995 | INFO | mlb_pipeline.crawler_statsapi |   Progress: 2 / 2 games fetched
2026-07-08 21:45:07,058 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=2, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-08 21:45:10,093 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-09. Catching up from 2026-07-10 to 2026-07-07
2026-07-08 21:45:10,093 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-08 window=2026-07-08T04:00:00Z..2026-07-09T04:00:00Z
2026-07-08 21:45:10,812 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-08 | events=2 | credits_remaining=86669
2026-07-08 21:45:10,843 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-09 window=2026-07-09T04:00:00Z..2026-07-10T04:00:00Z
2026-07-08 21:45:11,484 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-09 | events=12 | credits_remaining=86667
2026-07-08 21:45:11,515 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=86667
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-08 21:45:14,045 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-09. Catching up from 2026-07-10 to 2026-07-07
2026-07-08 21:45:14,624 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 2 events (game_date=2026-07-08, as_of=2026-07-08)
2026-07-08 21:45:15,312 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-08 event=8c0695707371420ef6b1cf89c2d1b262 (Arizona Diamondbacks@San Diego Padres) | credits=86661
2026-07-08 21:45:16,159 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-08 event=49de1a8b7d85007610fb077c920e206a (Colorado Rockies@Los Angeles Dodgers) | credits=86655
2026-07-08 21:45:16,963 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 12 events (game_date=2026-07-09, as_of=2026-07-08)
2026-07-08 21:45:17,838 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-08 event=be3485a7acdabede1a849330fcc226dc (New York Yankees@Tampa Bay Rays) | credits=86655
2026-07-08 21:45:18,140 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=86655
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-08 21:45:20,655 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1545 rows into odds.mlb_game_lines (live odds).
2026-07-08 21:45:20,655 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-08
2026-07-08 21:45:20,655 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-08 21:45:20,655 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-09T03:15:20.655968+00:00.
2026-07-08 21:45:20,655 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-07
2026-07-08 21:45:22,263 | INFO | mlb_pipeline.parse_oddsapi | Upserted 206 rows into odds.mlb_player_prop_lines.
2026-07-08 21:45:22,263 | INFO | mlb_pipeline.parse_oddsapi | Processed 206 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-08 21:45:22,263 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 7951,
  "run_ids": "all",
  "date_from": "2026-07-08",
  "date_to": "2026-07-08",
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
  "replay_rows": 20340,
  "examples": 20340
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 213449,
  "graded_rows": 191667,
  "valid_clv_rows": 177437,
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
  "active_rows": 2839,
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
  "rows": 21657,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 190841,
  "bucket_count": 144,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=78768
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 212548,
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
    "raw_api_one_sided": 17608
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 216-141 (60.5%) ROI: +15.5% | Total: 71-67 (51.4%) ROI: -1.8%
MLB CLV Run Line: beat close 2/65 (3%) avg CLV=+0.09 runs | CLV Total avg=+0.08 runs
MLB Price CLV Run Line: 63 bets  avg=+0.50%
```

**stderr (tail)**
```
2026-07-08 21:56:35,250 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 2 rows
2026-07-08 21:56:35,250 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 2 MLB game outcome rows
2026-07-08 21:56:35,265 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-08 21:56:42,218 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 306 prop rows
2026-07-08 21:56:42,249 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 306 MLB prop outcome rows
2026-07-08 21:56:42,359 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-08 21:56:42,374 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-08 21:56:42,515 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 4 game model-pick ledger rows
2026-07-08 21:56:43,030 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 45 prop model-pick ledger rows
2026-07-08 21:56:43,046 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 49 MLB model-pick ledger rows
```

### Grade daily forecast ledger

- rc: 0

**stdout (tail)**
```
{
  "schema_ready": true,
  "graded": 525
}
```

### Daily forecast projection audit

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 7251,
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
  "prospective_rows": 4405,
  "historical_rows": 21009,
  "tb_repair": "train_gated_direct_player_game_tb_head",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_forecast_repair_error_latest.md"
}
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-07-08T23:57:46-04:00
Range: 2026-06-25 to 2026-07-08

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
| 2026-07-08 | no | 3727 | 2579 | 7951 | 7951 | 73124 | 6833 | 85.9% | 0.0% | 3.0% | 3 | 32 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-07-07 | no | 4877 | 2689 | 8289 | 8289 | 75276 | 7151 | 86.3% | 0.0% | 3.8% | 3 | 31 | valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-07-06 | no | 2495 | 1469 | 4100 | 4100 | 38682 | 3580 | 87.3% | 0.0% | 2.8% | 3 | 31 | valid_close_coverage<0.90, stale_close_rate>0.02 |
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
```

### End-of-slate prop close diagnostic

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-08",
  "evaluation_status": "provisional",
  "strict_clean_slate": false,
  "valid_close_coverage": 0.8674332093337842,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_end_of_slate_close_latest.md"
}
```

### Frozen-release five-date checkpoint

- rc: 0

**stdout (tail)**
```
{
  "status": "collecting",
  "hitter_completed_dates": 1,
  "pitcher_completed_dates": 4,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_five_date_checkpoint_latest.md"
}
```

### Grade shadow prop replay

- rc: 0

**stdout (tail)**
```
{
  "graded_rows": 914,
  "run_ids": "all_pending",
  "regrade": false
}
```
