# SuperNovaBets MLB Daily Run (2026-07-26 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 4.3s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 4.9s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 7.5s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.8s)
- **Refresh prop replay CLV**: OK (rc=0, 28.8s)
- **Build prop market training table**: OK (rc=0, 153.8s)
- **Prop walk-forward accuracy report**: OK (rc=0, 9.1s)
- **Prop shadow selector report**: OK (rc=0, 10.2s)
- **Prop miss diagnostic report**: OK (rc=0, 138.5s)
- **Prop bucket repair report**: OK (rc=0, 114.3s)
- **TB prop repair report**: OK (rc=0, 43.3s)
- **Prop target quality report**: FAIL (rc=124, 215.5s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 30.6s)
- **Grade outcomes + ledgers**: OK (rc=0, 21.0s)
- **Grade daily forecast ledger**: OK (rc=0, 6.3s)
- **Daily forecast projection audit**: OK (rc=0, 29.7s)
- **Hitter live-vs-legacy forecast diff**: OK (rc=0, 27.5s)
- **Player-game bankroll proof**: OK (rc=0, 34.7s)
- **TB 1.5 line calibration**: OK (rc=0, 21.5s)
- **TB 1.5 close repair report**: OK (rc=0, 6.6s)
- **K-under repair report**: OK (rc=0, 4.4s)
- **DK K 4.5-6.0 under repair diagnostic**: OK (rc=0, 3.5s)
- **Exact-bucket CLV priors**: OK (rc=0, 14.7s)
- **Prop micro promotion evaluation**: OK (rc=0, 1.6s)
- **Prop micro gate sensitivity report**: OK (rc=0, 1.5s)
- **Prop micro bucket repair report**: OK (rc=0, 0.9s)
- **Prop trial candidate queue report**: OK (rc=0, 5.5s)
- **Prop drift guard diagnostic**: OK (rc=0, 52.8s)
- **Prop bettable-now scan**: OK (rc=0, 6.3s)
- **Lock micro projection ledger**: OK (rc=0, 10.4s)
- **Prop micro ledger report**: OK (rc=0, 3.6s)
- **Prop micro loss diagnostic**: OK (rc=0, 2.5s)
- **Prop micro probability calibrator**: OK (rc=0, 2.8s)
- **Prop post-gate candidate report**: OK (rc=0, 6.2s)
- **Prop layer promotion control**: OK (rc=0, 0.6s)
- **Forecast repair error decomposition**: OK (rc=0, 17.4s)
- **TB tail repair challenger**: OK (rc=0, 92.5s)
- **Prop snapshot coverage report**: FAIL (rc=124, 122.6s)
- **Real-money operational prop reports**: OK (rc=0, 146.9s)
- **Pitcher K-rate challenger diagnostic**: OK (rc=0, 74.2s)
- **End-of-slate prop close diagnostic**: OK (rc=0, 9.2s)
- **Frozen-release five-date checkpoint**: FAIL (rc=124, 190.2s)
- **Post-checkpoint micro promotion evaluation**: OK (rc=0, 19.2s)
- **Post-checkpoint micro gate sensitivity report**: OK (rc=0, 0.4s)
- **Post-checkpoint micro bucket repair report**: OK (rc=0, 0.4s)
- **Post-checkpoint prop trial candidate queue report**: OK (rc=0, 0.8s)
- **Post-checkpoint prop drift guard diagnostic**: OK (rc=0, 48.2s)
- **Post-checkpoint lock micro projection ledger**: OK (rc=0, 12.5s)
- **Post-checkpoint prop micro ledger report**: OK (rc=0, 5.3s)
- **Post-checkpoint prop micro probability calibrator**: OK (rc=0, 3.7s)
- **Post-checkpoint prop layer promotion control**: OK (rc=0, 0.8s)
- **Post-checkpoint real-money operational prop reports**: OK (rc=0, 55.8s)
- **Daily slate trust monitor**: FAIL (rc=124, 609.0s)
- **Grade shadow prop replay**: OK (rc=0, 215.4s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-26 17:45:20,798 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 30 teams today (2026-07-26)
2026-07-26 17:45:21,205 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 60 rows for 2026-07-26
2026-07-26 17:45:21,205 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 60 assignments for 2026-07-26
2026-07-26 17:45:21,220 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-26 17:45:21,552 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 15 unique games for season=2026-regular
2026-07-26 17:45:21,767 | INFO | mlb_pipeline.crawler_statsapi | Upserted 15 rows into raw.mlb_games
2026-07-26 17:45:21,830 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6430 completed games, 6421 already done, 0 to fetch
2026-07-26 17:45:21,986 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=0, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-26 17:45:25,408 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-27. Catching up from 2026-07-28 to 2026-07-25
2026-07-26 17:45:25,408 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-26 window=2026-07-26T04:00:00Z..2026-07-27T04:00:00Z
2026-07-26 17:45:26,080 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-26 | events=1 | credits_remaining=29078
2026-07-26 17:45:26,142 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-27 window=2026-07-27T04:00:00Z..2026-07-28T04:00:00Z
2026-07-26 17:45:26,799 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-27 | events=11 | credits_remaining=29076
2026-07-26 17:45:26,861 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=29076
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-26 17:45:29,454 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-27. Catching up from 2026-07-28 to 2026-07-25
2026-07-26 17:45:30,842 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 1 events (game_date=2026-07-26, as_of=2026-07-26)
2026-07-26 17:45:31,857 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-26 event=210e6566d3641ee245e8a099a1679244 (New York Yankees@Philadelphia Phillies) | credits=29070
2026-07-26 17:45:32,799 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 11 events (game_date=2026-07-27, as_of=2026-07-26)
2026-07-26 17:45:32,799 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=29070
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-26 17:45:37,642 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1593 rows into odds.mlb_game_lines (live odds).
2026-07-26 17:45:37,658 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-26
2026-07-26 17:45:37,658 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-26 17:45:37,658 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-26T23:15:37.658371+00:00.
2026-07-26 17:45:37,720 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-25
2026-07-26 17:45:39,190 | INFO | mlb_pipeline.parse_oddsapi | Upserted 172 rows into odds.mlb_player_prop_lines.
2026-07-26 17:45:39,190 | INFO | mlb_pipeline.parse_oddsapi | Processed 172 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-26 17:45:39,190 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 5175,
  "run_ids": "all",
  "markets": "all",
  "sides": "all",
  "bookmakers": "all",
  "line_buckets": "all",
  "limit": null,
  "batch_size": 100,
  "statement_timeout_ms": 30000,
  "lock_timeout_ms": 2000,
  "ensure_schema": false,
  "date_from": "2026-07-26",
  "date_to": "2026-07-26",
  "include_graded": true,
  "only_missing": false
}
```

**stderr (tail)**
```
2026-07-26 17:45:47,062 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=100 updated=100 skipped=0 last_id=318559
2026-07-26 17:45:48,015 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=200 updated=200 skipped=0 last_id=318659
2026-07-26 17:45:48,894 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=300 updated=300 skipped=0 last_id=318759
2026-07-26 17:45:49,570 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=400 updated=400 skipped=0 last_id=318859
2026-07-26 17:45:49,957 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=500 updated=500 skipped=0 last_id=318959
2026-07-26 17:45:50,720 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=600 updated=600 skipped=0 last_id=319059
2026-07-26 17:45:51,070 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=700 updated=700 skipped=0 last_id=319159
2026-07-26 17:45:51,376 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=800 updated=800 skipped=0 last_id=319259
2026-07-26 17:45:51,675 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=900 updated=900 skipped=0 last_id=319359
2026-07-26 17:45:51,958 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1000 updated=1000 skipped=0 last_id=319459
2026-07-26 17:45:52,362 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1100 updated=1100 skipped=0 last_id=319559
2026-07-26 17:45:52,659 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1200 updated=1200 skipped=0 last_id=319659
2026-07-26 17:45:52,938 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1300 updated=1300 skipped=0 last_id=319759
2026-07-26 17:45:53,252 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1400 updated=1400 skipped=0 last_id=319859
2026-07-26 17:45:53,533 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1500 updated=1500 skipped=0 last_id=319959
2026-07-26 17:45:53,923 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1600 updated=1600 skipped=0 last_id=320059
2026-07-26 17:45:54,862 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1700 updated=1700 skipped=0 last_id=320159
2026-07-26 17:45:55,158 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1800 updated=1800 skipped=0 last_id=320259
2026-07-26 17:45:56,113 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1900 updated=1900 skipped=0 last_id=320359
2026-07-26 17:45:56,546 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2000 updated=2000 skipped=0 last_id=320459
2026-07-26 17:45:57,431 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2100 updated=2100 skipped=0 last_id=320559
2026-07-26 17:45:57,792 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2200 updated=2200 skipped=0 last_id=320659
2026-07-26 17:45:58,119 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2300 updated=2300 skipped=0 last_id=320759
2026-07-26 17:45:58,432 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2400 updated=2400 skipped=0 last_id=320859
2026-07-26 17:45:58,851 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2500 updated=2500 skipped=0 last_id=320959
2026-07-26 17:45:59,166 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2600 updated=2600 skipped=0 last_id=321059
2026-07-26 17:46:00,079 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2700 updated=2700 skipped=0 last_id=321159
2026-07-26 17:46:00,764 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2800 updated=2800 skipped=0 last_id=321259
2026-07-26 17:46:01,049 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2900 updated=2900 skipped=0 last_id=321359
2026-07-26 17:46:01,455 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3000 updated=3000 skipped=0 last_id=321459
2026-07-26 17:46:01,821 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3100 updated=3100 skipped=0 last_id=321559
2026-07-26 17:46:02,081 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3200 updated=3200 skipped=0 last_id=321659
2026-07-26 17:46:02,396 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3300 updated=3300 skipped=0 last_id=321759
2026-07-26 17:46:02,627 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3400 updated=3400 skipped=0 last_id=321859
2026-07-26 17:46:02,876 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3500 updated=3500 skipped=0 last_id=321959
2026-07-26 17:46:03,173 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3600 updated=3600 skipped=0 last_id=322059
2026-07-26 17:46:03,455 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3700 updated=3700 skipped=0 last_id=322159
2026-07-26 17:46:03,799 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3800 updated=3800 skipped=0 last_id=322259
2026-07-26 17:46:04,096 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3900 updated=3900 skipped=0 last_id=322359
2026-07-26 17:46:04,359 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4000 updated=4000 skipped=0 last_id=322459
2026-07-26 17:46:04,643 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4100 updated=4100 skipped=0 last_id=322559
2026-07-26 17:46:04,924 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4200 updated=4200 skipped=0 last_id=322659
2026-07-26 17:46:05,204 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4300 updated=4300 skipped=0 last_id=322759
2026-07-26 17:46:05,580 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4400 updated=4400 skipped=0 last_id=322859
2026-07-26 17:46:05,931 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4500 updated=4500 skipped=0 last_id=322959
2026-07-26 17:46:06,236 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4600 updated=4600 skipped=0 last_id=323059
2026-07-26 17:46:06,517 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4700 updated=4700 skipped=0 last_id=323159
2026-07-26 17:46:06,767 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4800 updated=4800 skipped=0 last_id=323259
2026-07-26 17:46:07,036 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4900 updated=4900 skipped=0 last_id=323359
2026-07-26 17:46:07,330 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5000 updated=5000 skipped=0 last_id=323459
2026-07-26 17:46:07,726 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5100 updated=5100 skipped=0 last_id=323559
2026-07-26 17:46:07,965 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5175 updated=5175 skipped=0 last_id=323634
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 0,
  "replay_rows": 21167,
  "examples": 21167
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "reused_fresh_artifact",
  "artifact": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_walk_forward_accuracy_report.json",
  "age_minutes": 54.5,
  "rows": 259703,
  "graded_rows": 230123,
  "valid_clv_rows": 218563,
  "generated_at_utc": "2026-07-26T22:54:06+00:00"
}
```

### Prop shadow selector report

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "active_rows": 192,
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
  "rows": 24432,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 270306,
  "bucket_count": 155,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=112553
```

### Prop target quality report

- rc: 124

**stderr (tail)**
```
Timed out after 180s; killed process tree rooted at PID 3140
```

### FanDuel one-sided diagnostic

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "root_causes": {
    "raw_api_one_sided": 13380
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 231-157 (59.5%) ROI: +13.7% | Total: 72-69 (51.1%) ROI: -2.5%
MLB CLV Run Line: beat close 3/96 (3%) avg CLV=+0.09 runs | CLV Total avg=+0.10 runs
MLB Price CLV Run Line: 93 bets  avg=+0.42%
```

**stderr (tail)**
```
2026-07-26 17:58:19,517 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 28 pending predictions.
2026-07-26 17:58:19,517 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-07-26 17:58:19,552 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-26 17:58:22,996 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-07-26 17:58:23,018 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-07-26 17:58:23,376 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-26 17:58:23,376 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-26 17:58:23,595 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-07-26 17:58:24,381 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-07-26 17:58:24,403 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
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
  "rows": 29337,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_daily_forecast_projection_audit_latest.md"
}
```

### Hitter live-vs-legacy forecast diff

- rc: 0

**stdout (tail)**
```
{
  "rows": 5924,
  "phase": "day_pregame",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_hitter_live_vs_legacy_forecast_diff_latest.md"
}
```

### Player-game bankroll proof

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "allowed_stats": [],
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_player_game_bankroll_model_proof_latest.md"
}
```

### TB 1.5 line calibration

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 11132,
  "enabled_count": 6,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb15_line_calibration_latest.md"
}
```

### TB 1.5 close repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "focus_rows": 4711,
  "valid_close_coverage": 0.7960093398429208,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb15_close_repair_latest.md"
}
```

### K-under repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 2301,
  "micro_allowed_count": 1,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_k_under_repair_latest.md"
}
```

### DK K 4.5-6.0 under repair diagnostic

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 798,
  "roi": -0.04181591478696741,
  "clv_beat_rate": 0.47840531561461797,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_k_under_46_repair_diagnostic_latest.md"
}
```

### Exact-bucket CLV priors

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 56396,
  "micro_clv_confirmed_buckets": 4,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_exact_bucket_clv_priors_latest.md"
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

### Prop micro gate sensitivity report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "profiles": {
    "watch_only": 1,
    "micro_test": 0,
    "strict_micro": 0
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
}
```

### Prop micro bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "trial_ready_count": 0,
  "near_miss_count": 98,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_bucket_repair_latest.md"
}
```

### Prop trial candidate queue report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "trial_ready_count": 0,
  "near_miss_1_2_gate_count": 3,
  "proof_refresh_markets": [
    "pitcher_strikeouts",
    "batter_total_bases",
    "batter_hits"
  ],
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_trial_candidate_queue_latest.md"
}
```

### Prop drift guard diagnostic

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-26', 'micro_rows': 0, 'micro_bettable_now': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

### Prop bettable-now scan

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-26', 'approved_rows': 7, 'near_approved_rows': 2, 'bettable_now_before_cap': 0, 'bettable_now_inside_cap': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_bettable_now_scan.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bettable_now_scan_latest.md'}
```

### Lock micro projection ledger

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "game_date": "2026-07-26",
  "active_prediction_rows": 192,
  "locked_rows_attempted": 0,
  "ledger_before": [
    {
      "model_tier": "micro_projection",
      "rows": 4,
      "stake_usd": 4.0
    },
    {
      "model_tier": "watch",
      "rows": 103,
      "stake_usd": 0.0
    }
  ],
  "ledger_after": [
    {
      "model_tier": "micro_projection",
      "rows": 4,
      "stake_usd": 4.0
    },
    {
      "model_tier": "watch",
      "rows": 103,
      "stake_usd": 0.0
    }
  ],
  "micro_lock_audit": {
    "target_buckets": [
      "batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings",
      "batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings",
      "pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings"
    ],
    "target_ledger_locked": 0,
    "micro_ledger_locked": 0,
    "status_counts": {
      "selector_not_micro_projection": 192
    },
    "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_lock_audit.json",
    "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_lock_audit_latest.md"
  }
}
```

### Prop micro ledger report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "graded": 31,
  "pending": 3,
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_ledger_report.json",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_ledger_latest.md"
}
```

### Prop micro loss diagnostic

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 34,
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_loss_diagnostic.json",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_loss_diagnostic_latest.md"
}
```

### Prop micro probability calibrator

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "enabled": true,
  "graded_rows": 31,
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_probability_calibrator.json",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_probability_calibrator_latest.md"
}
```

### Prop post-gate candidate report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "active_rows": 192,
  "micro_projection_rows": 0,
  "near_misses": 30,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_post_gate_candidate_latest.md"
}
```

### Prop layer promotion control

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "auto_integrations": {
    "hitter_rate_challenger_production": {
      "enabled": false,
      "layer": "hitter_hits_hr_shadow",
      "target_mode": "production_scoring",
      "blockers": [
        "batter_hits_projection_not_better_than_baseline",
        "five_date_checkpoint_artifact_stale>36h"
      ]
    },
    "hitter_pa_v3_production": {
      "enabled": false,
      "layer": "hitter_pa_v3",
      "target_mode": "production_scoring",
      "blockers": [
        "five_date_checkpoint_artifact_stale>36h"
      ]
    },
    "exact_bucket_micro_ladder": {
      "enabled": false,
      "layer": "exact_bucket_micro",
      "target_mode": "micro",
      "blockers": [
        "micro_ready_exact_buckets=0",
        "real_money_kill_switch_active"
      ]
    }
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_layer_promotion_latest.md"
}
```

### Forecast repair error decomposition

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "active_source": "prospective_ledger",
  "prospective_rows": 19592,
  "historical_rows": 27016,
  "tb_repair": "train_gated_direct_player_game_tb_head",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_forecast_repair_error_latest.md"
}
```

### TB tail repair challenger

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 8986,
  "accepted": true,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb_tail_repair_challenger_latest.md"
}
```

### Prop snapshot coverage report

- rc: 124

**stderr (tail)**
```
Timed out after 120s; killed process tree rooted at PID 1380
```

### Real-money operational prop reports

- rc: 0

**stdout (tail)**
```
{
  "generated_at_utc": "2026-07-27T00:06:48Z",
  "slate_date": "2026-07-26",
  "lookback_days": 45,
  "reports": {
    "close_coverage_dashboard": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_close_coverage_dashboard_latest.md"
    },
    "player_game_projection_ledger_v2": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_player_game_projection_ledger_v2_latest.md"
    },
    "tb_error_decomposition": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_tb_error_decomposition_latest.md"
    },
    "true_pair_coverage_repair": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_true_pair_coverage_repair_latest.md"
    },
    "shadow_challenger_promotion": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_shadow_challenger_promotion_latest.md"
    },
    "micro_bucket_almost_there": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_micro_bucket_almost_there_latest.md"
    },
    "micro_gate_sensitivity": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
    }
  }
}
```

### Pitcher K-rate challenger diagnostic

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 1015,
  "mae_gain": 0.0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_pitcher_k_rate_challenger_latest.md"
}
```

### End-of-slate prop close diagnostic

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-26",
  "evaluation_status": "provisional",
  "strict_clean_slate": false,
  "valid_close_coverage": 0.857087975412985,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_end_of_slate_close_latest.md"
}
```

### Frozen-release five-date checkpoint

- rc: 124

**stderr (tail)**
```
Timed out after 180s; killed process tree rooted at PID 6200
```

### Post-checkpoint micro promotion evaluation

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "micro_ready_count": 0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_promotion_evaluation_latest.md"
}
```

### Post-checkpoint micro gate sensitivity report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "profiles": {
    "watch_only": 1,
    "micro_test": 0,
    "strict_micro": 0
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
}
```

### Post-checkpoint micro bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "trial_ready_count": 0,
  "near_miss_count": 98,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_bucket_repair_latest.md"
}
```

### Post-checkpoint prop trial candidate queue report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "trial_ready_count": 0,
  "near_miss_1_2_gate_count": 3,
  "proof_refresh_markets": [
    "pitcher_strikeouts",
    "batter_total_bases",
    "batter_hits"
  ],
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_trial_candidate_queue_latest.md"
}
```

### Post-checkpoint prop drift guard diagnostic

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-26', 'micro_rows': 0, 'micro_bettable_now': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

### Post-checkpoint lock micro projection ledger

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "game_date": "2026-07-26",
  "active_prediction_rows": 192,
  "locked_rows_attempted": 0,
  "ledger_before": [
    {
      "model_tier": "micro_projection",
      "rows": 4,
      "stake_usd": 4.0
    },
    {
      "model_tier": "watch",
      "rows": 103,
      "stake_usd": 0.0
    }
  ],
  "ledger_after": [
    {
      "model_tier": "micro_projection",
      "rows": 4,
      "stake_usd": 4.0
    },
    {
      "model_tier": "watch",
      "rows": 103,
      "stake_usd": 0.0
    }
  ],
  "micro_lock_audit": {
    "target_buckets": [
      "batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings",
      "batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings",
      "pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings"
    ],
    "target_ledger_locked": 0,
    "micro_ledger_locked": 0,
    "status_counts": {
      "selector_not_micro_projection": 192
    },
    "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_lock_audit.json",
    "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_lock_audit_latest.md"
  }
}
```

### Post-checkpoint prop micro ledger report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "graded": 31,
  "pending": 3,
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_ledger_report.json",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_ledger_latest.md"
}
```

### Post-checkpoint prop micro probability calibrator

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "enabled": true,
  "graded_rows": 31,
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_probability_calibrator.json",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_probability_calibrator_latest.md"
}
```

### Post-checkpoint prop layer promotion control

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "auto_integrations": {
    "hitter_rate_challenger_production": {
      "enabled": false,
      "layer": "hitter_hits_hr_shadow",
      "target_mode": "production_scoring",
      "blockers": [
        "batter_hits_projection_not_better_than_baseline",
        "five_date_checkpoint_artifact_stale>36h"
      ]
    },
    "hitter_pa_v3_production": {
      "enabled": false,
      "layer": "hitter_pa_v3",
      "target_mode": "production_scoring",
      "blockers": [
        "five_date_checkpoint_artifact_stale>36h"
      ]
    },
    "exact_bucket_micro_ladder": {
      "enabled": false,
      "layer": "exact_bucket_micro",
      "target_mode": "micro",
      "blockers": [
        "micro_ready_exact_buckets=0",
        "real_money_kill_switch_active"
      ]
    }
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_layer_promotion_latest.md"
}
```

### Post-checkpoint real-money operational prop reports

- rc: 0

**stdout (tail)**
```
{
  "generated_at_utc": "2026-07-27T00:14:53Z",
  "slate_date": "2026-07-26",
  "lookback_days": 45,
  "reports": {
    "close_coverage_dashboard": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_close_coverage_dashboard_latest.md"
    },
    "player_game_projection_ledger_v2": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_player_game_projection_ledger_v2_latest.md"
    },
    "tb_error_decomposition": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_tb_error_decomposition_latest.md"
    },
    "true_pair_coverage_repair": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_true_pair_coverage_repair_latest.md"
    },
    "shadow_challenger_promotion": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_shadow_challenger_promotion_latest.md"
    },
    "micro_bucket_almost_there": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_micro_bucket_almost_there_latest.md"
    },
    "micro_gate_sensitivity": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
    }
  }
}
```

### Daily slate trust monitor

- rc: 124

**stderr (tail)**
```
Timed out after 600s; killed process tree rooted at PID 4352
```

### Grade shadow prop replay

- rc: 0

**stdout (tail)**
```
{
  "graded_rows": 4309,
  "run_ids": "all_pending",
  "regrade": false
}
```
