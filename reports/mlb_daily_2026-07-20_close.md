# SuperNovaBets MLB Daily Run (2026-07-20 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 5.5s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 19.9s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 51.2s)
- **Refresh prop replay CLV**: OK (rc=0, 78.2s)
- **Build prop market training table**: OK (rc=0, 144.2s)
- **Prop walk-forward accuracy report**: OK (rc=0, 6.9s)
- **Prop shadow selector report**: OK (rc=0, 7.0s)
- **Prop miss diagnostic report**: OK (rc=0, 110.8s)
- **Prop bucket repair report**: OK (rc=0, 36.5s)
- **TB prop repair report**: OK (rc=0, 33.1s)
- **Prop target quality report**: OK (rc=0, 47.2s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 16.8s)
- **Grade outcomes + ledgers**: OK (rc=0, 16.3s)
- **Grade daily forecast ledger**: OK (rc=0, 7.0s)
- **Daily forecast projection audit**: OK (rc=0, 17.1s)
- **Hitter live-vs-legacy forecast diff**: OK (rc=0, 8.3s)
- **Player-game bankroll proof**: OK (rc=0, 18.4s)
- **TB 1.5 line calibration**: OK (rc=0, 8.5s)
- **K-under repair report**: OK (rc=0, 3.3s)
- **Exact-bucket CLV priors**: OK (rc=0, 9.6s)
- **Prop micro promotion evaluation**: OK (rc=0, 0.4s)
- **Prop micro gate sensitivity report**: OK (rc=0, 0.3s)
- **Prop micro bucket repair report**: OK (rc=0, 0.3s)
- **Prop drift guard diagnostic**: OK (rc=0, 19.9s)
- **Lock micro projection ledger**: OK (rc=0, 7.6s)
- **Prop micro ledger report**: OK (rc=0, 2.7s)
- **Prop micro loss diagnostic**: OK (rc=0, 2.4s)
- **Prop post-gate candidate report**: OK (rc=0, 8.0s)
- **Prop layer promotion control**: OK (rc=0, 0.9s)
- **Forecast repair error decomposition**: OK (rc=0, 7.4s)
- **TB tail repair challenger**: OK (rc=0, 30.4s)
- **Prop snapshot coverage report**: FAIL (rc=124, 123.8s)
- **Real-money operational prop reports**: OK (rc=0, 19.4s)
- **Pitcher K-rate challenger diagnostic**: OK (rc=0, 58.5s)
- **End-of-slate prop close diagnostic**: OK (rc=0, 6.7s)
- **Frozen-release five-date checkpoint**: FAIL (rc=124, 184.7s)
- **Post-checkpoint micro promotion evaluation**: OK (rc=0, 3.7s)
- **Post-checkpoint micro gate sensitivity report**: OK (rc=0, 0.4s)
- **Post-checkpoint micro bucket repair report**: OK (rc=0, 0.3s)
- **Post-checkpoint prop drift guard diagnostic**: OK (rc=0, 38.5s)
- **Post-checkpoint lock micro projection ledger**: OK (rc=0, 9.8s)
- **Post-checkpoint prop micro ledger report**: OK (rc=0, 3.4s)
- **Post-checkpoint prop layer promotion control**: OK (rc=0, 0.6s)
- **Post-checkpoint real-money operational prop reports**: OK (rc=0, 10.5s)
- **Daily slate trust monitor**: OK (rc=0, 237.3s)
- **Grade shadow prop replay**: OK (rc=0, 47.0s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-20 13:45:10,278 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-21. Catching up from 2026-07-22 to 2026-07-19
2026-07-20 13:45:10,294 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-20 window=2026-07-20T04:00:00Z..2026-07-21T04:00:00Z
2026-07-20 13:45:11,178 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-20 | events=15 | credits_remaining=55954
2026-07-20 13:45:11,466 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-21 window=2026-07-21T04:00:00Z..2026-07-22T04:00:00Z
2026-07-20 13:45:12,247 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-21 | events=11 | credits_remaining=55952
2026-07-20 13:45:12,262 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=55952
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-20 13:45:14,716 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-21. Catching up from 2026-07-22 to 2026-07-19
2026-07-20 13:45:15,309 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-07-20, as_of=2026-07-20)
2026-07-20 13:45:15,915 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=ed99d6d31957cf5543b0733a49085559 (Minnesota Twins@Cleveland Guardians) | credits=55946
2026-07-20 13:45:17,185 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=29d4af8b2386afc600f412243c22a53b (Pittsburgh Pirates@New York Yankees) | credits=55940
2026-07-20 13:45:18,028 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=f237d860c80df8632a4717d53fee1c25 (Tampa Bay Rays@Toronto Blue Jays) | credits=55934
2026-07-20 13:45:18,935 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=0dd84d01f2970e524c07c9ff2a747b7f (Baltimore Orioles@Boston Red Sox) | credits=55928
2026-07-20 13:45:19,904 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=128e4328bfe1ffa187a85b1e5e2b5ec6 (Los Angeles Dodgers@Philadelphia Phillies) | credits=55922
2026-07-20 13:45:20,888 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=b829b2871533d73401916f919c2b485a (San Diego Padres@Atlanta Braves) | credits=55916
2026-07-20 13:45:21,746 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=c9a69923caed0afdda9fd6b94171b189 (San Francisco Giants@Kansas City Royals) | credits=55910
2026-07-20 13:45:22,810 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=b6936af60f2ffed1b270c119f7127ae7 (New York Mets@Milwaukee Brewers) | credits=55904
2026-07-20 13:45:23,646 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=66d5a2a0202c6946683c1a8247be9ed4 (Detroit Tigers@Chicago Cubs) | credits=55898
2026-07-20 13:45:24,591 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=97617f61842d6317b6167cdf64f137a1 (Chicago White Sox@Texas Rangers) | credits=55892
2026-07-20 13:45:25,544 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=417e9c61c9e75d104917a558c350fd33 (Miami Marlins@Houston Astros) | credits=55886
2026-07-20 13:45:26,481 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=17257e28ab0601512288eeccc637c3d1 (Washington Nationals@Colorado Rockies) | credits=55880
2026-07-20 13:45:27,347 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=2d7808af8bc666b615b13864d5730354 (Athletics@Arizona Diamondbacks) | credits=55874
2026-07-20 13:45:28,403 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=85be05f7a6300ba953ba0dc345908c33 (Cincinnati Reds@Seattle Mariners) | credits=55868
2026-07-20 13:45:29,208 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=7a53b98354a4d009429693371637853d (St. Louis Cardinals@Los Angeles Angels) | credits=55862
2026-07-20 13:45:30,134 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 11 events (game_date=2026-07-21, as_of=2026-07-20)
2026-07-20 13:45:31,085 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=38b96dae7f6cab01cf931c9faefc2606 (Detroit Tigers@Chicago Cubs) | credits=55862
2026-07-20 13:45:31,960 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-20 event=3440c51cbfebc0ac81d574f6052d2eae (St. Louis Cardinals@Los Angeles Angels) | credits=55862
2026-07-20 13:45:32,212 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=55862
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-20 13:45:36,013 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1603 rows into odds.mlb_game_lines (live odds).
2026-07-20 13:45:36,045 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-20
2026-07-20 13:45:36,045 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-20 13:45:36,045 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-20T19:15:36.045256+00:00.
2026-07-20 13:45:36,163 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-19
2026-07-20 13:46:23,357 | INFO | mlb_pipeline.parse_oddsapi | Upserted 2732 rows into odds.mlb_player_prop_lines.
2026-07-20 13:46:23,357 | INFO | mlb_pipeline.parse_oddsapi | Processed 2732 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-20 13:46:23,357 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 4838,
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
  "date_from": "2026-07-20",
  "date_to": "2026-07-20",
  "include_graded": true,
  "only_missing": false
}
```

**stderr (tail)**
```
2026-07-20 13:46:28,278 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=100 updated=100 skipped=0 last_id=277342
2026-07-20 13:46:29,735 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=200 updated=200 skipped=0 last_id=277442
2026-07-20 13:46:31,184 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=300 updated=300 skipped=0 last_id=277542
2026-07-20 13:46:32,879 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=400 updated=400 skipped=0 last_id=277642
2026-07-20 13:46:34,185 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=500 updated=500 skipped=0 last_id=277742
2026-07-20 13:46:36,342 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=600 updated=600 skipped=0 last_id=277842
2026-07-20 13:46:38,231 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=700 updated=700 skipped=0 last_id=277942
2026-07-20 13:46:39,700 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=800 updated=800 skipped=0 last_id=278042
2026-07-20 13:46:41,014 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=900 updated=900 skipped=0 last_id=278142
2026-07-20 13:46:43,012 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1000 updated=1000 skipped=0 last_id=278242
2026-07-20 13:46:44,450 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1100 updated=1100 skipped=0 last_id=278342
2026-07-20 13:46:46,090 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1200 updated=1200 skipped=0 last_id=278442
2026-07-20 13:46:48,341 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1300 updated=1300 skipped=0 last_id=278542
2026-07-20 13:46:49,810 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1400 updated=1400 skipped=0 last_id=278642
2026-07-20 13:46:51,148 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1500 updated=1500 skipped=0 last_id=278742
2026-07-20 13:46:53,009 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1600 updated=1600 skipped=0 last_id=278842
2026-07-20 13:46:54,451 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1700 updated=1700 skipped=0 last_id=278942
2026-07-20 13:46:55,716 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1800 updated=1800 skipped=0 last_id=279042
2026-07-20 13:46:57,309 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1900 updated=1900 skipped=0 last_id=279142
2026-07-20 13:46:58,530 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2000 updated=2000 skipped=0 last_id=279242
2026-07-20 13:46:59,941 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2100 updated=2100 skipped=0 last_id=279342
2026-07-20 13:47:01,507 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2200 updated=2200 skipped=0 last_id=279442
2026-07-20 13:47:02,778 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2300 updated=2300 skipped=0 last_id=279542
2026-07-20 13:47:04,092 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2400 updated=2400 skipped=0 last_id=279642
2026-07-20 13:47:05,841 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2500 updated=2500 skipped=0 last_id=279742
2026-07-20 13:47:07,171 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2600 updated=2600 skipped=0 last_id=279842
2026-07-20 13:47:08,685 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2700 updated=2700 skipped=0 last_id=279942
2026-07-20 13:47:09,873 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2800 updated=2800 skipped=0 last_id=280042
2026-07-20 13:47:11,536 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2900 updated=2900 skipped=0 last_id=280142
2026-07-20 13:47:15,616 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3000 updated=3000 skipped=0 last_id=280242
2026-07-20 13:47:17,149 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3100 updated=3100 skipped=0 last_id=280342
2026-07-20 13:47:18,989 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3200 updated=3200 skipped=0 last_id=280442
2026-07-20 13:47:20,540 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3300 updated=3300 skipped=0 last_id=280542
2026-07-20 13:47:22,106 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3400 updated=3400 skipped=0 last_id=280642
2026-07-20 13:47:23,478 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3500 updated=3500 skipped=0 last_id=280742
2026-07-20 13:47:24,710 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3600 updated=3600 skipped=0 last_id=280842
2026-07-20 13:47:26,226 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3700 updated=3700 skipped=0 last_id=280942
2026-07-20 13:47:27,686 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3800 updated=3800 skipped=0 last_id=281042
2026-07-20 13:47:28,887 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3900 updated=3900 skipped=0 last_id=281142
2026-07-20 13:47:30,341 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4000 updated=4000 skipped=0 last_id=281242
2026-07-20 13:47:31,674 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4100 updated=4100 skipped=0 last_id=281342
2026-07-20 13:47:33,029 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4200 updated=4200 skipped=0 last_id=281442
2026-07-20 13:47:34,263 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4300 updated=4300 skipped=0 last_id=281542
2026-07-20 13:47:35,559 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4400 updated=4400 skipped=0 last_id=281642
2026-07-20 13:47:37,184 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4500 updated=4500 skipped=0 last_id=281742
2026-07-20 13:47:38,528 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4600 updated=4600 skipped=0 last_id=281842
2026-07-20 13:47:39,780 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4700 updated=4700 skipped=0 last_id=281942
2026-07-20 13:47:41,179 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4800 updated=4800 skipped=0 last_id=282042
2026-07-20 13:47:41,622 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4838 updated=4838 skipped=0 last_id=282080
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 0,
  "replay_rows": 23999,
  "examples": 23999
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "reused_fresh_artifact",
  "artifact": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_walk_forward_accuracy_report.json",
  "age_minutes": 355.0,
  "rows": 250017,
  "graded_rows": 223878,
  "valid_clv_rows": 208823,
  "generated_at_utc": "2026-07-20T13:54:54+00:00"
}
```

### Prop shadow selector report

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "active_rows": 2432,
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
  "rows": 23355,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 235766,
  "bucket_count": 151,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=97993
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 265270,
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
    "raw_api_one_sided": 16793
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 228-154 (59.7%) ROI: +13.9% | Total: 71-68 (51.1%) ROI: -2.5%
MLB CLV Run Line: beat close 3/90 (3%) avg CLV=+0.10 runs | CLV Total avg=+0.12 runs
MLB Price CLV Run Line: 87 bets  avg=+0.26%
```

**stderr (tail)**
```
2026-07-20 13:54:34,370 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 40 pending predictions.
2026-07-20 13:54:34,370 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-07-20 13:54:34,403 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-20 13:54:38,941 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-07-20 13:54:38,970 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-07-20 13:54:39,079 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-20 13:54:39,080 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-20 13:54:39,216 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-07-20 13:54:40,237 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-07-20 13:54:40,250 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
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
  "rows": 18815,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_daily_forecast_projection_audit_latest.md"
}
```

### Hitter live-vs-legacy forecast diff

- rc: 0

**stdout (tail)**
```
{
  "rows": 4338,
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
  "rows": 9830,
  "enabled_count": 7,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb15_line_calibration_latest.md"
}
```

### K-under repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 2021,
  "micro_allowed_count": 1,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_k_under_repair_latest.md"
}
```

### Exact-bucket CLV priors

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 49779,
  "micro_clv_confirmed_buckets": 3,
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
    "micro_test": 1,
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
  "trial_ready_count": 1,
  "near_miss_count": 94,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_bucket_repair_latest.md"
}
```

### Prop drift guard diagnostic

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-20', 'micro_rows': 0, 'micro_bettable_now': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

### Lock micro projection ledger

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "game_date": "2026-07-20",
  "active_prediction_rows": 2432,
  "locked_rows_attempted": 0,
  "ledger_before": [
    {
      "model_tier": "watch",
      "rows": 219,
      "stake_usd": 0.0
    }
  ],
  "ledger_after": [
    {
      "model_tier": "watch",
      "rows": 219,
      "stake_usd": 0.0
    }
  ]
}
```

### Prop micro ledger report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "graded": 16,
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
  "rows": 19,
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_loss_diagnostic.json",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_loss_diagnostic_latest.md"
}
```

### Prop post-gate candidate report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "active_rows": 2432,
  "micro_projection_rows": 0,
  "near_misses": 395,
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
        "batter_hits_projection_not_better_than_baseline"
      ]
    },
    "hitter_pa_v3_production": {
      "enabled": true,
      "layer": "hitter_pa_v3",
      "target_mode": "production_scoring",
      "blockers": []
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
  "prospective_rows": 12284,
  "historical_rows": 25248,
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
  "rows": 8399,
  "accepted": true,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb_tail_repair_challenger_latest.md"
}
```

### Prop snapshot coverage report

- rc: 124

**stderr (tail)**
```
Timed out after 120s; killed process tree rooted at PID 10744
```

### Real-money operational prop reports

- rc: 0

**stdout (tail)**
```
{
  "generated_at_utc": "2026-07-20T19:59:18Z",
  "slate_date": "2026-07-20",
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
  "rows": 898,
  "mae_gain": 0.0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_pitcher_k_rate_challenger_latest.md"
}
```

### End-of-slate prop close diagnostic

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-20",
  "evaluation_status": "provisional",
  "strict_clean_slate": false,
  "valid_close_coverage": 0.0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_end_of_slate_close_latest.md"
}
```

### Frozen-release five-date checkpoint

- rc: 124

**stderr (tail)**
```
Timed out after 180s; killed process tree rooted at PID 12300
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
    "micro_test": 1,
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
  "trial_ready_count": 1,
  "near_miss_count": 94,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_bucket_repair_latest.md"
}
```

### Post-checkpoint prop drift guard diagnostic

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-20', 'micro_rows': 0, 'micro_bettable_now': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

### Post-checkpoint lock micro projection ledger

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "game_date": "2026-07-20",
  "active_prediction_rows": 2432,
  "locked_rows_attempted": 0,
  "ledger_before": [
    {
      "model_tier": "watch",
      "rows": 219,
      "stake_usd": 0.0
    }
  ],
  "ledger_after": [
    {
      "model_tier": "watch",
      "rows": 219,
      "stake_usd": 0.0
    }
  ]
}
```

### Post-checkpoint prop micro ledger report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "graded": 16,
  "pending": 3,
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_ledger_report.json",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_ledger_latest.md"
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
        "batter_hits_projection_not_better_than_baseline"
      ]
    },
    "hitter_pa_v3_production": {
      "enabled": true,
      "layer": "hitter_pa_v3",
      "target_mode": "production_scoring",
      "blockers": []
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
  "generated_at_utc": "2026-07-20T20:04:43Z",
  "slate_date": "2026-07-20",
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

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-20",
  "status": "provisional",
  "decision": "wait_for_final_results_or_repair_failed_checks",
  "failures": [
    "all_games_finalized",
    "valid_close_coverage",
    "targeted_close_captures",
    "hitter_anchor_settled_or_voided",
    "pitcher_anchor_settled_or_voided"
  ],
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_daily_slate_trust_latest.md"
}
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
