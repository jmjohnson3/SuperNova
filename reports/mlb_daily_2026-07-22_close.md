# SuperNovaBets MLB Daily Run (2026-07-22 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 9.5s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 7.5s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 7.7s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 6.6s)
- **Refresh prop replay CLV**: OK (rc=0, 66.1s)
- **Build prop market training table**: OK (rc=0, 122.3s)
- **Prop walk-forward accuracy report**: OK (rc=0, 6.7s)
- **Prop shadow selector report**: OK (rc=0, 7.2s)
- **Prop miss diagnostic report**: OK (rc=0, 102.2s)
- **Prop bucket repair report**: OK (rc=0, 31.7s)
- **TB prop repair report**: OK (rc=0, 27.1s)
- **Prop target quality report**: OK (rc=0, 47.2s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 11.9s)
- **Grade outcomes + ledgers**: OK (rc=0, 48.5s)
- **Grade daily forecast ledger**: OK (rc=0, 10.4s)
- **Daily forecast projection audit**: OK (rc=0, 18.5s)
- **Hitter live-vs-legacy forecast diff**: OK (rc=0, 20.4s)
- **Player-game bankroll proof**: OK (rc=0, 29.3s)
- **TB 1.5 line calibration**: OK (rc=0, 16.5s)
- **TB 1.5 close repair report**: OK (rc=0, 5.7s)
- **K-under repair report**: OK (rc=0, 4.4s)
- **DK K 4.5-6.0 under repair diagnostic**: OK (rc=0, 3.4s)
- **Exact-bucket CLV priors**: OK (rc=0, 12.3s)
- **Prop micro promotion evaluation**: OK (rc=0, 0.5s)
- **Prop micro gate sensitivity report**: OK (rc=0, 0.4s)
- **Prop micro bucket repair report**: OK (rc=0, 0.3s)
- **Prop trial candidate queue report**: OK (rc=0, 0.4s)
- **Prop drift guard diagnostic**: OK (rc=0, 39.9s)
- **Prop bettable-now scan**: OK (rc=0, 6.7s)
- **Lock micro projection ledger**: OK (rc=0, 11.5s)
- **Prop micro ledger report**: OK (rc=0, 3.8s)
- **Prop micro loss diagnostic**: OK (rc=0, 2.5s)
- **Prop micro probability calibrator**: OK (rc=0, 2.4s)
- **Prop post-gate candidate report**: OK (rc=0, 7.7s)
- **Prop layer promotion control**: OK (rc=0, 2.2s)
- **Forecast repair error decomposition**: OK (rc=0, 9.6s)
- **TB tail repair challenger**: OK (rc=0, 51.8s)
- **Prop snapshot coverage report**: OK (rc=0, 82.3s)
- **Real-money operational prop reports**: OK (rc=0, 14.8s)
- **Pitcher K-rate challenger diagnostic**: OK (rc=0, 54.2s)
- **End-of-slate prop close diagnostic**: OK (rc=0, 10.3s)
- **Frozen-release five-date checkpoint**: FAIL (rc=124, 186.7s)
- **Post-checkpoint micro promotion evaluation**: OK (rc=0, 6.5s)
- **Post-checkpoint micro gate sensitivity report**: OK (rc=0, 0.7s)
- **Post-checkpoint micro bucket repair report**: OK (rc=0, 1.4s)
- **Post-checkpoint prop trial candidate queue report**: OK (rc=0, 2.2s)
- **Post-checkpoint prop drift guard diagnostic**: OK (rc=0, 103.2s)
- **Post-checkpoint lock micro projection ledger**: OK (rc=0, 20.5s)
- **Post-checkpoint prop micro ledger report**: OK (rc=0, 4.1s)
- **Post-checkpoint prop micro probability calibrator**: OK (rc=0, 2.5s)
- **Post-checkpoint prop layer promotion control**: OK (rc=0, 0.4s)
- **Post-checkpoint real-money operational prop reports**: OK (rc=0, 11.4s)
- **Daily slate trust monitor**: OK (rc=0, 308.4s)
- **Grade shadow prop replay**: OK (rc=0, 56.9s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-22 20:45:07,468 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 34 teams today (2026-07-22)
2026-07-22 20:45:07,886 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 68 rows for 2026-07-22
2026-07-22 20:45:07,886 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 68 assignments for 2026-07-22
2026-07-22 20:45:07,933 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-22 20:45:08,473 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 17 unique games for season=2026-regular
2026-07-22 20:45:08,674 | INFO | mlb_pipeline.crawler_statsapi | Upserted 17 rows into raw.mlb_games
2026-07-22 20:45:08,749 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6380 completed games, 6364 already done, 7 to fetch
2026-07-22 20:45:13,003 | INFO | mlb_pipeline.crawler_statsapi |   Progress: 7 / 7 games fetched
2026-07-22 20:45:13,190 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=7, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-22 20:45:18,409 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-23. Catching up from 2026-07-24 to 2026-07-21
2026-07-22 20:45:18,425 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-22 window=2026-07-22T04:00:00Z..2026-07-23T04:00:00Z
2026-07-22 20:45:19,127 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-22 | events=3 | credits_remaining=43391
2026-07-22 20:45:19,333 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-23 window=2026-07-23T04:00:00Z..2026-07-24T04:00:00Z
2026-07-22 20:45:20,687 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-23 | events=5 | credits_remaining=43389
2026-07-22 20:45:20,738 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=43389
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-22 20:45:23,522 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-23. Catching up from 2026-07-24 to 2026-07-21
2026-07-22 20:45:24,252 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 3 events (game_date=2026-07-22, as_of=2026-07-22)
2026-07-22 20:45:24,870 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-22 event=5a272ddcdf5d8b9d87abacf3a492c021 (Chicago White Sox@Texas Rangers) | credits=43386
2026-07-22 20:45:25,836 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-22 event=83d83fb84e3bfedb96b4ae81d9283bfb (Detroit Tigers@Chicago Cubs) | credits=43383
2026-07-22 20:45:26,714 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-22 event=46df00997efe9e30fd9b1130a3774851 (Miami Marlins@Houston Astros) | credits=43378
2026-07-22 20:45:28,425 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 5 events (game_date=2026-07-23, as_of=2026-07-22)
2026-07-22 20:45:28,549 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=43378
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-22 20:45:34,065 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1583 rows into odds.mlb_game_lines (live odds).
2026-07-22 20:45:34,111 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-22
2026-07-22 20:45:34,111 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-22 20:45:34,111 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-23T02:15:34.111290+00:00.
2026-07-22 20:45:34,236 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-21
2026-07-22 20:45:34,909 | INFO | mlb_pipeline.parse_oddsapi | Upserted 12 rows into odds.mlb_player_prop_lines.
2026-07-22 20:45:34,909 | INFO | mlb_pipeline.parse_oddsapi | Processed 12 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-22 20:45:34,909 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 7248,
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
  "date_from": "2026-07-22",
  "date_to": "2026-07-22",
  "include_graded": true,
  "only_missing": false
}
```

**stderr (tail)**
```
2026-07-22 20:45:41,374 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=100 updated=100 skipped=0 last_id=295319
2026-07-22 20:45:44,173 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=200 updated=200 skipped=0 last_id=295419
2026-07-22 20:45:45,439 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=300 updated=300 skipped=0 last_id=295519
2026-07-22 20:45:46,442 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=400 updated=400 skipped=0 last_id=295619
2026-07-22 20:45:47,705 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=500 updated=500 skipped=0 last_id=295719
2026-07-22 20:45:48,675 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=600 updated=600 skipped=0 last_id=295819
2026-07-22 20:45:49,330 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=700 updated=700 skipped=0 last_id=295919
2026-07-22 20:45:50,566 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=800 updated=800 skipped=0 last_id=296019
2026-07-22 20:45:51,528 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=900 updated=900 skipped=0 last_id=296119
2026-07-22 20:45:52,219 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1000 updated=1000 skipped=0 last_id=296219
2026-07-22 20:45:53,577 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1100 updated=1100 skipped=0 last_id=296319
2026-07-22 20:45:55,034 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1200 updated=1200 skipped=0 last_id=296419
2026-07-22 20:45:55,768 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1300 updated=1300 skipped=0 last_id=296519
2026-07-22 20:45:56,331 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1400 updated=1400 skipped=0 last_id=296619
2026-07-22 20:45:56,923 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1500 updated=1500 skipped=0 last_id=296719
2026-07-22 20:45:57,945 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1600 updated=1600 skipped=0 last_id=296819
2026-07-22 20:45:58,733 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1700 updated=1700 skipped=0 last_id=296919
2026-07-22 20:45:59,604 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1800 updated=1800 skipped=0 last_id=297019
2026-07-22 20:46:00,643 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1900 updated=1900 skipped=0 last_id=297119
2026-07-22 20:46:01,295 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2000 updated=2000 skipped=0 last_id=297219
2026-07-22 20:46:02,094 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2100 updated=2100 skipped=0 last_id=297319
2026-07-22 20:46:03,174 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2200 updated=2200 skipped=0 last_id=297419
2026-07-22 20:46:03,942 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2300 updated=2300 skipped=0 last_id=297519
2026-07-22 20:46:04,736 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2400 updated=2400 skipped=0 last_id=297619
2026-07-22 20:46:05,142 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2500 updated=2500 skipped=0 last_id=297719
2026-07-22 20:46:05,939 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2600 updated=2600 skipped=0 last_id=297819
2026-07-22 20:46:06,722 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2700 updated=2700 skipped=0 last_id=297919
2026-07-22 20:46:07,315 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2800 updated=2800 skipped=0 last_id=298019
2026-07-22 20:46:07,882 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2900 updated=2900 skipped=0 last_id=298119
2026-07-22 20:46:08,768 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3000 updated=3000 skipped=0 last_id=298219
2026-07-22 20:46:09,908 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3100 updated=3100 skipped=0 last_id=298319
2026-07-22 20:46:10,798 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3200 updated=3200 skipped=0 last_id=298419
2026-07-22 20:46:11,658 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3300 updated=3300 skipped=0 last_id=298519
2026-07-22 20:46:12,728 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3400 updated=3400 skipped=0 last_id=298619
2026-07-22 20:46:13,968 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3500 updated=3500 skipped=0 last_id=298719
2026-07-22 20:46:14,734 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3600 updated=3600 skipped=0 last_id=298819
2026-07-22 20:46:15,719 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3700 updated=3700 skipped=0 last_id=298919
2026-07-22 20:46:16,245 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3800 updated=3800 skipped=0 last_id=299019
2026-07-22 20:46:16,717 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3900 updated=3900 skipped=0 last_id=299119
2026-07-22 20:46:17,424 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4000 updated=4000 skipped=0 last_id=299219
2026-07-22 20:46:18,066 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4100 updated=4100 skipped=0 last_id=299319
2026-07-22 20:46:18,627 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4200 updated=4200 skipped=0 last_id=299419
2026-07-22 20:46:19,064 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4300 updated=4300 skipped=0 last_id=299519
2026-07-22 20:46:19,737 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4400 updated=4400 skipped=0 last_id=299619
2026-07-22 20:46:20,221 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4500 updated=4500 skipped=0 last_id=299719
2026-07-22 20:46:21,237 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4600 updated=4600 skipped=0 last_id=299819
2026-07-22 20:46:21,844 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4700 updated=4700 skipped=0 last_id=299919
2026-07-22 20:46:22,831 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4800 updated=4800 skipped=0 last_id=300019
2026-07-22 20:46:24,424 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4900 updated=4900 skipped=0 last_id=300119
2026-07-22 20:46:26,269 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5000 updated=5000 skipped=0 last_id=300219
2026-07-22 20:46:27,033 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5100 updated=5100 skipped=0 last_id=300319
2026-07-22 20:46:27,690 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5200 updated=5200 skipped=0 last_id=300419
2026-07-22 20:46:28,289 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5300 updated=5300 skipped=0 last_id=300519
2026-07-22 20:46:29,583 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5400 updated=5400 skipped=0 last_id=300619
2026-07-22 20:46:31,311 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5500 updated=5500 skipped=0 last_id=300719
2026-07-22 20:46:32,060 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5600 updated=5600 skipped=0 last_id=300819
2026-07-22 20:46:32,838 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5700 updated=5700 skipped=0 last_id=300919
2026-07-22 20:46:33,435 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5800 updated=5800 skipped=0 last_id=301019
2026-07-22 20:46:33,899 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5900 updated=5900 skipped=0 last_id=301119
2026-07-22 20:46:34,332 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6000 updated=6000 skipped=0 last_id=301219
2026-07-22 20:46:34,819 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6100 updated=6100 skipped=0 last_id=301319
2026-07-22 20:46:35,721 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6200 updated=6200 skipped=0 last_id=301419
2026-07-22 20:46:36,331 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6300 updated=6300 skipped=0 last_id=301519
2026-07-22 20:46:37,205 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6400 updated=6400 skipped=0 last_id=301619
2026-07-22 20:46:37,757 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6500 updated=6500 skipped=0 last_id=301719
2026-07-22 20:46:38,142 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6600 updated=6600 skipped=0 last_id=301819
2026-07-22 20:46:38,675 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6700 updated=6700 skipped=0 last_id=301919
2026-07-22 20:46:38,955 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6800 updated=6800 skipped=0 last_id=302019
2026-07-22 20:46:39,205 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6900 updated=6900 skipped=0 last_id=302119
2026-07-22 20:46:39,689 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=7000 updated=7000 skipped=0 last_id=302219
2026-07-22 20:46:40,159 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=7100 updated=7100 skipped=0 last_id=302319
2026-07-22 20:46:40,736 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=7200 updated=7200 skipped=0 last_id=302419
2026-07-22 20:46:40,941 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=7248 updated=7248 skipped=0 last_id=302467
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 0,
  "replay_rows": 22779,
  "examples": 22779
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "reused_fresh_artifact",
  "artifact": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_walk_forward_accuracy_report.json",
  "age_minutes": 342.0,
  "rows": 259307,
  "graded_rows": 227697,
  "valid_clv_rows": 215384,
  "generated_at_utc": "2026-07-22T21:05:37+00:00"
}
```

### Prop shadow selector report

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "active_rows": 1712,
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
  "rows": 23800,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 251533,
  "bucket_count": 155,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=104701
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 283211,
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
    "raw_api_one_sided": 14111
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 229-155 (59.6%) ROI: +13.8% | Total: 72-69 (51.1%) ROI: -2.5%
MLB CLV Run Line: beat close 3/92 (3%) avg CLV=+0.10 runs | CLV Total avg=+0.10 runs
MLB Price CLV Run Line: 89 bets  avg=+0.30%
```

**stderr (tail)**
```
2026-07-22 20:52:46,562 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 7 rows
2026-07-22 20:52:46,562 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 7 MLB game outcome rows
2026-07-22 20:52:46,578 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-22 20:53:20,239 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1248 prop rows
2026-07-22 20:53:20,269 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1248 MLB prop outcome rows
2026-07-22 20:53:21,549 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-22 20:53:21,549 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-22 20:53:24,064 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 14 game model-pick ledger rows
2026-07-22 20:53:25,987 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 67 prop model-pick ledger rows
2026-07-22 20:53:26,003 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 81 MLB model-pick ledger rows
```

### Grade daily forecast ledger

- rc: 0

**stdout (tail)**
```
{
  "schema_ready": true,
  "graded": 2907
}
```

### Daily forecast projection audit

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 23769,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_daily_forecast_projection_audit_latest.md"
}
```

### Hitter live-vs-legacy forecast diff

- rc: 0

**stdout (tail)**
```
{
  "rows": 4960,
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
  "rows": 10443,
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
  "focus_rows": 4381,
  "valid_close_coverage": 0.7943391919653047,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb15_close_repair_latest.md"
}
```

### K-under repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 2160,
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
  "rows": 748,
  "roi": -0.05835494652406416,
  "clv_beat_rate": 0.475177304964539,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_k_under_46_repair_diagnostic_latest.md"
}
```

### Exact-bucket CLV priors

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 52849,
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
  "near_miss_1_2_gate_count": 2,
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
{'status': 'ok', 'game_date': '2026-07-22', 'micro_rows': 10, 'micro_bettable_now': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

### Prop bettable-now scan

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-22', 'approved_rows': 74, 'near_approved_rows': 14, 'bettable_now_before_cap': 0, 'bettable_now_inside_cap': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_bettable_now_scan.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bettable_now_scan_latest.md'}
```

### Lock micro projection ledger

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "game_date": "2026-07-22",
  "active_prediction_rows": 1712,
  "locked_rows_attempted": 0,
  "ledger_before": [
    {
      "model_tier": "micro_projection",
      "rows": 4,
      "stake_usd": 4.0
    },
    {
      "model_tier": "watch",
      "rows": 148,
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
      "rows": 148,
      "stake_usd": 0.0
    }
  ],
  "micro_lock_audit": {
    "target_buckets": [
      "batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings",
      "batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings",
      "pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings"
    ],
    "target_ledger_locked": 3,
    "micro_ledger_locked": 3,
    "status_counts": {
      "selector_not_micro_projection": 1702,
      "stale_after_expired": 10
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
  "graded": 23,
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
  "rows": 26,
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
  "graded_rows": 23,
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
  "active_rows": 1712,
  "micro_projection_rows": 11,
  "near_misses": 489,
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
  "prospective_rows": 15728,
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

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-07-22T22:59:08-04:00
Range: 2026-07-09 to 2026-07-22

## Collection Status

**COLLECTING**

- Clean shadow slates: 1 / 10
- Additional clean slates needed: 9
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 90.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 2.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-07-22 | no | 5316 | 2883 | 7248 | 7248 | 129373 | 6127 | 84.5% | 0.0% | 0.5% | 3 | 54 | valid_close_coverage<0.90 |
| 2026-07-21 | no | 4683 | 2771 | 7893 | 7893 | 127679 | 6225 | 78.9% | 0.0% | 1.9% | 3 | 55 | valid_close_coverage<0.90 |
| 2026-07-20 | no | 4685 | 2740 | 7638 | 7638 | 123069 | 6711 | 87.9% | 0.0% | 0.0% | 3 | 49 | valid_close_coverage<0.90 |
| 2026-07-19 | no | 4978 | 2676 | 5423 | 5423 | 81193 | 4648 | 85.7% | 0.0% | 0.4% | 3 | 42 | valid_close_coverage<0.90 |
| 2026-07-18 | no | 4910 | 2748 | 6033 | 6033 | 202052 | 4896 | 81.2% | 0.0% | 0.9% | 3 | 101 | valid_close_coverage<0.90 |
| 2026-07-17 | no | 4447 | 2687 | 7705 | 7705 | 195834 | 6471 | 84.0% | 0.0% | 1.0% | 3 | 83 | valid_close_coverage<0.90 |
| 2026-07-16 | yes | 280 | 177 | 563 | 563 | 5265 | 551 | 97.9% | 0.0% | 0.0% | 3 | 29 |  |
| 2026-07-15 | no | 0 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | side_locks<100, captured_side_locks<100, close_capture_coverage<0.90, valid_close_coverage<0.90, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.02, no_close_snapshot_time |
| 2026-07-14 | no | 42 | 42 | 0 | 0 | 990 | 0 | - | - | - | 0 | 24 | side_locks<100, captured_side_locks<100, close_capture_coverage<0.90, valid_close_coverage<0.90, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.02 |
| 2026-07-13 | no | 42 | 42 | 0 | 0 | 42 | 0 | - | - | - | 0 | 1 | side_locks<100, captured_side_locks<100, close_capture_coverage<0.90, valid_close_coverage<0.90, no_training_rows, missing_lock_rate>0.02, stale_close_rate>0.02 |
| 2026-07-12 | no | 4577 | 2643 | 7667 | 7667 | 76486 | 4599 | 60.0% | 0.0% | 27.4% | 3 | 33 | close_capture_coverage<0.90, valid_close_coverage<0.90, stale_close_rate>0.02 |
| 2026-07-11 | no | 4850 | 2754 | 6643 | 6643 | 123974 | 5923 | 89.2% | 0.0% | 0.0% | 3 | 58 | valid_close_coverage<0.90 |
| 2026-07-10 | no | 4688 | 2855 | 8100 | 8100 | 97641 | 7024 | 86.7% | 0.0% | 0.0% | 3 | 44 | valid_close_coverage<0.90 |
| 2026-07-09 | no | 3913 | 2435 | 5964 | 5964 | 62333 | 5292 | 88.7% | 0.0% | 0.0% | 3 | 40 | valid_close_coverage<0.90 |
```

### Real-money operational prop reports

- rc: 0

**stdout (tail)**
```
{
  "generated_at_utc": "2026-07-23T02:59:10Z",
  "slate_date": "2026-07-22",
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
  "rows": 956,
  "mae_gain": 0.0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_pitcher_k_rate_challenger_latest.md"
}
```

### End-of-slate prop close diagnostic

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-22",
  "evaluation_status": "provisional",
  "strict_clean_slate": false,
  "valid_close_coverage": 0.8587311968606932,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_end_of_slate_close_latest.md"
}
```

### Frozen-release five-date checkpoint

- rc: 124

**stderr (tail)**
```
Timed out after 180s; killed process tree rooted at PID 2216
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
  "near_miss_1_2_gate_count": 2,
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
{'status': 'ok', 'game_date': '2026-07-22', 'micro_rows': 11, 'micro_bettable_now': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

### Post-checkpoint lock micro projection ledger

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "game_date": "2026-07-22",
  "active_prediction_rows": 1712,
  "locked_rows_attempted": 0,
  "ledger_before": [
    {
      "model_tier": "micro_projection",
      "rows": 4,
      "stake_usd": 4.0
    },
    {
      "model_tier": "watch",
      "rows": 148,
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
      "rows": 148,
      "stake_usd": 0.0
    }
  ],
  "micro_lock_audit": {
    "target_buckets": [
      "batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings",
      "batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings",
      "pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings"
    ],
    "target_ledger_locked": 3,
    "micro_ledger_locked": 3,
    "status_counts": {
      "selector_not_micro_projection": 1701,
      "stale_after_expired": 11
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
  "graded": 23,
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
  "graded_rows": 23,
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
  "generated_at_utc": "2026-07-23T03:05:57Z",
  "slate_date": "2026-07-22",
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
  "slate_date": "2026-07-22",
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
  "graded_rows": 3652,
  "run_ids": "all_pending",
  "regrade": false
}
```
