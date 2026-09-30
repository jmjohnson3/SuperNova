# SuperNovaBets MLB Daily Run (2026-07-24 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 4.8s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 4.5s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 18.7s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 23.4s)
- **Refresh prop replay CLV**: OK (rc=0, 67.2s)
- **Build prop market training table**: OK (rc=0, 157.5s)
- **Prop walk-forward accuracy report**: OK (rc=0, 13.4s)
- **Prop shadow selector report**: OK (rc=0, 9.9s)
- **Prop miss diagnostic report**: OK (rc=0, 126.3s)
- **Prop bucket repair report**: OK (rc=0, 94.9s)
- **TB prop repair report**: OK (rc=0, 47.7s)
- **Prop target quality report**: FAIL (rc=124, 190.2s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 21.0s)
- **Grade outcomes + ledgers**: OK (rc=0, 14.7s)
- **Grade daily forecast ledger**: OK (rc=0, 6.4s)
- **Daily forecast projection audit**: OK (rc=0, 25.5s)
- **Hitter live-vs-legacy forecast diff**: OK (rc=0, 22.8s)
- **Player-game bankroll proof**: OK (rc=0, 22.6s)
- **TB 1.5 line calibration**: OK (rc=0, 15.3s)
- **TB 1.5 close repair report**: OK (rc=0, 4.9s)
- **K-under repair report**: OK (rc=0, 4.2s)
- **DK K 4.5-6.0 under repair diagnostic**: OK (rc=0, 3.5s)
- **Exact-bucket CLV priors**: OK (rc=0, 10.6s)
- **Prop micro promotion evaluation**: OK (rc=0, 2.6s)
- **Prop micro gate sensitivity report**: OK (rc=0, 0.4s)
- **Prop micro bucket repair report**: OK (rc=0, 0.3s)
- **Prop trial candidate queue report**: OK (rc=0, 0.7s)
- **Prop drift guard diagnostic**: OK (rc=0, 27.5s)
- **Prop bettable-now scan**: OK (rc=0, 6.2s)
- **Lock micro projection ledger**: OK (rc=0, 12.7s)
- **Prop micro ledger report**: OK (rc=0, 5.3s)
- **Prop micro loss diagnostic**: OK (rc=0, 2.8s)
- **Prop micro probability calibrator**: OK (rc=0, 2.5s)
- **Prop post-gate candidate report**: OK (rc=0, 6.9s)
- **Prop layer promotion control**: OK (rc=0, 0.3s)
- **Forecast repair error decomposition**: OK (rc=0, 17.9s)
- **TB tail repair challenger**: OK (rc=0, 71.9s)
- **Prop snapshot coverage report**: FAIL (rc=124, 136.8s)
- **Real-money operational prop reports**: OK (rc=0, 49.6s)
- **Pitcher K-rate challenger diagnostic**: OK (rc=0, 74.8s)
- **End-of-slate prop close diagnostic**: OK (rc=0, 9.0s)
- **Frozen-release five-date checkpoint**: FAIL (rc=124, 214.6s)
- **Post-checkpoint micro promotion evaluation**: OK (rc=0, 15.6s)
- **Post-checkpoint micro gate sensitivity report**: OK (rc=0, 3.2s)
- **Post-checkpoint micro bucket repair report**: OK (rc=0, 0.6s)
- **Post-checkpoint prop trial candidate queue report**: OK (rc=0, 1.7s)
- **Post-checkpoint prop drift guard diagnostic**: FAIL (rc=124, 121.5s)
- **Post-checkpoint lock micro projection ledger**: OK (rc=0, 48.9s)
- **Post-checkpoint prop micro ledger report**: OK (rc=0, 6.2s)
- **Post-checkpoint prop micro probability calibrator**: OK (rc=0, 3.0s)
- **Post-checkpoint prop layer promotion control**: OK (rc=0, 0.5s)
- **Post-checkpoint real-money operational prop reports**: OK (rc=0, 44.2s)
- **Daily slate trust monitor**: OK (rc=0, 496.8s)
- **Grade shadow prop replay**: OK (rc=0, 31.0s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-24 16:45:18,764 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 30 teams today (2026-07-24)
2026-07-24 16:45:19,165 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 60 rows for 2026-07-24
2026-07-24 16:45:19,165 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 60 assignments for 2026-07-24
2026-07-24 16:45:19,165 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-24 16:45:19,507 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 15 unique games for season=2026-regular
2026-07-24 16:45:19,694 | INFO | mlb_pipeline.crawler_statsapi | Upserted 15 rows into raw.mlb_games
2026-07-24 16:45:19,741 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6386 completed games, 6377 already done, 0 to fetch
2026-07-24 16:45:19,892 | INFO | mlb_pipeline.crawler_statsapi | Updated SP IDs for 3 games in raw.mlb_games
2026-07-24 16:45:19,892 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=0, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-24 16:45:23,079 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-25. Catching up from 2026-07-26 to 2026-07-23
2026-07-24 16:45:23,079 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-24 window=2026-07-24T04:00:00Z..2026-07-25T04:00:00Z
2026-07-24 16:45:23,766 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-24 | events=15 | credits_remaining=38584
2026-07-24 16:45:23,812 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-25 window=2026-07-25T04:00:00Z..2026-07-26T04:00:00Z
2026-07-24 16:45:24,409 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-25 | events=15 | credits_remaining=38582
2026-07-24 16:45:24,424 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=38582
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-24 16:45:26,867 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-25. Catching up from 2026-07-26 to 2026-07-23
2026-07-24 16:45:27,693 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-07-24, as_of=2026-07-24)
2026-07-24 16:45:28,819 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=d7f4b3c3080d3f6612a4be54f7e0f798 (Colorado Rockies@Milwaukee Brewers) | credits=38579
2026-07-24 16:45:30,017 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=263aac2dfd923f6a65c5136418407715 (Chicago Cubs@Pittsburgh Pirates) | credits=38573
2026-07-24 16:45:31,158 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=74d637de5babda95c445e8cf11c32a15 (Kansas City Royals@Detroit Tigers) | credits=38567
2026-07-24 16:45:32,033 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=12a8e27a29d44d46e10583c9c875e5f5 (Arizona Diamondbacks@Washington Nationals) | credits=38561
2026-07-24 16:45:32,845 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=5859ce305cb0ae65e5e7f771db28dcdd (New York Yankees@Philadelphia Phillies) | credits=38555
2026-07-24 16:45:33,720 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=5dc243f04181b459d0cc18968be70660 (Atlanta Braves@Baltimore Orioles) | credits=38549
2026-07-24 16:45:34,549 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=29437f5a0e0b0af43535ee03d438df64 (Cleveland Guardians@Tampa Bay Rays) | credits=38543
2026-07-24 16:45:35,423 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=cee011b0978086fc2ebee48d455d495e (Los Angeles Dodgers@New York Mets) | credits=38537
2026-07-24 16:45:36,298 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=6a85ed309a708411951843fe95d213bb (San Diego Padres@Miami Marlins) | credits=38531
2026-07-24 16:45:37,205 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=7cb10d307b371751677d513fb168b3bd (Toronto Blue Jays@Boston Red Sox) | credits=38525
2026-07-24 16:45:38,018 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=e25ed72c9f5c072ef60c242c29ccf1a3 (Houston Astros@Chicago White Sox) | credits=38519
2026-07-24 16:45:38,799 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=7623b2ff95daaba1c14165f508c506b8 (Seattle Mariners@Texas Rangers) | credits=38513
2026-07-24 16:45:39,659 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=6f9f1f656034bccc0a22e55a976cdf3b (Athletics@Minnesota Twins) | credits=38507
2026-07-24 16:45:40,580 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=e52022e436fbab769c0e31e1eec70f53 (Cincinnati Reds@St. Louis Cardinals) | credits=38501
2026-07-24 16:45:41,788 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-24 event=2192276baf2c4e8e5358e9905c801e2f (Los Angeles Angels@San Francisco Giants) | credits=38495
2026-07-24 16:45:42,869 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-07-25, as_of=2026-07-24)
2026-07-24 16:45:42,888 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=38495
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-24 16:45:49,070 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1627 rows into odds.mlb_game_lines (live odds).
2026-07-24 16:45:49,073 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-24
2026-07-24 16:45:49,073 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-24 16:45:49,073 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-24T22:15:49.073021+00:00.
2026-07-24 16:45:49,142 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-23
2026-07-24 16:46:06,409 | INFO | mlb_pipeline.parse_oddsapi | Upserted 2841 rows into odds.mlb_player_prop_lines.
2026-07-24 16:46:06,409 | INFO | mlb_pipeline.parse_oddsapi | Processed 2841 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-24 16:46:06,409 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 7560,
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
  "date_from": "2026-07-24",
  "date_to": "2026-07-24",
  "include_graded": true,
  "only_missing": false
}
```

**stderr (tail)**
```
2026-07-24 16:46:18,191 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=100 updated=100 skipped=0 last_id=304136
2026-07-24 16:46:19,096 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=200 updated=200 skipped=0 last_id=304236
2026-07-24 16:46:20,426 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=300 updated=300 skipped=0 last_id=304336
2026-07-24 16:46:21,427 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=400 updated=400 skipped=0 last_id=304436
2026-07-24 16:46:22,175 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=500 updated=500 skipped=0 last_id=304536
2026-07-24 16:46:23,676 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=600 updated=600 skipped=0 last_id=304636
2026-07-24 16:46:24,658 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=700 updated=700 skipped=0 last_id=304736
2026-07-24 16:46:25,015 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=800 updated=800 skipped=0 last_id=304836
2026-07-24 16:46:25,720 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=900 updated=900 skipped=0 last_id=304936
2026-07-24 16:46:26,659 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1000 updated=1000 skipped=0 last_id=305036
2026-07-24 16:46:27,018 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1100 updated=1100 skipped=0 last_id=305136
2026-07-24 16:46:27,710 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1200 updated=1200 skipped=0 last_id=305236
2026-07-24 16:46:28,879 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1300 updated=1300 skipped=0 last_id=305336
2026-07-24 16:46:29,623 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1400 updated=1400 skipped=0 last_id=305436
2026-07-24 16:46:30,864 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1500 updated=1500 skipped=0 last_id=305536
2026-07-24 16:46:31,479 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1600 updated=1600 skipped=0 last_id=305636
2026-07-24 16:46:31,928 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1700 updated=1700 skipped=0 last_id=305736
2026-07-24 16:46:32,502 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1800 updated=1800 skipped=0 last_id=305836
2026-07-24 16:46:32,976 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1900 updated=1900 skipped=0 last_id=305936
2026-07-24 16:46:33,534 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2000 updated=2000 skipped=0 last_id=306036
2026-07-24 16:46:33,986 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2100 updated=2100 skipped=0 last_id=306136
2026-07-24 16:46:35,065 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2200 updated=2200 skipped=0 last_id=306236
2026-07-24 16:46:36,549 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2300 updated=2300 skipped=0 last_id=306336
2026-07-24 16:46:37,432 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2400 updated=2400 skipped=0 last_id=306436
2026-07-24 16:46:38,579 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2500 updated=2500 skipped=0 last_id=306536
2026-07-24 16:46:39,327 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2600 updated=2600 skipped=0 last_id=306636
2026-07-24 16:46:40,233 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2700 updated=2700 skipped=0 last_id=306736
2026-07-24 16:46:41,189 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2800 updated=2800 skipped=0 last_id=306836
2026-07-24 16:46:41,816 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2900 updated=2900 skipped=0 last_id=306936
2026-07-24 16:46:42,300 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3000 updated=3000 skipped=0 last_id=307036
2026-07-24 16:46:42,712 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3100 updated=3100 skipped=0 last_id=307136
2026-07-24 16:46:43,050 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3200 updated=3200 skipped=0 last_id=307236
2026-07-24 16:46:44,143 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3300 updated=3300 skipped=0 last_id=307336
2026-07-24 16:46:45,781 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3400 updated=3400 skipped=0 last_id=307436
2026-07-24 16:46:46,175 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3500 updated=3500 skipped=0 last_id=307536
2026-07-24 16:46:46,857 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3600 updated=3600 skipped=0 last_id=307636
2026-07-24 16:46:47,301 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3700 updated=3700 skipped=0 last_id=307736
2026-07-24 16:46:47,718 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3800 updated=3800 skipped=0 last_id=307836
2026-07-24 16:46:48,552 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3900 updated=3900 skipped=0 last_id=307936
2026-07-24 16:46:49,206 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4000 updated=4000 skipped=0 last_id=308036
2026-07-24 16:46:49,721 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4100 updated=4100 skipped=0 last_id=308136
2026-07-24 16:46:50,143 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4200 updated=4200 skipped=0 last_id=308236
2026-07-24 16:46:50,690 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4300 updated=4300 skipped=0 last_id=308336
2026-07-24 16:46:51,910 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4400 updated=4400 skipped=0 last_id=308436
2026-07-24 16:46:52,993 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4500 updated=4500 skipped=0 last_id=308536
2026-07-24 16:46:53,820 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4600 updated=4600 skipped=0 last_id=308636
2026-07-24 16:46:54,847 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4700 updated=4700 skipped=0 last_id=308736
2026-07-24 16:46:55,424 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4800 updated=4800 skipped=0 last_id=308836
2026-07-24 16:46:55,986 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4900 updated=4900 skipped=0 last_id=308936
2026-07-24 16:46:56,642 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5000 updated=5000 skipped=0 last_id=309036
2026-07-24 16:46:57,242 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5100 updated=5100 skipped=0 last_id=309136
2026-07-24 16:46:57,955 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5200 updated=5200 skipped=0 last_id=309236
2026-07-24 16:46:58,317 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5300 updated=5300 skipped=0 last_id=309336
2026-07-24 16:46:58,925 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5400 updated=5400 skipped=0 last_id=309436
2026-07-24 16:46:59,969 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5500 updated=5500 skipped=0 last_id=309536
2026-07-24 16:47:01,268 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5600 updated=5600 skipped=0 last_id=309636
2026-07-24 16:47:01,719 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5700 updated=5700 skipped=0 last_id=309736
2026-07-24 16:47:03,242 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5800 updated=5800 skipped=0 last_id=309836
2026-07-24 16:47:03,783 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5900 updated=5900 skipped=0 last_id=309936
2026-07-24 16:47:04,672 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6000 updated=6000 skipped=0 last_id=310036
2026-07-24 16:47:05,440 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6100 updated=6100 skipped=0 last_id=310136
2026-07-24 16:47:05,893 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6200 updated=6200 skipped=0 last_id=310236
2026-07-24 16:47:06,299 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6300 updated=6300 skipped=0 last_id=310336
2026-07-24 16:47:06,939 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6400 updated=6400 skipped=0 last_id=310436
2026-07-24 16:47:07,574 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6500 updated=6500 skipped=0 last_id=310536
2026-07-24 16:47:07,908 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6600 updated=6600 skipped=0 last_id=310636
2026-07-24 16:47:08,299 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6700 updated=6700 skipped=0 last_id=310736
2026-07-24 16:47:08,750 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6800 updated=6800 skipped=0 last_id=310836
2026-07-24 16:47:09,066 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6900 updated=6900 skipped=0 last_id=310936
2026-07-24 16:47:09,548 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=7000 updated=7000 skipped=0 last_id=311036
2026-07-24 16:47:09,923 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=7100 updated=7100 skipped=0 last_id=311136
2026-07-24 16:47:10,347 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=7200 updated=7200 skipped=0 last_id=311236
2026-07-24 16:47:10,892 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=7300 updated=7300 skipped=0 last_id=311336
2026-07-24 16:47:11,970 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=7400 updated=7400 skipped=0 last_id=311436
2026-07-24 16:47:12,802 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=7500 updated=7500 skipped=0 last_id=311536
2026-07-24 16:47:13,609 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=7560 updated=7560 skipped=0 last_id=311596
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 0,
  "replay_rows": 24270,
  "examples": 24270
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "reused_fresh_artifact",
  "artifact": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_walk_forward_accuracy_report.json",
  "age_minutes": 99.5,
  "rows": 258281,
  "graded_rows": 226184,
  "valid_clv_rows": 212031,
  "generated_at_utc": "2026-07-24T21:09:16+00:00"
}
```

### Prop shadow selector report

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "active_rows": 2697,
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
  "rows": 23917,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 257073,
  "bucket_count": 155,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=107037
```

### Prop target quality report

- rc: 124

**stderr (tail)**
```
Timed out after 180s; killed process tree rooted at PID 6432
```

### FanDuel one-sided diagnostic

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "root_causes": {
    "raw_api_one_sided": 16109
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 229-156 (59.5%) ROI: +13.6% | Total: 72-69 (51.1%) ROI: -2.5%
MLB CLV Run Line: beat close 3/93 (3%) avg CLV=+0.10 runs | CLV Total avg=+0.10 runs
MLB Price CLV Run Line: 90 bets  avg=+0.34%
```

**stderr (tail)**
```
2026-07-24 16:58:22,189 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 42 pending predictions.
2026-07-24 16:58:22,189 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-07-24 16:58:22,205 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-24 16:58:28,090 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-07-24 16:58:28,122 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-07-24 16:58:28,252 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-24 16:58:28,252 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-24 16:58:28,491 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-07-24 16:58:29,287 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-07-24 16:58:29,303 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
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
  "rows": 24346,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_daily_forecast_projection_audit_latest.md"
}
```

### Hitter live-vs-legacy forecast diff

- rc: 0

**stdout (tail)**
```
{
  "rows": 5341,
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
  "rows": 10657,
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
  "focus_rows": 4507,
  "valid_close_coverage": 0.7934324384291103,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb15_close_repair_latest.md"
}
```

### K-under repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 2204,
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
  "rows": 761,
  "roi": -0.05525729303547963,
  "clv_beat_rate": 0.47909407665505227,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_k_under_46_repair_diagnostic_latest.md"
}
```

### Exact-bucket CLV priors

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 53930,
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
{'status': 'ok', 'game_date': '2026-07-24', 'micro_rows': 1, 'micro_bettable_now': 1, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

### Prop bettable-now scan

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-24', 'approved_rows': 103, 'near_approved_rows': 21, 'bettable_now_before_cap': 1, 'bettable_now_inside_cap': 1, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_bettable_now_scan.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bettable_now_scan_latest.md'}
```

### Lock micro projection ledger

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "game_date": "2026-07-24",
  "active_prediction_rows": 2697,
  "locked_rows_attempted": 1,
  "ledger_before": [
    {
      "model_tier": "watch",
      "rows": 275,
      "stake_usd": 0.0
    }
  ],
  "ledger_after": [
    {
      "model_tier": "micro_projection",
      "rows": 1,
      "stake_usd": 1.0
    },
    {
      "model_tier": "watch",
      "rows": 275,
      "stake_usd": 0.0
    }
  ],
  "micro_lock_audit": {
    "target_buckets": [
      "batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings",
      "batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings",
      "pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings"
    ],
    "target_ledger_locked": 1,
    "micro_ledger_locked": 1,
    "status_counts": {
      "selector_not_micro_projection": 2696,
      "lockable_now": 1
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
  "graded": 24,
  "pending": 4,
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
  "rows": 28,
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
  "graded_rows": 24,
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
  "active_rows": 2697,
  "micro_projection_rows": 1,
  "near_misses": 185,
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
  "prospective_rows": 16111,
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
Timed out after 120s; killed process tree rooted at PID 11500
```

### Real-money operational prop reports

- rc: 0

**stdout (tail)**
```
{
  "generated_at_utc": "2026-07-24T23:05:49Z",
  "slate_date": "2026-07-24",
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
  "rows": 975,
  "mae_gain": 0.0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_pitcher_k_rate_challenger_latest.md"
}
```

### End-of-slate prop close diagnostic

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-24",
  "evaluation_status": "provisional",
  "strict_clean_slate": false,
  "valid_close_coverage": 0.8332756831546178,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_end_of_slate_close_latest.md"
}
```

### Frozen-release five-date checkpoint

- rc: 124

**stderr (tail)**
```
Timed out after 180s; killed process tree rooted at PID 7752
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

- rc: 124

**stderr (tail)**
```
Timed out after 120s; killed process tree rooted at PID 10908
```

### Post-checkpoint lock micro projection ledger

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "game_date": "2026-07-24",
  "active_prediction_rows": 2697,
  "locked_rows_attempted": 0,
  "ledger_before": [
    {
      "model_tier": "micro_projection",
      "rows": 1,
      "stake_usd": 1.0
    },
    {
      "model_tier": "watch",
      "rows": 275,
      "stake_usd": 0.0
    }
  ],
  "ledger_after": [
    {
      "model_tier": "micro_projection",
      "rows": 1,
      "stake_usd": 1.0
    },
    {
      "model_tier": "watch",
      "rows": 275,
      "stake_usd": 0.0
    }
  ],
  "micro_lock_audit": {
    "target_buckets": [
      "batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings",
      "batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings",
      "pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings"
    ],
    "target_ledger_locked": 1,
    "micro_ledger_locked": 1,
    "status_counts": {
      "selector_not_micro_projection": 2696,
      "lockable_now": 1
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
  "graded": 24,
  "pending": 4,
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
  "graded_rows": 24,
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
  "generated_at_utc": "2026-07-24T23:14:34Z",
  "slate_date": "2026-07-24",
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
  "slate_date": "2026-07-24",
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
