# SuperNovaBets MLB Daily Run (2026-07-28 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 37.2s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 24.8s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 33.4s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 34.9s)
- **Refresh prop replay CLV**: FAIL (rc=124, 322.5s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-28 19:46:03,638 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 32 teams today (2026-07-28)
2026-07-28 19:46:04,205 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 60 rows for 2026-07-28
2026-07-28 19:46:04,205 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 60 assignments for 2026-07-28
2026-07-28 19:46:04,236 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-28 19:46:04,601 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 16 unique games for season=2026-regular
2026-07-28 19:46:05,049 | INFO | mlb_pipeline.crawler_statsapi | Upserted 16 rows into raw.mlb_games
2026-07-28 19:46:05,112 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6447 completed games, 6434 already done, 4 to fetch
2026-07-28 19:46:13,721 | INFO | mlb_pipeline.crawler_statsapi |   Progress: 4 / 4 games fetched
2026-07-28 19:46:13,902 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=4, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-28 19:46:36,501 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-29. Catching up from 2026-07-30 to 2026-07-27
2026-07-28 19:46:36,533 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-28 window=2026-07-28T04:00:00Z..2026-07-29T04:00:00Z
2026-07-28 19:46:39,766 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-28 | events=10 | credits_remaining=20452
2026-07-28 19:46:39,923 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-29 window=2026-07-29T04:00:00Z..2026-07-30T04:00:00Z
2026-07-28 19:46:44,641 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-29 | events=16 | credits_remaining=20450
2026-07-28 19:46:44,676 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=20450
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-28 19:46:56,830 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-29. Catching up from 2026-07-30 to 2026-07-27
2026-07-28 19:46:57,579 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 10 events (game_date=2026-07-28, as_of=2026-07-28)
2026-07-28 19:47:02,521 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-28 event=3200119bc7fbf549a208a197223c9966 (Arizona Diamondbacks@Pittsburgh Pirates) | credits=20444
2026-07-28 19:47:03,390 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-28 event=6fb81f375a5fc10737ecb85381f1ac0b (New York Yankees@Chicago White Sox) | credits=20437
2026-07-28 19:47:04,840 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-28 event=8fa2b64a3a88b00af2cc037c70eaed6e (Kansas City Royals@Minnesota Twins) | credits=20433
2026-07-28 19:47:05,874 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-28 event=83b4bc6720bc5b4953002ee5327e881d (Chicago Cubs@St. Louis Cardinals) | credits=20426
2026-07-28 19:47:06,879 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-28 event=652c3471321a108cc9222b8e5b773ad2 (Toronto Blue Jays@Washington Nationals) | credits=20419
2026-07-28 19:47:08,591 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-28 event=a981fee543b7844ab5bf4928bdca43d2 (Houston Astros@Los Angeles Angels) | credits=20411
2026-07-28 19:47:09,448 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-28 event=0f350cb709d6394c946ef1df6fca40af (Boston Red Sox@Athletics) | credits=20403
2026-07-28 19:47:10,658 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-28 event=f829a782d542dc9d3eb459cddb29492c (Colorado Rockies@San Diego Padres) | credits=20395
2026-07-28 19:47:15,249 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-28 event=b77df217e739f24b8ab3440a9f013da0 (Milwaukee Brewers@San Francisco Giants) | credits=20387
2026-07-28 19:47:16,934 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-28 event=0c2627afd8b3d72890ab2725df592412 (Seattle Mariners@Los Angeles Dodgers) | credits=20379
2026-07-28 19:47:17,900 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 16 events (game_date=2026-07-29, as_of=2026-07-28)
2026-07-28 19:47:18,178 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=20379
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-28 19:47:24,497 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1627 rows into odds.mlb_game_lines (live odds).
2026-07-28 19:47:24,502 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-28
2026-07-28 19:47:24,503 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-28 19:47:24,503 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-29T01:17:24.503059+00:00.
2026-07-28 19:47:24,561 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-27
2026-07-28 19:47:52,818 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1816 rows into odds.mlb_player_prop_lines.
2026-07-28 19:47:52,818 | INFO | mlb_pipeline.parse_oddsapi | Processed 1816 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-28 19:47:52,819 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 124

**stderr (tail)**
```
2026-07-28 19:48:41,747 | WARNING | mlb_pipeline.modeling.prop_replay | Skipping prop replay CLV refresh row id=329521 after resolver/update failure: canceling statement due to statement timeout

2026-07-28 19:49:24,870 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=100 updated=99 skipped=1 last_id=329584
2026-07-28 19:49:29,453 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=200 updated=199 skipped=1 last_id=329684
2026-07-28 19:49:38,179 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=300 updated=299 skipped=1 last_id=329784
2026-07-28 19:49:41,126 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=400 updated=399 skipped=1 last_id=329884
2026-07-28 19:49:44,941 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=500 updated=499 skipped=1 last_id=329984
2026-07-28 19:49:51,287 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=600 updated=599 skipped=1 last_id=330084
2026-07-28 19:49:54,012 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=700 updated=699 skipped=1 last_id=330184
2026-07-28 19:49:55,870 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=800 updated=799 skipped=1 last_id=330284
2026-07-28 19:49:56,265 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=900 updated=899 skipped=1 last_id=330384
2026-07-28 19:49:56,578 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1000 updated=999 skipped=1 last_id=330484
2026-07-28 19:49:56,858 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1100 updated=1099 skipped=1 last_id=330584
2026-07-28 19:49:57,609 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1200 updated=1199 skipped=1 last_id=330684
2026-07-28 19:49:58,221 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1300 updated=1299 skipped=1 last_id=330784
2026-07-28 19:50:03,784 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1400 updated=1399 skipped=1 last_id=330884
2026-07-28 19:50:05,627 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1500 updated=1499 skipped=1 last_id=330984
2026-07-28 19:50:07,175 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1600 updated=1599 skipped=1 last_id=331084
2026-07-28 19:50:08,275 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1700 updated=1699 skipped=1 last_id=331184
2026-07-28 19:50:10,369 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1800 updated=1799 skipped=1 last_id=331284
2026-07-28 19:50:11,784 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=1900 updated=1899 skipped=1 last_id=331384
2026-07-28 19:50:17,371 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2000 updated=1999 skipped=1 last_id=331484
2026-07-28 19:50:19,768 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2100 updated=2099 skipped=1 last_id=331584
2026-07-28 19:50:21,243 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2200 updated=2199 skipped=1 last_id=331684
2026-07-28 19:50:26,332 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2300 updated=2299 skipped=1 last_id=331784
2026-07-28 19:50:27,652 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2400 updated=2399 skipped=1 last_id=331884
2026-07-28 19:50:29,384 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2500 updated=2499 skipped=1 last_id=331984
2026-07-28 19:50:30,288 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2600 updated=2599 skipped=1 last_id=332084
2026-07-28 19:50:33,322 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2700 updated=2699 skipped=1 last_id=332184
2026-07-28 19:50:38,064 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2800 updated=2799 skipped=1 last_id=332284
2026-07-28 19:50:46,412 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=2900 updated=2899 skipped=1 last_id=332384
2026-07-28 19:50:49,943 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3000 updated=2999 skipped=1 last_id=332484
2026-07-28 19:50:54,309 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3100 updated=3099 skipped=1 last_id=332584
2026-07-28 19:50:56,513 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3200 updated=3199 skipped=1 last_id=332684
2026-07-28 19:50:57,446 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3300 updated=3299 skipped=1 last_id=332784
2026-07-28 19:51:01,883 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3400 updated=3399 skipped=1 last_id=332884
2026-07-28 19:51:03,243 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3500 updated=3499 skipped=1 last_id=332984
2026-07-28 19:51:04,231 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3600 updated=3599 skipped=1 last_id=333084
2026-07-28 19:51:04,574 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3700 updated=3699 skipped=1 last_id=333184
2026-07-28 19:51:06,019 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3800 updated=3799 skipped=1 last_id=333284
2026-07-28 19:51:12,824 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=3900 updated=3899 skipped=1 last_id=333384
2026-07-28 19:51:14,847 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4000 updated=3999 skipped=1 last_id=333484
2026-07-28 19:51:19,641 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4100 updated=4099 skipped=1 last_id=333584
2026-07-28 19:51:26,297 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4200 updated=4199 skipped=1 last_id=333684
2026-07-28 19:51:32,372 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4300 updated=4299 skipped=1 last_id=333784
2026-07-28 19:51:35,012 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4400 updated=4399 skipped=1 last_id=333884
2026-07-28 19:51:40,433 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4500 updated=4499 skipped=1 last_id=333984
2026-07-28 19:51:41,533 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4600 updated=4599 skipped=1 last_id=334084
2026-07-28 19:51:43,862 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4700 updated=4699 skipped=1 last_id=334184
2026-07-28 19:51:49,461 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4800 updated=4799 skipped=1 last_id=334284
2026-07-28 19:51:52,100 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=4900 updated=4899 skipped=1 last_id=334384
2026-07-28 19:51:56,182 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5000 updated=4999 skipped=1 last_id=334484
2026-07-28 19:51:57,384 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5100 updated=5099 skipped=1 last_id=334584
2026-07-28 19:51:58,640 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5200 updated=5199 skipped=1 last_id=334684
2026-07-28 19:51:59,555 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5300 updated=5299 skipped=1 last_id=334784
2026-07-28 19:52:04,176 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5400 updated=5399 skipped=1 last_id=334884
2026-07-28 19:52:07,910 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5500 updated=5499 skipped=1 last_id=334984
2026-07-28 19:52:17,221 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5600 updated=5599 skipped=1 last_id=335084
2026-07-28 19:52:19,738 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5700 updated=5699 skipped=1 last_id=335184
2026-07-28 19:52:23,452 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5800 updated=5799 skipped=1 last_id=335284
2026-07-28 19:52:28,224 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=5900 updated=5899 skipped=1 last_id=335384
2026-07-28 19:52:30,887 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6000 updated=5999 skipped=1 last_id=335484
2026-07-28 19:52:31,978 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6100 updated=6099 skipped=1 last_id=335584
2026-07-28 19:52:33,408 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6200 updated=6199 skipped=1 last_id=335684
2026-07-28 19:52:38,084 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6300 updated=6299 skipped=1 last_id=335784
2026-07-28 19:52:41,143 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6400 updated=6399 skipped=1 last_id=335884
2026-07-28 19:52:42,853 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6500 updated=6499 skipped=1 last_id=335984
2026-07-28 19:52:49,578 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6600 updated=6599 skipped=1 last_id=336084
2026-07-28 19:52:56,512 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6700 updated=6699 skipped=1 last_id=336184
2026-07-28 19:53:03,111 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6800 updated=6799 skipped=1 last_id=336284
2026-07-28 19:53:04,306 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=6900 updated=6899 skipped=1 last_id=336384
2026-07-28 19:53:06,188 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=7000 updated=6999 skipped=1 last_id=336484
2026-07-28 19:53:07,989 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=7100 updated=7099 skipped=1 last_id=336584
2026-07-28 19:53:10,034 | INFO | mlb_pipeline.modeling.prop_replay | Prop replay CLV refresh progress: scanned=7200 updated=7199 skipped=1 last_id=336684
Timed out after 300s; killed process tree rooted at PID 1900
```
