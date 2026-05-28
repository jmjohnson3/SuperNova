# SuperNovaBets MLB Daily Run (2026-05-27 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 43.9s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 13.2s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 13.0s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-27 18:38:17,457 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-28. Catching up from 2026-05-29 to 2026-05-26
2026-05-27 18:38:17,458 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-27 window=2026-05-26T18:00:00Z..2026-05-28T04:00:00Z
2026-05-27 18:38:24,098 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-27 | events=9 | credits_remaining=97167
2026-05-27 18:38:25,763 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-28 window=2026-05-27T18:00:00Z..2026-05-29T04:00:00Z
2026-05-27 18:38:26,492 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-28 | events=15 | credits_remaining=97165
2026-05-27 18:38:27,041 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=97165
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-27 18:38:37,151 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1244 rows into odds.mlb_game_lines (live odds).
2026-05-27 18:38:37,185 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-27
2026-05-27 18:38:37,186 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-27 18:38:37,395 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-26
2026-05-27 18:38:40,593 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6155 rows into odds.mlb_player_prop_lines.
2026-05-27 18:38:40,596 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 166-108 (60.6%) ROI: +15.7% | Total: 56-58 (49.1%) ROI: -6.2%
MLB CLV Run Line: beat close 71/274 (26%) avg CLV=+0.91 runs | CLV Total avg=+0.53 runs
MLB Price CLV Run Line: 203 bets  avg=-1.81%
```

**stderr (tail)**
```
2026-05-27 18:38:48,455 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-05-27 18:38:48,455 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-05-27 18:38:48,932 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 141 rows
2026-05-27 18:38:48,935 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 141 historical rows
2026-05-27 18:38:53,712 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 847 prop rows
2026-05-27 18:38:53,721 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 847 MLB prop outcome rows
```
