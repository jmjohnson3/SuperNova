# SuperNovaBets MLB Daily Run (2026-05-20 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 61.0s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 23.0s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 39.4s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-20 18:34:57,414 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-21. Catching up from 2026-05-22 to 2026-05-19
2026-05-20 18:34:57,415 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-20 window=2026-05-19T18:00:00Z..2026-05-21T04:00:00Z
2026-05-20 18:35:11,536 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-20 | events=9 | credits_remaining=97870
2026-05-20 18:35:12,697 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-21 window=2026-05-20T18:00:00Z..2026-05-22T04:00:00Z
2026-05-20 18:35:13,544 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-21 | events=16 | credits_remaining=97868
2026-05-20 18:35:13,565 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=97868
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-20 18:35:31,136 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1138 rows into odds.mlb_game_lines (live odds).
2026-05-20 18:35:31,168 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-20
2026-05-20 18:35:31,170 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-20 18:35:32,128 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-19
2026-05-20 18:35:36,854 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5639 rows into odds.mlb_player_prop_lines.
2026-05-20 18:35:36,855 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 153-96 (61.4%) ROI: +17.3% | Total: 51-54 (48.6%) ROI: -7.3%
MLB CLV Run Line: beat close 61/249 (24%) avg CLV=+0.86 runs | CLV Total avg=+0.56 runs
MLB Price CLV Run Line: 178 bets  avg=-1.67%
```

**stderr (tail)**
```
2026-05-20 18:35:59,415 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 17 rows
2026-05-20 18:35:59,415 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 17 MLB game outcome rows
2026-05-20 18:36:00,726 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 137 rows
2026-05-20 18:36:00,726 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 137 historical rows
2026-05-20 18:36:16,262 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1221 prop rows
2026-05-20 18:36:16,269 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1221 MLB prop outcome rows
```
