# SuperNovaBets MLB Daily Run (2026-05-18 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 73.6s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 8.5s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 8.5s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-18 18:33:13,030 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-19. Catching up from 2026-05-20 to 2026-05-17
2026-05-18 18:33:13,030 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-18 window=2026-05-17T18:00:00Z..2026-05-19T04:00:00Z
2026-05-18 18:33:15,246 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-18 | events=14 | credits_remaining=98061
2026-05-18 18:33:15,944 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-19 window=2026-05-18T18:00:00Z..2026-05-20T04:00:00Z
2026-05-18 18:33:16,581 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-19 | events=28 | credits_remaining=98059
2026-05-18 18:33:16,617 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=98059
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-18 18:33:21,883 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1114 rows into odds.mlb_game_lines (live odds).
2026-05-18 18:33:21,908 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-18
2026-05-18 18:33:21,908 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-18 18:33:22,079 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-17
2026-05-18 18:33:25,093 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6235 rows into odds.mlb_player_prop_lines.
2026-05-18 18:33:25,095 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 152-93 (62.0%) ROI: +18.4% | Total: 51-53 (49.0%) ROI: -6.4%
MLB CLV Run Line: beat close 61/245 (25%) avg CLV=+0.87 runs | CLV Total avg=+0.55 runs
MLB Price CLV Run Line: 174 bets  avg=-1.64%
```

**stderr (tail)**
```
2026-05-18 18:33:29,919 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 26 pending predictions.
2026-05-18 18:33:29,919 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-05-18 18:33:30,464 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 136 rows
2026-05-18 18:33:30,464 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 136 historical rows
2026-05-18 18:33:33,562 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-05-18 18:33:33,568 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
```
