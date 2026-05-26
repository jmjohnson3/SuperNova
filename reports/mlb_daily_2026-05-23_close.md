# SuperNovaBets MLB Daily Run (2026-05-23 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 11.5s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 5.8s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 5.9s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-23 18:40:44,457 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-24. Catching up from 2026-05-25 to 2026-05-22
2026-05-23 18:40:44,457 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-23 window=2026-05-22T18:00:00Z..2026-05-24T04:00:00Z
2026-05-23 18:40:45,612 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-23 | events=6 | credits_remaining=97629
2026-05-23 18:40:46,262 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-24 window=2026-05-23T18:00:00Z..2026-05-25T04:00:00Z
2026-05-23 18:40:46,807 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-24 | events=21 | credits_remaining=97627
2026-05-23 18:40:46,849 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=97627
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-23 18:40:50,672 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1193 rows into odds.mlb_game_lines (live odds).
2026-05-23 18:40:50,731 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-23
2026-05-23 18:40:50,732 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-23 18:40:50,883 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-22
2026-05-23 18:40:52,660 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5954 rows into odds.mlb_player_prop_lines.
2026-05-23 18:40:52,662 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 157-100 (61.1%) ROI: +16.6% | Total: 52-55 (48.6%) ROI: -7.2%
MLB CLV Run Line: beat close 63/257 (25%) avg CLV=+0.86 runs | CLV Total avg=+0.54 runs
MLB Price CLV Run Line: 186 bets  avg=-1.58%
```

**stderr (tail)**
```
2026-05-23 18:40:55,546 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 14 rows
2026-05-23 18:40:55,546 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 14 MLB game outcome rows
2026-05-23 18:40:55,763 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 138 rows
2026-05-23 18:40:55,763 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 138 historical rows
2026-05-23 18:40:58,390 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 789 prop rows
2026-05-23 18:40:58,396 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 789 MLB prop outcome rows
```
