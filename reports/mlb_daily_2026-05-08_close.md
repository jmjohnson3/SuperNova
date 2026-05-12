# SuperNovaBets MLB Daily Run (2026-05-08 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 6.5s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 5.6s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 5.4s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-08 18:30:09,484 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-09. Catching up from 2026-05-10 to 2026-05-07
2026-05-08 18:30:09,484 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-08 window=2026-05-07T18:00:00Z..2026-05-09T04:00:00Z
2026-05-08 18:30:10,171 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-08 | events=15 | credits_remaining=99133
2026-05-08 18:30:10,504 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-09 window=2026-05-08T18:00:00Z..2026-05-10T04:00:00Z
2026-05-08 18:30:11,061 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-09 | events=30 | credits_remaining=99131
2026-05-08 18:30:11,113 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99131
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-08 18:30:15,124 | INFO | mlb_pipeline.parse_oddsapi | Upserted 943 rows into odds.mlb_game_lines (live odds).
2026-05-08 18:30:15,136 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-08
2026-05-08 18:30:15,136 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-08 18:30:15,262 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-07
2026-05-08 18:30:16,702 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5305 rows into odds.mlb_player_prop_lines.
2026-05-08 18:30:16,704 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 123-77 (61.5%) ROI: +17.4% | Total: 40-45 (47.1%) ROI: -10.2%
MLB CLV Run Line: beat close 42/200 (21%) avg CLV=+0.67 runs | CLV Total avg=+0.49 runs
MLB Price CLV Run Line: 129 bets  avg=-1.11%
```

**stderr (tail)**
```
2026-05-08 18:30:19,516 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 9 rows
2026-05-08 18:30:19,516 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 9 MLB game outcome rows
2026-05-08 18:30:19,772 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 124 rows
2026-05-08 18:30:19,772 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 124 historical rows
2026-05-08 18:30:22,016 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 678 prop rows
2026-05-08 18:30:22,021 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 678 MLB prop outcome rows
```
