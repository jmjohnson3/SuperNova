# SuperNovaBets MLB Daily Run (2026-04-29 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 254.0s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 8.0s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 4.1s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-04-29 18:36:58,490 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-04-30. Catching up from 2026-05-01 to 2026-04-28
2026-04-29 18:36:58,503 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-04-29 window=2026-04-28T18:00:00Z..2026-04-30T04:00:00Z
2026-04-29 18:37:08,797 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-04-29 | events=6 | credits_remaining=3719
2026-04-29 18:37:09,280 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-04-30 window=2026-04-29T18:00:00Z..2026-05-01T04:00:00Z
2026-04-29 18:37:09,937 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-04-30 | events=15 | credits_remaining=3717
2026-04-29 18:37:09,948 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=3717
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-04-29 18:37:15,996 | INFO | mlb_pipeline.parse_oddsapi | Upserted 760 rows into odds.mlb_game_lines (live odds).
2026-04-29 18:37:16,035 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-04-29
2026-04-29 18:37:16,036 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-04-29 18:37:16,161 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-04-28
2026-04-29 18:37:18,126 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6254 rows into odds.mlb_player_prop_lines.
2026-04-29 18:37:18,128 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 99-63 (61.1%) ROI: +16.7% | Total: 25-31 (44.6%) ROI: -14.8%
MLB CLV Run Line: beat close 24/162 (15%) avg CLV=+0.41 runs | CLV Total avg=+0.58 runs
MLB Price CLV Run Line: 91 bets  avg=-0.95%
```

**stderr (tail)**
```
2026-04-29 18:37:20,804 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 21 pending predictions.
2026-04-29 18:37:20,804 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-04-29 18:37:21,036 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 106 rows
2026-04-29 18:37:21,036 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 106 historical rows
2026-04-29 18:37:22,158 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-04-29 18:37:22,162 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
```
