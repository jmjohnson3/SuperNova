# SuperNovaBets MLB Daily Run (2026-06-01 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 46.7s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 19.8s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 15.1s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-01 18:30:23,559 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-02. Catching up from 2026-06-03 to 2026-05-31
2026-06-01 18:30:23,559 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-01 window=2026-05-31T18:00:00Z..2026-06-02T04:00:00Z
2026-06-01 18:30:43,938 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-01 | events=9 | credits_remaining=99936
2026-06-01 18:30:48,131 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-02 window=2026-06-01T18:00:00Z..2026-06-03T04:00:00Z
2026-06-01 18:30:48,775 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-02 | events=24 | credits_remaining=99934
2026-06-01 18:30:48,794 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99934
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-01 18:31:06,747 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1326 rows into odds.mlb_game_lines (live odds).
2026-06-01 18:31:06,810 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-01
2026-06-01 18:31:06,811 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-01 18:31:06,902 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-31
2026-06-01 18:31:11,594 | INFO | mlb_pipeline.parse_oddsapi | Upserted 4945 rows into odds.mlb_player_prop_lines.
2026-06-01 18:31:11,596 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 176-119 (59.7%) ROI: +13.9% | Total: 56-58 (49.1%) ROI: -6.2%
MLB CLV Run Line: beat close 78/295 (26%) avg CLV=+0.93 runs | CLV Total avg=+0.53 runs
MLB Price CLV Run Line: 224 bets  avg=-1.94%
```

**stderr (tail)**
```
2026-06-01 18:31:16,842 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-06-01 18:31:16,842 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-06-01 18:31:17,279 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 141 rows
2026-06-01 18:31:17,280 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 141 historical rows
2026-06-01 18:31:26,510 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 824 prop rows
2026-06-01 18:31:26,516 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 824 MLB prop outcome rows
```
