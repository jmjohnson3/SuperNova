# SuperNovaBets MLB Daily Run (2026-05-26 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 13.6s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 25.7s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 27.6s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-26 18:30:17,824 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-27. Catching up from 2026-05-28 to 2026-05-25
2026-05-26 18:30:17,954 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-26 window=2026-05-25T18:00:00Z..2026-05-27T04:00:00Z
2026-05-26 18:30:20,447 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-26 | events=15 | credits_remaining=97264
2026-05-26 18:30:21,326 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-27 window=2026-05-26T18:00:00Z..2026-05-28T04:00:00Z
2026-05-26 18:30:21,956 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-27 | events=29 | credits_remaining=97262
2026-05-26 18:30:22,062 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=97262
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-26 18:30:28,996 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1250 rows into odds.mlb_game_lines (live odds).
2026-05-26 18:30:29,069 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-26
2026-05-26 18:30:29,069 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-26 18:30:29,726 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-25
2026-05-26 18:30:47,870 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5761 rows into odds.mlb_player_prop_lines.
2026-05-26 18:30:47,872 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 165-106 (60.9%) ROI: +16.2% | Total: 56-58 (49.1%) ROI: -6.2%
MLB CLV Run Line: beat close 70/271 (26%) avg CLV=+0.92 runs | CLV Total avg=+0.53 runs
MLB Price CLV Run Line: 200 bets  avg=-1.73%
```

**stderr (tail)**
```
2026-05-26 18:31:06,924 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 31 pending predictions.
2026-05-26 18:31:06,924 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-05-26 18:31:07,352 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 141 rows
2026-05-26 18:31:07,352 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 141 historical rows
2026-05-26 18:31:15,328 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-05-26 18:31:15,336 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
```
