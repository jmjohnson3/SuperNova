# SuperNovaBets MLB Daily Run (2026-05-15 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 8.0s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.8s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 4.1s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-15 18:30:11,025 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-16. Catching up from 2026-05-17 to 2026-05-14
2026-05-15 18:30:11,026 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-15 window=2026-05-14T18:00:00Z..2026-05-16T04:00:00Z
2026-05-15 18:30:11,770 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-15 | events=15 | credits_remaining=98409
2026-05-15 18:30:12,082 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-16 window=2026-05-15T18:00:00Z..2026-05-17T04:00:00Z
2026-05-15 18:30:12,640 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-16 | events=28 | credits_remaining=98407
2026-05-15 18:30:12,652 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=98407
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-15 18:30:16,210 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1067 rows into odds.mlb_game_lines (live odds).
2026-05-15 18:30:16,212 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-15
2026-05-15 18:30:16,212 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-15 18:30:16,272 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-14
2026-05-15 18:30:17,496 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5372 rows into odds.mlb_player_prop_lines.
2026-05-15 18:30:17,498 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 147-90 (62.0%) ROI: +18.4% | Total: 45-52 (46.4%) ROI: -11.4%
MLB CLV Run Line: beat close 57/237 (24%) avg CLV=+0.84 runs | CLV Total avg=+0.46 runs
MLB Price CLV Run Line: 166 bets  avg=-1.66%
```

**stderr (tail)**
```
2026-05-15 18:30:20,122 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 11 rows
2026-05-15 18:30:20,123 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 11 MLB game outcome rows
2026-05-15 18:30:20,336 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 130 rows
2026-05-15 18:30:20,336 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 130 historical rows
2026-05-15 18:30:21,617 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 785 prop rows
2026-05-15 18:30:21,623 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 785 MLB prop outcome rows
```
