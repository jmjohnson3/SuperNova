# SuperNovaBets MLB Daily Run (2026-05-22 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 11.9s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 5.7s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 3.6s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-22 18:30:42,321 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-23. Catching up from 2026-05-24 to 2026-05-21
2026-05-22 18:30:42,322 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-22 window=2026-05-21T18:00:00Z..2026-05-23T04:00:00Z
2026-05-22 18:30:43,876 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-22 | events=13 | credits_remaining=97725
2026-05-22 18:30:44,384 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-23 window=2026-05-22T18:00:00Z..2026-05-24T04:00:00Z
2026-05-22 18:30:44,974 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-23 | events=28 | credits_remaining=97723
2026-05-22 18:30:45,010 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=97723
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-22 18:30:49,627 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1195 rows into odds.mlb_game_lines (live odds).
2026-05-22 18:30:49,630 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-22
2026-05-22 18:30:49,631 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-22 18:30:49,678 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-21
2026-05-22 18:30:50,774 | INFO | mlb_pipeline.parse_oddsapi | Upserted 4305 rows into odds.mlb_player_prop_lines.
2026-05-22 18:30:50,775 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 155-98 (61.3%) ROI: +17.0% | Total: 51-55 (48.1%) ROI: -8.1%
MLB CLV Run Line: beat close 62/253 (25%) avg CLV=+0.85 runs | CLV Total avg=+0.55 runs
MLB Price CLV Run Line: 182 bets  avg=-1.64%
```

**stderr (tail)**
```
2026-05-22 18:30:53,406 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 7 rows
2026-05-22 18:30:53,406 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 7 MLB game outcome rows
2026-05-22 18:30:53,619 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 138 rows
2026-05-22 18:30:53,619 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 138 historical rows
2026-05-22 18:30:54,287 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 386 prop rows
2026-05-22 18:30:54,294 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 386 MLB prop outcome rows
```
