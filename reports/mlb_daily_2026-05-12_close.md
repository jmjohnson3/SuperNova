# SuperNovaBets MLB Daily Run (2026-05-12 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 13.2s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 6.9s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 9.2s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-12 18:31:16,352 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-13. Catching up from 2026-05-14 to 2026-05-11
2026-05-12 18:31:16,352 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-12 window=2026-05-11T18:00:00Z..2026-05-13T04:00:00Z
2026-05-12 18:31:17,387 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-12 | events=15 | credits_remaining=98720
2026-05-12 18:31:17,925 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-13 window=2026-05-12T18:00:00Z..2026-05-14T04:00:00Z
2026-05-12 18:31:18,572 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-13 | events=29 | credits_remaining=98718
2026-05-12 18:31:18,606 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=98718
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-12 18:31:23,322 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1005 rows into odds.mlb_game_lines (live odds).
2026-05-12 18:31:23,362 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-12
2026-05-12 18:31:23,362 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-12 18:31:23,591 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-11
2026-05-12 18:31:25,575 | INFO | mlb_pipeline.parse_oddsapi | Upserted 4302 rows into odds.mlb_player_prop_lines.
2026-05-12 18:31:25,576 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 137-81 (62.8%) ROI: +20.0% | Total: 41-51 (44.6%) ROI: -14.9%
MLB CLV Run Line: beat close 50/218 (23%) avg CLV=+0.78 runs | CLV Total avg=+0.53 runs
MLB Price CLV Run Line: 147 bets  avg=-1.56%
```

**stderr (tail)**
```
2026-05-12 18:31:29,118 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 6 rows
2026-05-12 18:31:29,118 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 6 MLB game outcome rows
2026-05-12 18:31:29,389 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 128 rows
2026-05-12 18:31:29,389 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 128 historical rows
2026-05-12 18:31:34,688 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 422 prop rows
2026-05-12 18:31:34,693 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 422 MLB prop outcome rows
```
