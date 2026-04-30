# SuperNovaBets MLB Daily Run (2026-04-28 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 7.0s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 5.2s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 4.6s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-04-28 18:30:47,349 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-04-29. Catching up from 2026-04-30 to 2026-04-27
2026-04-28 18:30:47,350 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-04-28 window=2026-04-27T18:00:00Z..2026-04-29T04:00:00Z
2026-04-28 18:30:48,162 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-04-28 | events=15 | credits_remaining=3840
2026-04-28 18:30:48,550 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-04-29 window=2026-04-28T18:00:00Z..2026-04-30T04:00:00Z
2026-04-28 18:30:49,096 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-04-29 | events=29 | credits_remaining=3838
2026-04-28 18:30:49,117 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=3838
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-04-28 18:30:52,252 | INFO | mlb_pipeline.parse_oddsapi | Upserted 775 rows into odds.mlb_game_lines (live odds).
2026-04-28 18:30:52,255 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-04-28
2026-04-28 18:30:52,255 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-04-28 18:30:52,363 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-04-27
2026-04-28 18:30:54,340 | INFO | mlb_pipeline.parse_oddsapi | Upserted 4729 rows into odds.mlb_player_prop_lines.
2026-04-28 18:30:54,341 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 98-59 (62.4%) ROI: +19.2% | Total: 23-31 (42.6%) ROI: -18.7%
MLB CLV Run Line: beat close 23/157 (15%) avg CLV=+0.46 runs | CLV Total avg=+0.53 runs
MLB Price CLV Run Line: 86 bets  avg=+0.20%
```

**stderr (tail)**
```
2026-04-28 18:30:57,077 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 8 rows
2026-04-28 18:30:57,077 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 8 MLB game outcome rows
2026-04-28 18:30:57,177 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 105 rows
2026-04-28 18:30:57,177 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 105 historical rows
2026-04-28 18:30:58,914 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 596 prop rows
2026-04-28 18:30:58,918 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 596 MLB prop outcome rows
```
