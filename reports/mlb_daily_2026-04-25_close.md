# SuperNovaBets MLB Daily Run (2026-04-25 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 6.2s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.0s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 5.5s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-04-25 18:30:16,549 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-04-26. Catching up from 2026-04-27 to 2026-04-24
2026-04-25 18:30:16,563 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-04-25 window=2026-04-24T18:00:00Z..2026-04-26T04:00:00Z
2026-04-25 18:30:17,525 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-04-25 | events=8 | credits_remaining=4157
2026-04-25 18:30:17,891 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-04-26 window=2026-04-25T18:00:00Z..2026-04-27T04:00:00Z
2026-04-25 18:30:18,522 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-04-26 | events=24 | credits_remaining=4155
2026-04-25 18:30:18,534 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=4155
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-04-25 18:30:21,295 | INFO | mlb_pipeline.parse_oddsapi | Upserted 720 rows into odds.mlb_game_lines (live odds).
2026-04-25 18:30:21,297 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-04-25
2026-04-25 18:30:21,298 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-04-25 18:30:21,359 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-04-24
2026-04-25 18:30:22,556 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5787 rows into odds.mlb_player_prop_lines.
2026-04-25 18:30:22,557 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 93-50 (65.0%) ROI: +24.2% | Total: 19-28 (40.4%) ROI: -22.8%
MLB CLV Run Line: beat close 18/143 (13%) avg CLV=+0.41 runs | CLV Total avg=+0.34 runs
MLB Price CLV Run Line: 72 bets  avg=-0.68%
```

**stderr (tail)**
```
2026-04-25 18:30:25,233 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 132 rows
2026-04-25 18:30:25,233 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 132 MLB game outcome rows
2026-04-25 18:30:25,282 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 102 rows
2026-04-25 18:30:25,282 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 102 historical rows
2026-04-25 18:30:27,998 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 10419 prop rows
2026-04-25 18:30:28,004 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 10419 MLB prop outcome rows
```
