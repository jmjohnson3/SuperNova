# SuperNovaBets MLB Daily Run (2026-04-26 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 20.8s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.8s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 4.5s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-04-26 18:31:12,291 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-04-27. Catching up from 2026-04-28 to 2026-04-25
2026-04-26 18:31:12,305 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-04-26 window=2026-04-25T18:00:00Z..2026-04-27T04:00:00Z
2026-04-26 18:31:13,243 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-04-26 | events=1 | credits_remaining=4029
2026-04-26 18:31:13,464 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-04-27 window=2026-04-26T18:00:00Z..2026-04-28T04:00:00Z
2026-04-26 18:31:14,071 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-04-27 | events=9 | credits_remaining=4027
2026-04-26 18:31:14,128 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=4027
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-04-26 18:31:17,066 | INFO | mlb_pipeline.parse_oddsapi | Upserted 690 rows into odds.mlb_game_lines (live odds).
2026-04-26 18:31:17,094 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-04-26
2026-04-26 18:31:17,095 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-04-26 18:31:17,135 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-04-25
2026-04-26 18:31:18,894 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6455 rows into odds.mlb_player_prop_lines.
2026-04-26 18:31:18,895 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 95-55 (63.3%) ROI: +20.9% | Total: 21-29 (42.0%) ROI: -19.8%
MLB CLV Run Line: beat close 19/150 (13%) avg CLV=+0.39 runs | CLV Total avg=+0.46 runs
MLB Price CLV Run Line: 79 bets  avg=-0.33%
```

**stderr (tail)**
```
2026-04-26 18:31:21,588 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 14 rows
2026-04-26 18:31:21,588 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 14 MLB game outcome rows
2026-04-26 18:31:21,789 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 103 rows
2026-04-26 18:31:21,789 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 103 historical rows
2026-04-26 18:31:23,407 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1052 prop rows
2026-04-26 18:31:23,411 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1052 MLB prop outcome rows
```
