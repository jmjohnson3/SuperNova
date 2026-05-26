# SuperNovaBets MLB Daily Run (2026-05-24 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 19.6s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 9.2s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 49.3s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-24 18:31:20,081 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-25. Catching up from 2026-05-26 to 2026-05-23
2026-05-24 18:31:20,082 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-24 window=2026-05-23T18:00:00Z..2026-05-25T04:00:00Z
2026-05-24 18:31:22,996 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-24 | events=3 | credits_remaining=97528
2026-05-24 18:31:24,052 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-25 window=2026-05-24T18:00:00Z..2026-05-26T04:00:00Z
2026-05-24 18:31:24,667 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-25 | events=16 | credits_remaining=97526
2026-05-24 18:31:24,707 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=97526
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-24 18:31:30,247 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1192 rows into odds.mlb_game_lines (live odds).
2026-05-24 18:31:30,318 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-24
2026-05-24 18:31:30,319 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-24 18:31:30,589 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-23
2026-05-24 18:31:34,028 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6354 rows into odds.mlb_player_prop_lines.
2026-05-24 18:31:34,029 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 160-103 (60.8%) ROI: +16.1% | Total: 54-58 (48.2%) ROI: -8.0%
MLB CLV Run Line: beat close 68/263 (26%) avg CLV=+0.92 runs | CLV Total avg=+0.54 runs
MLB Price CLV Run Line: 192 bets  avg=-1.63%
```

**stderr (tail)**
```
2026-05-24 18:31:38,666 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 14 rows
2026-05-24 18:31:38,667 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 14 MLB game outcome rows
2026-05-24 18:31:38,741 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 140 rows
2026-05-24 18:31:38,741 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 140 historical rows
2026-05-24 18:32:06,051 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 742 prop rows
2026-05-24 18:32:06,058 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 742 MLB prop outcome rows
```
