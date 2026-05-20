# SuperNovaBets MLB Daily Run (2026-05-13 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 6.9s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 5.2s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 3.8s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-13 18:30:08,367 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-14. Catching up from 2026-05-15 to 2026-05-12
2026-05-13 18:30:08,367 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-13 window=2026-05-12T18:00:00Z..2026-05-14T04:00:00Z
2026-05-13 18:30:09,120 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-13 | events=13 | credits_remaining=98607
2026-05-13 18:30:09,461 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-14 window=2026-05-13T18:00:00Z..2026-05-15T04:00:00Z
2026-05-13 18:30:10,098 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-14 | events=24 | credits_remaining=98605
2026-05-13 18:30:10,102 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=98605
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-13 18:30:13,620 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1022 rows into odds.mlb_game_lines (live odds).
2026-05-13 18:30:13,631 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-13
2026-05-13 18:30:13,632 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-13 18:30:13,678 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-12
2026-05-13 18:30:15,306 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6167 rows into odds.mlb_player_prop_lines.
2026-05-13 18:30:15,307 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 142-83 (63.1%) ROI: +20.5% | Total: 42-52 (44.7%) ROI: -14.7%
MLB CLV Run Line: beat close 53/225 (24%) avg CLV=+0.84 runs | CLV Total avg=+0.47 runs
MLB Price CLV Run Line: 154 bets  avg=-1.64%
```

**stderr (tail)**
```
2026-05-13 18:30:17,920 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-05-13 18:30:17,920 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-05-13 18:30:18,087 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 129 rows
2026-05-13 18:30:18,087 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 129 historical rows
2026-05-13 18:30:19,093 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1132 prop rows
2026-05-13 18:30:19,099 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1132 MLB prop outcome rows
```
