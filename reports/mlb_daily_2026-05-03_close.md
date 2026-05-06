# SuperNovaBets MLB Daily Run (2026-05-03 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 6.8s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.5s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 3.4s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-03 18:30:07,974 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-04. Catching up from 2026-05-05 to 2026-05-02
2026-05-03 18:30:07,974 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-03 window=2026-05-02T18:00:00Z..2026-05-04T04:00:00Z
2026-05-03 18:30:08,857 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-03 | events=1 | credits_remaining=99653
2026-05-03 18:30:09,095 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-04 window=2026-05-03T18:00:00Z..2026-05-05T04:00:00Z
2026-05-03 18:30:09,625 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-04 | events=13 | credits_remaining=99651
2026-05-03 18:30:09,708 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99651
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-03 18:30:12,573 | INFO | mlb_pipeline.parse_oddsapi | Upserted 803 rows into odds.mlb_game_lines (live odds).
2026-05-03 18:30:12,575 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-03
2026-05-03 18:30:12,576 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-03 18:30:12,624 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-02
2026-05-03 18:30:14,250 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6307 rows into odds.mlb_player_prop_lines.
2026-05-03 18:30:14,252 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 107-72 (59.8%) ROI: +14.1% | Total: 32-36 (47.1%) ROI: -10.2%
MLB CLV Run Line: beat close 33/179 (18%) avg CLV=+0.58 runs | CLV Total avg=+0.49 runs
MLB Price CLV Run Line: 108 bets  avg=-1.55%
```

**stderr (tail)**
```
2026-05-03 18:30:16,791 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-05-03 18:30:16,791 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-05-03 18:30:16,859 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 113 rows
2026-05-03 18:30:16,860 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 113 historical rows
2026-05-03 18:30:17,683 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1121 prop rows
2026-05-03 18:30:17,688 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1121 MLB prop outcome rows
```
