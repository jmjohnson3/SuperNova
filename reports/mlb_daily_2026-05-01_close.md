# SuperNovaBets MLB Daily Run (2026-05-01 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 11.1s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.3s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 3.5s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-01 18:30:15,320 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-02. Catching up from 2026-05-03 to 2026-04-30
2026-05-01 18:30:15,320 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-01 window=2026-04-30T18:00:00Z..2026-05-02T04:00:00Z
2026-05-01 18:30:16,104 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-01 | events=14 | credits_remaining=99885
2026-05-01 18:30:16,401 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-02 window=2026-05-01T18:00:00Z..2026-05-03T04:00:00Z
2026-05-01 18:30:16,969 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-02 | events=28 | credits_remaining=99883
2026-05-01 18:30:17,013 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99883
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-01 18:30:19,932 | INFO | mlb_pipeline.parse_oddsapi | Upserted 817 rows into odds.mlb_game_lines (live odds).
2026-05-01 18:30:19,948 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-01
2026-05-01 18:30:19,949 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-01 18:30:19,979 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-04-30
2026-05-01 18:30:21,310 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5364 rows into odds.mlb_player_prop_lines.
2026-05-01 18:30:21,311 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 105-67 (61.0%) ROI: +16.5% | Total: 29-32 (47.5%) ROI: -9.2%
MLB CLV Run Line: beat close 33/172 (19%) avg CLV=+0.61 runs | CLV Total avg=+0.58 runs
MLB Price CLV Run Line: 101 bets  avg=-1.56%
```

**stderr (tail)**
```
2026-05-01 18:30:23,792 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 11 rows
2026-05-01 18:30:23,792 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 11 MLB game outcome rows
2026-05-01 18:30:23,913 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 108 rows
2026-05-01 18:30:23,913 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 108 historical rows
2026-05-01 18:30:24,800 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 840 prop rows
2026-05-01 18:30:24,804 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 840 MLB prop outcome rows
```
