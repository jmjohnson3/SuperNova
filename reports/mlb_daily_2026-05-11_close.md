# SuperNovaBets MLB Daily Run (2026-05-11 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 158.6s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 8.8s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 12.4s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-11 18:33:55,031 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-12. Catching up from 2026-05-13 to 2026-05-10
2026-05-11 18:33:55,044 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-11 window=2026-05-10T18:00:00Z..2026-05-12T04:00:00Z
2026-05-11 18:33:57,247 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-11 | events=6 | credits_remaining=98833
2026-05-11 18:33:58,070 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-12 window=2026-05-11T18:00:00Z..2026-05-13T04:00:00Z
2026-05-11 18:33:58,958 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-12 | events=20 | credits_remaining=98831
2026-05-11 18:33:58,974 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=98831
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-11 18:34:06,116 | INFO | mlb_pipeline.parse_oddsapi | Upserted 955 rows into odds.mlb_game_lines (live odds).
2026-05-11 18:34:06,168 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-11
2026-05-11 18:34:06,170 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-11 18:34:06,527 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-10
2026-05-11 18:34:08,441 | INFO | mlb_pipeline.parse_oddsapi | Upserted 4462 rows into odds.mlb_player_prop_lines.
2026-05-11 18:34:08,442 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 136-81 (62.7%) ROI: +19.6% | Total: 41-50 (45.1%) ROI: -14.0%
MLB CLV Run Line: beat close 50/217 (23%) avg CLV=+0.79 runs | CLV Total avg=+0.54 runs
MLB Price CLV Run Line: 146 bets  avg=-1.56%
```

**stderr (tail)**
```
2026-05-11 18:34:16,074 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-05-11 18:34:16,075 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-05-11 18:34:16,209 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 128 rows
2026-05-11 18:34:16,209 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 128 historical rows
2026-05-11 18:34:20,815 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1130 prop rows
2026-05-11 18:34:20,822 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1130 MLB prop outcome rows
```
