# SuperNovaBets MLB Daily Run (2026-05-29 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 11.2s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 6.1s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 7.3s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-29 18:31:41,893 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-30. Catching up from 2026-05-31 to 2026-05-28
2026-05-29 18:31:41,893 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-29 window=2026-05-28T18:00:00Z..2026-05-30T04:00:00Z
2026-05-29 18:31:43,565 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-29 | events=15 | credits_remaining=97034
2026-05-29 18:31:44,118 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-30 window=2026-05-29T18:00:00Z..2026-05-31T04:00:00Z
2026-05-29 18:31:44,731 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-30 | events=28 | credits_remaining=97032
2026-05-29 18:31:44,754 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=97032
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-29 18:31:48,828 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1305 rows into odds.mlb_game_lines (live odds).
2026-05-29 18:31:48,876 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-29
2026-05-29 18:31:48,877 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-29 18:31:49,016 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-28
2026-05-29 18:31:50,867 | INFO | mlb_pipeline.parse_oddsapi | Upserted 4023 rows into odds.mlb_player_prop_lines.
2026-05-29 18:31:50,868 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 169-111 (60.4%) ROI: +15.2% | Total: 56-58 (49.1%) ROI: -6.2%
MLB CLV Run Line: beat close 72/280 (26%) avg CLV=+0.90 runs | CLV Total avg=+0.53 runs
MLB Price CLV Run Line: 209 bets  avg=-1.76%
```

**stderr (tail)**
```
2026-05-29 18:31:54,196 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 6 rows
2026-05-29 18:31:54,196 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 6 MLB game outcome rows
2026-05-29 18:31:54,606 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 141 rows
2026-05-29 18:31:54,607 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 141 historical rows
2026-05-29 18:31:58,075 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 323 prop rows
2026-05-29 18:31:58,082 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 323 MLB prop outcome rows
```
