# SuperNovaBets MLB Daily Run (2026-05-31 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 33.2s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 20.4s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 10.6s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-31 18:32:06,254 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-01. Catching up from 2026-06-02 to 2026-05-30
2026-05-31 18:32:06,255 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-31 window=2026-05-30T18:00:00Z..2026-06-01T04:00:00Z
2026-05-31 18:32:07,952 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-31 | events=1 | credits_remaining=99998
2026-05-31 18:32:08,653 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-01 window=2026-05-31T18:00:00Z..2026-06-02T04:00:00Z
2026-05-31 18:32:09,199 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-01 | events=10 | credits_remaining=99996
2026-05-31 18:32:09,321 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99996
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-31 18:32:22,451 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1284 rows into odds.mlb_game_lines (live odds).
2026-05-31 18:32:22,499 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-31
2026-05-31 18:32:22,500 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-31 18:32:23,025 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-30
2026-05-31 18:32:29,857 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6198 rows into odds.mlb_player_prop_lines.
2026-05-31 18:32:29,858 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 173-115 (60.1%) ROI: +14.7% | Total: 56-58 (49.1%) ROI: -6.2%
MLB CLV Run Line: beat close 76/288 (26%) avg CLV=+0.92 runs | CLV Total avg=+0.53 runs
MLB Price CLV Run Line: 217 bets  avg=-1.99%
```

**stderr (tail)**
```
2026-05-31 18:32:34,266 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-05-31 18:32:34,266 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-05-31 18:32:34,407 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 141 rows
2026-05-31 18:32:34,408 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 141 historical rows
2026-05-31 18:32:40,360 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 822 prop rows
2026-05-31 18:32:40,369 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 822 MLB prop outcome rows
```
