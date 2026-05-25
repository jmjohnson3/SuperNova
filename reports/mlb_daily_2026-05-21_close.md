# SuperNovaBets MLB Daily Run (2026-05-21 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 14.2s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 8.0s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 5.8s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-21 18:30:40,295 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-22. Catching up from 2026-05-23 to 2026-05-20
2026-05-21 18:30:40,295 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-21 window=2026-05-20T18:00:00Z..2026-05-22T04:00:00Z
2026-05-21 18:30:42,457 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-21 | events=4 | credits_remaining=97820
2026-05-21 18:30:43,097 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-22 window=2026-05-21T18:00:00Z..2026-05-23T04:00:00Z
2026-05-21 18:30:43,649 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-22 | events=16 | credits_remaining=97818
2026-05-21 18:30:43,684 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=97818
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-21 18:30:49,338 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1148 rows into odds.mlb_game_lines (live odds).
2026-05-21 18:30:49,358 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-21
2026-05-21 18:30:49,359 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-21 18:30:49,729 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-20
2026-05-21 18:30:51,899 | INFO | mlb_pipeline.parse_oddsapi | Upserted 3978 rows into odds.mlb_player_prop_lines.
2026-05-21 18:30:51,900 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 153-98 (61.0%) ROI: +16.4% | Total: 51-55 (48.1%) ROI: -8.1%
MLB CLV Run Line: beat close 61/251 (24%) avg CLV=+0.85 runs | CLV Total avg=+0.55 runs
MLB Price CLV Run Line: 180 bets  avg=-1.66%
```

**stderr (tail)**
```
2026-05-21 18:30:55,112 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 12 rows
2026-05-21 18:30:55,112 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 12 MLB game outcome rows
2026-05-21 18:30:55,453 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 138 rows
2026-05-21 18:30:55,454 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 138 historical rows
2026-05-21 18:30:57,630 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 672 prop rows
2026-05-21 18:30:57,636 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 672 MLB prop outcome rows
```
