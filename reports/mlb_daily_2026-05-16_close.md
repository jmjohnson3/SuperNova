# SuperNovaBets MLB Daily Run (2026-05-16 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 6.8s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 5.2s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 4.0s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-16 18:30:09,618 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-17. Catching up from 2026-05-18 to 2026-05-15
2026-05-16 18:30:09,618 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-16 window=2026-05-15T18:00:00Z..2026-05-17T04:00:00Z
2026-05-16 18:30:10,428 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-16 | events=9 | credits_remaining=98294
2026-05-16 18:30:10,663 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-17 window=2026-05-16T18:00:00Z..2026-05-18T04:00:00Z
2026-05-16 18:30:11,215 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-17 | events=22 | credits_remaining=98292
2026-05-16 18:30:11,226 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=98292
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-16 18:30:15,063 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1073 rows into odds.mlb_game_lines (live odds).
2026-05-16 18:30:15,065 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-16
2026-05-16 18:30:15,066 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-16 18:30:15,138 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-15
2026-05-16 18:30:16,460 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6258 rows into odds.mlb_player_prop_lines.
2026-05-16 18:30:16,462 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 147-92 (61.5%) ROI: +17.4% | Total: 47-53 (47.0%) ROI: -10.3%
MLB CLV Run Line: beat close 57/239 (24%) avg CLV=+0.83 runs | CLV Total avg=+0.47 runs
MLB Price CLV Run Line: 168 bets  avg=-1.60%
```

**stderr (tail)**
```
2026-05-16 18:30:19,030 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-05-16 18:30:19,030 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-05-16 18:30:19,137 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 133 rows
2026-05-16 18:30:19,137 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 133 historical rows
2026-05-16 18:30:20,437 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1108 prop rows
2026-05-16 18:30:20,444 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1108 MLB prop outcome rows
```
