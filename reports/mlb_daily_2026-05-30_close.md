# SuperNovaBets MLB Daily Run (2026-05-30 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 40.7s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 15.3s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 17.9s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-30 18:32:07,349 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-31. Catching up from 2026-06-01 to 2026-05-29
2026-05-30 18:32:07,354 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-30 window=2026-05-29T18:00:00Z..2026-05-31T04:00:00Z
2026-05-30 18:32:10,016 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-30 | events=6 | credits_remaining=96936
2026-05-30 18:32:10,706 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-31 window=2026-05-30T18:00:00Z..2026-06-01T04:00:00Z
2026-05-30 18:32:11,277 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-31 | events=21 | credits_remaining=96934
2026-05-30 18:32:11,355 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=96934
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-30 18:32:22,141 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1303 rows into odds.mlb_game_lines (live odds).
2026-05-30 18:32:22,202 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-30
2026-05-30 18:32:22,203 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-30 18:32:22,459 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-29
2026-05-30 18:32:27,238 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5892 rows into odds.mlb_player_prop_lines.
2026-05-30 18:32:27,240 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 171-114 (60.0%) ROI: +14.5% | Total: 56-58 (49.1%) ROI: -6.2%
MLB CLV Run Line: beat close 74/285 (26%) avg CLV=+0.88 runs | CLV Total avg=+0.53 runs
MLB Price CLV Run Line: 214 bets  avg=-2.02%
```

**stderr (tail)**
```
2026-05-30 18:32:42,345 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-05-30 18:32:42,345 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-05-30 18:32:42,545 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 141 rows
2026-05-30 18:32:42,547 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 141 historical rows
2026-05-30 18:32:45,053 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 815 prop rows
2026-05-30 18:32:45,060 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 815 MLB prop outcome rows
```
