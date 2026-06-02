# SuperNovaBets MLB Daily Run (2026-05-28 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 14.7s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 6.7s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 8.7s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-28 18:30:20,550 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-29. Catching up from 2026-05-30 to 2026-05-27
2026-05-28 18:30:20,550 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-28 window=2026-05-27T18:00:00Z..2026-05-29T04:00:00Z
2026-05-28 18:30:21,408 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-28 | events=3 | credits_remaining=97123
2026-05-28 18:30:21,939 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-29 window=2026-05-28T18:00:00Z..2026-05-30T04:00:00Z
2026-05-28 18:30:22,483 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-29 | events=17 | credits_remaining=97121
2026-05-28 18:30:22,535 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=97121
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-28 18:30:27,362 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1253 rows into odds.mlb_game_lines (live odds).
2026-05-28 18:30:27,386 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-28
2026-05-28 18:30:27,386 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-28 18:30:27,571 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-27
2026-05-28 18:30:29,328 | INFO | mlb_pipeline.parse_oddsapi | Upserted 4291 rows into odds.mlb_player_prop_lines.
2026-05-28 18:30:29,329 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 168-110 (60.4%) ROI: +15.4% | Total: 56-58 (49.1%) ROI: -6.2%
MLB CLV Run Line: beat close 72/278 (26%) avg CLV=+0.91 runs | CLV Total avg=+0.53 runs
MLB Price CLV Run Line: 207 bets  avg=-1.78%
```

**stderr (tail)**
```
2026-05-28 18:30:32,515 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-05-28 18:30:32,515 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-05-28 18:30:32,929 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 141 rows
2026-05-28 18:30:32,930 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 141 historical rows
2026-05-28 18:30:37,986 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 840 prop rows
2026-05-28 18:30:37,993 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 840 MLB prop outcome rows
```
