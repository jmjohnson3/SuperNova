# SuperNovaBets MLB Daily Run (2026-05-17 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 5.8s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.6s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 3.5s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-17 18:30:15,724 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-18. Catching up from 2026-05-19 to 2026-05-16
2026-05-17 18:30:15,725 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-17 window=2026-05-16T18:00:00Z..2026-05-18T04:00:00Z
2026-05-17 18:30:16,693 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-17 | events=1 | credits_remaining=98170
2026-05-17 18:30:17,034 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-18 window=2026-05-17T18:00:00Z..2026-05-19T04:00:00Z
2026-05-17 18:30:17,576 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-18 | events=15 | credits_remaining=98168
2026-05-17 18:30:17,616 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=98168
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-17 18:30:20,576 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1061 rows into odds.mlb_game_lines (live odds).
2026-05-17 18:30:20,589 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-17
2026-05-17 18:30:20,590 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-17 18:30:20,692 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-16
2026-05-17 18:30:22,252 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6472 rows into odds.mlb_player_prop_lines.
2026-05-17 18:30:22,255 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 152-93 (62.0%) ROI: +18.4% | Total: 51-53 (49.0%) ROI: -6.4%
MLB CLV Run Line: beat close 61/245 (25%) avg CLV=+0.87 runs | CLV Total avg=+0.55 runs
MLB Price CLV Run Line: 174 bets  avg=-1.64%
```

**stderr (tail)**
```
2026-05-17 18:30:24,807 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-05-17 18:30:24,807 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-05-17 18:30:24,915 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 136 rows
2026-05-17 18:30:24,915 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 136 historical rows
2026-05-17 18:30:25,720 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1118 prop rows
2026-05-17 18:30:25,726 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1118 MLB prop outcome rows
```
