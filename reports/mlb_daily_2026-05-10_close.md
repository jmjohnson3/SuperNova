# SuperNovaBets MLB Daily Run (2026-05-10 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 7.1s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.9s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 3.8s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-10 18:30:08,516 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-11. Catching up from 2026-05-12 to 2026-05-09
2026-05-10 18:30:08,516 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-10 window=2026-05-09T18:00:00Z..2026-05-11T04:00:00Z
2026-05-10 18:30:09,289 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-10 | events=1 | credits_remaining=98882
2026-05-10 18:30:09,634 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-11 window=2026-05-10T18:00:00Z..2026-05-12T04:00:00Z
2026-05-10 18:30:10,179 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-11 | events=6 | credits_remaining=98880
2026-05-10 18:30:10,213 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=98880
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-10 18:30:13,447 | INFO | mlb_pipeline.parse_oddsapi | Upserted 917 rows into odds.mlb_game_lines (live odds).
2026-05-10 18:30:13,488 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-10
2026-05-10 18:30:13,490 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-10 18:30:13,516 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-09
2026-05-10 18:30:15,140 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6610 rows into odds.mlb_player_prop_lines.
2026-05-10 18:30:15,142 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 130-79 (62.2%) ROI: +18.7% | Total: 41-49 (45.6%) ROI: -13.0%
MLB CLV Run Line: beat close 48/209 (23%) avg CLV=+0.78 runs | CLV Total avg=+0.51 runs
MLB Price CLV Run Line: 138 bets  avg=-1.52%
```

**stderr (tail)**
```
2026-05-10 18:30:17,705 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 14 rows
2026-05-10 18:30:17,705 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 14 MLB game outcome rows
2026-05-10 18:30:17,806 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 127 rows
2026-05-10 18:30:17,806 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 127 historical rows
2026-05-10 18:30:18,889 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1068 prop rows
2026-05-10 18:30:18,894 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1068 MLB prop outcome rows
```
