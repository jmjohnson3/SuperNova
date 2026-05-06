# SuperNovaBets MLB Daily Run (2026-05-05 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 8.2s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.5s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 3.9s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-05 18:30:14,231 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-06. Catching up from 2026-05-07 to 2026-05-04
2026-05-05 18:30:14,231 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-05 window=2026-05-04T18:00:00Z..2026-05-06T04:00:00Z
2026-05-05 18:30:14,905 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-05 | events=13 | credits_remaining=99446
2026-05-05 18:30:15,212 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-06 window=2026-05-05T18:00:00Z..2026-05-07T04:00:00Z
2026-05-05 18:30:15,833 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-06 | events=27 | credits_remaining=99444
2026-05-05 18:30:15,838 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99444
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-05 18:30:18,763 | INFO | mlb_pipeline.parse_oddsapi | Upserted 878 rows into odds.mlb_game_lines (live odds).
2026-05-05 18:30:18,773 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-05
2026-05-05 18:30:18,774 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-05 18:30:18,821 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-04
2026-05-05 18:30:20,303 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5428 rows into odds.mlb_player_prop_lines.
2026-05-05 18:30:20,305 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 115-75 (60.5%) ROI: +15.6% | Total: 36-43 (45.6%) ROI: -13.0%
MLB CLV Run Line: beat close 37/190 (19%) avg CLV=+0.62 runs | CLV Total avg=+0.47 runs
MLB Price CLV Run Line: 119 bets  avg=-0.96%
```

**stderr (tail)**
```
2026-05-05 18:30:22,843 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 12 rows
2026-05-05 18:30:22,843 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 12 MLB game outcome rows
2026-05-05 18:30:22,941 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 119 rows
2026-05-05 18:30:22,941 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 119 historical rows
2026-05-05 18:30:24,210 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 858 prop rows
2026-05-05 18:30:24,215 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 858 MLB prop outcome rows
```
