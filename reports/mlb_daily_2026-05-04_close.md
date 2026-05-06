# SuperNovaBets MLB Daily Run (2026-05-04 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 10.6s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.5s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 3.5s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-04 18:30:20,292 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-05. Catching up from 2026-05-06 to 2026-05-03
2026-05-04 18:30:20,292 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-04 window=2026-05-03T18:00:00Z..2026-05-05T04:00:00Z
2026-05-04 18:30:21,183 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-04 | events=11 | credits_remaining=99557
2026-05-04 18:30:21,449 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-05 window=2026-05-04T18:00:00Z..2026-05-06T04:00:00Z
2026-05-04 18:30:22,086 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-05 | events=24 | credits_remaining=99555
2026-05-04 18:30:22,090 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99555
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-04 18:30:25,114 | INFO | mlb_pipeline.parse_oddsapi | Upserted 845 rows into odds.mlb_game_lines (live odds).
2026-05-04 18:30:25,117 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-04
2026-05-04 18:30:25,118 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-04 18:30:25,153 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-03
2026-05-04 18:30:26,608 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5665 rows into odds.mlb_player_prop_lines.
2026-05-04 18:30:26,610 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 110-74 (59.8%) ROI: +14.1% | Total: 33-42 (44.0%) ROI: -16.0%
MLB CLV Run Line: beat close 35/184 (19%) avg CLV=+0.60 runs | CLV Total avg=+0.47 runs
MLB Price CLV Run Line: 113 bets  avg=-1.13%
```

**stderr (tail)**
```
2026-05-04 18:30:29,100 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-05-04 18:30:29,101 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-05-04 18:30:29,191 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 118 rows
2026-05-04 18:30:29,192 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 118 historical rows
2026-05-04 18:30:30,140 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1109 prop rows
2026-05-04 18:30:30,144 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1109 MLB prop outcome rows
```
