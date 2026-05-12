# SuperNovaBets MLB Daily Run (2026-05-07 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 8.8s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 5.7s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 3.9s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-07 18:30:16,275 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-08. Catching up from 2026-05-09 to 2026-05-06
2026-05-07 18:30:16,275 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-07 window=2026-05-06T18:00:00Z..2026-05-08T04:00:00Z
2026-05-07 18:30:17,021 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-07 | events=4 | credits_remaining=99252
2026-05-07 18:30:17,303 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-08 window=2026-05-07T18:00:00Z..2026-05-09T04:00:00Z
2026-05-07 18:30:17,925 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-08 | events=17 | credits_remaining=99250
2026-05-07 18:30:17,929 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99250
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-07 18:30:21,431 | INFO | mlb_pipeline.parse_oddsapi | Upserted 887 rows into odds.mlb_game_lines (live odds).
2026-05-07 18:30:21,482 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-07
2026-05-07 18:30:21,483 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-07 18:30:21,538 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-06
2026-05-07 18:30:23,628 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5213 rows into odds.mlb_player_prop_lines.
2026-05-07 18:30:23,630 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 120-77 (60.9%) ROI: +16.3% | Total: 38-44 (46.3%) ROI: -11.5%
MLB CLV Run Line: beat close 40/197 (20%) avg CLV=+0.63 runs | CLV Total avg=+0.45 runs
MLB Price CLV Run Line: 126 bets  avg=-1.04%
```

**stderr (tail)**
```
2026-05-07 18:30:26,259 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-05-07 18:30:26,260 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-05-07 18:30:26,460 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 122 rows
2026-05-07 18:30:26,460 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 122 historical rows
2026-05-07 18:30:27,441 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1122 prop rows
2026-05-07 18:30:27,446 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1122 MLB prop outcome rows
```
