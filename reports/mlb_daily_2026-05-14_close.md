# SuperNovaBets MLB Daily Run (2026-05-14 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 7.1s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 5.0s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 4.0s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-14 18:30:08,859 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-15. Catching up from 2026-05-16 to 2026-05-13
2026-05-14 18:30:08,859 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-14 window=2026-05-13T18:00:00Z..2026-05-15T04:00:00Z
2026-05-14 18:30:09,727 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-14 | events=4 | credits_remaining=98522
2026-05-14 18:30:10,124 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-15 window=2026-05-14T18:00:00Z..2026-05-16T04:00:00Z
2026-05-14 18:30:10,732 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-15 | events=16 | credits_remaining=98520
2026-05-14 18:30:10,736 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=98520
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-14 18:30:14,312 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1013 rows into odds.mlb_game_lines (live odds).
2026-05-14 18:30:14,322 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-14
2026-05-14 18:30:14,323 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-14 18:30:14,355 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-13
2026-05-14 18:30:15,761 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5357 rows into odds.mlb_player_prop_lines.
2026-05-14 18:30:15,763 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 147-84 (63.6%) ROI: +21.5% | Total: 44-52 (45.8%) ROI: -12.5%
MLB CLV Run Line: beat close 55/231 (24%) avg CLV=+0.84 runs | CLV Total avg=+0.46 runs
MLB Price CLV Run Line: 160 bets  avg=-1.60%
```

**stderr (tail)**
```
2026-05-14 18:30:18,404 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-05-14 18:30:18,404 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-05-14 18:30:18,593 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 130 rows
2026-05-14 18:30:18,593 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 130 historical rows
2026-05-14 18:30:19,771 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1118 prop rows
2026-05-14 18:30:19,776 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1118 MLB prop outcome rows
```
