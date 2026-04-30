# SuperNovaBets MLB Daily Run (2026-04-27 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 11.3s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 5.5s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 4.4s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-04-27 18:30:19,943 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-04-28. Catching up from 2026-04-29 to 2026-04-26
2026-04-27 18:30:19,943 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-04-27 window=2026-04-26T18:00:00Z..2026-04-28T04:00:00Z
2026-04-27 18:30:21,063 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-04-27 | events=8 | credits_remaining=3961
2026-04-27 18:30:21,489 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-04-28 window=2026-04-27T18:00:00Z..2026-04-29T04:00:00Z
2026-04-27 18:30:22,071 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-04-28 | events=20 | credits_remaining=3959
2026-04-27 18:30:22,081 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=3959
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-04-27 18:30:25,380 | INFO | mlb_pipeline.parse_oddsapi | Upserted 728 rows into odds.mlb_game_lines (live odds).
2026-04-27 18:30:25,398 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-04-27
2026-04-27 18:30:25,399 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-04-27 18:30:25,493 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-04-26
2026-04-27 18:30:27,638 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5112 rows into odds.mlb_player_prop_lines.
2026-04-27 18:30:27,640 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 98-59 (62.4%) ROI: +19.2% | Total: 23-31 (42.6%) ROI: -18.7%
MLB CLV Run Line: beat close 23/157 (15%) avg CLV=+0.46 runs | CLV Total avg=+0.53 runs
MLB Price CLV Run Line: 86 bets  avg=+0.20%
```

**stderr (tail)**
```
2026-04-27 18:30:30,700 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 16 rows
2026-04-27 18:30:30,700 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 16 MLB game outcome rows
2026-04-27 18:30:30,747 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 105 rows
2026-04-27 18:30:30,747 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 105 historical rows
2026-04-27 18:30:31,985 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1155 prop rows
2026-04-27 18:30:31,989 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1155 MLB prop outcome rows
```
