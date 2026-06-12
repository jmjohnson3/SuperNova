# SuperNovaBets MLB Daily Run (2026-06-03 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 24.4s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 30.1s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 75.3s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-03 18:30:43,353 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-04. Catching up from 2026-06-05 to 2026-06-02
2026-06-03 18:30:43,358 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-03 window=2026-06-02T18:00:00Z..2026-06-04T04:00:00Z
2026-06-03 18:30:47,426 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-03 | events=11 | credits_remaining=99736
2026-06-03 18:30:48,242 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-04 window=2026-06-03T18:00:00Z..2026-06-05T04:00:00Z
2026-06-03 18:30:48,792 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-04 | events=20 | credits_remaining=99734
2026-06-03 18:30:48,816 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99734
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-03 18:31:03,621 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1370 rows into odds.mlb_game_lines (live odds).
2026-06-03 18:31:03,647 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-03
2026-06-03 18:31:03,649 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-03 18:31:04,007 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-02
2026-06-03 18:31:19,675 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6193 rows into odds.mlb_player_prop_lines.
2026-06-03 18:31:19,676 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 180-124 (59.2%) ROI: +13.0% | Total: 56-58 (49.1%) ROI: -6.2%
MLB CLV Run Line: beat close 84/304 (28%) avg CLV=+0.98 runs | CLV Total avg=+0.53 runs
MLB Price CLV Run Line: 233 bets  avg=-1.87%
```

**stderr (tail)**
```
2026-06-03 18:32:26,906 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 31 pending predictions.
2026-06-03 18:32:26,906 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-03 18:32:27,469 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 141 rows
2026-06-03 18:32:27,469 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 141 historical rows
2026-06-03 18:32:32,903 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-03 18:32:32,916 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-03 18:32:33,262 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-03 18:32:33,264 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-03 18:32:33,281 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-03 18:32:33,374 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-03 18:32:33,388 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```
