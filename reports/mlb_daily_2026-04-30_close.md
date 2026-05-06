# SuperNovaBets MLB Daily Run (2026-04-30 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 7.7s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 5.1s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 3.6s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-04-30 18:30:11,839 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-01. Catching up from 2026-05-02 to 2026-04-29
2026-04-30 18:30:11,840 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-04-30 window=2026-04-29T18:00:00Z..2026-05-01T04:00:00Z
2026-04-30 18:30:12,549 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-04-30 | events=2 | credits_remaining=99998
2026-04-30 18:30:12,828 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-01 window=2026-04-30T18:00:00Z..2026-05-02T04:00:00Z
2026-04-30 18:30:13,370 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-01 | events=17 | credits_remaining=99996
2026-04-30 18:30:13,399 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99996
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-04-30 18:30:16,586 | INFO | mlb_pipeline.parse_oddsapi | Upserted 768 rows into odds.mlb_game_lines (live odds).
2026-04-30 18:30:16,588 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-04-30
2026-04-30 18:30:16,588 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-04-30 18:30:16,641 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-04-29
2026-04-30 18:30:18,540 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5468 rows into odds.mlb_player_prop_lines.
2026-04-30 18:30:18,541 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 104-64 (61.9%) ROI: +18.2% | Total: 28-32 (46.7%) ROI: -10.9%
MLB CLV Run Line: beat close 30/168 (18%) avg CLV=+0.57 runs | CLV Total avg=+0.58 runs
MLB Price CLV Run Line: 97 bets  avg=-1.63%
```

**stderr (tail)**
```
2026-04-30 18:30:21,020 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 13 rows
2026-04-30 18:30:21,021 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 13 MLB game outcome rows
2026-04-30 18:30:21,179 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 107 rows
2026-04-30 18:30:21,179 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 107 historical rows
2026-04-30 18:30:22,108 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 970 prop rows
2026-04-30 18:30:22,112 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 970 MLB prop outcome rows
```
