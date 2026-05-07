# SuperNovaBets MLB Daily Run (2026-05-06 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 64.6s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 6.5s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 5.0s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-06 18:32:30,274 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-07. Catching up from 2026-05-08 to 2026-05-05
2026-05-06 18:32:30,274 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-06 window=2026-05-05T18:00:00Z..2026-05-07T04:00:00Z
2026-05-06 18:32:31,599 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-06 | events=9 | credits_remaining=99334
2026-05-06 18:32:32,097 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-07 window=2026-05-06T18:00:00Z..2026-05-08T04:00:00Z
2026-05-06 18:32:32,663 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-07 | events=19 | credits_remaining=99332
2026-05-06 18:32:32,676 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99332
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-06 18:32:36,667 | INFO | mlb_pipeline.parse_oddsapi | Upserted 882 rows into odds.mlb_game_lines (live odds).
2026-05-06 18:32:36,670 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-06
2026-05-06 18:32:36,671 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-06 18:32:36,964 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-05
2026-05-06 18:32:39,235 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6038 rows into odds.mlb_player_prop_lines.
2026-05-06 18:32:39,237 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 117-77 (60.3%) ROI: +15.1% | Total: 37-43 (46.2%) ROI: -11.7%
MLB CLV Run Line: beat close 39/194 (20%) avg CLV=+0.63 runs | CLV Total avg=+0.46 runs
MLB Price CLV Run Line: 123 bets  avg=-1.07%
```

**stderr (tail)**
```
2026-05-06 18:32:42,136 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 13 rows
2026-05-06 18:32:42,136 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 13 MLB game outcome rows
2026-05-06 18:32:42,258 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 120 rows
2026-05-06 18:32:42,259 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 120 historical rows
2026-05-06 18:32:44,229 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 922 prop rows
2026-05-06 18:32:44,234 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 922 MLB prop outcome rows
```
