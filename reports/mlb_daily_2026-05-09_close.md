# SuperNovaBets MLB Daily Run (2026-05-09 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 5.5s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.7s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 3.5s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-09 18:30:06,153 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-10. Catching up from 2026-05-11 to 2026-05-08
2026-05-09 18:30:06,153 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-09 window=2026-05-08T18:00:00Z..2026-05-10T04:00:00Z
2026-05-09 18:30:06,842 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-09 | events=10 | credits_remaining=99005
2026-05-09 18:30:07,144 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-10 window=2026-05-09T18:00:00Z..2026-05-11T04:00:00Z
2026-05-09 18:30:07,680 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-10 | events=20 | credits_remaining=99003
2026-05-09 18:30:07,684 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99003
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-09 18:30:10,922 | INFO | mlb_pipeline.parse_oddsapi | Upserted 943 rows into odds.mlb_game_lines (live odds).
2026-05-09 18:30:10,948 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-09
2026-05-09 18:30:10,949 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-09 18:30:10,983 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-08
2026-05-09 18:30:12,443 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6570 rows into odds.mlb_player_prop_lines.
2026-05-09 18:30:12,445 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 126-77 (62.1%) ROI: +18.5% | Total: 40-48 (45.5%) ROI: -13.2%
MLB CLV Run Line: beat close 44/203 (22%) avg CLV=+0.73 runs | CLV Total avg=+0.49 runs
MLB Price CLV Run Line: 132 bets  avg=-1.35%
```

**stderr (tail)**
```
2026-05-09 18:30:14,840 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-05-09 18:30:14,841 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-05-09 18:30:15,044 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 126 rows
2026-05-09 18:30:15,044 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 126 historical rows
2026-05-09 18:30:15,967 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1085 prop rows
2026-05-09 18:30:15,973 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1085 MLB prop outcome rows
```
