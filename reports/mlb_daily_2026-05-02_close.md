# SuperNovaBets MLB Daily Run (2026-05-02 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 8.1s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.1s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 3.4s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-02 18:30:08,794 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-03. Catching up from 2026-05-04 to 2026-05-01
2026-05-02 18:30:08,795 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-02 window=2026-05-01T18:00:00Z..2026-05-03T04:00:00Z
2026-05-02 18:30:10,977 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-02 | events=7 | credits_remaining=99769
2026-05-02 18:30:11,240 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-03 window=2026-05-02T18:00:00Z..2026-05-04T04:00:00Z
2026-05-02 18:30:11,864 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-03 | events=20 | credits_remaining=99767
2026-05-02 18:30:11,882 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99767
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-02 18:30:14,752 | INFO | mlb_pipeline.parse_oddsapi | Upserted 817 rows into odds.mlb_game_lines (live odds).
2026-05-02 18:30:14,755 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-02
2026-05-02 18:30:14,756 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-02 18:30:14,803 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-01
2026-05-02 18:30:15,982 | INFO | mlb_pipeline.parse_oddsapi | Upserted 6213 rows into odds.mlb_player_prop_lines.
2026-05-02 18:30:15,984 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 107-69 (60.8%) ROI: +16.1% | Total: 32-34 (48.5%) ROI: -7.4%
MLB CLV Run Line: beat close 33/176 (19%) avg CLV=+0.60 runs | CLV Total avg=+0.52 runs
MLB Price CLV Run Line: 105 bets  avg=-1.51%
```

**stderr (tail)**
```
2026-05-02 18:30:18,590 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 15 rows
2026-05-02 18:30:18,590 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 15 MLB game outcome rows
2026-05-02 18:30:18,670 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 111 rows
2026-05-02 18:30:18,670 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 111 historical rows
2026-05-02 18:30:19,352 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 1132 prop rows
2026-05-02 18:30:19,356 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 1132 MLB prop outcome rows
```
