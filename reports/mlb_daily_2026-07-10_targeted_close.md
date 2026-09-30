# SuperNovaBets MLB Daily Run (2026-07-10 ET)

## Summary

- **Targeted prop close odds capture**: OK (rc=0, 18.2s)
- **Parse targeted prop close snapshot**: OK (rc=0, 6.1s)
- **Refresh prop replay CLV after targeted close**: OK (rc=0, 71.2s)

## Outputs (tails)

### Targeted prop close odds capture

- rc: 0

**stderr (tail)**
```
2026-07-10 21:45:16,813 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-11. Catching up from 2026-07-12 to 2026-07-09
2026-07-10 21:45:17,625 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 5 events (game_date=2026-07-10, as_of=2026-07-10)
2026-07-10 21:45:18,232 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-10 event=2ac65fed3f431f0601c2e788aee18cab (Atlanta Braves@St. Louis Cardinals) | credits=80984
2026-07-10 21:45:19,299 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-10 event=250b0373676b10f51ed1c59c93714245 (New York Yankees@Washington Nationals) | credits=80982
2026-07-10 21:45:20,150 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-10 event=cc0b13e921789925a1c0502553d4ec9b (Toronto Blue Jays@San Diego Padres) | credits=80977
2026-07-10 21:45:21,229 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-10 event=ca5e8cb9be664ff91b121c15379d7ce9 (Arizona Diamondbacks@Los Angeles Dodgers) | credits=80971
2026-07-10 21:45:22,112 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-10 event=976819aa47554213e70bc968d9356fc4 (Colorado Rockies@San Francisco Giants) | credits=80965
2026-07-10 21:45:22,960 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 14 events (game_date=2026-07-11, as_of=2026-07-10)
2026-07-10 21:45:22,976 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=80965
```

### Parse targeted prop close snapshot

- rc: 0

**stderr (tail)**
```
2026-07-10 21:45:25,646 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1566 rows into odds.mlb_game_lines (live odds).
2026-07-10 21:45:25,646 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-10
2026-07-10 21:45:25,646 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-10 21:45:25,646 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-11T03:15:25.646196+00:00.
2026-07-10 21:45:25,646 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-09
2026-07-10 21:45:29,152 | INFO | mlb_pipeline.parse_oddsapi | Upserted 500 rows into odds.mlb_player_prop_lines.
2026-07-10 21:45:29,152 | INFO | mlb_pipeline.parse_oddsapi | Processed 500 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-10 21:45:29,152 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV after targeted close

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 8100,
  "run_ids": "all",
  "date_from": "2026-07-10",
  "date_to": "2026-07-10",
  "include_graded": true,
  "only_missing": false
}
```
