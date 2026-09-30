# NFL Manual Odds CSV Fallback

When API credits run out, put FanDuel/DraftKings exports in this folder and rerun the NFL daily script.

Accepted player prop filenames for a slate:

- `nfl_player_props_YYYY-MM-DD.csv`
- `player_props_YYYY-MM-DD.csv`
- `YYYY-MM-DD_nfl_player_props.csv`
- `YYYY-MM-DD_player_props.csv`

Accepted game odds filenames:

- `nfl_game_odds_YYYY-MM-DD.csv`
- `game_odds_YYYY-MM-DD.csv`
- `YYYY-MM-DD_nfl_game_odds.csv`
- `YYYY-MM-DD_game_odds.csv`

Player prop columns:

```csv
snapshot_role,event_id,commence_time_utc,book,home_team,away_team,player_name,stat,line,over_price,under_price,over_link,under_link
lock,2026_01_DEN_KC,2026-09-15T00:15:00Z,draftkings,Kansas City Chiefs,Denver Broncos,Patrick Mahomes,passing_yards,260.5,-115,-115,,
```

You can also provide one side per row:

```csv
snapshot_role,event_id,commence_time_utc,book,home_team,away_team,player_name,stat,line,side,price,link
lock,2026_01_DEN_KC,2026-09-15T00:15:00Z,fanduel,Kansas City Chiefs,Denver Broncos,Patrick Mahomes,passing_yards,260.5,over,-112,
lock,2026_01_DEN_KC,2026-09-15T00:15:00Z,fanduel,Kansas City Chiefs,Denver Broncos,Patrick Mahomes,passing_yards,260.5,under,-108,
```

Supported `stat` values:

- `passing_yards`
- `rushing_yards`
- `passing_tds`
- `rushing_tds`
- `receiving_yards`
- `receiving_tds`

Game odds columns:

```csv
snapshot_role,event_id,commence_time_utc,book,home_team,away_team,spread_home_points,spread_home_price,spread_away_points,spread_away_price,total_points,total_over_price,total_under_price,spread_home_link,spread_away_link,total_over_link,total_under_link
lock,2026_01_DEN_KC,2026-09-15T00:15:00Z,draftkings,Kansas City Chiefs,Denver Broncos,-4.5,-110,4.5,-110,47.5,-110,-110,,,,
```

