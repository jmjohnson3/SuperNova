-- Indexes for the LEFT JOIN LATERAL lookups in features._load_logs.
--
-- Each lateral runs once per row of raw.nfl_player_gamelogs (~40k) and filters on
-- (season, player_id, team_abbr). The pre-existing indexes lead with
-- (season, team_abbr) or index player_name_norm instead of player_id, so Postgres
-- had to read every row for the season/team and discard almost all of them: a
-- single depth-chart lookup measured 13.8s, with 15,285 of 15,418 rows removed by
-- filter. _load_logs was 1337s of a 1426s feature build (93.7%).
--
-- Trailing columns follow each lateral's ORDER BY so the LIMIT 1 can be answered
-- from the index without a sort. CONCURRENTLY keeps the 5-minute scanner tasks
-- running while these build; it cannot be used inside a transaction.

-- WHERE season/player_id/team_abbr, week <= p.week
-- ORDER BY COALESCE(week,0) DESC, snapshot_ts_utc DESC NULLS LAST, pos_rank
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_depth_player_asof
    ON raw.nfl_depth_charts (season, player_id, team_abbr, week DESC, snapshot_ts_utc DESC);

-- WHERE season/player_id/team_abbr, week = p.week
-- ORDER BY COALESCE(week,0) DESC, updated_at_utc DESC
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_injuries_player_asof
    ON raw.nfl_injuries (season, player_id, team_abbr, week, updated_at_utc DESC);

-- WHERE season/player_id/team_abbr, COALESCE(week,0) <= COALESCE(p.week,999)
-- ORDER BY COALESCE(week,0) DESC, roster_status priority, updated_at_utc DESC
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_rosters_player_asof
    ON raw.nfl_rosters (season, player_id, team_abbr, week DESC, updated_at_utc DESC);
