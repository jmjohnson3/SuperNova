-- MLB007: Materialized views for rolling stats + standings
-- These cache expensive rolling window computations so that prediction queries
-- (which join to these views) are fast. Refresh daily after data load.
-- REFRESH order: mat rolling views first, then standings (depends on mlb_games).

-- ============================================================
-- Batting rolling materialized view
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_team_batting_rolling_mat AS
SELECT * FROM features.mlb_team_batting_rolling;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_batting_rolling_mat_pk
    ON features.mlb_team_batting_rolling_mat (game_slug, team_abbr);
CREATE INDEX IF NOT EXISTS idx_mlb_batting_rolling_mat_team_date
    ON features.mlb_team_batting_rolling_mat (team_abbr, game_date_et DESC, game_slug DESC);

-- ============================================================
-- Pitching rolling materialized view
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_team_pitching_rolling_mat AS
SELECT * FROM features.mlb_team_pitching_rolling;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_pitching_rolling_mat_pk
    ON features.mlb_team_pitching_rolling_mat (game_slug, team_abbr);
CREATE INDEX IF NOT EXISTS idx_mlb_pitching_rolling_mat_team_date
    ON features.mlb_team_pitching_rolling_mat (team_abbr, game_date_et DESC, game_slug DESC);

-- ============================================================
-- Pitcher (SP) rolling materialized view
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_pitcher_rolling_mat AS
SELECT * FROM features.mlb_pitcher_rolling;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_pitcher_rolling_mat_pk
    ON features.mlb_pitcher_rolling_mat (game_slug, player_id);
CREATE INDEX IF NOT EXISTS idx_mlb_pitcher_rolling_mat_player_date
    ON features.mlb_pitcher_rolling_mat (player_id, game_date_et DESC, game_slug DESC);

-- ============================================================
-- Individual batter rolling materialized view (MLB008)
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_player_batting_rolling_mat AS
SELECT * FROM features.mlb_player_batting_rolling;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_batting_player_mat_pk
    ON features.mlb_player_batting_rolling_mat (game_slug, player_id);
CREATE INDEX IF NOT EXISTS idx_mlb_batting_player_mat_player_date
    ON features.mlb_player_batting_rolling_mat (player_id, game_date_et DESC, game_slug DESC);

-- ============================================================
-- Standings + rest materialized view
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_standings_rest_mat AS
SELECT * FROM features.mlb_standings_rest;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_standings_rest_mat_pk
    ON features.mlb_standings_rest_mat (game_slug, team_abbr);
CREATE INDEX IF NOT EXISTS idx_mlb_standings_rest_mat_date
    ON features.mlb_standings_rest_mat (game_date_et, game_slug, team_abbr);

-- ============================================================
-- Umpire rolling materialized view (MLB009)
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_umpire_rolling_mat AS
SELECT * FROM features.mlb_umpire_rolling;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_umpire_rolling_mat_pk
    ON features.mlb_umpire_rolling_mat (game_slug, umpire_id);
CREATE INDEX IF NOT EXISTS idx_mlb_umpire_rolling_mat_ump_date
    ON features.mlb_umpire_rolling_mat (umpire_id, game_date_et DESC);

-- ============================================================
-- Batter vs SP handedness rolling materialized view (MLB012)
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_batting_vs_hand_mat AS
SELECT * FROM features.mlb_batting_vs_hand;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_batting_vs_hand_mat_pk
    ON features.mlb_batting_vs_hand_mat (game_slug, player_id);
CREATE INDEX IF NOT EXISTS idx_mlb_batting_vs_hand_mat_player_date
    ON features.mlb_batting_vs_hand_mat (player_id, game_date_et DESC, game_slug DESC);

-- ============================================================
-- Cross-season batter rolling materialized view (MLB013)
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_player_batting_rolling_cross_mat AS
SELECT * FROM features.mlb_player_batting_rolling_cross;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_batting_rolling_cross_mat_pk
    ON features.mlb_player_batting_rolling_cross_mat (game_slug, player_id);
CREATE INDEX IF NOT EXISTS idx_mlb_batting_rolling_cross_mat_player_date
    ON features.mlb_player_batting_rolling_cross_mat (player_id, game_date_et DESC, game_slug DESC);

-- ============================================================
-- Prior season per-player summary materialized view (MLB014)
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_player_prev_season_stats_mat AS
SELECT * FROM features.mlb_player_prev_season_stats;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_player_prev_season_stats_mat_pk
    ON features.mlb_player_prev_season_stats_mat (player_id, season);

-- ============================================================
-- Batter vs specific SP career H2H materialized view (MLB015)
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_batter_vs_sp_mat
    AS SELECT * FROM features.mlb_batter_vs_sp
    WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS mlb_bvsp_pk
    ON features.mlb_batter_vs_sp_mat (batter_id, pitcher_id, game_slug);

-- Secondary index for inference LATERAL (most-recent career stats before date)
CREATE INDEX IF NOT EXISTS mlb_bvsp_lookup
    ON features.mlb_batter_vs_sp_mat (batter_id, pitcher_id, game_date_et DESC);

-- ============================================================
-- SP venue career stats materialized view (MLB017)
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_sp_venue_stats_mat AS
SELECT * FROM features.mlb_sp_venue_stats;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_sp_venue_stats_mat_pk
    ON features.mlb_sp_venue_stats_mat (game_slug, player_id);
CREATE INDEX IF NOT EXISTS idx_mlb_sp_venue_stats_mat_player_venue
    ON features.mlb_sp_venue_stats_mat (player_id, venue_id, game_date_et DESC, game_slug DESC);

-- ============================================================
-- Team batting vs SP handedness materialized view (MLB018)
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_team_batting_vs_hand_mat AS
SELECT * FROM features.mlb_team_batting_vs_hand;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_team_batting_vs_hand_mat_pk
    ON features.mlb_team_batting_vs_hand_mat (game_slug, team_abbr);
CREATE INDEX IF NOT EXISTS idx_mlb_team_batting_vs_hand_mat_team_date
    ON features.mlb_team_batting_vs_hand_mat (team_abbr, game_date_et DESC, game_slug DESC);
