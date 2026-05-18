-- MLB023b: Materialized view for batter career stats at specific venue (MLB023)
-- Caches expensive window function computation so training JOINs are fast.
-- Unique key: (player_id, venue_id, game_slug)

CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_batter_venue_stats_mat AS
SELECT * FROM features.mlb_batter_venue_stats;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_batter_venue_stats_mat_pk
    ON features.mlb_batter_venue_stats_mat (player_id, venue_id, game_slug);
CREATE INDEX IF NOT EXISTS idx_mlb_batter_venue_stats_mat_player_venue_date
    ON features.mlb_batter_venue_stats_mat (player_id, venue_id, game_date_et DESC, game_slug DESC);
