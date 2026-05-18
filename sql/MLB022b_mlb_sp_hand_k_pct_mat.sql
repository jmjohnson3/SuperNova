-- MLB022b: Materialized view for SP K% by batter handedness (MLB022)
-- Caches expensive window function computation so training JOINs are fast.
-- Unique key: (pitcher_id, game_slug)

CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_sp_hand_k_pct_mat AS
SELECT * FROM features.mlb_sp_hand_k_pct;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_sp_hand_k_pct_mat_pk
    ON features.mlb_sp_hand_k_pct_mat (pitcher_id, game_slug);
CREATE INDEX IF NOT EXISTS idx_mlb_sp_hand_k_pct_mat_pitcher_date
    ON features.mlb_sp_hand_k_pct_mat (pitcher_id, game_date_et DESC, game_slug DESC);
