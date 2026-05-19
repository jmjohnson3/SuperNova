-- MLB021b: Materialized view for SP velocity rolling (MLB021)
-- Caches the window-function computation so training/prediction JOINs are fast.
-- Unique key: (player_id, game_date) — matches the source table PK.

CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_sp_velocity_rolling_mat AS
SELECT * FROM features.mlb_sp_velocity_rolling;

CREATE UNIQUE INDEX IF NOT EXISTS idx_sp_velocity_rolling_mat_pk
    ON features.mlb_sp_velocity_rolling_mat (player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_sp_velocity_rolling_mat_pitcher_date
    ON features.mlb_sp_velocity_rolling_mat (player_id, game_date DESC);
