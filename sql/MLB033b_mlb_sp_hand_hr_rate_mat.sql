CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_sp_hand_hr_rate_mat AS
SELECT * FROM features.mlb_sp_hand_hr_rate;

CREATE UNIQUE INDEX IF NOT EXISTS idx_sp_hand_hr_rate_mat_pk
    ON features.mlb_sp_hand_hr_rate_mat (pitcher_id, game_slug);
