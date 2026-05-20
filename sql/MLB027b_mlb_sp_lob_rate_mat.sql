-- MLB027b: Materialized view for SP strand rate (fast training/prediction joins)
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_sp_lob_rate_mat AS
SELECT * FROM features.mlb_sp_lob_rate;

CREATE UNIQUE INDEX IF NOT EXISTS idx_sp_lob_rate_mat_pk
    ON features.mlb_sp_lob_rate_mat (player_id, game_slug);
