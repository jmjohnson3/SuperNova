-- MLB026b: Materialized view for batter vs relief pitching (fast training/prediction joins)
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_batter_vs_rp_mat AS
SELECT * FROM features.mlb_batter_vs_rp;

CREATE UNIQUE INDEX IF NOT EXISTS idx_batter_vs_rp_mat_pk
    ON features.mlb_batter_vs_rp_mat (batter_id, game_slug);
