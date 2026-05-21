-- MLB030b: Materialized view for mlb_batter_babip_rolling (fast training/prediction joins)
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_batter_babip_rolling_mat AS
SELECT * FROM features.mlb_batter_babip_rolling;

CREATE UNIQUE INDEX IF NOT EXISTS idx_batter_babip_rolling_mat_pk
    ON features.mlb_batter_babip_rolling_mat (player_id, game_slug);
