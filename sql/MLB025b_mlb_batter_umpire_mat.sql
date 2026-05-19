-- MLB025b: Materialized view for mlb_batter_umpire (fast training/prediction joins)

CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_batter_umpire_mat AS
SELECT * FROM features.mlb_batter_umpire;

CREATE UNIQUE INDEX IF NOT EXISTS idx_batter_umpire_mat_pk
    ON features.mlb_batter_umpire_mat (batter_id, umpire_id, game_slug);
