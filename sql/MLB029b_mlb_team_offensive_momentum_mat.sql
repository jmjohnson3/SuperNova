-- MLB029b: Materialized view for team offensive momentum (fast training/prediction joins)
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_team_offensive_momentum_mat AS
SELECT * FROM features.mlb_team_offensive_momentum;

CREATE UNIQUE INDEX IF NOT EXISTS idx_team_offensive_momentum_mat_pk
    ON features.mlb_team_offensive_momentum_mat (team_abbr, game_slug);
