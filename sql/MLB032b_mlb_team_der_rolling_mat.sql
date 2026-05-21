-- MLB032b: Materialized view for mlb_team_der_rolling (fast training/prediction joins)
CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_team_der_rolling_mat AS
SELECT * FROM features.mlb_team_der_rolling;

CREATE UNIQUE INDEX IF NOT EXISTS idx_team_der_rolling_mat_pk
    ON features.mlb_team_der_rolling_mat (team_abbr, game_slug);
