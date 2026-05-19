-- MLB024b: Materialized view for mlb_sp_vs_team (fast training/prediction joins)

CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_sp_vs_team_mat AS
SELECT * FROM features.mlb_sp_vs_team;

CREATE UNIQUE INDEX IF NOT EXISTS idx_sp_vs_team_mat_pk
    ON features.mlb_sp_vs_team_mat (pitcher_id, opp_team_abbr, game_slug);
