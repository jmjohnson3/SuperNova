-- MLB029: Opponent offensive momentum — team runs scored in last 3/5 games
-- Keys: (team_abbr, game_slug)
-- Leakage-safe: ROWS BETWEEN 3/5 PRECEDING AND 1 PRECEDING
CREATE OR REPLACE VIEW features.mlb_team_offensive_momentum AS
WITH team_runs AS (
    SELECT game_slug, game_date_et, season,
           home_team_abbr AS team_abbr, home_score AS runs_scored
    FROM raw.mlb_games WHERE status = 'final' AND home_score IS NOT NULL
    UNION ALL
    SELECT game_slug, game_date_et, season,
           away_team_abbr, away_score
    FROM raw.mlb_games WHERE status = 'final' AND away_score IS NOT NULL
)
SELECT
    team_abbr, game_slug, game_date_et, season,
    COALESCE(SUM(runs_scored) OVER w3, 0)  AS team_runs_last3,
    COALESCE(AVG(runs_scored) OVER w3)     AS team_runs_avg3,
    COALESCE(SUM(runs_scored) OVER w5, 0)  AS team_runs_last5,
    COUNT(*)                 OVER w3       AS team_games_counted_last3
FROM team_runs
WINDOW w3 AS (PARTITION BY team_abbr, season ORDER BY game_date_et, game_slug
              ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING),
       w5 AS (PARTITION BY team_abbr, season ORDER BY game_date_et, game_slug
              ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING);
