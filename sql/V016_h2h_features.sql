-- ============================================================================
-- V016: Head-to-Head (H2H) Features
-- ============================================================================
-- Adds features.team_h2h_features — last 5 prior meetings between the two
-- teams, expressed from today's home team perspective.
--
-- Features:
--   h2h_meetings_5:       count of prior meetings used (0–5)
--   h2h_home_margin_avg5: avg score differential for today's home team
--                         (positive = home team won those meetings on avg)
--   h2h_home_win_pct5:    fraction of those meetings won by today's home team
--
-- Leakage-safe: inner filter is strictly prev.game_date_et < g.game_date_et.
-- NULL when no prior meetings exist (median-filled at training time).
--
-- Usage: JOIN this view into training/inference queries by (season, game_slug).
-- See SQL_GAME_TRAINING_FEATURES in train_game_models.py and
--     SQL_GAMES_FOR_DATE in predict_today.py.
-- ============================================================================

CREATE OR REPLACE VIEW features.team_h2h_features AS
SELECT
    g.season,
    g.game_slug,
    g.game_date_et,
    g.home_team_abbr,
    g.away_team_abbr,

    COALESCE(h2h.n_meetings, 0)::int   AS h2h_meetings_5,
    h2h.home_margin_avg                AS h2h_home_margin_avg5,
    h2h.home_win_pct                   AS h2h_home_win_pct5

FROM raw.nba_games g
LEFT JOIN LATERAL (
    -- Step 1: pull the 5 most recent prior meetings, then aggregate.
    -- ORDER+LIMIT must be in a subquery; aggregates go in the outer SELECT.
    SELECT
        COUNT(*)                                                        AS n_meetings,
        AVG(
            CASE
                WHEN prev.home_team_abbr = g.home_team_abbr
                THEN (prev.home_score - prev.away_score)::numeric
                ELSE (prev.away_score - prev.home_score)::numeric
            END
        )                                                               AS home_margin_avg,
        AVG(
            CASE
                WHEN prev.home_team_abbr = g.home_team_abbr
                 AND prev.home_score > prev.away_score THEN 1.0
                WHEN prev.home_team_abbr = g.away_team_abbr
                 AND prev.away_score > prev.home_score THEN 1.0
                ELSE 0.0
            END
        )                                                               AS home_win_pct
    FROM (
        SELECT inner_prev.*
        FROM raw.nba_games inner_prev
        WHERE inner_prev.game_date_et < g.game_date_et
          AND inner_prev.home_score IS NOT NULL
          AND inner_prev.away_score IS NOT NULL
          AND (
                  (inner_prev.home_team_abbr = g.home_team_abbr AND inner_prev.away_team_abbr = g.away_team_abbr)
               OR (inner_prev.home_team_abbr = g.away_team_abbr AND inner_prev.away_team_abbr = g.home_team_abbr)
          )
        ORDER BY inner_prev.game_date_et DESC
        LIMIT 5
    ) prev
) h2h ON TRUE;
