-- MLB005: Standings + rest days per team per game
-- rest_days = calendar days since team's last game (NULL if no prior game in season)
-- is_b2b = TRUE when rest_days = 1 (back-to-back)
CREATE OR REPLACE VIEW features.mlb_standings_rest AS
WITH game_dates AS (
    -- One row per (season, team, game) combining home + away appearances
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        g.home_team_abbr AS team_abbr,
        'home'::text      AS side
    FROM raw.mlb_games g
    UNION ALL
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        g.away_team_abbr AS team_abbr,
        'away'::text      AS side
    FROM raw.mlb_games g
),
game_dates_with_prev AS (
    SELECT
        season,
        game_slug,
        game_date_et,
        team_abbr,
        side,
        -- Most recent prior game date for this team in the same season
        MAX(game_date_et) OVER (
            PARTITION BY season, team_abbr
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prev_game_date
    FROM game_dates
),
rest_calc AS (
    SELECT
        season,
        game_slug,
        game_date_et,
        team_abbr,
        side,
        prev_game_date,
        CASE
            WHEN prev_game_date IS NULL THEN NULL
            ELSE (game_date_et - prev_game_date)::int
        END AS rest_days
    FROM game_dates_with_prev
),
-- Get standings for each team per season.
-- raw.mlb_standings has one row per (season, team_abbr) — the latest snapshot.
standings_latest AS (
    SELECT
        s.season,
        s.team_abbr,
        g.game_slug,
        g.game_date_et,
        s.wins,
        s.losses,
        s.win_pct,
        s.run_differential,
        s.division_rank
    FROM raw.mlb_standings s
    JOIN game_dates g
        ON g.team_abbr = s.team_abbr
       AND g.season    = s.season
)
SELECT
    r.season,
    r.game_slug,
    r.game_date_et,
    r.team_abbr,
    r.side,
    r.prev_game_date,
    r.rest_days,
    CASE WHEN r.rest_days = 1 THEN TRUE ELSE FALSE END AS is_b2b,
    -- Standings at game time
    sl.wins,
    sl.losses,
    sl.win_pct,
    sl.run_differential                           AS run_diff,
    sl.division_rank,
    -- Computed convenience fields
    CASE
        WHEN (sl.wins + sl.losses) > 0
        THEN sl.wins::float / (sl.wins + sl.losses)
        ELSE NULL
    END AS computed_win_pct,
    sl.wins + sl.losses                           AS games_played,
    -- Run diff per game
    CASE
        WHEN (sl.wins + sl.losses) > 0
        THEN sl.run_differential::float / (sl.wins + sl.losses)
        ELSE NULL
    END AS run_diff_per_game
FROM rest_calc r
LEFT JOIN standings_latest sl
    ON sl.game_slug  = r.game_slug
   AND sl.team_abbr  = r.team_abbr
;
