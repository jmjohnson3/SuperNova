-- MLB005: Standings + rest days per team per game
-- rest_days = calendar days since team's last game (NULL if no prior game in season)
-- is_b2b = TRUE when rest_days = 1 (back-to-back)
-- Standings (wins/losses/run_diff) computed from raw.mlb_games results directly
-- since raw.mlb_standings has NULL data (MSF subscription doesn't cover MLB game data).
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
-- Compute wins/losses/run_diff from actual completed game results.
-- One row per (season, team_abbr) per completed game they played.
completed_game_records AS (
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        g.home_team_abbr                         AS team_abbr,
        CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END AS won,
        g.home_score - g.away_score              AS run_diff_game
    FROM raw.mlb_games g
    WHERE g.status = 'final'
      AND g.home_score IS NOT NULL
    UNION ALL
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        g.away_team_abbr                         AS team_abbr,
        CASE WHEN g.away_score > g.home_score THEN 1 ELSE 0 END AS won,
        g.away_score - g.home_score              AS run_diff_game
    FROM raw.mlb_games g
    WHERE g.status = 'final'
      AND g.home_score IS NOT NULL
),
-- Cumulative standings THROUGH each completed game (inclusive of that game's result).
-- wins_through / losses_through / run_diff_through include this game's result.
-- LATERAL join will use `game_date_et < target_date` to get state before target game.
cumulative_standings AS (
    SELECT
        season,
        game_slug,
        game_date_et,
        team_abbr,
        SUM(won) OVER (
            PARTITION BY season, team_abbr
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS wins_through,
        SUM(1 - won) OVER (
            PARTITION BY season, team_abbr
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS losses_through,
        SUM(run_diff_game) OVER (
            PARTITION BY season, team_abbr
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS run_differential_through
    FROM completed_game_records
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
    -- Standings entering this game: wins/losses/run_diff from all prior completed games
    sw.wins,
    sw.losses,
    CASE
        WHEN (sw.wins + sw.losses) > 0
        THEN sw.wins::float / (sw.wins + sw.losses)
        ELSE NULL
    END AS win_pct,
    sw.run_differential                               AS run_diff,
    -- Division rank not available (would require additional standings data)
    NULL::integer                                     AS division_rank,
    -- Computed convenience fields (alias for win_pct)
    CASE
        WHEN (sw.wins + sw.losses) > 0
        THEN sw.wins::float / (sw.wins + sw.losses)
        ELSE NULL
    END AS computed_win_pct,
    sw.wins + sw.losses                               AS games_played,
    -- Run diff per game
    CASE
        WHEN (sw.wins + sw.losses) > 0
        THEN sw.run_differential::float / (sw.wins + sw.losses)
        ELSE NULL
    END AS run_diff_per_game
FROM rest_calc r
-- Join to most recent cumulative standings row before this game
LEFT JOIN LATERAL (
    SELECT
        cs.wins_through      AS wins,
        cs.losses_through    AS losses,
        cs.run_differential_through AS run_differential
    FROM cumulative_standings cs
    WHERE cs.team_abbr   = r.team_abbr
      AND cs.season      = r.season
      AND cs.game_date_et < r.game_date_et
    ORDER BY cs.game_date_et DESC, cs.game_slug DESC
    LIMIT 1
) sw ON true
;
