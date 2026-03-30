-- MLB005: Standings + rest days per team per game
-- rest_days = calendar days since team's last game (NULL if no prior game in season)
-- is_b2b = TRUE when rest_days = 1 (back-to-back)
-- Standings (wins/losses/run_diff) computed from raw.mlb_games results directly
-- since raw.mlb_standings has NULL data (MSF subscription doesn't cover MLB game data).
-- New (Group B): wins_last_5, win_pct_last_5, wins_last_10, win_pct_last_10,
--                run_diff_avg_last_5 — rolling form windows capturing team momentum.
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
-- Cumulative standings THROUGH each completed game (inclusive of that game's result)
-- PLUS rolling form windows (5 and 10 prior games, leakage-safe via PRECEDING AND 1 PRECEDING).
-- LATERAL join will use `game_date_et < target_date` to get state before target game.
cumulative_standings AS (
    SELECT
        season,
        game_slug,
        game_date_et,
        team_abbr,
        -- ── Cumulative (season-to-date, inclusive of this game) ──────────────
        SUM(won)           OVER w_cum AS wins_through,
        SUM(1 - won)       OVER w_cum AS losses_through,
        SUM(run_diff_game) OVER w_cum AS run_differential_through,
        -- ── Rolling form: last 5 completed games (leakage-safe) ──────────────
        SUM(won)           OVER w5    AS wins_last_5,
        COUNT(*)           OVER w5    AS games_last_5,
        AVG(run_diff_game) OVER w5    AS run_diff_avg_last_5,
        -- ── Rolling form: last 10 completed games ─────────────────────────────
        SUM(won)           OVER w10   AS wins_last_10,
        COUNT(*)           OVER w10   AS games_last_10
    FROM completed_game_records
    WINDOW
        w_cum AS (PARTITION BY season, team_abbr
                  ORDER BY game_date_et, game_slug
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
        w5    AS (PARTITION BY season, team_abbr
                  ORDER BY game_date_et, game_slug
                  ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING),
        w10   AS (PARTITION BY season, team_abbr
                  ORDER BY game_date_et, game_slug
                  ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING)
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
    -- Alias for win_pct
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
    END AS run_diff_per_game,
    -- ── Rolling form (last 5/10 games) ────────────────────────────────────────
    sw.wins_last_5,
    sw.games_last_5,
    CASE WHEN sw.games_last_5 > 0
         THEN sw.wins_last_5::float / sw.games_last_5
         ELSE NULL END AS win_pct_last_5,
    sw.wins_last_10,
    sw.games_last_10,
    CASE WHEN sw.games_last_10 > 0
         THEN sw.wins_last_10::float / sw.games_last_10
         ELSE NULL END AS win_pct_last_10,
    sw.run_diff_avg_last_5
FROM rest_calc r
-- Join to most recent cumulative standings row before this game
LEFT JOIN LATERAL (
    SELECT
        cs.wins_through      AS wins,
        cs.losses_through    AS losses,
        cs.run_differential_through AS run_differential,
        cs.wins_last_5,
        cs.games_last_5,
        cs.run_diff_avg_last_5,
        cs.wins_last_10,
        cs.games_last_10
    FROM cumulative_standings cs
    WHERE cs.team_abbr   = r.team_abbr
      AND cs.season      = r.season
      AND cs.game_date_et < r.game_date_et
    ORDER BY cs.game_date_et DESC, cs.game_slug DESC
    LIMIT 1
) sw ON true
;
