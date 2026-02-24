-- ============================================================================
-- V009: Standings Detail Features
-- ============================================================================
-- Extracts underutilised data from the raw.nba_standings table.
-- Currently only win_pct, conference_rank, division_rank, playoff_rank
-- and games_back are used, but the table also stores:
--   • overall_rank, conference_name, division_name
--   • stats JSONB with home/away record, streak, last-10, etc.
--
-- This view provides POINT-IN-TIME standings for each team-game by finding
-- the most recent standings snapshot BEFORE the game date, enabling proper
-- historical training without look-ahead bias.
-- ============================================================================

CREATE OR REPLACE VIEW features.team_standings_detail AS
WITH spine AS (
    SELECT season, team_abbr, game_slug, game_date_et, start_ts_utc
    FROM features.team_game_spine
),

-- For each spine row, find the most recent standings snapshot <= game_date_et
standings_pit AS (
    SELECT DISTINCT ON (sp.season, sp.team_abbr, sp.game_slug)
        sp.season,
        sp.team_abbr,
        sp.game_slug,
        sp.game_date_et,

        -- Already-extracted columns
        s.wins,
        s.losses,
        s.win_pct,
        s.conference_rank,
        s.division_rank,
        s.playoff_rank,
        s.games_back,
        s.overall_rank,
        s.conference_name,
        s.division_name,

        -- JSONB: home/away record
        -- Try multiple paths for MySportsFeeds response structure
        COALESCE(
            NULLIF(s.stats->'standings'->>'homeWins', '')::int,
            NULLIF(s.stats->'stats'->'standings'->>'homeWins', '')::int,
            NULLIF(s.stats->>'homeWins', '')::int
        ) AS home_wins,
        COALESCE(
            NULLIF(s.stats->'standings'->>'homeLosses', '')::int,
            NULLIF(s.stats->'stats'->'standings'->>'homeLosses', '')::int,
            NULLIF(s.stats->>'homeLosses', '')::int
        ) AS home_losses,
        COALESCE(
            NULLIF(s.stats->'standings'->>'awayWins', '')::int,
            NULLIF(s.stats->'stats'->'standings'->>'awayWins', '')::int,
            NULLIF(s.stats->>'awayWins', '')::int
        ) AS away_wins,
        COALESCE(
            NULLIF(s.stats->'standings'->>'awayLosses', '')::int,
            NULLIF(s.stats->'stats'->'standings'->>'awayLosses', '')::int,
            NULLIF(s.stats->>'awayLosses', '')::int
        ) AS away_losses,

        -- JSONB: streak
        COALESCE(
            s.stats->'standings'->'streak'->>'streakType',
            s.stats->'stats'->'standings'->'streak'->>'streakType',
            s.stats->'streak'->>'streakType',
            s.stats->'standings'->'streak'->>'type',
            s.stats->'streak'->>'type'
        ) AS streak_type_raw,
        COALESCE(
            NULLIF(s.stats->'standings'->'streak'->>'streakCount', '')::int,
            NULLIF(s.stats->'stats'->'standings'->'streak'->>'streakCount', '')::int,
            NULLIF(s.stats->'streak'->>'streakCount', '')::int,
            NULLIF(s.stats->'standings'->'streak'->>'count', '')::int,
            NULLIF(s.stats->'streak'->>'count', '')::int
        ) AS streak_count_raw,

        -- JSONB: last 10
        COALESCE(
            NULLIF(s.stats->'standings'->>'last10Wins', '')::int,
            NULLIF(s.stats->'stats'->'standings'->>'last10Wins', '')::int,
            NULLIF(s.stats->>'last10Wins', '')::int
        ) AS last10_wins,
        COALESCE(
            NULLIF(s.stats->'standings'->>'last10Losses', '')::int,
            NULLIF(s.stats->'stats'->'standings'->>'last10Losses', '')::int,
            NULLIF(s.stats->>'last10Losses', '')::int
        ) AS last10_losses

    FROM spine sp
    JOIN raw.nba_standings s
      ON s.season = sp.season
     AND UPPER(s.team_abbr) = sp.team_abbr
     AND s.source_fetched_at_utc::date <= sp.game_date_et
    ORDER BY sp.season, sp.team_abbr, sp.game_slug,
             s.source_fetched_at_utc DESC
)

SELECT
    season,
    team_abbr,
    game_slug,
    game_date_et,

    -- Core standings (point-in-time)
    win_pct,
    conference_rank,
    division_rank,
    playoff_rank,
    games_back,
    overall_rank,

    -- Conference flag
    CASE WHEN UPPER(conference_name) LIKE '%EAST%' THEN TRUE ELSE FALSE END AS is_eastern_conf,

    -- Division (for intra-division matchup detection downstream)
    division_name,

    -- Home / away record
    home_wins,
    home_losses,
    CASE WHEN COALESCE(home_wins, 0) + COALESCE(home_losses, 0) > 0
         THEN home_wins::numeric / (home_wins + home_losses)
         ELSE NULL
    END AS home_win_pct,

    away_wins,
    away_losses,
    CASE WHEN COALESCE(away_wins, 0) + COALESCE(away_losses, 0) > 0
         THEN away_wins::numeric / (away_wins + away_losses)
         ELSE NULL
    END AS away_win_pct,

    -- Home-away split (positive = much better at home)
    CASE WHEN COALESCE(home_wins, 0) + COALESCE(home_losses, 0) > 0
          AND COALESCE(away_wins, 0) + COALESCE(away_losses, 0) > 0
         THEN home_wins::numeric / (home_wins + home_losses)
            - away_wins::numeric / (away_wins + away_losses)
         ELSE NULL
    END AS home_away_split,

    -- Streak: +N for winning streak, -N for losing streak
    CASE
        WHEN UPPER(streak_type_raw) IN ('W', 'WIN')  THEN  COALESCE(streak_count_raw, 0)
        WHEN UPPER(streak_type_raw) IN ('L', 'LOSS') THEN -COALESCE(streak_count_raw, 0)
        ELSE 0
    END AS streak_signed,

    streak_count_raw AS streak_length,

    -- Last 10 games
    last10_wins,
    last10_losses,
    CASE WHEN COALESCE(last10_wins, 0) + COALESCE(last10_losses, 0) > 0
         THEN last10_wins::numeric / (last10_wins + last10_losses)
         ELSE NULL
    END AS last10_win_pct

FROM standings_pit;
