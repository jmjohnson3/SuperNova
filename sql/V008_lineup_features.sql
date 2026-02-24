-- ============================================================================
-- V008: Team Lineup Features from raw.nba_game_lineups
-- ============================================================================
-- Extracts lineup composition and stability signals from the previously
-- unused raw.nba_game_lineups table.
--
-- Features produced:
--   - starter_count, bench_count, scratch_count, roster_available
--   - starters_changed (vs prior game), starter_continuity_pct
--   - Rolling averages: scratch health, roster size, turnover, continuity
--
-- Depends on:
--   - raw.nba_game_lineups   (starters / bench / scratches JSONB arrays)
--   - features.team_game_spine (season, team_abbr, game_slug ordering)
-- ============================================================================


CREATE OR REPLACE VIEW features.team_lineup_features AS
WITH
-- -----------------------------------------------------------------------
-- 1. Join lineups to the game spine for reliable game ordering
-- -----------------------------------------------------------------------
lineup_base AS (
    SELECT
        sp.season,
        sp.team_abbr,
        sp.game_slug,
        sp.game_date_et,
        sp.is_home,
        sp.start_ts_utc,
        COALESCE(sp.start_ts_utc, sp.game_date_et::timestamp with time zone) AS order_ts,
        lu.starters,
        lu.bench,
        lu.scratches
    FROM features.team_game_spine sp
    INNER JOIN raw.nba_game_lineups lu
        ON  lu.season    = sp.season
        AND lu.game_slug = sp.game_slug
        AND lu.team_abbr = sp.team_abbr
),

-- -----------------------------------------------------------------------
-- 2. Compute roster counts and extract sorted starter player-ID arrays.
--    The JSONB player objects may use either the nested format
--      {"player": {"id": 123, ...}}
--    or the flat format
--      {"id": 123, ...}
--    We handle both with COALESCE.
-- -----------------------------------------------------------------------
with_counts AS (
    SELECT
        lb.*,

        -- Roster counts (NULL-safe: treat NULL/non-array as 0)
        COALESCE(jsonb_array_length(lb.starters),  0) AS starter_count,
        COALESCE(jsonb_array_length(lb.bench),      0) AS bench_count,
        COALESCE(jsonb_array_length(lb.scratches),  0) AS scratch_count,

        -- Available players = starters + bench
        COALESCE(jsonb_array_length(lb.starters), 0)
          + COALESCE(jsonb_array_length(lb.bench), 0)   AS roster_available,

        -- Sorted array of starter player IDs for continuity comparison
        COALESCE(
            (SELECT array_agg(
                COALESCE(
                    (elem -> 'player' ->> 'id')::int,
                    (elem ->> 'id')::int
                ) ORDER BY COALESCE(
                    (elem -> 'player' ->> 'id')::int,
                    (elem ->> 'id')::int
                )
            )
            FROM jsonb_array_elements(lb.starters) AS elem
            WHERE COALESCE(
                (elem -> 'player' ->> 'id')::int,
                (elem ->> 'id')::int
            ) IS NOT NULL),
            ARRAY[]::int[]
        ) AS starter_ids

    FROM lineup_base lb
),

-- -----------------------------------------------------------------------
-- 3. Starter continuity: compare each game's starter IDs to the previous
--    game's starter IDs (LAG window).  Count how many IDs are shared.
-- -----------------------------------------------------------------------
with_continuity AS (
    SELECT
        wc.*,

        LAG(wc.starter_ids) OVER (
            PARTITION BY wc.season, wc.team_abbr
            ORDER BY wc.order_ts, wc.game_slug
        ) AS prev_starter_ids

    FROM with_counts wc
),

with_changes AS (
    SELECT
        wct.*,

        -- Number of starters in common with the previous game
        -- Uses array intersection via unnest + intersect
        CASE
            WHEN wct.prev_starter_ids IS NULL THEN NULL  -- first game of season
            WHEN array_length(wct.starter_ids, 1) IS NULL
              OR array_length(wct.prev_starter_ids, 1) IS NULL THEN NULL
            ELSE (
                SELECT COUNT(*)::int
                FROM (
                    SELECT unnest(wct.starter_ids)
                    INTERSECT
                    SELECT unnest(wct.prev_starter_ids)
                ) shared
            )
        END AS starters_in_common,

        -- Expected starter slots (use current game's count, normally 5)
        COALESCE(array_length(wct.starter_ids, 1), 0) AS n_starters

    FROM with_continuity wct
),

with_derived AS (
    SELECT
        wd.*,

        -- starters_changed: how many starters differ from previous game (0-5)
        CASE
            WHEN wd.starters_in_common IS NULL THEN NULL
            ELSE GREATEST(wd.n_starters, COALESCE(array_length(wd.prev_starter_ids, 1), 0))
                 - wd.starters_in_common
        END AS starters_changed,

        -- starter_continuity_pct: fraction of starters same as prior game
        CASE
            WHEN wd.starters_in_common IS NULL THEN NULL
            WHEN GREATEST(wd.n_starters, COALESCE(array_length(wd.prev_starter_ids, 1), 0)) = 0 THEN NULL
            ELSE wd.starters_in_common::numeric
                 / GREATEST(wd.n_starters, COALESCE(array_length(wd.prev_starter_ids, 1), 0))
        END AS starter_continuity_pct

    FROM with_changes wd
)

-- -----------------------------------------------------------------------
-- 4. Final SELECT: expose raw counts, continuity, and rolling averages.
--    Rolling windows use ROWS BETWEEN N PRECEDING AND 1 PRECEDING so that
--    only prior-game data is included (no data leakage).
-- -----------------------------------------------------------------------
SELECT
    d.season,
    d.team_abbr,
    d.game_slug,
    d.game_date_et,

    -- Raw per-game counts
    d.starter_count,
    d.bench_count,
    d.scratch_count,
    d.roster_available,

    -- Starter continuity vs previous game
    d.starters_changed,
    d.starter_continuity_pct,

    -- Rolling 5-game averages (roster health & turnover)
    AVG(d.scratch_count)         OVER w5  AS avg_scratch_count_5,
    AVG(d.roster_available)      OVER w5  AS avg_roster_available_5,
    AVG(d.starters_changed)      OVER w5  AS avg_starters_changed_5,

    -- Rolling 10-game average of starter continuity
    AVG(d.starter_continuity_pct) OVER w10 AS starter_continuity_avg_10

FROM with_derived d
WINDOW
    w5  AS (PARTITION BY d.season, d.team_abbr
            ORDER BY d.order_ts, d.game_slug
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY d.season, d.team_abbr
            ORDER BY d.order_ts, d.game_slug
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING);
