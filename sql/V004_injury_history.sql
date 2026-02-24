-- ============================================================================
-- V004: Injury History Table & Historicised team_injury_impact View
-- ============================================================================
-- 1. Creates raw.nba_injuries_history to store daily injury snapshots.
-- 2. Replaces features.team_injury_impact with a version that works for BOTH
--    historical training (joins nba_injuries_history by as_of_date) and
--    current-day prediction (falls back to raw.nba_injuries when no history
--    row exists for today).
-- ============================================================================


-- ============================================================================
-- 1. INJURY HISTORY TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS raw.nba_injuries_history (
    as_of_date              date            NOT NULL,
    player_id               int4            NOT NULL,
    first_name              text            NULL,
    last_name               text            NULL,
    primary_position        text            NULL,
    jersey_number           text            NULL,
    team_abbr               text            NULL,
    roster_status           text            NULL,
    injury_description      text            NULL,
    playing_probability     text            NULL,
    source_fetched_at_utc   timestamptz     NOT NULL,
    created_at_utc          timestamptz     DEFAULT now() NOT NULL,
    CONSTRAINT nba_injuries_history_pkey PRIMARY KEY (as_of_date, player_id)
);

CREATE INDEX IF NOT EXISTS idx_injuries_history_team_date
    ON raw.nba_injuries_history (team_abbr, as_of_date);

CREATE INDEX IF NOT EXISTS idx_injuries_history_date
    ON raw.nba_injuries_history (as_of_date);


-- ============================================================================
-- 2. REPLACE features.team_injury_impact
-- ============================================================================
-- The new view is keyed by (season, team_abbr, game_date_et) so downstream
-- training and prediction views can join on all three columns.
--
-- Logic:
--   a) Start from features.team_game_spine to enumerate every team-game.
--   b) For each game_date_et, look up injuries from nba_injuries_history.
--      If no history rows exist for that date (e.g. today before the snapshot
--      job runs), fall back to raw.nba_injuries (the live snapshot).
--   c) For each injured player, compute their average production over their
--      last 10 games played BEFORE the injury date, using nba_player_gamelogs.
--   d) Weight that production by miss probability:
--         OUT / O            = 1.0
--         DOUBTFUL / D       = 0.8
--         QUESTIONABLE / Q   = 0.4
--         PROBABLE / P       = 0.1
--   e) Aggregate per (season, team_abbr, game_date_et).
-- ============================================================================

CREATE OR REPLACE VIEW features.team_injury_impact AS
WITH spine AS (
    -- Every team-game with its date
    SELECT DISTINCT
        sp.season,
        sp.team_abbr,
        sp.game_date_et
    FROM features.team_game_spine sp
),

-- Unified injury source: prefer history, fall back to live snapshot for today
injuries_unified AS (
    -- Historical injuries keyed by date
    SELECT
        h.as_of_date,
        h.player_id,
        UPPER(h.team_abbr)         AS team_abbr,
        h.playing_probability
    FROM raw.nba_injuries_history h
    WHERE h.team_abbr IS NOT NULL

    UNION ALL

    -- Live injuries for dates that have NO history rows yet.
    -- This covers "today" before the daily snapshot job has run,
    -- and any future dates used for prediction.
    SELECT
        s.game_date_et              AS as_of_date,
        i.player_id,
        UPPER(i.team_abbr)         AS team_abbr,
        i.playing_probability
    FROM raw.nba_injuries i
    CROSS JOIN (
        -- Distinct game dates that are NOT covered by the history table
        SELECT DISTINCT sp.game_date_et
        FROM features.team_game_spine sp
        WHERE NOT EXISTS (
            SELECT 1
            FROM raw.nba_injuries_history h2
            WHERE h2.as_of_date = sp.game_date_et
        )
    ) s
    WHERE i.team_abbr IS NOT NULL
),

-- For each injured player on a given date, compute their recent production
-- from the last 10 games they played BEFORE that date.
player_production AS (
    SELECT
        iu.as_of_date,
        iu.player_id,
        iu.team_abbr,
        iu.playing_probability,
        AVG(pg.minutes)             AS min_avg_10,
        AVG(pg.points::numeric)     AS pts_avg_10,
        AVG(pg.rebounds::numeric)   AS reb_avg_10,
        AVG(pg.assists::numeric)    AS ast_avg_10
    FROM injuries_unified iu
    INNER JOIN LATERAL (
        SELECT
            pg2.minutes,
            pg2.points,
            pg2.rebounds,
            pg2.assists
        FROM raw.nba_player_gamelogs pg2
        WHERE pg2.player_id = iu.player_id
          AND pg2.as_of_date < iu.as_of_date
          AND pg2.minutes IS NOT NULL
          AND pg2.minutes > 0
        ORDER BY pg2.as_of_date DESC
        LIMIT 10
    ) pg ON TRUE
    GROUP BY
        iu.as_of_date,
        iu.player_id,
        iu.team_abbr,
        iu.playing_probability
),

-- Apply miss-probability weight
impact AS (
    SELECT
        s.season,
        s.team_abbr,
        s.game_date_et,
        pp.player_id,
        pp.playing_probability,
        pp.min_avg_10,
        pp.pts_avg_10,
        pp.reb_avg_10,
        pp.ast_avg_10,
        CASE
            WHEN UPPER(pp.playing_probability) IN ('OUT', 'O')           THEN 1.0
            WHEN UPPER(pp.playing_probability) IN ('DOUBTFUL', 'D')      THEN 0.8
            WHEN UPPER(pp.playing_probability) IN ('QUESTIONABLE', 'Q')  THEN 0.4
            WHEN UPPER(pp.playing_probability) IN ('PROBABLE', 'P')      THEN 0.1
            ELSE 0.0
        END AS miss_weight
    FROM spine s
    INNER JOIN player_production pp
        ON  pp.team_abbr   = s.team_abbr
        AND pp.as_of_date  = s.game_date_et
)

SELECT
    season,
    team_abbr,
    game_date_et,
    -- Total weighted minutes missing from injured players
    COALESCE(SUM(min_avg_10 * miss_weight), 0)          AS injured_min_lost,
    -- Total weighted points missing
    COALESCE(SUM(pts_avg_10 * miss_weight), 0)          AS injured_pts_lost,
    -- Total weighted rebounds missing
    COALESCE(SUM(reb_avg_10 * miss_weight), 0)          AS injured_reb_lost,
    -- Total weighted assists missing
    COALESCE(SUM(ast_avg_10 * miss_weight), 0)          AS injured_ast_lost,
    -- Count of players OUT or DOUBTFUL
    COUNT(*) FILTER (WHERE miss_weight >= 0.8)          AS injured_out_count,
    -- Count of players QUESTIONABLE or worse
    COUNT(*) FILTER (WHERE miss_weight >= 0.4)          AS injured_questionable_plus_count
FROM impact
GROUP BY season, team_abbr, game_date_et;
