-- ============================================================================
-- V013: Referee Features — Foul Tendencies & Player-Referee History
-- ============================================================================
-- Extracts referee/official data from boxscore API responses and builds
-- features for predicting foul trouble based on referee assignments.
--
-- Tables created:
--   raw.nba_game_referees — game-referee assignments
--
-- Views created:
--   features.referee_foul_tendencies     — per-referee rolling foul stats
--   features.player_referee_foul_history — player-specific foul rates with/without each ref
--   features.game_referee_features       — game-level crew aggregates
--   features.player_game_referee_foul_risk — player-game level foul risk
-- ============================================================================


-- -------------------------------------------------------------------------
-- TABLE: raw.nba_game_referees
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.nba_game_referees (
    game_slug       TEXT NOT NULL,
    season          TEXT NOT NULL,
    referee_id      INT  NOT NULL,
    first_name      TEXT,
    last_name       TEXT,
    title           TEXT,    -- 'Referee', 'Umpire', 'Crew Chief', etc.
    source_fetched_at_utc TIMESTAMPTZ,
    updated_at_utc  TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (game_slug, referee_id)
);
CREATE INDEX IF NOT EXISTS idx_game_referees_ref_id
    ON raw.nba_game_referees (referee_id);
CREATE INDEX IF NOT EXISTS idx_game_referees_season
    ON raw.nba_game_referees (season);


-- =========================================================================
-- VIEW 1: Referee foul tendencies (rolling per-referee stats)
-- =========================================================================
-- For each referee × game, compute the total fouls called in that game,
-- then rolling-average over the referee's previous 20 games.
-- =========================================================================
CREATE OR REPLACE VIEW features.referee_foul_tendencies AS

WITH ref_game_fouls AS (
    -- Total fouls in each game a referee officiated, split home/away
    SELECT
        r.referee_id,
        r.first_name,
        r.last_name,
        r.game_slug,
        r.season,
        g.game_date_et,
        g.start_ts_utc,
        COALESCE(
            NULLIF(hts.stats->'miscellaneous'->>'foulsTotal', '')::numeric, 0
        ) AS home_fouls,
        COALESCE(
            NULLIF(ats.stats->'miscellaneous'->>'foulsTotal', '')::numeric, 0
        ) AS away_fouls,
        COALESCE(
            NULLIF(hts.stats->'miscellaneous'->>'foulsTotal', '')::numeric, 0
        ) + COALESCE(
            NULLIF(ats.stats->'miscellaneous'->>'foulsTotal', '')::numeric, 0
        ) AS total_fouls
    FROM raw.nba_game_referees r
    JOIN raw.nba_games g
      ON g.game_slug = r.game_slug AND g.season = r.season
    LEFT JOIN raw.nba_boxscore_team_stats hts
      ON hts.game_slug = r.game_slug AND hts.season = r.season
     AND UPPER(hts.team_abbr) = UPPER(g.home_team_abbr)
    LEFT JOIN raw.nba_boxscore_team_stats ats
      ON ats.game_slug = r.game_slug AND ats.season = r.season
     AND UPPER(ats.team_abbr) = UPPER(g.away_team_abbr)
    WHERE g.status = 'final'
)

SELECT
    referee_id,
    first_name,
    last_name,
    game_slug,
    season,
    game_date_et,
    total_fouls,
    home_fouls,
    away_fouls,

    -- Rolling 20-game averages (pregame)
    AVG(total_fouls) OVER w20 AS avg_total_fouls_20,
    AVG(home_fouls)  OVER w20 AS avg_home_fouls_20,
    AVG(away_fouls)  OVER w20 AS avg_away_fouls_20,

    -- Home/away foul bias (positive = calls more on away team)
    AVG(away_fouls - home_fouls) OVER w20 AS avg_away_foul_bias_20,

    -- Games officiated (experience / sample size)
    COUNT(*) OVER w20 AS games_officiated_prev_20

FROM ref_game_fouls

WINDOW w20 AS (
    PARTITION BY referee_id
    ORDER BY COALESCE(start_ts_utc, game_date_et::timestamp with time zone), game_slug
    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
);


-- =========================================================================
-- VIEW 2: Player-referee foul history
-- =========================================================================
-- For each player, compute avg fouls WITH vs WITHOUT each referee.
-- This reveals which refs consistently send a player into foul trouble.
-- =========================================================================
CREATE OR REPLACE VIEW features.player_referee_foul_history AS

WITH player_game_fouls AS (
    -- Each player's foul count per game
    SELECT
        ps.game_slug,
        ps.season,
        ps.player_id,
        UPPER(ps.team_abbr) AS team_abbr,
        ps.first_name,
        ps.last_name,
        g.game_date_et,
        g.start_ts_utc,
        COALESCE(
            NULLIF(ps.stats->'miscellaneous'->>'foulsTotal', '')::numeric, 0
        ) AS fouls,
        COALESCE(
            NULLIF(ps.stats->'miscellaneous'->>'minSeconds', '')::numeric, 0
        ) / 60.0 AS minutes_played
    FROM raw.nba_boxscore_player_stats ps
    JOIN raw.nba_games g
      ON g.game_slug = ps.game_slug AND g.season = ps.season
    WHERE g.status = 'final'
      AND COALESCE(NULLIF(ps.stats->'miscellaneous'->>'minSeconds', '')::numeric, 0) > 0
),

-- Join players to referees for each game
player_ref_games AS (
    SELECT
        pgf.player_id,
        pgf.team_abbr,
        pgf.first_name AS player_first,
        pgf.last_name  AS player_last,
        r.referee_id,
        r.first_name   AS ref_first,
        r.last_name    AS ref_last,
        pgf.game_slug,
        pgf.season,
        pgf.game_date_et,
        pgf.fouls,
        pgf.minutes_played,
        CASE WHEN pgf.minutes_played > 0
             THEN pgf.fouls / pgf.minutes_played * 36.0  -- fouls per 36 min
             ELSE NULL
        END AS fouls_per_36
    FROM player_game_fouls pgf
    JOIN raw.nba_game_referees r
      ON r.game_slug = pgf.game_slug AND r.season = pgf.season
),

-- Aggregate: player stats WITH each referee (career/season)
with_ref AS (
    SELECT
        player_id,
        team_abbr,
        player_first,
        player_last,
        referee_id,
        ref_first,
        ref_last,
        COUNT(*)             AS games_with_ref,
        AVG(fouls)           AS avg_fouls_with_ref,
        AVG(fouls_per_36)    AS avg_fouls_per_36_with_ref
    FROM player_ref_games
    GROUP BY player_id, team_abbr, player_first, player_last,
             referee_id, ref_first, ref_last
    HAVING COUNT(*) >= 2  -- need at least 2 games for meaningful signal
),

-- Aggregate: player overall foul baseline
baseline AS (
    SELECT
        player_id,
        AVG(fouls)        AS avg_fouls_overall,
        AVG(fouls_per_36) AS avg_fouls_per_36_overall,
        COUNT(*)          AS total_games
    FROM player_game_fouls
    GROUP BY player_id
)

SELECT
    wr.player_id,
    wr.team_abbr,
    wr.player_first,
    wr.player_last,
    wr.referee_id,
    wr.ref_first,
    wr.ref_last,
    wr.games_with_ref,
    ROUND(wr.avg_fouls_with_ref::numeric, 2)        AS avg_fouls_with_ref,
    ROUND(wr.avg_fouls_per_36_with_ref::numeric, 2)  AS avg_fouls_per_36_with_ref,
    ROUND(bl.avg_fouls_overall::numeric, 2)           AS avg_fouls_overall,
    ROUND(bl.avg_fouls_per_36_overall::numeric, 2)    AS avg_fouls_per_36_overall,
    bl.total_games,
    -- Foul uplift: positive = more fouls with this ref than usual
    ROUND((wr.avg_fouls_with_ref - bl.avg_fouls_overall)::numeric, 2)
        AS foul_uplift_with_ref,
    ROUND((wr.avg_fouls_per_36_with_ref - bl.avg_fouls_per_36_overall)::numeric, 2)
        AS foul_per_36_uplift_with_ref
FROM with_ref wr
JOIN baseline bl ON bl.player_id = wr.player_id;


-- =========================================================================
-- VIEW 3: Game-level referee features (crew aggregates)
-- =========================================================================
-- For each game, aggregate the assigned referee crew's tendencies.
-- Joinable to game_prediction_features on (season, game_slug).
-- =========================================================================
CREATE OR REPLACE VIEW features.game_referee_features AS

WITH crew_stats AS (
    SELECT
        rft.game_slug,
        rft.season,
        rft.game_date_et,
        AVG(rft.avg_total_fouls_20)    AS crew_avg_fouls_per_game,
        AVG(rft.avg_away_foul_bias_20) AS crew_away_foul_bias,
        SUM(rft.games_officiated_prev_20) AS crew_total_experience,
        COUNT(*)                          AS crew_size
    FROM features.referee_foul_tendencies rft
    GROUP BY rft.game_slug, rft.season, rft.game_date_et
)

SELECT
    season,
    game_slug,
    game_date_et,
    ROUND(crew_avg_fouls_per_game::numeric, 1)  AS crew_avg_fouls_per_game,
    ROUND(crew_away_foul_bias::numeric, 2)      AS crew_away_foul_bias,
    crew_total_experience,
    crew_size
FROM crew_stats;


-- =========================================================================
-- VIEW 4: Player-game referee foul risk
-- =========================================================================
-- For each player in a game with assigned referees, compute their
-- historical foul uplift with the officiating crew.
-- Used by player prop models to adjust foul/minutes projections.
-- =========================================================================
CREATE OR REPLACE VIEW features.player_game_referee_foul_risk AS

WITH player_ref_uplift AS (
    SELECT
        r.game_slug,
        r.season,
        g.game_date_et,
        prh.player_id,
        prh.team_abbr,
        prh.player_first,
        prh.player_last,
        prh.referee_id,
        prh.foul_uplift_with_ref,
        prh.foul_per_36_uplift_with_ref,
        prh.games_with_ref
    FROM raw.nba_game_referees r
    JOIN raw.nba_games g
      ON g.game_slug = r.game_slug AND g.season = r.season
    JOIN features.player_referee_foul_history prh
      ON prh.referee_id = r.referee_id
)

SELECT
    game_slug,
    season,
    game_date_et,
    player_id,
    team_abbr,
    player_first,
    player_last,

    -- Average foul uplift across all refs in this game's crew
    ROUND(AVG(foul_uplift_with_ref)::numeric, 2)          AS avg_foul_uplift_crew,
    ROUND(AVG(foul_per_36_uplift_with_ref)::numeric, 2)   AS avg_foul_per_36_uplift_crew,

    -- Worst-case ref for this player
    ROUND(MAX(foul_uplift_with_ref)::numeric, 2)           AS max_foul_uplift_crew,

    -- Total games with this crew's refs (sample size indicator)
    SUM(games_with_ref)                                     AS total_games_with_crew,

    -- Number of refs in crew that this player has history with
    COUNT(*)                                                AS refs_with_history

FROM player_ref_uplift
GROUP BY game_slug, season, game_date_et,
         player_id, team_abbr, player_first, player_last;
