-- MLB009: Home plate umpire DDL + rolling K/BB/RPG stats view
-- Creates raw.mlb_game_umpires table and features.mlb_umpire_rolling view.
-- Populated by crawler_statsapi.py from boxscore officials array.
-- Leakage-safe: ROWS BETWEEN N PRECEDING AND 1 PRECEDING (prior games only).

-- ============================================================
-- DDL: raw.mlb_game_umpires
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.mlb_game_umpires (
    game_slug             TEXT    NOT NULL,
    ump_position          TEXT    NOT NULL,   -- 'Home Plate', 'First Base', etc.
    umpire_id             INTEGER NOT NULL,
    umpire_name           TEXT,
    source_fetched_at_utc TIMESTAMPTZ DEFAULT now(),
    updated_at_utc        TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (game_slug, ump_position)
);

CREATE INDEX IF NOT EXISTS idx_mlb_game_umpires_slug
    ON raw.mlb_game_umpires (game_slug);

-- ============================================================
-- VIEW: features.mlb_umpire_rolling
-- Home plate ump K9/BB9/RPG rolling stats (5 and 10 prior games)
-- ============================================================
CREATE OR REPLACE VIEW features.mlb_umpire_rolling AS
WITH ump_game_stats AS (
    SELECT
        u.game_slug,
        g.game_date_et,
        g.start_ts_utc,
        u.umpire_id,
        u.umpire_name,
        (b.home_runs + b.away_runs)::FLOAT                  AS total_runs,
        COALESCE(SUM(gl.strikeouts_pitcher), 0)::FLOAT      AS total_k,
        COALESCE(SUM(gl.walks_allowed),      0)::FLOAT      AS total_bb,
        GREATEST(COALESCE(SUM(gl.innings_pitched), 0), 1.0) AS total_ip
    FROM raw.mlb_game_umpires u
    JOIN raw.mlb_games g
        ON g.game_slug = u.game_slug
    JOIN raw.mlb_boxscore_games b
        ON b.game_slug = u.game_slug
    LEFT JOIN raw.mlb_player_gamelogs gl
        ON gl.game_slug = u.game_slug
       AND gl.innings_pitched IS NOT NULL
       AND gl.innings_pitched > 0
    WHERE u.ump_position = 'Home Plate'
      AND g.status = 'final'
    GROUP BY
        u.game_slug, g.game_date_et, g.start_ts_utc,
        u.umpire_id, u.umpire_name,
        b.home_runs, b.away_runs
)
SELECT
    game_slug,
    game_date_et,
    start_ts_utc,
    umpire_id,
    umpire_name,
    -- 5-game rolling (prior games only — leakage-safe)
    COUNT(*)                              OVER w5   AS n_ump_games_prev_5,
    AVG(total_runs)                       OVER w5   AS ump_rpg_5,
    AVG(total_k * 9.0 / total_ip)        OVER w5   AS ump_k9_5,
    AVG(total_bb * 9.0 / total_ip)       OVER w5   AS ump_bb9_5,
    -- 10-game rolling
    COUNT(*)                              OVER w10  AS n_ump_games_prev_10,
    AVG(total_runs)                       OVER w10  AS ump_rpg_10,
    AVG(total_k * 9.0 / total_ip)        OVER w10  AS ump_k9_10,
    AVG(total_bb * 9.0 / total_ip)       OVER w10  AS ump_bb9_10
FROM ump_game_stats
WINDOW
    w5  AS (PARTITION BY umpire_id ORDER BY start_ts_utc
            ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY umpire_id ORDER BY start_ts_utc
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING)
;
