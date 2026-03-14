-- ============================================================================
-- V025: Player On/Off Splits — Game-Level Net Impact Metrics
-- ============================================================================
-- Computes per-player on-court net rating (per36) and on/off differential
-- using only game-level data (no PBP required).
--
-- Formula:
--   on_net_per36  = plus_minus / minutes * 36
--   off_net_per36 = (team_net - plus_minus) / (48 - minutes) * 36
--   on_off_diff   = on_net_per36 - off_net_per36
--
-- Rolling w5 and w10 pregame averages become training + inference features.
--
-- Guard rails:
--   - minutes_played >= 5: avoids noisy garbage-time stints
--   - minutes_played < 43: avoids near-full-game cases where off-court sample is tiny
--   - team_net derived from home_score/away_score (authoritative final scores)
--   - Window ROWS BETWEEN N PRECEDING AND 1 PRECEDING = strictly pregame (no leakage)
--   - Partitioned by player_id only (no season), matching V007/V023 pattern
-- ============================================================================

CREATE OR REPLACE VIEW features.player_on_off_splits AS
WITH pm_base AS (
    SELECT
        gl.player_id,
        gl.game_slug,
        gl.season,
        gl.team_abbr,
        (gl.stats -> 'miscellaneous' ->> 'plusMinus')::numeric AS pm,
        gl.minutes AS minutes_played,
        CASE WHEN LOWER(gl.team_abbr) = LOWER(g.home_team_abbr)
             THEN g.home_score - g.away_score
             ELSE g.away_score - g.home_score
        END AS team_net,
        g.game_date_et,
        COALESCE(g.start_ts_utc, g.game_date_et::timestamp with time zone) AS order_ts
    FROM raw.nba_player_gamelogs gl
    JOIN raw.nba_games g ON g.game_slug = gl.game_slug
    WHERE (gl.stats -> 'miscellaneous' ->> 'plusMinus') IS NOT NULL
      AND g.home_score IS NOT NULL
      AND gl.minutes IS NOT NULL
      AND gl.minutes > 0
),
on_off_derived AS (
    SELECT
        player_id, game_slug, season, game_date_et, order_ts,
        CASE WHEN minutes_played >= 5
             THEN pm / minutes_played * 36.0
             ELSE NULL END AS on_net_per36,
        CASE WHEN minutes_played >= 5 AND minutes_played < 43
             THEN (team_net - pm) / NULLIF(48.0 - minutes_played, 0) * 36.0
             ELSE NULL END AS off_net_per36
    FROM pm_base
)
SELECT
    player_id, game_slug, season, game_date_et,
    AVG(on_net_per36)                     OVER w10 AS on_net_per36_avg_10,
    AVG(on_net_per36)                     OVER w5  AS on_net_per36_avg_5,
    AVG(on_net_per36 - off_net_per36)     OVER w10 AS on_off_diff_avg_10,
    AVG(on_net_per36 - off_net_per36)     OVER w5  AS on_off_diff_avg_5
FROM on_off_derived
WINDOW
    w5  AS (PARTITION BY player_id ORDER BY order_ts, game_slug
            ROWS BETWEEN  5 PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY player_id ORDER BY order_ts, game_slug
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING);
