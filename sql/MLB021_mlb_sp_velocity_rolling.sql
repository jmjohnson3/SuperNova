-- MLB021: SP per-start velocity rolling view
-- Depends on: raw.mlb_sp_start_velocity (populated by crawler_statcast_velocity)
--
-- Computes 5-start rolling average and trend for fastball (FF) and sinker (SI)
-- velocity.  The window looks BACK at the 5 starts preceding the current row
-- (ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING), so the trend value for a given
-- start reflects how the SP's velocity changed going INTO that start vs. recent
-- form — no lookahead leakage.
--
-- fb_velo_trend_5 > 0 → SP throwing harder than their recent 5-start average.
-- fb_velo_trend_5 < 0 → SP velocity declining (fatigue / injury signal).
--
-- This view is safe to CREATE OR REPLACE when the underlying table is empty;
-- the view simply returns no rows until data is populated by the crawler.

-- Ensure the underlying table exists (idempotent DDL so MLB021 is self-contained)
CREATE TABLE IF NOT EXISTS raw.mlb_sp_start_velocity (
    player_id      INTEGER  NOT NULL,
    player_name    TEXT,
    season_year    INTEGER  NOT NULL,
    game_date      DATE     NOT NULL,
    game_pk        INTEGER,
    ff_avg_speed   FLOAT,
    si_avg_speed   FLOAT,
    ff_n_pitches   INTEGER,
    si_n_pitches   INTEGER,
    fetched_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, game_date)
);

CREATE OR REPLACE VIEW features.mlb_sp_velocity_rolling AS
SELECT
    v.player_id,
    v.season_year,
    v.game_date,
    v.ff_avg_speed,
    v.si_avg_speed,
    AVG(v.ff_avg_speed) OVER w5                AS fb_velo_avg_5,
    v.ff_avg_speed
        - AVG(v.ff_avg_speed) OVER w5          AS fb_velo_trend_5,
    AVG(v.si_avg_speed) OVER w5                AS si_velo_avg_5,
    v.si_avg_speed
        - AVG(v.si_avg_speed) OVER w5          AS si_velo_trend_5
FROM raw.mlb_sp_start_velocity v
WINDOW
    w5 AS (
        PARTITION BY v.player_id, v.season_year
        ORDER BY v.game_date
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )
;
