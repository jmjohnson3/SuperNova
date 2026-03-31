-- MLB010: Weather data table DDL
-- Populated by mlb_pipeline.crawler_weather using Open-Meteo API (free, no key).
-- Historical games use archive-api.open-meteo.com; today/future use api.open-meteo.com.

CREATE TABLE IF NOT EXISTS raw.mlb_weather (
    game_slug           TEXT        PRIMARY KEY,
    temperature_f       NUMERIC(5,1),
    wind_speed_mph      NUMERIC(5,1),
    wind_direction_deg  NUMERIC(5,1),   -- 0=N, 90=E, 180=S, 270=W
    precip_prob_pct     NUMERIC(5,1),   -- 0-100
    fetched_at_utc      TIMESTAMPTZ DEFAULT now(),
    updated_at_utc      TIMESTAMPTZ DEFAULT now()
);
