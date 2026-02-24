-- ============================================================================
-- V010: Travel Distance & Altitude Features
-- ============================================================================
-- Uses the completely unused raw.nba_venues table (latitude, longitude)
-- to compute travel distance for away teams and altitude effects.
--
-- Travel distance is a well-documented fatigue factor in the NBA, especially
-- combined with back-to-back games and cross-country flights.
--
-- Haversine formula computes great-circle distance in miles.
-- ============================================================================

CREATE OR REPLACE VIEW features.game_travel_features AS
WITH games AS (
    SELECT
        g.game_slug,
        g.season,
        g.game_date_et,
        g.start_ts_utc,
        UPPER(g.home_team_abbr) AS home_team_abbr,
        UPPER(g.away_team_abbr) AS away_team_abbr
    FROM raw.nba_games g
    WHERE g.game_date_et IS NOT NULL
),

-- Get venue lat/lon for each team's home arena
venues AS (
    SELECT DISTINCT ON (UPPER(home_team_abbr))
        UPPER(home_team_abbr) AS team_abbr,
        latitude,
        longitude
    FROM raw.nba_venues
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    ORDER BY UPPER(home_team_abbr), updated_at_utc DESC
),

-- Base game with venue coordinates
game_venues AS (
    SELECT
        g.*,
        hv.latitude  AS home_lat,
        hv.longitude AS home_lon,
        av.latitude  AS away_home_lat,
        av.longitude AS away_home_lon
    FROM games g
    LEFT JOIN venues hv ON hv.team_abbr = g.home_team_abbr
    LEFT JOIN venues av ON av.team_abbr = g.away_team_abbr
),

-- Compute distances
with_distance AS (
    SELECT
        gv.*,
        -- Distance from away team's home arena to this game's arena (miles)
        CASE
            WHEN home_lat IS NOT NULL AND away_home_lat IS NOT NULL THEN
                3959.0 * acos(
                    LEAST(1.0, GREATEST(-1.0,
                        cos(radians(away_home_lat)) * cos(radians(home_lat))
                          * cos(radians(home_lon) - radians(away_home_lon))
                        + sin(radians(away_home_lat)) * sin(radians(home_lat))
                    ))
                )
            ELSE NULL
        END AS travel_distance_miles
    FROM game_venues gv
),

-- For consecutive travel: find the away team's previous game location
-- (was it at home or away? Get that venue's coordinates)
away_prev AS (
    SELECT
        d.game_slug,
        d.away_team_abbr,
        LAG(d.home_team_abbr) OVER (
            PARTITION BY d.away_team_abbr
            ORDER BY d.game_date_et, d.start_ts_utc
        ) AS prev_game_host
    FROM (
        -- All games involving this team (as home or away)
        SELECT game_slug, game_date_et, start_ts_utc, home_team_abbr, away_team_abbr,
               away_team_abbr AS the_team
        FROM games
        UNION ALL
        SELECT game_slug, game_date_et, start_ts_utc, home_team_abbr, away_team_abbr,
               home_team_abbr AS the_team
        FROM games
    ) d
    -- We only care about the away team's travel path
    -- This gives us all games for each team in chronological order
),

-- Actually, let's simplify: for each team, find their previous game's venue
team_games_ordered AS (
    SELECT
        game_slug,
        game_date_et,
        start_ts_utc,
        home_team_abbr AS venue_team,     -- the game is played at home_team's arena
        home_team_abbr,
        away_team_abbr,
        'home' AS side
    FROM games
    UNION ALL
    SELECT
        game_slug,
        game_date_et,
        start_ts_utc,
        home_team_abbr AS venue_team,
        home_team_abbr,
        away_team_abbr,
        'away' AS side
    FROM games
),

away_travel_chain AS (
    SELECT
        tg.game_slug,
        tg.away_team_abbr AS team_abbr,
        tg.venue_team AS current_venue_team,
        LAG(tg.venue_team) OVER (
            PARTITION BY tg.away_team_abbr
            ORDER BY tg.game_date_et, tg.start_ts_utc
        ) AS prev_venue_team
    FROM team_games_ordered tg
    WHERE tg.side = 'away'  -- away team's games only
       OR tg.home_team_abbr = tg.away_team_abbr  -- also include when they're at home
),

-- This is getting complex, so let's do a cleaner approach:
-- For EVERY team, track the sequence of venues they play at
all_team_venues AS (
    -- Home games: team plays at their own arena
    SELECT game_slug, season, game_date_et, start_ts_utc,
           home_team_abbr AS team_abbr, home_team_abbr AS venue_host
    FROM games
    UNION ALL
    -- Away games: team plays at opponent's arena
    SELECT game_slug, season, game_date_et, start_ts_utc,
           away_team_abbr AS team_abbr, home_team_abbr AS venue_host
    FROM games
),

team_venue_seq AS (
    SELECT
        atv.*,
        LAG(atv.venue_host) OVER (
            PARTITION BY atv.team_abbr
            ORDER BY atv.game_date_et, atv.start_ts_utc
        ) AS prev_venue_host
    FROM all_team_venues atv
),

-- Now compute prev-game travel for the away team in each game
prev_travel AS (
    SELECT
        tvs.game_slug,
        tvs.team_abbr,
        -- Distance from previous venue to current venue
        CASE
            WHEN pv.latitude IS NOT NULL AND cv.latitude IS NOT NULL THEN
                3959.0 * acos(
                    LEAST(1.0, GREATEST(-1.0,
                        cos(radians(pv.latitude)) * cos(radians(cv.latitude))
                          * cos(radians(cv.longitude) - radians(pv.longitude))
                        + sin(radians(pv.latitude)) * sin(radians(cv.latitude))
                    ))
                )
            ELSE NULL
        END AS prev_leg_miles
    FROM team_venue_seq tvs
    LEFT JOIN venues cv ON cv.team_abbr = tvs.venue_host
    LEFT JOIN venues pv ON pv.team_abbr = tvs.prev_venue_host
),

-- Rolling 5-game travel for away team
away_rolling_travel AS (
    SELECT
        pt.game_slug,
        pt.team_abbr,
        pt.prev_leg_miles,
        SUM(pt.prev_leg_miles) OVER (
            PARTITION BY pt.team_abbr
            ORDER BY g.game_date_et, g.start_ts_utc
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) AS total_travel_miles_5
    FROM prev_travel pt
    JOIN games g ON g.game_slug = pt.game_slug
        AND (g.home_team_abbr = pt.team_abbr OR g.away_team_abbr = pt.team_abbr)
)

SELECT
    wd.season,
    wd.game_slug,
    wd.game_date_et,
    wd.home_team_abbr,
    wd.away_team_abbr,

    -- Primary: distance from away team's home city to game venue
    ROUND(wd.travel_distance_miles::numeric, 0) AS travel_distance_miles,

    -- Is this a cross-country trip? (> 1500 miles)
    CASE WHEN wd.travel_distance_miles > 1500 THEN TRUE ELSE FALSE END AS is_cross_country,

    -- Previous leg distance for the away team
    ROUND(apt.prev_leg_miles::numeric, 0) AS away_prev_leg_miles,

    -- Rolling 5-game total travel for away team
    ROUND(art.total_travel_miles_5::numeric, 0) AS away_total_travel_miles_5,

    -- Altitude effect (feet above sea level)
    CASE
        WHEN wd.home_team_abbr = 'DEN' THEN 5280
        WHEN wd.home_team_abbr = 'UTA' THEN 4226
        WHEN wd.home_team_abbr = 'PHX' THEN 1086
        WHEN wd.home_team_abbr = 'ATL' THEN 1050
        WHEN wd.home_team_abbr = 'OKC' THEN 1201
        ELSE 0
    END AS home_altitude_ft,

    -- Combined fatigue indicator: long travel + altitude
    CASE
        WHEN wd.travel_distance_miles > 1500
         AND wd.home_team_abbr IN ('DEN', 'UTA')
        THEN TRUE ELSE FALSE
    END AS altitude_travel_fatigue

FROM with_distance wd
LEFT JOIN prev_travel apt
  ON apt.game_slug = wd.game_slug
 AND apt.team_abbr = wd.away_team_abbr
LEFT JOIN away_rolling_travel art
  ON art.game_slug = wd.game_slug
 AND art.team_abbr = wd.away_team_abbr;
