-- ============================================================================
-- V001: New Feature Views for Improved Predictive Power
-- ============================================================================
-- Run this script against the nba database to create all new feature views.
-- Views are ordered by dependency (leaf views first, composite views last).
-- ============================================================================

-- ============================================================================
-- 1. INJURY / LINEUP IMPACT
-- ============================================================================
-- Measures how much production is missing from a team's rotation based on
-- the nba_injuries table cross-referenced with recent player usage.
-- ============================================================================

CREATE OR REPLACE VIEW features.team_injury_impact AS
WITH latest_injuries AS (
    SELECT
        i.player_id,
        UPPER(i.team_abbr) AS team_abbr,
        i.playing_probability,     -- e.g. 'Out', 'Doubtful', 'Questionable', 'Probable'
        i.injury_description
    FROM raw.nba_injuries i
    WHERE i.team_abbr IS NOT NULL
),
-- Rolling player production (last 10 games) to quantify what each player is worth
player_recent AS (
    SELECT
        pg.season,
        pg.as_of_date AS game_date_et,
        pg.player_id,
        UPPER(pg.team_abbr) AS team_abbr,
        AVG(pg.minutes)  OVER w10 AS min_avg_10,
        AVG(pg.points)   OVER w10 AS pts_avg_10,
        AVG(pg.rebounds)  OVER w10 AS reb_avg_10,
        AVG(pg.assists)   OVER w10 AS ast_avg_10,
        ROW_NUMBER() OVER (
            PARTITION BY pg.season, UPPER(pg.team_abbr), pg.player_id
            ORDER BY pg.as_of_date DESC
        ) AS recency_rn
    FROM raw.nba_player_gamelogs pg
    WHERE pg.as_of_date IS NOT NULL
      AND pg.minutes IS NOT NULL
      AND pg.minutes > 0
    WINDOW w10 AS (
        PARTITION BY pg.season, UPPER(pg.team_abbr), pg.player_id
        ORDER BY pg.as_of_date
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    )
),
-- Most recent snapshot per player per season
player_latest AS (
    SELECT season, player_id, team_abbr,
           min_avg_10, pts_avg_10, reb_avg_10, ast_avg_10
    FROM player_recent
    WHERE recency_rn = 1
      AND min_avg_10 IS NOT NULL
),
-- Join injuries to player production
impact AS (
    SELECT
        pl.season,
        pl.team_abbr,
        li.player_id,
        li.playing_probability,
        pl.min_avg_10,
        pl.pts_avg_10,
        pl.reb_avg_10,
        pl.ast_avg_10,
        CASE
            WHEN UPPER(li.playing_probability) IN ('OUT', 'O')           THEN 1.0
            WHEN UPPER(li.playing_probability) IN ('DOUBTFUL', 'D')      THEN 0.8
            WHEN UPPER(li.playing_probability) IN ('QUESTIONABLE', 'Q')  THEN 0.4
            WHEN UPPER(li.playing_probability) IN ('PROBABLE', 'P')      THEN 0.1
            ELSE 0.0
        END AS miss_weight
    FROM latest_injuries li
    JOIN player_latest pl
      ON pl.team_abbr = li.team_abbr
     AND pl.player_id = li.player_id
)
SELECT
    season,
    team_abbr,
    -- Total weighted minutes missing from injured players
    COALESCE(SUM(min_avg_10 * miss_weight), 0) AS injured_min_lost,
    -- Total weighted points missing
    COALESCE(SUM(pts_avg_10 * miss_weight), 0) AS injured_pts_lost,
    -- Total weighted rebounds missing
    COALESCE(SUM(reb_avg_10 * miss_weight), 0) AS injured_reb_lost,
    -- Total weighted assists missing
    COALESCE(SUM(ast_avg_10 * miss_weight), 0) AS injured_ast_lost,
    -- Count of players OUT or DOUBTFUL
    COUNT(*) FILTER (WHERE miss_weight >= 0.8) AS injured_out_count,
    -- Count of players QUESTIONABLE+
    COUNT(*) FILTER (WHERE miss_weight >= 0.4) AS injured_questionable_plus_count
FROM impact
GROUP BY season, team_abbr;


-- ============================================================================
-- 2. ADVANCED EFFICIENCY (3PT%, TS%, eFG%, 3PA rate)
-- ============================================================================
-- Extracts three-point shooting and true shooting metrics from boxscore data,
-- then computes rolling pregame averages.
-- ============================================================================

CREATE OR REPLACE VIEW features.team_advanced_efficiency AS
WITH box AS (
    SELECT
        s.season,
        s.game_slug,
        s.team_abbr,
        sp.game_date_et,
        sp.start_ts_utc,
        -- Raw counting stats from JSONB
        NULLIF((s.stats->'fieldGoals'->>'fgAtt'),'')::numeric       AS fga,
        NULLIF((s.stats->'fieldGoals'->>'fgMade'),'')::numeric      AS fgm,
        NULLIF((s.stats->'fieldGoals'->>'fg3PtAtt'),'')::numeric    AS fg3a,
        NULLIF((s.stats->'fieldGoals'->>'fg3PtMade'),'')::numeric   AS fg3m,
        NULLIF((s.stats->'freeThrows'->>'ftAtt'),'')::numeric       AS fta,
        NULLIF((s.stats->'freeThrows'->>'ftMade'),'')::numeric      AS ftm,
        NULLIF((s.stats->'defense'->>'tov'),'')::numeric             AS tov,
        NULLIF((s.stats->'rebounds'->>'offReb'),'')::numeric         AS oreb,
        -- Points from offense
        NULLIF((s.stats->'offense'->>'pts'),'')::numeric             AS pts
    FROM raw.nba_boxscore_team_stats s
    JOIN features.team_game_spine sp
      ON sp.season = s.season
     AND sp.game_slug = s.game_slug
     AND sp.team_abbr = s.team_abbr
),
derived AS (
    SELECT
        season, game_slug, team_abbr, game_date_et, start_ts_utc,
        fga, fgm, fg3a, fg3m, fta, ftm, pts, tov, oreb,
        -- 3PT percentage
        CASE WHEN fg3a > 0 THEN fg3m / fg3a ELSE NULL END AS fg3_pct,
        -- 3PT attempt rate (share of FGA that are threes)
        CASE WHEN fga > 0 THEN fg3a / fga ELSE NULL END AS fg3a_rate,
        -- effective FG% = (FGM + 0.5 * 3PM) / FGA
        CASE WHEN fga > 0 THEN (fgm + 0.5 * fg3m) / fga ELSE NULL END AS efg_pct,
        -- true shooting% = PTS / (2 * (FGA + 0.44 * FTA))
        CASE WHEN (fga + 0.44 * fta) > 0 THEN pts / (2.0 * (fga + 0.44 * fta)) ELSE NULL END AS ts_pct,
        -- Turnover rate = TOV / (FGA + 0.44*FTA + TOV)
        CASE WHEN (fga + 0.44 * fta + tov) > 0 THEN tov / (fga + 0.44 * fta + tov) ELSE NULL END AS tov_rate,
        -- Offensive rebound percentage approximation (team OREBs)
        oreb AS oreb_total
    FROM box
),
ordered AS (
    SELECT *,
        COALESCE(start_ts_utc, game_date_et::timestamp with time zone) AS order_ts
    FROM derived
)
SELECT
    season, game_slug, team_abbr, game_date_et,
    -- 5-game rolling
    AVG(fg3_pct)   OVER w5 AS fg3_pct_avg_5,
    AVG(fg3a_rate) OVER w5 AS fg3a_rate_avg_5,
    AVG(efg_pct)   OVER w5 AS efg_pct_avg_5,
    AVG(ts_pct)    OVER w5 AS ts_pct_avg_5,
    AVG(tov_rate)  OVER w5 AS tov_rate_avg_5,
    -- 10-game rolling
    AVG(fg3_pct)   OVER w10 AS fg3_pct_avg_10,
    AVG(fg3a_rate) OVER w10 AS fg3a_rate_avg_10,
    AVG(efg_pct)   OVER w10 AS efg_pct_avg_10,
    AVG(ts_pct)    OVER w10 AS ts_pct_avg_10,
    AVG(tov_rate)  OVER w10 AS tov_rate_avg_10
FROM ordered
WINDOW
    w5  AS (PARTITION BY season, team_abbr ORDER BY order_ts, game_slug ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY season, team_abbr ORDER BY order_ts, game_slug ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING);


-- ============================================================================
-- 3. OPPONENT-ADJUSTED RATINGS
-- ============================================================================
-- For each team-game, computes offensive/defensive rating (per 100 poss)
-- and weights the rolling averages by opponent strength.
-- ============================================================================

CREATE OR REPLACE VIEW features.team_opp_adjusted_roll AS
WITH game_ratings AS (
    SELECT
        sp.season,
        sp.game_slug,
        sp.team_abbr,
        sp.opponent_abbr,
        sp.game_date_et,
        sp.start_ts_utc,
        sp.points_for,
        sp.points_against,
        bm.poss_est,
        -- Offensive rating = points_for / poss * 100
        CASE WHEN bm.poss_est > 0 THEN sp.points_for::numeric / bm.poss_est * 100 ELSE NULL END AS off_rtg,
        -- Defensive rating = points_against / poss * 100
        CASE WHEN bm.poss_est > 0 THEN sp.points_against::numeric / bm.poss_est * 100 ELSE NULL END AS def_rtg
    FROM features.team_game_spine sp
    LEFT JOIN features.team_boxscore_metrics bm
      ON bm.season = sp.season
     AND bm.team_abbr = sp.team_abbr
     AND bm.game_slug = sp.game_slug
    WHERE sp.points_for IS NOT NULL
),
-- Season-level opponent strength (used as weights)
opp_season_stats AS (
    SELECT
        season,
        team_abbr,
        AVG(CASE WHEN poss_est > 0 THEN points_for::numeric / poss_est * 100 ELSE NULL END) AS season_off_rtg,
        AVG(CASE WHEN poss_est > 0 THEN points_against::numeric / poss_est * 100 ELSE NULL END) AS season_def_rtg
    FROM (
        SELECT sp.season, sp.team_abbr, sp.points_for, sp.points_against, bm.poss_est
        FROM features.team_game_spine sp
        LEFT JOIN features.team_boxscore_metrics bm
          ON bm.season = sp.season AND bm.team_abbr = sp.team_abbr AND bm.game_slug = sp.game_slug
        WHERE sp.points_for IS NOT NULL AND bm.poss_est > 0
    ) sub
    GROUP BY season, team_abbr
),
enriched AS (
    SELECT
        gr.*,
        COALESCE(gr.start_ts_utc, gr.game_date_et::timestamp with time zone) AS order_ts,
        -- Opponent's season def rating (higher = worse defense = easier opponent for offense)
        opp.season_def_rtg AS opp_season_def_rtg,
        -- Opponent's season off rating (higher = better offense = harder opponent for defense)
        opp.season_off_rtg AS opp_season_off_rtg
    FROM game_ratings gr
    LEFT JOIN opp_season_stats opp
      ON opp.season = gr.season AND opp.team_abbr = gr.opponent_abbr
)
SELECT
    season, game_slug, team_abbr, game_date_et,
    -- Raw off/def rating rolling averages
    AVG(off_rtg)  OVER w10 AS off_rtg_avg_10,
    AVG(def_rtg)  OVER w10 AS def_rtg_avg_10,
    AVG(off_rtg)  OVER w10 - AVG(def_rtg) OVER w10 AS net_rtg_avg_10,
    AVG(off_rtg)  OVER w5  AS off_rtg_avg_5,
    AVG(def_rtg)  OVER w5  AS def_rtg_avg_5,
    -- Opponent strength over last 10 (schedule difficulty proxy)
    AVG(opp_season_def_rtg) OVER w10 AS opp_def_rtg_faced_avg_10,
    AVG(opp_season_off_rtg) OVER w10 AS opp_off_rtg_faced_avg_10
FROM enriched
WINDOW
    w5  AS (PARTITION BY season, team_abbr ORDER BY order_ts, game_slug ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY season, team_abbr ORDER BY order_ts, game_slug ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING);


-- ============================================================================
-- 4. STANDINGS / SEASON CONTEXT
-- ============================================================================
-- Pulls the most recent standings snapshot for each team in each season.
-- ============================================================================

CREATE OR REPLACE VIEW features.team_standings_features AS
WITH ranked AS (
    SELECT
        s.season,
        s.team_abbr,
        s.wins,
        s.losses,
        s.win_pct,
        s.conference_rank,
        s.division_rank,
        s.playoff_rank,
        s.games_back,
        s.source_fetched_at_utc,
        ROW_NUMBER() OVER (
            PARTITION BY s.season, s.team_abbr
            ORDER BY s.source_fetched_at_utc DESC
        ) AS rn
    FROM raw.nba_standings s
)
SELECT
    season,
    team_abbr,
    wins,
    losses,
    win_pct,
    conference_rank,
    division_rank,
    playoff_rank,
    games_back
FROM ranked
WHERE rn = 1;


-- ============================================================================
-- 5. REST INTERACTION FEATURES
-- ============================================================================
-- Extends team_rest_features with B2B+away flag, rest differential,
-- and 3-in-4-nights detection.
-- ============================================================================

CREATE OR REPLACE VIEW features.team_rest_interactions AS
WITH base AS (
    SELECT
        sp.season,
        sp.team_abbr,
        sp.game_slug,
        sp.game_date_et,
        sp.is_home,
        LAG(sp.game_date_et, 1) OVER w AS prev_game_1,
        LAG(sp.game_date_et, 2) OVER w AS prev_game_2,
        LAG(sp.game_date_et, 3) OVER w AS prev_game_3
    FROM features.team_game_spine sp
    WINDOW w AS (PARTITION BY sp.season, sp.team_abbr ORDER BY sp.game_date_et, sp.game_slug)
)
SELECT
    season,
    team_abbr,
    game_slug,
    game_date_et,
    is_home,
    -- Basic rest
    CASE WHEN prev_game_1 IS NOT NULL THEN game_date_et - prev_game_1 ELSE NULL END AS rest_days,
    -- Back-to-back flag
    CASE WHEN prev_game_1 IS NOT NULL THEN (game_date_et - prev_game_1) = 1 ELSE NULL END AS is_b2b,
    -- Back-to-back AND away (worst case)
    CASE WHEN prev_game_1 IS NOT NULL
         THEN (game_date_et - prev_game_1) = 1 AND NOT is_home
         ELSE NULL
    END AS is_b2b_away,
    -- 3 games in 4 nights
    CASE WHEN prev_game_2 IS NOT NULL
         THEN (game_date_et - prev_game_2) <= 3
         ELSE FALSE
    END AS is_3_in_4,
    -- 4 games in 5 nights
    CASE WHEN prev_game_3 IS NOT NULL
         THEN (game_date_et - prev_game_3) <= 4
         ELSE FALSE
    END AS is_4_in_5
FROM base;


-- ============================================================================
-- 6. HOME / AWAY SPLITS
-- ============================================================================
-- Rolling averages computed separately for home games and away games.
-- ============================================================================

CREATE OR REPLACE VIEW features.team_home_away_splits AS
WITH ordered AS (
    SELECT
        sp.season,
        sp.team_abbr,
        sp.game_slug,
        sp.game_date_et,
        sp.is_home,
        sp.points_for,
        sp.points_against,
        CASE WHEN sp.points_for > sp.points_against THEN 1.0 ELSE 0.0 END AS is_win,
        COALESCE(sp.start_ts_utc, sp.game_date_et::timestamp with time zone) AS order_ts
    FROM features.team_game_spine sp
    WHERE sp.points_for IS NOT NULL
)
SELECT
    season, team_abbr, game_slug, game_date_et, is_home,
    -- Home-only rolling (only computed for home games, NULL for away)
    CASE WHEN is_home THEN AVG(points_for)     OVER wh10 ELSE NULL END AS home_pts_for_avg_10,
    CASE WHEN is_home THEN AVG(points_against)  OVER wh10 ELSE NULL END AS home_pts_against_avg_10,
    CASE WHEN is_home THEN AVG(is_win)          OVER wh10 ELSE NULL END AS home_win_pct_10,
    -- Away-only rolling (only computed for away games, NULL for home)
    CASE WHEN NOT is_home THEN AVG(points_for)     OVER wa10 ELSE NULL END AS away_pts_for_avg_10,
    CASE WHEN NOT is_home THEN AVG(points_against)  OVER wa10 ELSE NULL END AS away_pts_against_avg_10,
    CASE WHEN NOT is_home THEN AVG(is_win)          OVER wa10 ELSE NULL END AS away_win_pct_10,
    -- Overall rolling win%
    AVG(is_win) OVER wall10 AS overall_win_pct_10
FROM ordered
WINDOW
    wh10   AS (PARTITION BY season, team_abbr, CASE WHEN is_home THEN 1 ELSE 0 END
               ORDER BY order_ts, game_slug ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
    wa10   AS (PARTITION BY season, team_abbr, CASE WHEN is_home THEN 1 ELSE 0 END
               ORDER BY order_ts, game_slug ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
    wall10 AS (PARTITION BY season, team_abbr
               ORDER BY order_ts, game_slug ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING);


-- ============================================================================
-- 7. CLUTCH PERFORMANCE (reworked)
-- ============================================================================
-- Net points in clutch time (last 5 min of Q4 when margin <= 5)
-- rather than just counting clutch plays.
-- ============================================================================

CREATE OR REPLACE VIEW features.team_clutch_performance AS
WITH clutch_events AS (
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        p.team_abbr,
        p.period,
        p.clock,
        p.points_home,
        p.points_away,
        p.event_type,
        -- Parse the clock to identify last 5 minutes of Q4
        CASE
            WHEN p.period = 4
             AND p.clock ~ '^\d{1,2}:\d{2}$'
             AND split_part(p.clock, ':', 1)::int <= 5
             AND ABS(p.points_home - p.points_away) <= 5
            THEN TRUE
            ELSE FALSE
        END AS is_clutch
    FROM raw.nba_pbp_plays p
    JOIN raw.nba_games g USING (game_slug)
    WHERE g.game_date_et IS NOT NULL
),
-- Get final clutch score differential per team per game
game_clutch_summary AS (
    SELECT
        season,
        game_slug,
        game_date_et,
        -- We track from the home team perspective then flip
        MAX(points_home) FILTER (WHERE is_clutch) AS clutch_end_home,
        MIN(points_home) FILTER (WHERE is_clutch) AS clutch_start_home,
        MAX(points_away) FILTER (WHERE is_clutch) AS clutch_end_away,
        MIN(points_away) FILTER (WHERE is_clutch) AS clutch_start_away,
        COUNT(*) FILTER (WHERE is_clutch) AS clutch_play_count
    FROM clutch_events
    GROUP BY season, game_slug, game_date_et
),
-- Per team: how many clutch points did they score vs allow?
team_clutch AS (
    SELECT
        gs.season,
        sp.team_abbr,
        sp.game_slug,
        sp.game_date_et,
        gs.clutch_play_count,
        CASE WHEN sp.is_home THEN
            COALESCE(gs.clutch_end_home - gs.clutch_start_home, 0)
        ELSE
            COALESCE(gs.clutch_end_away - gs.clutch_start_away, 0)
        END AS clutch_pts_scored,
        CASE WHEN sp.is_home THEN
            COALESCE(gs.clutch_end_away - gs.clutch_start_away, 0)
        ELSE
            COALESCE(gs.clutch_end_home - gs.clutch_start_home, 0)
        END AS clutch_pts_allowed
    FROM game_clutch_summary gs
    JOIN features.team_game_spine sp
      ON sp.season = gs.season AND sp.game_slug = gs.game_slug
    WHERE gs.clutch_play_count > 0
),
ordered AS (
    SELECT *,
        clutch_pts_scored - clutch_pts_allowed AS clutch_net,
        COALESCE(
            (SELECT start_ts_utc FROM features.team_game_spine s
             WHERE s.season = team_clutch.season AND s.game_slug = team_clutch.game_slug AND s.team_abbr = team_clutch.team_abbr),
            team_clutch.game_date_et::timestamp with time zone
        ) AS order_ts
    FROM team_clutch
)
SELECT
    season, team_abbr, game_slug, game_date_et,
    AVG(clutch_net)         OVER w10 AS clutch_net_avg_10,
    AVG(clutch_pts_scored)  OVER w10 AS clutch_pts_scored_avg_10,
    AVG(clutch_pts_allowed) OVER w10 AS clutch_pts_allowed_avg_10,
    AVG(clutch_play_count)  OVER w10 AS clutch_play_cnt_avg_10
FROM ordered
WINDOW w10 AS (
    PARTITION BY season, team_abbr
    ORDER BY order_ts, game_slug
    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
);


-- ============================================================================
-- 8. PLAYER MATCHUP FEATURES
-- ============================================================================
-- For each player-game, adds the opponent's defensive rating, pace,
-- and the team implied total from market lines.
-- This enriches player_training_features with matchup context.
-- ============================================================================

CREATE OR REPLACE VIEW features.player_matchup_features AS
WITH player_base AS (
    SELECT
        pg.season,
        pg.game_slug,
        pg.as_of_date AS game_date_et,
        pg.player_id,
        UPPER(pg.team_abbr)     AS team_abbr,
        UPPER(pg.opponent_abbr) AS opponent_abbr,
        pg.is_home,
        pg.minutes,
        pg.points::numeric   AS points,
        pg.rebounds::numeric AS rebounds,
        pg.assists::numeric  AS assists,
        pg.threes_made::numeric AS threes_made,
        pg.fga::numeric AS fga,
        pg.fta::numeric AS fta
    FROM raw.nba_player_gamelogs pg
    WHERE pg.minutes IS NOT NULL AND pg.minutes > 0
      AND pg.as_of_date IS NOT NULL
),
game_info AS (
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        g.start_ts_utc,
        g.status,
        g.home_team_abbr,
        g.away_team_abbr
    FROM raw.nba_games g
),
-- Opponent's defensive profile (rolling 10)
opp_defense AS (
    SELECT
        oar.season,
        oar.team_abbr,
        oar.game_slug,
        oar.game_date_et,
        oar.def_rtg_avg_10  AS opp_def_rtg_10,
        oar.off_rtg_avg_10  AS opp_off_rtg_10
    FROM features.team_opp_adjusted_roll oar
),
-- Opponent's pace
opp_pace AS (
    SELECT
        pr.season,
        pr.team_abbr,
        pr.game_slug,
        pr.game_date_et,
        pr.pace_avg_5 AS opp_pace_avg_5,
        pr.pace_avg_10 AS opp_pace_avg_10
    FROM features.team_pregame_rolling_boxscore pr
),
-- Market-derived implied team totals
mkt AS (
    SELECT
        oc.as_of_date,
        oc.home_team_abbr,
        oc.away_team_abbr,
        oc.close_total AS market_total,
        oc.close_spread_home_points AS market_spread_home
    FROM odds.nba_game_lines_open_close oc
    WHERE oc.bookmaker_key = 'draftkings'
),
joined AS (
    SELECT
        pb.season,
        pb.game_slug,
        pb.game_date_et,
        gi.start_ts_utc,
        pb.player_id,
        pb.team_abbr,
        pb.opponent_abbr,
        pb.is_home,
        pb.minutes,
        pb.points,
        pb.rebounds,
        pb.assists,
        pb.threes_made,
        pb.fga,
        pb.fta,
        -- Opponent's defensive rating the player faces
        od.opp_def_rtg_10,
        od.opp_off_rtg_10,
        -- Opponent pace
        op.opp_pace_avg_5,
        op.opp_pace_avg_10,
        -- Market implied team total
        CASE
            WHEN mk.market_total IS NULL OR mk.market_spread_home IS NULL THEN NULL
            WHEN pb.team_abbr = gi.home_team_abbr THEN (mk.market_total / 2.0) - (mk.market_spread_home / 2.0)
            WHEN pb.team_abbr = gi.away_team_abbr THEN (mk.market_total / 2.0) + (mk.market_spread_home / 2.0)
            ELSE NULL
        END AS team_implied_total,
        mk.market_total AS game_market_total,
        mk.market_spread_home AS game_market_spread
    FROM player_base pb
    JOIN game_info gi
      ON gi.season = pb.season AND gi.game_slug = pb.game_slug
    -- Opponent defensive profile for this game
    LEFT JOIN opp_defense od
      ON od.season = pb.season
     AND od.team_abbr = pb.opponent_abbr
     AND od.game_slug = pb.game_slug
    -- Opponent pace
    LEFT JOIN opp_pace op
      ON op.season = pb.season
     AND op.team_abbr = pb.opponent_abbr
     AND op.game_slug = pb.game_slug
    -- Market data
    LEFT JOIN mkt mk
      ON mk.as_of_date = gi.game_date_et
     AND mk.home_team_abbr = gi.home_team_abbr
     AND mk.away_team_abbr = gi.away_team_abbr
),
-- Player rolling stats + matchup context
rolled AS (
    SELECT
        j.*,
        -- Player rolling stats
        COUNT(*) OVER w10 AS n_games_prev_10,
        AVG(j.minutes)  OVER w5  AS min_avg_5,
        AVG(j.minutes)  OVER w10 AS min_avg_10,
        AVG(j.points)   OVER w5  AS pts_avg_5,
        AVG(j.points)   OVER w10 AS pts_avg_10,
        AVG(j.rebounds)  OVER w5  AS reb_avg_5,
        AVG(j.rebounds)  OVER w10 AS reb_avg_10,
        AVG(j.assists)   OVER w5  AS ast_avg_5,
        AVG(j.assists)   OVER w10 AS ast_avg_10,
        AVG(j.fga)       OVER w10 AS fga_avg_10,
        AVG(j.fta)       OVER w10 AS fta_avg_10,
        AVG(j.threes_made) OVER w10 AS threes_avg_10,
        STDDEV_SAMP(j.points)   OVER w10 AS pts_sd_10,
        STDDEV_SAMP(j.rebounds) OVER w10 AS reb_sd_10,
        STDDEV_SAMP(j.assists)  OVER w10 AS ast_sd_10,
        -- Usage rate proxy: (FGA + 0.44*FTA + TOV) / minutes * 48
        -- (approximation since we don't have team possessions here)
        CASE WHEN j.minutes > 0
            THEN (j.fga + 0.44 * j.fta) / j.minutes
            ELSE NULL
        END AS usage_proxy_per_min,
        -- Player rest days
        EXTRACT(EPOCH FROM (j.start_ts_utc - LAG(j.start_ts_utc) OVER (PARTITION BY j.player_id ORDER BY j.start_ts_utc))) / 86400.0 AS player_rest_days
    FROM joined j
    WINDOW
        w5  AS (PARTITION BY j.player_id ORDER BY j.start_ts_utc ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
        w10 AS (PARTITION BY j.player_id ORDER BY j.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING)
)
SELECT
    season, game_slug, game_date_et, start_ts_utc,
    player_id, team_abbr, opponent_abbr, is_home,
    -- Actuals (targets for training)
    points, rebounds, assists, minutes,
    -- Player rolling
    n_games_prev_10,
    min_avg_5, min_avg_10,
    pts_avg_5, pts_avg_10, pts_sd_10,
    reb_avg_5, reb_avg_10, reb_sd_10,
    ast_avg_5, ast_avg_10, ast_sd_10,
    fga_avg_10, fta_avg_10, threes_avg_10,
    player_rest_days,
    -- Usage proxy rolling
    AVG(usage_proxy_per_min) OVER (PARTITION BY player_id ORDER BY start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS usage_proxy_avg_10,
    -- Matchup features (opponent context)
    opp_def_rtg_10,
    opp_off_rtg_10,
    opp_pace_avg_5,
    opp_pace_avg_10,
    -- Market context
    team_implied_total,
    game_market_total,
    game_market_spread
FROM rolled;


-- ============================================================================
-- 9. MARKET-DERIVED FEATURES
-- ============================================================================
-- Implied team scores, spread pricing signal, multi-book consensus.
-- ============================================================================

CREATE OR REPLACE VIEW features.team_market_derived AS
WITH dk AS (
    SELECT
        oc.as_of_date,
        oc.home_team_abbr,
        oc.away_team_abbr,
        oc.close_spread_home_points,
        oc.close_total,
        oc.open_spread_home_points,
        oc.open_total,
        oc.line_move_margin,
        oc.line_move_total
    FROM odds.nba_game_lines_open_close oc
    WHERE oc.bookmaker_key = 'draftkings'
),
-- Multi-book consensus: average closing spread & total across all books
consensus AS (
    SELECT
        oc.as_of_date,
        oc.home_team_abbr,
        oc.away_team_abbr,
        AVG(oc.close_spread_home_points)  AS consensus_spread_home,
        AVG(oc.close_total)               AS consensus_total,
        STDDEV_SAMP(oc.close_spread_home_points) AS spread_book_disagreement,
        STDDEV_SAMP(oc.close_total)       AS total_book_disagreement,
        COUNT(DISTINCT oc.bookmaker_key)  AS num_books
    FROM odds.nba_game_lines_open_close oc
    WHERE oc.close_spread_home_points IS NOT NULL
    GROUP BY oc.as_of_date, oc.home_team_abbr, oc.away_team_abbr
)
SELECT
    dk.as_of_date,
    dk.home_team_abbr,
    dk.away_team_abbr,
    -- DK primary lines
    dk.close_spread_home_points AS market_spread_home,
    dk.close_total              AS market_total,
    dk.open_spread_home_points  AS market_open_spread_home,
    dk.open_total               AS market_open_total,
    dk.line_move_margin         AS market_line_move_margin,
    dk.line_move_total          AS market_line_move_total,
    -- Implied team scores
    CASE WHEN dk.close_total IS NOT NULL AND dk.close_spread_home_points IS NOT NULL
         THEN (dk.close_total / 2.0) - (dk.close_spread_home_points / 2.0)
         ELSE NULL
    END AS home_implied_score,
    CASE WHEN dk.close_total IS NOT NULL AND dk.close_spread_home_points IS NOT NULL
         THEN (dk.close_total / 2.0) + (dk.close_spread_home_points / 2.0)
         ELSE NULL
    END AS away_implied_score,
    -- Multi-book consensus
    c.consensus_spread_home,
    c.consensus_total,
    c.spread_book_disagreement,
    c.total_book_disagreement,
    c.num_books,
    -- DraftKings deviation from consensus (signal of "sharp" vs "public" divergence)
    dk.close_spread_home_points - c.consensus_spread_home AS dk_vs_consensus_spread,
    dk.close_total - c.consensus_total                     AS dk_vs_consensus_total
FROM dk
LEFT JOIN consensus c
  ON c.as_of_date = dk.as_of_date
 AND c.home_team_abbr = dk.home_team_abbr
 AND c.away_team_abbr = dk.away_team_abbr;
