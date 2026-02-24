-- ============================================================================
-- V002: Updated Game Prediction & Training Feature Views
-- ============================================================================
-- These replace the existing game_prediction_features and game_training_features
-- views with versions that incorporate all the new feature views from V001.
-- ============================================================================

-- ============================================================================
-- GAME PREDICTION FEATURES (used at inference time)
-- ============================================================================

CREATE OR REPLACE VIEW features.game_prediction_features AS
WITH base AS (
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        g.start_ts_utc,
        g.home_team_abbr,
        g.away_team_abbr
    FROM raw.nba_games g
)
SELECT
    b.season,
    b.game_slug,
    b.game_date_et,
    b.start_ts_utc,
    b.home_team_abbr,
    b.away_team_abbr,

    -- ===== REST FEATURES =====
    hr.rest_days                AS home_rest_days,
    hr.is_b2b                  AS home_is_b2b,
    ar.rest_days                AS away_rest_days,
    ar.is_b2b                  AS away_is_b2b,

    -- ===== REST INTERACTIONS (NEW) =====
    hri.is_b2b_away             AS home_is_b2b_away,
    ari.is_b2b_away             AS away_is_b2b_away,
    hri.is_3_in_4               AS home_is_3_in_4,
    ari.is_3_in_4               AS away_is_3_in_4,
    hri.is_4_in_5               AS home_is_4_in_5,
    ari.is_4_in_5               AS away_is_4_in_5,
    COALESCE(hr.rest_days, 2) - COALESCE(ar.rest_days, 2) AS rest_advantage_home,

    -- ===== TEAM FORM (existing) =====
    hf.pts_for_avg_5            AS home_pts_for_avg_5,
    hf.pts_against_avg_5        AS home_pts_against_avg_5,
    hf.pts_for_avg_10           AS home_pts_for_avg_10,
    hf.pts_against_avg_10       AS home_pts_against_avg_10,
    hf.pts_for_sd_10            AS home_pts_for_sd_10,
    af.pts_for_avg_5            AS away_pts_for_avg_5,
    af.pts_against_avg_5        AS away_pts_against_avg_5,
    af.pts_for_avg_10           AS away_pts_for_avg_10,
    af.pts_against_avg_10       AS away_pts_against_avg_10,
    af.pts_for_sd_10            AS away_pts_for_sd_10,

    -- ===== PACE & BOXSCORE ROLLING (existing) =====
    hroll.pace_avg_5            AS home_pace_avg_5,
    hroll.pace_avg_10           AS home_pace_avg_10,
    aroll.pace_avg_5            AS away_pace_avg_5,
    aroll.pace_avg_10           AS away_pace_avg_10,
    hroll.fg_pct_avg_5          AS home_fg_pct_avg_5,
    aroll.fg_pct_avg_5          AS away_fg_pct_avg_5,
    hroll.ft_pct_avg_5          AS home_ft_pct_avg_5,
    aroll.ft_pct_avg_5          AS away_ft_pct_avg_5,
    hroll.tov_avg_5             AS home_tov_avg_5,
    aroll.tov_avg_5             AS away_tov_avg_5,
    hroll.oreb_avg_5            AS home_oreb_avg_5,
    aroll.oreb_avg_5            AS away_oreb_avg_5,

    -- ===== CROSS-TEAM INTERACTIONS (existing) =====
    hf.pts_for_avg_10 - af.pts_against_avg_10 AS home_off_vs_away_def,
    af.pts_for_avg_10 - hf.pts_against_avg_10 AS away_off_vs_home_def,
    (hroll.pace_avg_5 + aroll.pace_avg_5) / 2.0 AS game_pace_est_5,

    -- ===== ADVANCED EFFICIENCY (NEW) =====
    hae.fg3_pct_avg_5           AS home_fg3_pct_avg_5,
    hae.fg3a_rate_avg_5         AS home_fg3a_rate_avg_5,
    hae.efg_pct_avg_5           AS home_efg_pct_avg_5,
    hae.ts_pct_avg_5            AS home_ts_pct_avg_5,
    hae.tov_rate_avg_5          AS home_tov_rate_avg_5,
    hae.efg_pct_avg_10          AS home_efg_pct_avg_10,
    hae.ts_pct_avg_10           AS home_ts_pct_avg_10,
    aae.fg3_pct_avg_5           AS away_fg3_pct_avg_5,
    aae.fg3a_rate_avg_5         AS away_fg3a_rate_avg_5,
    aae.efg_pct_avg_5           AS away_efg_pct_avg_5,
    aae.ts_pct_avg_5            AS away_ts_pct_avg_5,
    aae.tov_rate_avg_5          AS away_tov_rate_avg_5,
    aae.efg_pct_avg_10          AS away_efg_pct_avg_10,
    aae.ts_pct_avg_10           AS away_ts_pct_avg_10,

    -- ===== OPPONENT-ADJUSTED RATINGS (NEW) =====
    hoar.off_rtg_avg_10         AS home_off_rtg_avg_10,
    hoar.def_rtg_avg_10         AS home_def_rtg_avg_10,
    hoar.net_rtg_avg_10         AS home_net_rtg_avg_10,
    aoar.off_rtg_avg_10         AS away_off_rtg_avg_10,
    aoar.def_rtg_avg_10         AS away_def_rtg_avg_10,
    aoar.net_rtg_avg_10         AS away_net_rtg_avg_10,
    hoar.net_rtg_avg_10 - aoar.net_rtg_avg_10 AS net_rtg_diff_10,
    hoar.opp_def_rtg_faced_avg_10 AS home_sos_def_10,
    aoar.opp_def_rtg_faced_avg_10 AS away_sos_def_10,

    -- ===== STANDINGS / SEASON CONTEXT (NEW) =====
    hst.win_pct                 AS home_win_pct,
    ast_st.win_pct              AS away_win_pct,
    hst.conference_rank         AS home_conf_rank,
    ast_st.conference_rank      AS away_conf_rank,
    COALESCE(hst.win_pct, 0.5) - COALESCE(ast_st.win_pct, 0.5) AS win_pct_diff,

    -- ===== HOME/AWAY SPLITS (NEW) =====
    hhas.home_win_pct_10        AS home_home_win_pct_10,
    aaas.away_win_pct_10        AS away_away_win_pct_10,

    -- ===== INJURY IMPACT (NEW) =====
    COALESCE(hinj.injured_pts_lost, 0)    AS home_injured_pts_lost,
    COALESCE(ainj.injured_pts_lost, 0)    AS away_injured_pts_lost,
    COALESCE(hinj.injured_min_lost, 0)    AS home_injured_min_lost,
    COALESCE(ainj.injured_min_lost, 0)    AS away_injured_min_lost,
    COALESCE(hinj.injured_out_count, 0)   AS home_injured_out_count,
    COALESCE(ainj.injured_out_count, 0)   AS away_injured_out_count,
    COALESCE(ainj.injured_pts_lost, 0) - COALESCE(hinj.injured_pts_lost, 0) AS injury_pts_advantage_home,

    -- ===== CLUTCH PERFORMANCE (NEW) =====
    hclutch.clutch_net_avg_10   AS home_clutch_net_avg_10,
    aclutch.clutch_net_avg_10   AS away_clutch_net_avg_10,

    -- ===== MARKET DATA (existing + new derived) =====
    md.market_spread_home,
    md.market_total,
    md.market_open_spread_home,
    md.market_open_total,
    md.market_line_move_margin,
    md.market_line_move_total,
    md.home_implied_score,
    md.away_implied_score,
    md.consensus_spread_home,
    md.consensus_total,
    md.spread_book_disagreement,
    md.total_book_disagreement,
    md.dk_vs_consensus_spread,
    md.dk_vs_consensus_total,

    -- ===== PLAYER USAGE (existing) =====
    hu.top8_min_avg_5           AS home_top8_min_avg_5,
    hu.top8_min_avg_10          AS home_top8_min_avg_10,
    hu.rotation_depth_5         AS home_rotation_depth_5,
    hu.rotation_depth_10        AS home_rotation_depth_10,
    hu.top3_pts_avg_5           AS home_top3_pts_avg_5,
    hu.top3_pts_avg_10          AS home_top3_pts_avg_10,
    au.top8_min_avg_5           AS away_top8_min_avg_5,
    au.top8_min_avg_10          AS away_top8_min_avg_10,
    au.rotation_depth_5         AS away_rotation_depth_5,
    au.rotation_depth_10        AS away_rotation_depth_10,
    au.top3_pts_avg_5           AS away_top3_pts_avg_5,
    au.top3_pts_avg_10          AS away_top3_pts_avg_10

FROM base b

-- Existing joins
LEFT JOIN features.team_rest_features hr
  ON hr.season = b.season AND hr.game_slug = b.game_slug AND hr.team_abbr = b.home_team_abbr
LEFT JOIN features.team_rest_features ar
  ON ar.season = b.season AND ar.game_slug = b.game_slug AND ar.team_abbr = b.away_team_abbr
LEFT JOIN features.team_form_features hf
  ON hf.season = b.season AND hf.game_slug = b.game_slug AND hf.team_abbr = b.home_team_abbr
LEFT JOIN features.team_form_features af
  ON af.season = b.season AND af.game_slug = b.game_slug AND af.team_abbr = b.away_team_abbr
LEFT JOIN features.team_pregame_rolling_boxscore hroll
  ON hroll.season = b.season AND hroll.game_slug = b.game_slug AND hroll.team_abbr = b.home_team_abbr
LEFT JOIN features.team_pregame_rolling_boxscore aroll
  ON aroll.season = b.season AND aroll.game_slug = b.game_slug AND aroll.team_abbr = b.away_team_abbr
LEFT JOIN features.team_player_usage_roll hu
  ON hu.season = b.season AND hu.team_abbr = b.home_team_abbr AND hu.game_date_et = b.game_date_et
LEFT JOIN features.team_player_usage_roll au
  ON au.season = b.season AND au.team_abbr = b.away_team_abbr AND au.game_date_et = b.game_date_et

-- NEW joins: rest interactions
LEFT JOIN features.team_rest_interactions hri
  ON hri.season = b.season AND hri.game_slug = b.game_slug AND hri.team_abbr = b.home_team_abbr
LEFT JOIN features.team_rest_interactions ari
  ON ari.season = b.season AND ari.game_slug = b.game_slug AND ari.team_abbr = b.away_team_abbr

-- NEW joins: advanced efficiency
LEFT JOIN features.team_advanced_efficiency hae
  ON hae.season = b.season AND hae.game_slug = b.game_slug AND hae.team_abbr = b.home_team_abbr
LEFT JOIN features.team_advanced_efficiency aae
  ON aae.season = b.season AND aae.game_slug = b.game_slug AND aae.team_abbr = b.away_team_abbr

-- NEW joins: opponent-adjusted ratings
LEFT JOIN features.team_opp_adjusted_roll hoar
  ON hoar.season = b.season AND hoar.game_slug = b.game_slug AND hoar.team_abbr = b.home_team_abbr
LEFT JOIN features.team_opp_adjusted_roll aoar
  ON aoar.season = b.season AND aoar.game_slug = b.game_slug AND aoar.team_abbr = b.away_team_abbr

-- NEW joins: standings
LEFT JOIN features.team_standings_features hst
  ON hst.season = b.season AND hst.team_abbr = b.home_team_abbr
LEFT JOIN features.team_standings_features ast_st
  ON ast_st.season = b.season AND ast_st.team_abbr = b.away_team_abbr

-- NEW joins: home/away splits (latest home game for home team, latest away game for away team)
LEFT JOIN LATERAL (
    SELECT home_win_pct_10
    FROM features.team_home_away_splits s
    WHERE s.season = b.season AND s.team_abbr = b.home_team_abbr
      AND s.is_home = TRUE AND s.game_date_et < b.game_date_et
    ORDER BY s.game_date_et DESC
    LIMIT 1
) hhas ON TRUE
LEFT JOIN LATERAL (
    SELECT away_win_pct_10
    FROM features.team_home_away_splits s
    WHERE s.season = b.season AND s.team_abbr = b.away_team_abbr
      AND s.is_home = FALSE AND s.game_date_et < b.game_date_et
    ORDER BY s.game_date_et DESC
    LIMIT 1
) aaas ON TRUE

-- NEW joins: injury impact
LEFT JOIN features.team_injury_impact hinj
  ON hinj.season = b.season AND hinj.team_abbr = b.home_team_abbr
LEFT JOIN features.team_injury_impact ainj
  ON ainj.season = b.season AND ainj.team_abbr = b.away_team_abbr

-- NEW joins: clutch performance
LEFT JOIN features.team_clutch_performance hclutch
  ON hclutch.season = b.season AND hclutch.game_slug = b.game_slug AND hclutch.team_abbr = b.home_team_abbr
LEFT JOIN features.team_clutch_performance aclutch
  ON aclutch.season = b.season AND aclutch.game_slug = b.game_slug AND aclutch.team_abbr = b.away_team_abbr

-- NEW joins: market derived (replaces old mkt CTE)
LEFT JOIN features.team_market_derived md
  ON md.as_of_date = b.game_date_et
 AND md.home_team_abbr = b.home_team_abbr
 AND md.away_team_abbr = b.away_team_abbr;


-- ============================================================================
-- GAME TRAINING FEATURES (used at training time, includes targets)
-- ============================================================================

CREATE OR REPLACE VIEW features.game_training_features AS
WITH base AS (
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        g.start_ts_utc,
        UPPER(BTRIM(g.home_team_abbr)) AS home_team_abbr,
        UPPER(BTRIM(g.away_team_abbr)) AS away_team_abbr,
        g.home_score,
        g.away_score,
        g.home_score + g.away_score AS total_points,
        g.home_score - g.away_score AS margin
    FROM raw.nba_games g
    WHERE g.status = 'final'
)
SELECT
    b.season,
    b.game_slug,
    b.game_date_et,
    b.start_ts_utc,
    b.home_team_abbr,
    b.away_team_abbr,
    b.home_score,
    b.away_score,
    b.total_points,
    b.margin,

    -- ===== REST FEATURES =====
    hr.rest_days                AS home_rest_days,
    hr.is_b2b                  AS home_is_b2b,
    ar.rest_days                AS away_rest_days,
    ar.is_b2b                  AS away_is_b2b,

    -- ===== REST INTERACTIONS (NEW) =====
    hri.is_b2b_away             AS home_is_b2b_away,
    ari.is_b2b_away             AS away_is_b2b_away,
    hri.is_3_in_4               AS home_is_3_in_4,
    ari.is_3_in_4               AS away_is_3_in_4,
    hri.is_4_in_5               AS home_is_4_in_5,
    ari.is_4_in_5               AS away_is_4_in_5,
    COALESCE(hr.rest_days, 2) - COALESCE(ar.rest_days, 2) AS rest_advantage_home,

    -- ===== TEAM FORM (existing) =====
    hf.pts_for_avg_5            AS home_pts_for_avg_5,
    hf.pts_against_avg_5        AS home_pts_against_avg_5,
    hf.pts_for_avg_10           AS home_pts_for_avg_10,
    hf.pts_against_avg_10       AS home_pts_against_avg_10,
    hf.pts_for_sd_10            AS home_pts_for_sd_10,
    af.pts_for_avg_5            AS away_pts_for_avg_5,
    af.pts_against_avg_5        AS away_pts_against_avg_5,
    af.pts_for_avg_10           AS away_pts_for_avg_10,
    af.pts_against_avg_10       AS away_pts_against_avg_10,
    af.pts_for_sd_10            AS away_pts_for_sd_10,

    -- ===== PACE & BOXSCORE ROLLING (existing) =====
    hroll.pace_n_5              AS home_pace_n_5,
    hroll.pace_avg_5            AS home_pace_avg_5,
    hroll.pace_avg_10           AS home_pace_avg_10,
    hroll.fg_pct_avg_5          AS home_fg_pct_avg_5,
    hroll.ft_pct_avg_5          AS home_ft_pct_avg_5,
    hroll.tov_avg_5             AS home_tov_avg_5,
    hroll.oreb_avg_5            AS home_oreb_avg_5,
    aroll.pace_n_5              AS away_pace_n_5,
    aroll.pace_avg_5            AS away_pace_avg_5,
    aroll.pace_avg_10           AS away_pace_avg_10,
    aroll.fg_pct_avg_5          AS away_fg_pct_avg_5,
    aroll.ft_pct_avg_5          AS away_ft_pct_avg_5,
    aroll.tov_avg_5             AS away_tov_avg_5,
    aroll.oreb_avg_5            AS away_oreb_avg_5,

    -- ===== CROSS-TEAM INTERACTIONS (existing) =====
    hf.pts_for_avg_10 - af.pts_against_avg_10 AS home_off_vs_away_def,
    af.pts_for_avg_10 - hf.pts_against_avg_10 AS away_off_vs_home_def,
    (hroll.pace_avg_5 + aroll.pace_avg_5) / 2.0 AS game_pace_est_5,

    -- ===== ADVANCED EFFICIENCY (NEW) =====
    hae.fg3_pct_avg_5           AS home_fg3_pct_avg_5,
    hae.fg3a_rate_avg_5         AS home_fg3a_rate_avg_5,
    hae.efg_pct_avg_5           AS home_efg_pct_avg_5,
    hae.ts_pct_avg_5            AS home_ts_pct_avg_5,
    hae.tov_rate_avg_5          AS home_tov_rate_avg_5,
    hae.efg_pct_avg_10          AS home_efg_pct_avg_10,
    hae.ts_pct_avg_10           AS home_ts_pct_avg_10,
    aae.fg3_pct_avg_5           AS away_fg3_pct_avg_5,
    aae.fg3a_rate_avg_5         AS away_fg3a_rate_avg_5,
    aae.efg_pct_avg_5           AS away_efg_pct_avg_5,
    aae.ts_pct_avg_5            AS away_ts_pct_avg_5,
    aae.tov_rate_avg_5          AS away_tov_rate_avg_5,
    aae.efg_pct_avg_10          AS away_efg_pct_avg_10,
    aae.ts_pct_avg_10           AS away_ts_pct_avg_10,

    -- ===== OPPONENT-ADJUSTED RATINGS (NEW) =====
    hoar.off_rtg_avg_10         AS home_off_rtg_avg_10,
    hoar.def_rtg_avg_10         AS home_def_rtg_avg_10,
    hoar.net_rtg_avg_10         AS home_net_rtg_avg_10,
    aoar.off_rtg_avg_10         AS away_off_rtg_avg_10,
    aoar.def_rtg_avg_10         AS away_def_rtg_avg_10,
    aoar.net_rtg_avg_10         AS away_net_rtg_avg_10,
    hoar.net_rtg_avg_10 - aoar.net_rtg_avg_10 AS net_rtg_diff_10,
    hoar.opp_def_rtg_faced_avg_10 AS home_sos_def_10,
    aoar.opp_def_rtg_faced_avg_10 AS away_sos_def_10,

    -- ===== STANDINGS / SEASON CONTEXT (NEW) =====
    hst.win_pct                 AS home_win_pct,
    ast_st.win_pct              AS away_win_pct,
    hst.conference_rank         AS home_conf_rank,
    ast_st.conference_rank      AS away_conf_rank,
    COALESCE(hst.win_pct, 0.5) - COALESCE(ast_st.win_pct, 0.5) AS win_pct_diff,

    -- ===== HOME/AWAY SPLITS (NEW) =====
    hhas.home_win_pct_10        AS home_home_win_pct_10,
    aaas.away_win_pct_10        AS away_away_win_pct_10,

    -- ===== INJURY IMPACT (NEW) =====
    COALESCE(hinj.injured_pts_lost, 0)    AS home_injured_pts_lost,
    COALESCE(ainj.injured_pts_lost, 0)    AS away_injured_pts_lost,
    COALESCE(hinj.injured_min_lost, 0)    AS home_injured_min_lost,
    COALESCE(ainj.injured_min_lost, 0)    AS away_injured_min_lost,
    COALESCE(hinj.injured_out_count, 0)   AS home_injured_out_count,
    COALESCE(ainj.injured_out_count, 0)   AS away_injured_out_count,
    COALESCE(ainj.injured_pts_lost, 0) - COALESCE(hinj.injured_pts_lost, 0) AS injury_pts_advantage_home,

    -- ===== CLUTCH PERFORMANCE (NEW) =====
    hclutch.clutch_net_avg_10   AS home_clutch_net_avg_10,
    aclutch.clutch_net_avg_10   AS away_clutch_net_avg_10,

    -- ===== MARKET DATA (existing + new derived) =====
    md.market_spread_home,
    md.market_total,
    md.market_open_spread_home,
    md.market_open_total,
    md.market_line_move_margin,
    md.market_line_move_total,
    md.home_implied_score,
    md.away_implied_score,
    md.consensus_spread_home,
    md.consensus_total,
    md.spread_book_disagreement,
    md.total_book_disagreement,
    md.dk_vs_consensus_spread,
    md.dk_vs_consensus_total,

    -- ===== PLAYER USAGE (existing) =====
    hu.top8_min_avg_5           AS home_top8_min_avg_5,
    hu.top8_min_avg_10          AS home_top8_min_avg_10,
    hu.rotation_depth_5         AS home_rotation_depth_5,
    hu.rotation_depth_10        AS home_rotation_depth_10,
    hu.top3_pts_avg_5           AS home_top3_pts_avg_5,
    hu.top3_pts_avg_10          AS home_top3_pts_avg_10,
    au.top8_min_avg_5           AS away_top8_min_avg_5,
    au.top8_min_avg_10          AS away_top8_min_avg_10,
    au.rotation_depth_5         AS away_rotation_depth_5,
    au.rotation_depth_10        AS away_rotation_depth_10,
    au.top3_pts_avg_5           AS away_top3_pts_avg_5,
    au.top3_pts_avg_10          AS away_top3_pts_avg_10,

    -- ===== RESIDUAL TARGETS =====
    CASE WHEN md.market_spread_home IS NOT NULL
         THEN b.margin::numeric - md.market_spread_home
         ELSE NULL
    END AS spread_residual,
    CASE WHEN md.market_total IS NOT NULL
         THEN b.total_points::numeric - md.market_total
         ELSE NULL
    END AS total_residual

FROM base b

-- Existing joins
LEFT JOIN features.team_rest_features hr
  ON hr.season = b.season AND hr.game_slug = b.game_slug AND hr.team_abbr = b.home_team_abbr
LEFT JOIN features.team_rest_features ar
  ON ar.season = b.season AND ar.game_slug = b.game_slug AND ar.team_abbr = b.away_team_abbr
LEFT JOIN features.team_form_features hf
  ON hf.season = b.season AND hf.game_slug = b.game_slug AND hf.team_abbr = b.home_team_abbr
LEFT JOIN features.team_form_features af
  ON af.season = b.season AND af.game_slug = b.game_slug AND af.team_abbr = b.away_team_abbr
LEFT JOIN features.team_pregame_rolling_boxscore hroll
  ON hroll.season = b.season AND hroll.game_slug = b.game_slug AND hroll.team_abbr = b.home_team_abbr
LEFT JOIN features.team_pregame_rolling_boxscore aroll
  ON aroll.season = b.season AND aroll.game_slug = b.game_slug AND aroll.team_abbr = b.away_team_abbr
LEFT JOIN features.team_player_usage_roll hu
  ON hu.season = b.season AND hu.team_abbr = b.home_team_abbr AND hu.game_date_et = b.game_date_et
LEFT JOIN features.team_player_usage_roll au
  ON au.season = b.season AND au.team_abbr = b.away_team_abbr AND au.game_date_et = b.game_date_et

-- NEW joins
LEFT JOIN features.team_rest_interactions hri
  ON hri.season = b.season AND hri.game_slug = b.game_slug AND hri.team_abbr = b.home_team_abbr
LEFT JOIN features.team_rest_interactions ari
  ON ari.season = b.season AND ari.game_slug = b.game_slug AND ari.team_abbr = b.away_team_abbr
LEFT JOIN features.team_advanced_efficiency hae
  ON hae.season = b.season AND hae.game_slug = b.game_slug AND hae.team_abbr = b.home_team_abbr
LEFT JOIN features.team_advanced_efficiency aae
  ON aae.season = b.season AND aae.game_slug = b.game_slug AND aae.team_abbr = b.away_team_abbr
LEFT JOIN features.team_opp_adjusted_roll hoar
  ON hoar.season = b.season AND hoar.game_slug = b.game_slug AND hoar.team_abbr = b.home_team_abbr
LEFT JOIN features.team_opp_adjusted_roll aoar
  ON aoar.season = b.season AND aoar.game_slug = b.game_slug AND aoar.team_abbr = b.away_team_abbr
LEFT JOIN features.team_standings_features hst
  ON hst.season = b.season AND hst.team_abbr = b.home_team_abbr
LEFT JOIN features.team_standings_features ast_st
  ON ast_st.season = b.season AND ast_st.team_abbr = b.away_team_abbr
LEFT JOIN LATERAL (
    SELECT home_win_pct_10
    FROM features.team_home_away_splits s
    WHERE s.season = b.season AND s.team_abbr = b.home_team_abbr
      AND s.is_home = TRUE AND s.game_date_et < b.game_date_et
    ORDER BY s.game_date_et DESC
    LIMIT 1
) hhas ON TRUE
LEFT JOIN LATERAL (
    SELECT away_win_pct_10
    FROM features.team_home_away_splits s
    WHERE s.season = b.season AND s.team_abbr = b.away_team_abbr
      AND s.is_home = FALSE AND s.game_date_et < b.game_date_et
    ORDER BY s.game_date_et DESC
    LIMIT 1
) aaas ON TRUE
LEFT JOIN features.team_injury_impact hinj
  ON hinj.season = b.season AND hinj.team_abbr = b.home_team_abbr
LEFT JOIN features.team_injury_impact ainj
  ON ainj.season = b.season AND ainj.team_abbr = b.away_team_abbr
LEFT JOIN features.team_clutch_performance hclutch
  ON hclutch.season = b.season AND hclutch.game_slug = b.game_slug AND hclutch.team_abbr = b.home_team_abbr
LEFT JOIN features.team_clutch_performance aclutch
  ON aclutch.season = b.season AND aclutch.game_slug = b.game_slug AND aclutch.team_abbr = b.away_team_abbr
LEFT JOIN features.team_market_derived md
  ON md.as_of_date = b.game_date_et
 AND md.home_team_abbr = b.home_team_abbr
 AND md.away_team_abbr = b.away_team_abbr;
