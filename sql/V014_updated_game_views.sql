-- ============================================================================
-- V014: Updated Game Prediction & Training Feature Views (V2)
-- ============================================================================
-- Replaces V002's game_prediction_features and game_training_features with
-- versions that incorporate ALL new feature views V005-V013:
--   V005  odds_juice_features         — vig/juice signals
--   V006  team_style_profile          — steals, blocks, paint pts, bench pts
--   V007  player_expanded_rolling     — (player-level, used in V015)
--   V008  team_lineup_features        — starter continuity, bench depth
--   V009  team_standings_detail       — point-in-time streak, home/away record
--   V010  game_travel_features        — travel distance, altitude
--   V011  team_pbp_profile            — 3PT rate, foul rate from PBP
--         team_quarter_scoring        — quarter scoring tendencies
--   V012  game_context_features       — attendance, overtime, game duration
--   V013  game_referee_features       — crew foul tendency
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

    -- ===== REST INTERACTIONS =====
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

    -- ===== ADVANCED EFFICIENCY =====
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

    -- ===== OPPONENT-ADJUSTED RATINGS =====
    hoar.off_rtg_avg_10         AS home_off_rtg_avg_10,
    hoar.def_rtg_avg_10         AS home_def_rtg_avg_10,
    hoar.net_rtg_avg_10         AS home_net_rtg_avg_10,
    aoar.off_rtg_avg_10         AS away_off_rtg_avg_10,
    aoar.def_rtg_avg_10         AS away_def_rtg_avg_10,
    aoar.net_rtg_avg_10         AS away_net_rtg_avg_10,
    hoar.net_rtg_avg_10 - aoar.net_rtg_avg_10 AS net_rtg_diff_10,
    hoar.opp_def_rtg_faced_avg_10 AS home_sos_def_10,
    aoar.opp_def_rtg_faced_avg_10 AS away_sos_def_10,

    -- ===== STANDINGS / SEASON CONTEXT =====
    hst.win_pct                 AS home_win_pct,
    ast_st.win_pct              AS away_win_pct,
    hst.conference_rank         AS home_conf_rank,
    ast_st.conference_rank      AS away_conf_rank,
    COALESCE(hst.win_pct, 0.5) - COALESCE(ast_st.win_pct, 0.5) AS win_pct_diff,

    -- ===== HOME/AWAY SPLITS =====
    hhas.home_win_pct_10        AS home_home_win_pct_10,
    aaas.away_win_pct_10        AS away_away_win_pct_10,

    -- ===== INJURY IMPACT =====
    COALESCE(hinj.injured_pts_lost, 0)    AS home_injured_pts_lost,
    COALESCE(ainj.injured_pts_lost, 0)    AS away_injured_pts_lost,
    COALESCE(hinj.injured_min_lost, 0)    AS home_injured_min_lost,
    COALESCE(ainj.injured_min_lost, 0)    AS away_injured_min_lost,
    COALESCE(hinj.injured_out_count, 0)   AS home_injured_out_count,
    COALESCE(ainj.injured_out_count, 0)   AS away_injured_out_count,
    COALESCE(ainj.injured_pts_lost, 0) - COALESCE(hinj.injured_pts_lost, 0) AS injury_pts_advantage_home,

    -- ===== CLUTCH PERFORMANCE =====
    hclutch.clutch_net_avg_10   AS home_clutch_net_avg_10,
    aclutch.clutch_net_avg_10   AS away_clutch_net_avg_10,

    -- ===== MARKET DATA =====
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

    -- ======================== NEW V005-V013 FEATURES ========================

    -- ===== V005: ODDS JUICE / VIG SIGNALS =====
    oj.dk_spread_home_juice,
    oj.dk_spread_away_juice,
    oj.dk_total_over_juice,
    oj.dk_total_under_juice,
    oj.dk_spread_juice_move,
    oj.dk_total_over_juice_move,
    oj.avg_spread_home_juice,
    oj.avg_total_over_juice,
    oj.spread_juice_skew,
    oj.total_juice_skew,
    oj.spread_home_implied_prob,
    oj.total_over_implied_prob,

    -- ===== V006: TEAM STYLE PROFILE =====
    -- Home team style
    hstyle.stl_avg_10           AS home_stl_avg_10,
    hstyle.blk_avg_10           AS home_blk_avg_10,
    hstyle.ast_avg_10           AS home_ast_avg_10,
    hstyle.pts_fast_break_avg_10 AS home_pts_fast_break_avg_10,
    hstyle.pts_paint_avg_10     AS home_pts_paint_avg_10,
    hstyle.pts_bench_avg_10     AS home_pts_bench_avg_10,
    hstyle.fouls_avg_10         AS home_fouls_avg_10,
    hstyle.stl_plus_blk_avg_10  AS home_stocks_avg_10,
    hstyle.ast_to_tov_ratio_avg_10 AS home_ast_tov_ratio_10,
    hstyle.fast_break_pct_avg_10   AS home_fast_break_pct_10,
    hstyle.paint_pct_avg_10     AS home_paint_pct_10,
    hstyle.bench_pct_avg_10     AS home_bench_pct_10,
    -- Away team style
    astyle.stl_avg_10           AS away_stl_avg_10,
    astyle.blk_avg_10           AS away_blk_avg_10,
    astyle.ast_avg_10           AS away_ast_avg_10,
    astyle.pts_fast_break_avg_10 AS away_pts_fast_break_avg_10,
    astyle.pts_paint_avg_10     AS away_pts_paint_avg_10,
    astyle.pts_bench_avg_10     AS away_pts_bench_avg_10,
    astyle.fouls_avg_10         AS away_fouls_avg_10,
    astyle.stl_plus_blk_avg_10  AS away_stocks_avg_10,
    astyle.ast_to_tov_ratio_avg_10 AS away_ast_tov_ratio_10,
    astyle.fast_break_pct_avg_10   AS away_fast_break_pct_10,
    astyle.paint_pct_avg_10     AS away_paint_pct_10,
    astyle.bench_pct_avg_10     AS away_bench_pct_10,

    -- ===== V008: LINEUP STABILITY =====
    hlu.starter_continuity_pct   AS home_starter_continuity,
    hlu.bench_count              AS home_bench_count,
    hlu.scratch_count            AS home_scratch_count,
    hlu.starter_continuity_avg_10 AS home_starter_continuity_avg_10,
    alu.starter_continuity_pct   AS away_starter_continuity,
    alu.bench_count              AS away_bench_count,
    alu.scratch_count            AS away_scratch_count,
    alu.starter_continuity_avg_10 AS away_starter_continuity_avg_10,

    -- ===== V009: STANDINGS DETAIL (point-in-time) =====
    hsd.home_win_pct             AS home_home_record_pct,
    hsd.away_win_pct             AS home_away_record_pct,
    hsd.home_away_split          AS home_home_away_split,
    hsd.streak_signed            AS home_streak,
    hsd.last10_win_pct           AS home_last10_pct,
    asd.home_win_pct             AS away_home_record_pct,
    asd.away_win_pct             AS away_away_record_pct,
    asd.home_away_split          AS away_home_away_split,
    asd.streak_signed            AS away_streak,
    asd.last10_win_pct           AS away_last10_pct,

    -- ===== V010: TRAVEL DISTANCE =====
    tv.travel_distance_miles,
    tv.is_cross_country::int     AS is_cross_country,
    tv.away_prev_leg_miles,
    tv.away_total_travel_miles_5,
    tv.home_altitude_ft,
    tv.altitude_travel_fatigue::int AS altitude_travel_fatigue,

    -- ===== V011: PBP PROFILE & QUARTER SCORING =====
    hpbp.three_pt_rate_avg_10   AS home_three_pt_rate_avg_10,
    hpbp.foul_rate_avg_10       AS home_foul_rate_avg_10,
    hpbp.turnover_rate_avg_10   AS home_pbp_tov_rate_avg_10,
    apbp.three_pt_rate_avg_10   AS away_three_pt_rate_avg_10,
    apbp.foul_rate_avg_10       AS away_foul_rate_avg_10,
    apbp.turnover_rate_avg_10   AS away_pbp_tov_rate_avg_10,
    hqs.q1_pts_avg_10           AS home_q1_pts_avg_10,
    hqs.q4_pts_avg_10           AS home_q4_pts_avg_10,
    hqs.first_half_pct_avg_10   AS home_first_half_pct_10,
    hqs.strong_closer_pct_10    AS home_strong_closer_pct_10,
    aqs.q1_pts_avg_10           AS away_q1_pts_avg_10,
    aqs.q4_pts_avg_10           AS away_q4_pts_avg_10,
    aqs.first_half_pct_avg_10   AS away_first_half_pct_10,
    aqs.strong_closer_pct_10    AS away_strong_closer_pct_10,

    -- ===== V012: ATTENDANCE & OVERTIME =====
    ctx.attendance,
    ctx.attendance_pct_capacity,
    ctx.attendance_vs_avg,
    ctx.home_ot_tendency_10,
    ctx.away_ot_tendency_10,

    -- ===== V013: REFEREE FEATURES =====
    ref.crew_avg_fouls_per_game,
    ref.crew_away_foul_bias,
    ref.crew_size

FROM base b

-- ===== Existing joins =====
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

-- ===== V001 joins =====
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
    ORDER BY s.game_date_et DESC LIMIT 1
) hhas ON TRUE
LEFT JOIN LATERAL (
    SELECT away_win_pct_10
    FROM features.team_home_away_splits s
    WHERE s.season = b.season AND s.team_abbr = b.away_team_abbr
      AND s.is_home = FALSE AND s.game_date_et < b.game_date_et
    ORDER BY s.game_date_et DESC LIMIT 1
) aaas ON TRUE
LEFT JOIN features.team_injury_impact hinj
  ON hinj.season = b.season AND hinj.team_abbr = b.home_team_abbr AND hinj.game_date_et = b.game_date_et
LEFT JOIN features.team_injury_impact ainj
  ON ainj.season = b.season AND ainj.team_abbr = b.away_team_abbr AND ainj.game_date_et = b.game_date_et
LEFT JOIN features.team_clutch_performance hclutch
  ON hclutch.season = b.season AND hclutch.game_slug = b.game_slug AND hclutch.team_abbr = b.home_team_abbr
LEFT JOIN features.team_clutch_performance aclutch
  ON aclutch.season = b.season AND aclutch.game_slug = b.game_slug AND aclutch.team_abbr = b.away_team_abbr
LEFT JOIN features.team_market_derived md
  ON md.as_of_date = b.game_date_et
 AND md.home_team_abbr = b.home_team_abbr
 AND md.away_team_abbr = b.away_team_abbr

-- ===== V005: Odds juice =====
LEFT JOIN features.odds_juice_features oj
  ON oj.as_of_date = b.game_date_et
 AND oj.home_team_abbr = b.home_team_abbr
 AND oj.away_team_abbr = b.away_team_abbr

-- ===== V006: Team style profile =====
LEFT JOIN features.team_style_profile hstyle
  ON hstyle.season = b.season AND hstyle.game_slug = b.game_slug AND hstyle.team_abbr = b.home_team_abbr
LEFT JOIN features.team_style_profile astyle
  ON astyle.season = b.season AND astyle.game_slug = b.game_slug AND astyle.team_abbr = b.away_team_abbr

-- ===== V008: Lineup stability =====
LEFT JOIN features.team_lineup_features hlu
  ON hlu.season = b.season AND hlu.game_slug = b.game_slug AND hlu.team_abbr = b.home_team_abbr
LEFT JOIN features.team_lineup_features alu
  ON alu.season = b.season AND alu.game_slug = b.game_slug AND alu.team_abbr = b.away_team_abbr

-- ===== V009: Standings detail (point-in-time) =====
LEFT JOIN features.team_standings_detail hsd
  ON hsd.season = b.season AND hsd.game_slug = b.game_slug AND hsd.team_abbr = b.home_team_abbr
LEFT JOIN features.team_standings_detail asd
  ON asd.season = b.season AND asd.game_slug = b.game_slug AND asd.team_abbr = b.away_team_abbr

-- ===== V010: Travel distance =====
LEFT JOIN features.game_travel_features tv
  ON tv.game_slug = b.game_slug AND tv.season = b.season

-- ===== V011: PBP profile & quarter scoring =====
LEFT JOIN features.team_pbp_profile hpbp
  ON hpbp.season = b.season AND hpbp.game_slug = b.game_slug AND hpbp.team_abbr = b.home_team_abbr
LEFT JOIN features.team_pbp_profile apbp
  ON apbp.season = b.season AND apbp.game_slug = b.game_slug AND apbp.team_abbr = b.away_team_abbr
LEFT JOIN features.team_quarter_scoring hqs
  ON hqs.season = b.season AND hqs.game_slug = b.game_slug AND hqs.team_abbr = b.home_team_abbr
LEFT JOIN features.team_quarter_scoring aqs
  ON aqs.season = b.season AND aqs.game_slug = b.game_slug AND aqs.team_abbr = b.away_team_abbr

-- ===== V012: Attendance & game context =====
LEFT JOIN features.game_context_features ctx
  ON ctx.game_slug = b.game_slug AND ctx.season = b.season

-- ===== V013: Referee features =====
LEFT JOIN features.game_referee_features ref
  ON ref.game_slug = b.game_slug AND ref.season = b.season;


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

    -- ===== REST INTERACTIONS =====
    hri.is_b2b_away             AS home_is_b2b_away,
    ari.is_b2b_away             AS away_is_b2b_away,
    hri.is_3_in_4               AS home_is_3_in_4,
    ari.is_3_in_4               AS away_is_3_in_4,
    hri.is_4_in_5               AS home_is_4_in_5,
    ari.is_4_in_5               AS away_is_4_in_5,
    COALESCE(hr.rest_days, 2) - COALESCE(ar.rest_days, 2) AS rest_advantage_home,

    -- ===== TEAM FORM =====
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

    -- ===== PACE & BOXSCORE ROLLING =====
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

    -- ===== CROSS-TEAM INTERACTIONS =====
    hf.pts_for_avg_10 - af.pts_against_avg_10 AS home_off_vs_away_def,
    af.pts_for_avg_10 - hf.pts_against_avg_10 AS away_off_vs_home_def,
    (hroll.pace_avg_5 + aroll.pace_avg_5) / 2.0 AS game_pace_est_5,

    -- ===== ADVANCED EFFICIENCY =====
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

    -- ===== OPPONENT-ADJUSTED RATINGS =====
    hoar.off_rtg_avg_10         AS home_off_rtg_avg_10,
    hoar.def_rtg_avg_10         AS home_def_rtg_avg_10,
    hoar.net_rtg_avg_10         AS home_net_rtg_avg_10,
    aoar.off_rtg_avg_10         AS away_off_rtg_avg_10,
    aoar.def_rtg_avg_10         AS away_def_rtg_avg_10,
    aoar.net_rtg_avg_10         AS away_net_rtg_avg_10,
    hoar.net_rtg_avg_10 - aoar.net_rtg_avg_10 AS net_rtg_diff_10,
    hoar.opp_def_rtg_faced_avg_10 AS home_sos_def_10,
    aoar.opp_def_rtg_faced_avg_10 AS away_sos_def_10,

    -- ===== STANDINGS =====
    hst.win_pct                 AS home_win_pct,
    ast_st.win_pct              AS away_win_pct,
    hst.conference_rank         AS home_conf_rank,
    ast_st.conference_rank      AS away_conf_rank,
    COALESCE(hst.win_pct, 0.5) - COALESCE(ast_st.win_pct, 0.5) AS win_pct_diff,

    -- ===== HOME/AWAY SPLITS =====
    hhas.home_win_pct_10        AS home_home_win_pct_10,
    aaas.away_win_pct_10        AS away_away_win_pct_10,

    -- ===== INJURY IMPACT =====
    COALESCE(hinj.injured_pts_lost, 0)    AS home_injured_pts_lost,
    COALESCE(ainj.injured_pts_lost, 0)    AS away_injured_pts_lost,
    COALESCE(hinj.injured_min_lost, 0)    AS home_injured_min_lost,
    COALESCE(ainj.injured_min_lost, 0)    AS away_injured_min_lost,
    COALESCE(hinj.injured_out_count, 0)   AS home_injured_out_count,
    COALESCE(ainj.injured_out_count, 0)   AS away_injured_out_count,
    COALESCE(ainj.injured_pts_lost, 0) - COALESCE(hinj.injured_pts_lost, 0) AS injury_pts_advantage_home,

    -- ===== CLUTCH =====
    hclutch.clutch_net_avg_10   AS home_clutch_net_avg_10,
    aclutch.clutch_net_avg_10   AS away_clutch_net_avg_10,

    -- ===== MARKET DATA =====
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

    -- ===== PLAYER USAGE =====
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

    -- ======================== NEW V005-V013 FEATURES ========================

    -- ===== V005: ODDS JUICE =====
    oj.dk_spread_home_juice,
    oj.dk_spread_away_juice,
    oj.dk_total_over_juice,
    oj.dk_total_under_juice,
    oj.dk_spread_juice_move,
    oj.dk_total_over_juice_move,
    oj.avg_spread_home_juice,
    oj.avg_total_over_juice,
    oj.spread_juice_skew,
    oj.total_juice_skew,
    oj.spread_home_implied_prob,
    oj.total_over_implied_prob,

    -- ===== V006: TEAM STYLE =====
    hstyle.stl_avg_10           AS home_stl_avg_10,
    hstyle.blk_avg_10           AS home_blk_avg_10,
    hstyle.ast_avg_10           AS home_ast_avg_10,
    hstyle.pts_fast_break_avg_10 AS home_pts_fast_break_avg_10,
    hstyle.pts_paint_avg_10     AS home_pts_paint_avg_10,
    hstyle.pts_bench_avg_10     AS home_pts_bench_avg_10,
    hstyle.fouls_avg_10         AS home_fouls_avg_10,
    hstyle.stl_plus_blk_avg_10  AS home_stocks_avg_10,
    hstyle.ast_to_tov_ratio_avg_10 AS home_ast_tov_ratio_10,
    hstyle.fast_break_pct_avg_10   AS home_fast_break_pct_10,
    hstyle.paint_pct_avg_10     AS home_paint_pct_10,
    hstyle.bench_pct_avg_10     AS home_bench_pct_10,
    astyle.stl_avg_10           AS away_stl_avg_10,
    astyle.blk_avg_10           AS away_blk_avg_10,
    astyle.ast_avg_10           AS away_ast_avg_10,
    astyle.pts_fast_break_avg_10 AS away_pts_fast_break_avg_10,
    astyle.pts_paint_avg_10     AS away_pts_paint_avg_10,
    astyle.pts_bench_avg_10     AS away_pts_bench_avg_10,
    astyle.fouls_avg_10         AS away_fouls_avg_10,
    astyle.stl_plus_blk_avg_10  AS away_stocks_avg_10,
    astyle.ast_to_tov_ratio_avg_10 AS away_ast_tov_ratio_10,
    astyle.fast_break_pct_avg_10   AS away_fast_break_pct_10,
    astyle.paint_pct_avg_10     AS away_paint_pct_10,
    astyle.bench_pct_avg_10     AS away_bench_pct_10,

    -- ===== V008: LINEUP =====
    hlu.starter_continuity_pct   AS home_starter_continuity,
    hlu.bench_count              AS home_bench_count,
    hlu.scratch_count            AS home_scratch_count,
    hlu.starter_continuity_avg_10 AS home_starter_continuity_avg_10,
    alu.starter_continuity_pct   AS away_starter_continuity,
    alu.bench_count              AS away_bench_count,
    alu.scratch_count            AS away_scratch_count,
    alu.starter_continuity_avg_10 AS away_starter_continuity_avg_10,

    -- ===== V009: STANDINGS DETAIL =====
    hsd.home_win_pct             AS home_home_record_pct,
    hsd.away_win_pct             AS home_away_record_pct,
    hsd.home_away_split          AS home_home_away_split,
    hsd.streak_signed            AS home_streak,
    hsd.last10_win_pct           AS home_last10_pct,
    asd.home_win_pct             AS away_home_record_pct,
    asd.away_win_pct             AS away_away_record_pct,
    asd.home_away_split          AS away_home_away_split,
    asd.streak_signed            AS away_streak,
    asd.last10_win_pct           AS away_last10_pct,

    -- ===== V010: TRAVEL =====
    tv.travel_distance_miles,
    tv.is_cross_country::int     AS is_cross_country,
    tv.away_prev_leg_miles,
    tv.away_total_travel_miles_5,
    tv.home_altitude_ft,
    tv.altitude_travel_fatigue::int AS altitude_travel_fatigue,

    -- ===== V011: PBP =====
    hpbp.three_pt_rate_avg_10   AS home_three_pt_rate_avg_10,
    hpbp.foul_rate_avg_10       AS home_foul_rate_avg_10,
    hpbp.turnover_rate_avg_10   AS home_pbp_tov_rate_avg_10,
    apbp.three_pt_rate_avg_10   AS away_three_pt_rate_avg_10,
    apbp.foul_rate_avg_10       AS away_foul_rate_avg_10,
    apbp.turnover_rate_avg_10   AS away_pbp_tov_rate_avg_10,
    hqs.q1_pts_avg_10           AS home_q1_pts_avg_10,
    hqs.q4_pts_avg_10           AS home_q4_pts_avg_10,
    hqs.first_half_pct_avg_10   AS home_first_half_pct_10,
    hqs.strong_closer_pct_10    AS home_strong_closer_pct_10,
    aqs.q1_pts_avg_10           AS away_q1_pts_avg_10,
    aqs.q4_pts_avg_10           AS away_q4_pts_avg_10,
    aqs.first_half_pct_avg_10   AS away_first_half_pct_10,
    aqs.strong_closer_pct_10    AS away_strong_closer_pct_10,

    -- ===== V012: CONTEXT =====
    ctx.attendance,
    ctx.attendance_pct_capacity,
    ctx.attendance_vs_avg,
    ctx.home_ot_tendency_10,
    ctx.away_ot_tendency_10,

    -- ===== V013: REFEREES =====
    ref.crew_avg_fouls_per_game,
    ref.crew_away_foul_bias,
    ref.crew_size,

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

-- ===== Existing joins =====
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
    ORDER BY s.game_date_et DESC LIMIT 1
) hhas ON TRUE
LEFT JOIN LATERAL (
    SELECT away_win_pct_10
    FROM features.team_home_away_splits s
    WHERE s.season = b.season AND s.team_abbr = b.away_team_abbr
      AND s.is_home = FALSE AND s.game_date_et < b.game_date_et
    ORDER BY s.game_date_et DESC LIMIT 1
) aaas ON TRUE
LEFT JOIN features.team_injury_impact hinj
  ON hinj.season = b.season AND hinj.team_abbr = b.home_team_abbr AND hinj.game_date_et = b.game_date_et
LEFT JOIN features.team_injury_impact ainj
  ON ainj.season = b.season AND ainj.team_abbr = b.away_team_abbr AND ainj.game_date_et = b.game_date_et
LEFT JOIN features.team_clutch_performance hclutch
  ON hclutch.season = b.season AND hclutch.game_slug = b.game_slug AND hclutch.team_abbr = b.home_team_abbr
LEFT JOIN features.team_clutch_performance aclutch
  ON aclutch.season = b.season AND aclutch.game_slug = b.game_slug AND aclutch.team_abbr = b.away_team_abbr
LEFT JOIN features.team_market_derived md
  ON md.as_of_date = b.game_date_et
 AND md.home_team_abbr = b.home_team_abbr
 AND md.away_team_abbr = b.away_team_abbr

-- ===== V005: Odds juice =====
LEFT JOIN features.odds_juice_features oj
  ON oj.as_of_date = b.game_date_et
 AND oj.home_team_abbr = b.home_team_abbr
 AND oj.away_team_abbr = b.away_team_abbr

-- ===== V006: Team style =====
LEFT JOIN features.team_style_profile hstyle
  ON hstyle.season = b.season AND hstyle.game_slug = b.game_slug AND hstyle.team_abbr = b.home_team_abbr
LEFT JOIN features.team_style_profile astyle
  ON astyle.season = b.season AND astyle.game_slug = b.game_slug AND astyle.team_abbr = b.away_team_abbr

-- ===== V008: Lineup =====
LEFT JOIN features.team_lineup_features hlu
  ON hlu.season = b.season AND hlu.game_slug = b.game_slug AND hlu.team_abbr = b.home_team_abbr
LEFT JOIN features.team_lineup_features alu
  ON alu.season = b.season AND alu.game_slug = b.game_slug AND alu.team_abbr = b.away_team_abbr

-- ===== V009: Standings detail =====
LEFT JOIN features.team_standings_detail hsd
  ON hsd.season = b.season AND hsd.game_slug = b.game_slug AND hsd.team_abbr = b.home_team_abbr
LEFT JOIN features.team_standings_detail asd
  ON asd.season = b.season AND asd.game_slug = b.game_slug AND asd.team_abbr = b.away_team_abbr

-- ===== V010: Travel =====
LEFT JOIN features.game_travel_features tv
  ON tv.game_slug = b.game_slug AND tv.season = b.season

-- ===== V011: PBP =====
LEFT JOIN features.team_pbp_profile hpbp
  ON hpbp.season = b.season AND hpbp.game_slug = b.game_slug AND hpbp.team_abbr = b.home_team_abbr
LEFT JOIN features.team_pbp_profile apbp
  ON apbp.season = b.season AND apbp.game_slug = b.game_slug AND apbp.team_abbr = b.away_team_abbr
LEFT JOIN features.team_quarter_scoring hqs
  ON hqs.season = b.season AND hqs.game_slug = b.game_slug AND hqs.team_abbr = b.home_team_abbr
LEFT JOIN features.team_quarter_scoring aqs
  ON aqs.season = b.season AND aqs.game_slug = b.game_slug AND aqs.team_abbr = b.away_team_abbr

-- ===== V012: Context =====
LEFT JOIN features.game_context_features ctx
  ON ctx.game_slug = b.game_slug AND ctx.season = b.season

-- ===== V013: Referees =====
LEFT JOIN features.game_referee_features ref
  ON ref.game_slug = b.game_slug AND ref.season = b.season;
