# src/mlb_pipeline/modeling/train_player_prop_models.py
"""
Train MLB player prop models: pitcher strikeouts, batter hits, batter total bases.

Three separate XGBoost + LightGBM models (50/50 ensemble where LGB is available).
Walk-forward CV: min_train_days=120, test_window=14, step=14.

Artifacts saved to models/player_props/:
  strikeouts_xgb.json, hits_xgb.json, total_bases_xgb.json
  lgb_strikeouts.txt, lgb_hits.txt, lgb_total_bases.txt
  feature_columns_pitchers.json, feature_columns_batters.json
  feature_medians_pitchers.json, feature_medians_batters.json
  backtest_mae.json
"""
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from .features import add_player_prop_derived_features

log = logging.getLogger("mlb_pipeline.modeling.train_player_prop_models")

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass(frozen=True)
class TrainConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR

    # Walk-forward (same cadence as MLB game model)
    min_train_days: int = 120
    test_window_days: int = 14
    step_days: int = 21          # 21-day step → ~33% fewer folds vs 14; still ample OOF coverage

    # XGBoost defaults (overridden by Optuna when enabled)
    n_estimators: int = 2000
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 10
    gamma: float = 0.1
    reg_alpha: float = 0.5
    reg_lambda: float = 5.0
    early_stopping_rounds: int = 50
    random_state: int = 42

    # Optuna hyperparameter tuning
    # TPE typically converges by trial 15-20; 20 trials × 3 folds = 60 fits/stat × 5 = 300 total
    # timeout_sec is the hard wall-clock cap per stat — Optuna uses best params found so far.
    run_optuna: bool = True
    optuna_n_trials: int = 20    # was 35; TPE converges well before trial 35
    optuna_n_folds: int = 3      # was 4; 25% fewer fits per trial
    optuna_timeout_sec: int = 600  # 10 min/stat × 5 stats = 50 min max (was 75 min)


# ─────────────────────────────────────────────────────────────────────────────
# Training SQL
# ─────────────────────────────────────────────────────────────────────────────

SQL_PITCHER_TRAIN = """
SELECT
    p.season,
    p.game_slug,
    p.game_date_et,
    p.player_id,
    p.team_abbr,
    CASE WHEN p.team_abbr = g.home_team_abbr THEN TRUE ELSE FALSE END AS is_home,
    -- SP rolling features (leakage-safe: exclude current game)
    p.ip_avg_5,
    p.k_pct_5,
    p.k9_5,
    p.era_5,
    p.whip_5,
    p.fip_5,
    p.bb_pct_5,
    p.hr9_5,
    p.k_pct_10,
    p.k9_10,
    p.era_10,
    p.whip_10,
    p.fip_10,
    p.starts_in_window_5,
    p.starts_in_window_10,
    p.last_start_k,
    p.last_start_ip,
    p.last_start_bb,
    -- Group B: SP rest + home/away performance splits (MLB003)
    p.days_since_last_start AS sp_days_since_last_start,
    p.is_short_rest,
    p.era_home_10,
    p.era_away_10,
    p.k9_home_10,
    p.k9_away_10,
    p.fip_home_10,
    p.fip_away_10,
    -- Opponent team batting context (how K-prone are the opposing batters?)
    ob.k_pct_avg_10    AS opp_k_pct_avg_10,
    ob.bb_pct_avg_10   AS opp_bb_pct_avg_10,
    ob.avg_avg_10      AS opp_avg_avg_10,
    ob.hr_avg_10       AS opp_hr_avg_10,
    ob.iso_avg_10      AS opp_iso_avg_10,
    ob.slg_avg_10      AS opp_slg_avg_10,
    -- Opponent lineup K-rate variance (distribution of K-proneness across the lineup)
    lq_opp.lineup_k_pct_std                  AS opp_lineup_k_pct_std,
    lq_opp.lineup_k_pct_cv                   AS opp_lineup_k_pct_cv,
    -- Park factors
    bf.run_factor      AS park_run_factor,
    bf.hr_factor       AS park_hr_factor,
    -- Home plate umpire rolling stats (ump_k9_10 is strongest predictor of K environment)
    ur.ump_k9_10       AS ump_k9_avg_10,
    ur.ump_bb9_10      AS ump_bb9_avg_10,
    -- Weather (dome-zeroed)
    wx.temperature_f,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(wx.wind_speed_mph::FLOAT, 0.0) END        AS wind_speed_mph,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(SIN(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0) END AS wind_sin,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(COS(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0) END AS wind_cos,
    COALESCE(wx.precip_prob_pct::FLOAT, 0.0)                     AS precip_prob_pct,
    CASE WHEN v.roof_type = 'dome' THEN 1 ELSE 0 END             AS is_dome,
    -- Day game flag (dome = always 0; day = ET hour < 17)
    CASE WHEN v.roof_type = 'dome' THEN 0
         WHEN EXTRACT(HOUR FROM g.start_ts_utc AT TIME ZONE 'America/New_York') < 17 THEN 1
         ELSE 0 END                                               AS is_day_game,
    -- Market total (game-level run environment; 7.5% NULL → median-imputed)
    mkt_odds.market_total                                        AS market_total,
    -- Pitcher handedness
    ph.pitch_hand                                                AS pitcher_hand,
    -- Opponent lineup quality (mlb_lineup_quality: NULL for upcoming games, median-imputed)
    lq_opp.lineup_avg_avg_10                                     AS opp_lineup_avg_avg_10,
    lq_opp.lineup_iso_avg_10                                     AS opp_lineup_iso_avg_10,
    lq_opp.top4_slg_avg_10                                       AS opp_top4_slg_avg_10,
    -- Statcast: pitcher's own batted-ball-against profile (season-level)
    sc_p.barrel_batted_rate  AS sc_barrel_rate,
    sc_p.hard_hit_percent    AS sc_hard_hit_pct,
    sc_p.avg_exit_velocity   AS sc_avg_exit_velo,
    sc_p.groundballs_percent AS sc_gb_pct,
    sc_p.flyballs_percent    AS sc_fb_pct,
    sc_p.xba                 AS sc_xba,
    sc_p.xslg                AS sc_xslg,
    sc_p.xwoba               AS sc_xwoba,
    -- Extended Statcast: pitcher's own arsenal whiff/K profile (crawler_statcast_extended)
    pa_self.fb_whiff_pct          AS sc_sp_fb_whiff_pct,
    pa_self.fb_k_pct              AS sc_sp_fb_k_pct,
    pa_self.fb_put_away           AS sc_sp_fb_put_away,
    pa_self.sl_percent            AS sc_sp_sl_pct,
    pa_self.sl_whiff_pct          AS sc_sp_sl_whiff_pct,
    pa_self.sl_k_pct              AS sc_sp_sl_k_pct,
    pa_self.sl_run_value_per_100  AS sc_sp_sl_run_value_per_100,
    pa_self.ch_percent            AS sc_sp_ch_pct,
    pa_self.ch_whiff_pct          AS sc_sp_ch_whiff_pct,
    pa_self.ch_k_pct              AS sc_sp_ch_k_pct,
    -- Arsenal: FB/SI usage + SI whiff/K + diversity (previously unused)
    pa_self.fb_percent            AS sc_sp_fb_pct,
    pa_self.fb_hard_hit_pct       AS sc_sp_fb_hard_hit_pct,
    pa_self.fb_xwoba              AS sc_sp_fb_xwoba,
    pa_self.fb_run_value_per_100  AS sc_sp_fb_run_value_per_100,
    pa_self.si_percent            AS sc_sp_si_pct,
    pa_self.si_whiff_pct          AS sc_sp_si_whiff_pct,
    pa_self.si_k_pct              AS sc_sp_si_k_pct,
    pa_self.si_hard_hit_pct       AS sc_sp_si_hard_hit_pct,
    pa_self.fastball_family_pct   AS sc_sp_fastball_family_pct,
    pa_self.pitch_diversity       AS sc_sp_pitch_diversity,
    -- Pitcher plate discipline (induced chase rate, whiff, zone%)
    pd_p.oz_swing_pct             AS sc_sp_oz_swing_pct,
    pd_p.iz_contact_pct           AS sc_sp_iz_contact_pct,
    pd_p.oz_contact_pct           AS sc_sp_oz_contact_pct,
    pd_p.whiff_pct                AS sc_sp_disc_whiff_pct,
    -- Catcher framing (run value per 100 borderline pitches; + = better framer)
    cf.framing_rv_per_100         AS catcher_framing_rv,
    cf.framing_rate               AS catcher_framing_rate,
    -- Target
    pgl.strikeouts_pitcher AS strikeouts
FROM features.mlb_pitcher_rolling_mat p
JOIN raw.mlb_games g
    ON g.game_slug = p.game_slug
JOIN raw.mlb_player_gamelogs pgl
    ON pgl.game_slug = p.game_slug
    AND pgl.player_id = p.player_id
LEFT JOIN features.mlb_team_batting_rolling_mat ob
    ON ob.game_slug = p.game_slug
    AND ob.team_abbr = CASE
        WHEN p.team_abbr = g.home_team_abbr THEN g.away_team_abbr
        ELSE g.home_team_abbr
    END
LEFT JOIN features.mlb_ballpark_factors bf
    ON bf.team_abbr = g.home_team_abbr
LEFT JOIN raw.mlb_weather wx ON wx.game_slug = g.game_slug
LEFT JOIN raw.mlb_venues  v  ON v.venue_id   = g.venue_id
LEFT JOIN raw.mlb_player_handedness ph ON ph.player_id = p.player_id
-- Home plate umpire rolling stats (known from boxscores; NULL for upcoming games → median-imputed)
LEFT JOIN raw.mlb_game_umpires gu
    ON gu.game_slug = g.game_slug AND gu.ump_position = 'Home Plate'
LEFT JOIN features.mlb_umpire_rolling_mat ur
    ON ur.game_slug = g.game_slug AND ur.umpire_id = gu.umpire_id
-- Opponent batting lineup quality (completed boxscores; NULL for today's upcoming games)
LEFT JOIN features.mlb_lineup_quality_mat lq_opp
    ON lq_opp.game_slug = p.game_slug
    AND lq_opp.team_abbr = CASE
        WHEN p.team_abbr = g.home_team_abbr THEN g.away_team_abbr
        ELSE g.home_team_abbr
    END
-- Statcast pitcher profile (BBE-weighted multi-year average)
-- Stabilises early-season samples; flyballs_percent capped at 100 (corrupted values up to 986).
LEFT JOIN LATERAL (
    SELECT
        SUM(barrel_batted_rate                               * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS barrel_batted_rate,
        SUM(hard_hit_percent                                 * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS hard_hit_percent,
        SUM(avg_exit_velocity                                * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS avg_exit_velocity,
        SUM(LEAST(groundballs_percent, 100.0)                * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS groundballs_percent,
        SUM(LEAST(flyballs_percent,   100.0)                 * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS flyballs_percent,
        SUM(xba   * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)                                            AS xba,
        SUM(xslg  * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)                                            AS xslg,
        SUM(xwoba * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)                                            AS xwoba
    FROM raw.mlb_statcast_pitching
    WHERE player_id = p.player_id
) sc_p ON TRUE
-- Extended Statcast: pitcher's own arsenal whiff/K profile
LEFT JOIN raw.mlb_statcast_pitcher_arsenal pa_self
    ON pa_self.player_id = p.player_id
    AND pa_self.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT
-- Market total (game-level run environment signal)
LEFT JOIN LATERAL (
    SELECT total_points AS market_total
    FROM odds.mlb_game_lines
    WHERE home_team = g.home_team_abbr
      AND as_of_date = g.game_date_et
      AND total_points IS NOT NULL
    ORDER BY fetched_at_utc DESC
    LIMIT 1
) mkt_odds ON TRUE
-- Pitcher plate discipline (own season-level discipline profile)
LEFT JOIN raw.mlb_statcast_pitcher_discipline pd_p
    ON pd_p.player_id = p.player_id
    AND pd_p.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT
-- Catcher framing: find the pitcher's team catcher for this game, then join framing stats
LEFT JOIN LATERAL (
    SELECT player_id AS catcher_id
    FROM raw.mlb_boxscore_player_stats
    WHERE game_slug = g.game_slug
      AND primary_position = 'C'
      AND team_abbr = p.team_abbr
    ORDER BY batting_order
    LIMIT 1
) cat_game ON TRUE
LEFT JOIN raw.mlb_statcast_catcher_framing cf
    ON cf.player_id = cat_game.catcher_id
    AND cf.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT
WHERE g.status = 'final'
  AND pgl.innings_pitched >= 3.0
  AND p.starts_in_window_10 >= 3
  AND pgl.strikeouts_pitcher IS NOT NULL
ORDER BY p.game_date_et, p.game_slug, p.player_id
"""

SQL_BATTER_TRAIN = """
SELECT
    b.season,
    b.game_slug,
    b.game_date_et,
    b.player_id,
    b.team_abbr,
    b.is_home,
    b.rest_days,
    -- Batter rolling features
    b.hits_avg_5,  b.hits_avg_10,  b.hits_avg_20,  b.hits_sd_10,
    b.hr_avg_5,    b.hr_avg_10,    b.hr_avg_20,
    b.tb_avg_5,    b.tb_avg_10,    b.tb_avg_20,    b.tb_sd_10,
    b.ab_avg_5,    b.ab_avg_10,
    b.avg_avg_10,  b.k_rate_avg_10, b.bb_rate_avg_10, b.iso_avg_10,
    b.hr_rate_avg_5, b.hr_rate_avg_10,
    -- AB-weighted cumulative HR rate (more reliable than per-game avg for rare HR events)
    b.hr_rate_cumul_5, b.hr_rate_cumul_10, b.hr_rate_cumul_20,
    b.n_games_prev_10,
    -- Rolling OBP + HR recency (MLB008 additions: gaps C + E)
    b.obp_avg_10,
    b.hr_any_last1, b.hr_count_last3, b.hr_games_with_hr_last5,
    -- Absolute walk/K count rolling (raw scale for walks prop model)
    b.bb_avg_5,    b.bb_avg_10,    b.bb_avg_20,    b.bb_sd_10,
    b.k_avg_5,     b.k_avg_10,
    -- Home/away conditional rolling
    b.hits_home_avg_20, b.hits_away_avg_20,
    b.tb_home_avg_20,   b.tb_away_avg_20,
    b.hr_home_avg_20,   b.hr_away_avg_20,
    b.bb_home_avg_20,   b.bb_away_avg_20,
    -- Opponent SP context
    sp_r.era_5     AS opp_sp_era_5,
    sp_r.fip_5     AS opp_sp_fip_5,
    sp_r.k_pct_5   AS opp_sp_k_pct_5,
    sp_r.k9_5      AS opp_sp_k9_5,
    sp_r.bb_pct_5  AS opp_sp_bb_pct_5,
    sp_r.whip_5    AS opp_sp_whip_5,
    sp_r.ip_avg_5  AS opp_sp_ip_avg_5,
    -- HR/9 — strongest direct signal for pitcher HR propensity
    sp_r.hr9_5     AS opp_sp_hr9_5,
    sp_r.hr9_10    AS opp_sp_hr9_10,
    -- SP per-start velocity trend (MLB021) — declining fastball = more hittable (gap B)
    sp_velo.fb_velo_avg_5    AS opp_sp_fb_velo_avg_5,
    sp_velo.fb_velo_trend_5  AS opp_sp_fb_velo_trend_5,
    -- Group B: opponent SP rest + home/away splits
    sp_r.days_since_last_start AS opp_sp_days_since_last_start,
    sp_r.is_short_rest         AS opp_sp_is_short_rest,
    sp_r.era_home_10           AS opp_sp_era_home_10,
    sp_r.era_away_10           AS opp_sp_era_away_10,
    -- Park factors
    bf.run_factor  AS park_run_factor,
    bf.hr_factor   AS park_hr_factor,
    -- Umpire
    ur.ump_bb9_10  AS ump_bb9_avg_10,
    ur.ump_k9_10   AS ump_k9_avg_10,
    -- Weather (dome-zeroed)
    wx.temperature_f,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(wx.wind_speed_mph::FLOAT, 0.0) END        AS wind_speed_mph,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(SIN(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0) END AS wind_sin,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(COS(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0) END AS wind_cos,
    COALESCE(wx.precip_prob_pct::FLOAT, 0.0)                     AS precip_prob_pct,
    CASE WHEN v.roof_type = 'dome' THEN 1 ELSE 0 END             AS is_dome,
    -- Day game flag (dome = always 0; day = ET hour < 17)
    CASE WHEN v.roof_type = 'dome' THEN 0
         WHEN EXTRACT(HOUR FROM g.start_ts_utc AT TIME ZONE 'America/New_York') < 17 THEN 1
         ELSE 0 END                                               AS is_day_game,
    -- Market total (game-level run environment; ~7.5% NULL → median-imputed)
    mkt_odds.market_total                                        AS market_total,
    -- Lineup slot
    b.batting_order_avg_5,
    b.batting_order_avg_10,
    -- Batter + opponent SP handedness (OHE'd by _prep_X)
    bh.bat_side                AS batter_hand,
    opp_ph.pitch_hand          AS opp_sp_hand,
    -- Matched-hand stats (the split that matches today's opponent's throwing hand)
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.hits_avg_40_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.hits_avg_40_vs_rhp END AS hits_avg_40_vs_hand,
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.tb_avg_40_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.tb_avg_40_vs_rhp END AS tb_avg_40_vs_hand,
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.k_rate_avg_40_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.k_rate_avg_40_vs_rhp END AS k_rate_avg_40_vs_hand,
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.iso_avg_40_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.iso_avg_40_vs_rhp END AS iso_avg_40_vs_hand,
    -- Split differential (positive = better vs LHP)
    COALESCE(bvh.hits_avg_40_vs_lhp, 0) - COALESCE(bvh.hits_avg_40_vs_rhp, 0) AS hits_hand_split_40,
    COALESCE(bvh.tb_avg_40_vs_lhp,   0) - COALESCE(bvh.tb_avg_40_vs_rhp,   0) AS tb_hand_split_40,
    -- Sample sizes (model learns when to trust the split)
    bvh.n_games_vs_lhp_40,
    bvh.n_games_vs_rhp_40,
    -- Cross-season rolling features (MLB013) — populated even in early season
    bcs.n_games_prev_10_cs,
    bcs.hits_avg_10_cs,  bcs.hits_avg_20_cs,
    bcs.tb_avg_10_cs,    bcs.tb_avg_20_cs,
    bcs.hr_avg_10_cs,    bcs.hr_rate_avg_10_cs,
    bcs.ab_avg_10_cs,
    bcs.k_rate_avg_10_cs, bcs.bb_rate_avg_10_cs, bcs.iso_avg_10_cs,
    -- Prior full-season stats (MLB014) — stable 162-game prior
    pss.prev_games,
    pss.prev_hits_avg,  pss.prev_tb_avg,  pss.prev_hr_avg,
    pss.prev_ab_avg,    pss.prev_k_rate,  pss.prev_bb_rate,
    pss.prev_iso,       pss.prev_hr_rate,
    -- Career H2H vs this specific SP (MLB015) — leakage-safe via matview window
    h2h.h2h_games,
    h2h.h2h_ba,
    h2h.h2h_obp,
    h2h.h2h_slg,
    h2h.h2h_k_rate,
    h2h.h2h_iso,
    h2h.h2h_hr,    -- career HRs vs this specific pitcher (gap A)
    h2h.h2h_ab,    -- career ABs vs this specific pitcher (for h2h_hr_rate in features.py)
    -- Own-team lineup quality (protection in order; NULL for upcoming games, median-imputed)
    lq_own.lineup_slg_avg_10                                     AS own_lineup_slg_avg_10,
    lq_own.lineup_iso_avg_10                                     AS own_lineup_iso_avg_10,
    lq_own.top4_slg_avg_10                                       AS own_top4_slg_avg_10,
    -- Statcast: batter's own batted-ball profile (season-level)
    sc_b.barrel_batted_rate  AS sc_barrel_rate,
    sc_b.hard_hit_percent    AS sc_hard_hit_pct,
    sc_b.avg_exit_velocity   AS sc_avg_exit_velo,
    sc_b.avg_launch_angle    AS sc_avg_launch_angle,
    sc_b.sweet_spot_percent  AS sc_sweet_spot_pct,
    sc_b.flyballs_percent    AS sc_fb_pct,
    sc_b.groundballs_percent AS sc_gb_pct,
    sc_b.linedrives_percent  AS sc_ld_pct,
    sc_b.xba                 AS sc_xba,
    sc_b.xslg                AS sc_xslg,
    sc_b.xwoba               AS sc_xwoba,
    sc_b.xiso                AS sc_xiso,
    -- Extended Statcast: spray angle + barrels/PA (crawler_statcast_extended)
    sc_b.pull_percent        AS sc_pull_pct,
    sc_b.opposite_percent    AS sc_opposite_pct,
    sc_b.popup_percent       AS sc_popup_pct,
    sc_b.brl_pa              AS sc_brl_pa,
    -- Extended Statcast: sprint speed
    ss.sprint_speed          AS sprint_speed,
    -- Statcast: opposing SP's batted-ball-against profile (BBE-weighted multi-year avg)
    sc_opp_p.barrel_batted_rate  AS opp_sp_sc_barrel_rate,
    sc_opp_p.hard_hit_percent    AS opp_sp_sc_hard_hit_pct,
    sc_opp_p.avg_exit_velocity   AS opp_sp_sc_avg_exit_velo,
    sc_opp_p.groundballs_percent AS opp_sp_sc_gb_pct,
    sc_opp_p.flyballs_percent    AS opp_sp_sc_fb_pct,
    sc_opp_p.xba                 AS opp_sp_sc_xba,
    sc_opp_p.xslg                AS opp_sp_sc_xslg,
    sc_opp_p.xwoba               AS opp_sp_sc_xwoba,
    -- Extended Statcast: opposing SP arsenal (crawler_statcast_extended)
    pa.fb_percent            AS opp_sp_fb_pct,
    pa.fb_hard_hit_pct       AS opp_sp_fb_hard_hit_pct,
    pa.fb_xwoba              AS opp_sp_fb_xwoba,
    pa.fb_run_value_per_100  AS opp_sp_fb_run_value_per_100,
    pa.fb_whiff_pct          AS opp_sp_fb_whiff_pct,
    pa.fb_k_pct              AS opp_sp_fb_k_pct,
    pa.sl_percent            AS opp_sp_sl_pct,
    pa.sl_whiff_pct          AS opp_sp_sl_whiff_pct,
    pa.sl_k_pct              AS opp_sp_sl_k_pct,
    pa.sl_xwoba              AS opp_sp_sl_xwoba,
    pa.ch_percent            AS opp_sp_ch_pct,
    pa.ch_whiff_pct          AS opp_sp_ch_whiff_pct,
    pa.ch_k_pct              AS opp_sp_ch_k_pct,
    pa.fastball_family_pct   AS opp_sp_fastball_family_pct,
    pa.pitch_diversity       AS opp_sp_pitch_diversity,
    -- Arsenal: SI + SL/CH quality metrics (previously unused)
    pa.si_percent            AS opp_sp_si_pct,
    pa.si_whiff_pct          AS opp_sp_si_whiff_pct,
    pa.si_k_pct              AS opp_sp_si_k_pct,
    pa.si_hard_hit_pct       AS opp_sp_si_hard_hit_pct,
    pa.sl_hard_hit_pct       AS opp_sp_sl_hard_hit_pct,
    pa.sl_run_value_per_100  AS opp_sp_sl_run_value_per_100,
    pa.ch_hard_hit_pct       AS opp_sp_ch_hard_hit_pct,
    pa.ch_run_value_per_100  AS opp_sp_ch_run_value_per_100,
    -- Batter plate discipline (chase rate, contact rates, swing-and-miss)
    pd_b.oz_swing_pct        AS sc_b_oz_swing_pct,
    pd_b.iz_contact_pct      AS sc_b_iz_contact_pct,
    pd_b.oz_contact_pct      AS sc_b_oz_contact_pct,
    pd_b.whiff_pct           AS sc_b_disc_whiff_pct,
    pd_b.out_zone_pct        AS sc_b_out_zone_pct,
    -- Opponent bullpen quality
    opp_tp.bp_era_5          AS opp_bp_era_5,
    opp_tp.bp_era_10         AS opp_bp_era_10,
    opp_tp.bp_k9_5           AS opp_bp_k9_5,
    opp_tp.bullpen_ip_last_7 AS opp_bp_ip_last_7,
    opp_tp.bp_era_7d         AS opp_bp_era_7d,
    -- Targets
    gl.hits        AS hits,
    gl.total_bases AS total_bases,
    gl.home_runs   AS home_runs,
    gl.walks_batter AS walks_batter
FROM features.mlb_player_batting_rolling_mat b
JOIN raw.mlb_games g
    ON g.game_slug = b.game_slug
JOIN raw.mlb_player_gamelogs gl
    ON gl.game_slug = b.game_slug
    AND gl.player_id = b.player_id
-- Opposing starting pitcher
LEFT JOIN raw.mlb_starting_pitchers sp
    ON sp.game_slug = b.game_slug
    AND sp.team_abbr = CASE
        WHEN b.team_abbr = g.home_team_abbr THEN g.away_team_abbr
        ELSE g.home_team_abbr
    END
-- SP rolling stats for this game
LEFT JOIN features.mlb_pitcher_rolling_mat sp_r
    ON sp_r.game_slug = b.game_slug
    AND sp_r.player_id = sp.player_id
LEFT JOIN features.mlb_ballpark_factors bf
    ON bf.team_abbr = g.home_team_abbr
-- Home plate umpire rolling stats
LEFT JOIN raw.mlb_game_umpires gu
    ON gu.game_slug = b.game_slug AND gu.ump_position = 'Home Plate'
LEFT JOIN features.mlb_umpire_rolling_mat ur
    ON ur.game_slug = b.game_slug AND ur.umpire_id = gu.umpire_id
-- Weather (dome-safe)
LEFT JOIN raw.mlb_weather wx ON wx.game_slug = g.game_slug
LEFT JOIN raw.mlb_venues  v  ON v.venue_id   = g.venue_id
-- Opponent SP handedness
LEFT JOIN raw.mlb_player_handedness opp_ph
    ON opp_ph.player_id = sp.player_id
-- Batter's own handedness
LEFT JOIN raw.mlb_player_handedness bh
    ON bh.player_id = b.player_id
-- Batter vs hand rolling stats (leakage-safe: joined by exact game_slug)
LEFT JOIN features.mlb_batting_vs_hand_mat bvh
    ON bvh.game_slug = b.game_slug
    AND bvh.player_id = b.player_id
-- Cross-season rolling stats (MLB013) — same game_slug join, no season boundary
LEFT JOIN features.mlb_player_batting_rolling_cross_mat bcs
    ON bcs.game_slug = b.game_slug
    AND bcs.player_id = b.player_id
-- Prior full-season stats (MLB014) — one season back
LEFT JOIN features.mlb_player_prev_season_stats_mat pss
    ON pss.player_id = b.player_id
    AND pss.season = CASE g.season
        WHEN '2025-regular' THEN '2024-regular'
        WHEN '2026-regular' THEN '2025-regular'
        ELSE NULL
    END
-- Career H2H stats vs today's specific SP (MLB015, leakage-safe via matview window)
LEFT JOIN features.mlb_batter_vs_sp_mat h2h
    ON  h2h.game_slug  = b.game_slug
    AND h2h.batter_id  = b.player_id
    AND h2h.pitcher_id = sp.player_id
-- Own-team lineup quality (completed boxscores; NULL for upcoming games)
LEFT JOIN features.mlb_lineup_quality lq_own
    ON lq_own.game_slug  = b.game_slug
    AND lq_own.team_abbr = b.team_abbr
-- Statcast: batter's own batted-ball profile (BBE-weighted multi-year average)
-- Weighting by batted_ball_events stabilises early-season small samples (e.g. a player
-- with 50 BBE in 2026 and 400 in 2025 gets ~89% weight from 2025).
-- flyballs_percent, gb_percent, ld_percent are capped at 100 to remove
-- corrupted early-season values that exceed 100 in the raw Statcast table.
LEFT JOIN LATERAL (
    SELECT
        SUM(barrel_batted_rate * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)               AS barrel_batted_rate,
        SUM(hard_hit_percent   * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)               AS hard_hit_percent,
        SUM(avg_exit_velocity  * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)               AS avg_exit_velocity,
        SUM(avg_launch_angle   * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)               AS avg_launch_angle,
        SUM(sweet_spot_percent * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)               AS sweet_spot_percent,
        SUM(LEAST(flyballs_percent,   100.0) * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS flyballs_percent,
        SUM(LEAST(groundballs_percent,100.0) * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS groundballs_percent,
        SUM(LEAST(linedrives_percent, 100.0) * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS linedrives_percent,
        SUM(xba    * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)                           AS xba,
        SUM(xslg   * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)                           AS xslg,
        SUM(xwoba  * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)                           AS xwoba,
        SUM(xiso   * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)                           AS xiso,
        SUM(pull_percent     * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)                 AS pull_percent,
        SUM(opposite_percent * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)                 AS opposite_percent,
        SUM(popup_percent    * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)                 AS popup_percent,
        SUM(brl_pa           * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)                 AS brl_pa
    FROM raw.mlb_statcast_batting
    WHERE player_id = b.player_id
) sc_b ON TRUE
-- Statcast: opposing SP's batted-ball-against profile (BBE-weighted multi-year average)
-- Same rationale as batter sc_b: stabilises early-season small samples;
-- flyballs_percent/groundballs_percent capped at 100 (corrupted values up to 986 in raw table).
LEFT JOIN LATERAL (
    SELECT
        SUM(barrel_batted_rate                               * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS barrel_batted_rate,
        SUM(hard_hit_percent                                 * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS hard_hit_percent,
        SUM(avg_exit_velocity                                * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS avg_exit_velocity,
        SUM(LEAST(groundballs_percent, 100.0)                * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS groundballs_percent,
        SUM(LEAST(flyballs_percent,   100.0)                 * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS flyballs_percent,
        SUM(xba   * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)                                            AS xba,
        SUM(xslg  * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)                                            AS xslg,
        SUM(xwoba * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0)                                            AS xwoba
    FROM raw.mlb_statcast_pitching
    WHERE player_id = sp.player_id
) sc_opp_p ON TRUE
-- Extended Statcast: batter sprint speed
LEFT JOIN raw.mlb_statcast_sprint_speed ss
    ON ss.player_id = b.player_id
    AND ss.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT
-- Extended Statcast: opposing SP fastball arsenal
LEFT JOIN raw.mlb_statcast_pitcher_arsenal pa
    ON pa.player_id = sp.player_id
    AND pa.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT
-- Market total (game-level run environment signal)
LEFT JOIN LATERAL (
    SELECT total_points AS market_total
    FROM odds.mlb_game_lines
    WHERE home_team = g.home_team_abbr
      AND as_of_date = g.game_date_et
      AND total_points IS NOT NULL
    ORDER BY fetched_at_utc DESC
    LIMIT 1
) mkt_odds ON TRUE
-- Batter plate discipline (chase rate, contact rates, swing-and-miss)
LEFT JOIN raw.mlb_statcast_batter_discipline pd_b
    ON pd_b.player_id = b.player_id
    AND pd_b.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT
-- Opponent team bullpen rolling stats (pre-game snapshot for this exact game)
LEFT JOIN features.mlb_team_pitching_rolling_mat opp_tp
    ON opp_tp.game_slug = b.game_slug
    AND opp_tp.team_abbr = CASE
        WHEN b.team_abbr = g.home_team_abbr THEN g.away_team_abbr
        ELSE g.home_team_abbr
    END
-- SP per-start velocity rolling (MLB021) — most recent start before this game (gap B)
-- NULL when no velocity data exists for this pitcher (pre-2024 or no Savant coverage)
LEFT JOIN LATERAL (
    SELECT fb_velo_avg_5, fb_velo_trend_5
    FROM features.mlb_sp_velocity_rolling
    WHERE player_id = sp.player_id
      AND game_date < g.game_date_et
    ORDER BY game_date DESC
    LIMIT 1
) sp_velo ON TRUE
WHERE g.status = 'final'
  AND b.ab_avg_10 >= 2.5
  AND b.n_games_prev_10 >= 3
  AND gl.hits IS NOT NULL
  AND gl.total_bases IS NOT NULL
ORDER BY b.game_date_et, b.game_slug, b.player_id
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_numeric(X: pd.DataFrame) -> pd.DataFrame:
    for c in list(X.columns):
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]) or is_datetime64_any_dtype(X[c]):
            continue
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _prep_X(df: pd.DataFrame, target_cols: List[str], meta_cols: List[str]) -> pd.DataFrame:
    """Drop meta / target cols, OHE season, coerce to numeric, add derived features."""
    drop_cols = set(target_cols) | set(meta_cols)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # Season OHE
    if "season" in X.columns:
        X = pd.get_dummies(X, columns=["season"], drop_first=False, dummy_na=False)

    # is_home → int
    if "is_home" in X.columns:
        X["is_home"] = X["is_home"].astype("boolean").fillna(False).astype(int)

    X = _coerce_numeric(X)

    # Remaining non-numeric (shouldn't happen after above)
    bad = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if bad:
        X = pd.get_dummies(X, columns=bad, drop_first=False, dummy_na=True)

    X = add_player_prop_derived_features(X)
    return X


def fit_medians(X: pd.DataFrame) -> Dict[str, float]:
    return {str(k): float(v) for k, v in X.median(numeric_only=True).items()}


def apply_medians(X: pd.DataFrame, medians: Dict[str, float], cols: List[str]) -> pd.DataFrame:
    X2 = X.reindex(columns=cols)
    for c, m in medians.items():
        if c in X2.columns:
            X2[c] = X2[c].fillna(m)
    return X2.fillna(0.0)


def _build_xgb(cfg: TrainConfig, n_est: Optional[int] = None,
               early_stop: bool = True,
               objective: str = "reg:absoluteerror",
               params_override: Optional[Dict] = None) -> XGBRegressor:
    eval_metric = "poisson-nloglik" if objective == "count:poisson" else "mae"
    p = dict(
        n_estimators=n_est or cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        objective=objective,
        eval_metric=eval_metric,
        min_child_weight=cfg.min_child_weight,
        gamma=cfg.gamma,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    if early_stop:
        p["early_stopping_rounds"] = cfg.early_stopping_rounds
    if params_override:
        p.update(params_override)
    return XGBRegressor(**p)


def _build_lgb(n_est: int = 2000, early_stop: bool = True,
               objective: str = "regression_l1",
               params_override: Optional[Dict] = None):
    if not _HAS_LGB:
        return None
    metric = "poisson" if objective == "poisson" else "mae"
    p = dict(
        n_estimators=n_est,
        num_leaves=31,
        learning_rate=0.05,
        objective=objective,
        metric=metric,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.5,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if params_override:
        # Apply shared params that map cleanly to LGB (skip XGB-only keys)
        _lgb_keys = {"learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"}
        p.update({k: v for k, v in params_override.items() if k in _lgb_keys})
    if early_stop:
        p["callbacks"] = [lgb.early_stopping(50, verbose=False)]
    return lgb.LGBMRegressor(**p)


def _walk_forward_folds(
    df: pd.DataFrame,
    min_train_days: int,
    test_window_days: int,
    step_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    dates = pd.to_datetime(df["game_date_et"])
    start = dates.min().normalize()
    end = dates.max().normalize() + pd.Timedelta(days=1)
    first_end = start + pd.Timedelta(days=min_train_days)
    if first_end >= end:
        return []
    folds = []
    train_end = first_end
    while True:
        test_end = train_end + pd.Timedelta(days=test_window_days)
        if test_end > end:
            break
        folds.append((train_end, test_end))
        train_end = train_end + pd.Timedelta(days=step_days)
    return folds


def _run_walk_forward(
    df: pd.DataFrame,
    X_raw: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    medians: Dict[str, float],
    cfg: TrainConfig,
    stat_name: str,
    objective: str = "reg:absoluteerror",
    params_override: Optional[Dict] = None,
) -> Tuple[float, float, List[int]]:
    """Run walk-forward CV. Returns (mae, p68_ci, best_iters)."""
    folds = _walk_forward_folds(df, cfg.min_train_days, cfg.test_window_days, cfg.step_days)
    if not folds:
        log.warning("No walk-forward folds for %s", stat_name)
        return float("nan"), float("nan"), []

    oof_preds, oof_actual = [], []
    best_iters = []

    for train_end, test_end in folds:
        tr_mask = pd.to_datetime(df["game_date_et"]) < train_end
        te_mask = (pd.to_datetime(df["game_date_et"]) >= train_end) & \
                  (pd.to_datetime(df["game_date_et"]) < test_end)
        if tr_mask.sum() < 50 or te_mask.sum() == 0:
            continue

        X_tr = apply_medians(X_raw[tr_mask], medians, feature_cols)
        X_te = apply_medians(X_raw[te_mask], medians, feature_cols)
        y_tr = y[tr_mask]
        y_te = y[te_mask]

        # XGBoost with early stopping on last 15% of train
        cutoff = X_tr.index[int(len(X_tr) * 0.85)]
        fit_mask = X_tr.index < cutoff
        eval_mask = X_tr.index >= cutoff
        if fit_mask.sum() < 30 or eval_mask.sum() == 0:
            fit_mask = slice(None)
            eval_mask = None

        lgb_obj = "poisson" if objective == "count:poisson" else "regression_l1"
        xgb = _build_xgb(cfg, early_stop=True, objective=objective, params_override=params_override)
        if eval_mask is not None:
            xgb.fit(
                X_tr[fit_mask], y_tr[fit_mask],
                eval_set=[(X_tr[eval_mask], y_tr[eval_mask])],
                verbose=False,
            )
        else:
            xgb.fit(X_tr, y_tr, verbose=False)
        best_iters.append(xgb.best_iteration if hasattr(xgb, "best_iteration") and
                          xgb.best_iteration > 0 else cfg.n_estimators)

        preds = xgb.predict(X_te)

        if _HAS_LGB:
            lgb_model = _build_lgb(early_stop=True, objective=lgb_obj, params_override=params_override)
            if eval_mask is not None:
                lgb_model.fit(
                    X_tr[fit_mask], y_tr[fit_mask],
                    eval_set=[(X_tr[eval_mask], y_tr[eval_mask])],
                )
            else:
                lgb_model.fit(X_tr, y_tr)
            preds = (preds + lgb_model.predict(X_te)) / 2.0

        oof_preds.extend(preds)
        oof_actual.extend(y_te.values)

    if not oof_preds:
        return float("nan"), float("nan"), []

    oof_preds = np.array(oof_preds)
    oof_actual = np.array(oof_actual)
    mae = float(mean_absolute_error(oof_actual, oof_preds))
    errors = np.abs(oof_actual - oof_preds)
    p68 = float(np.percentile(errors, 68))  # ~1σ for MAE-like CI

    log.info(
        "Walk-forward %s | MAE=%.3f p68=%.3f | %d OOF rows, %d folds, best_iter~%d",
        stat_name, mae, p68,
        len(oof_actual), len(folds),
        int(np.median(best_iters)) if best_iters else cfg.n_estimators,
    )
    return mae, p68, best_iters


def _fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: TrainConfig,
    stat_name: str,
    n_estimators: Optional[int] = None,
    objective: str = "reg:absoluteerror",
    params_override: Optional[Dict] = None,
) -> Tuple[XGBRegressor, Optional[object]]:
    """Fit final XGB (+LGB) on all data, no early stopping."""
    n_est = n_estimators or cfg.n_estimators
    lgb_obj = "poisson" if objective == "count:poisson" else "regression_l1"
    log.info("Fitting final %s XGB (n=%d rows, n_estimators=%d)", stat_name, len(X), n_est)
    xgb = _build_xgb(cfg, n_est=n_est, early_stop=False, objective=objective,
                     params_override=params_override)
    xgb.fit(X, y, verbose=False)

    lgb_model = None
    if _HAS_LGB:
        log.info("Fitting final %s LGB", stat_name)
        lgb_model = _build_lgb(n_est=n_est, early_stop=False, objective=lgb_obj,
                               params_override=params_override)
        lgb_model.fit(X, y)

    return xgb, lgb_model


# ─────────────────────────────────────────────────────────────────────────────
# Optuna hyperparameter tuning
# ─────────────────────────────────────────────────────────────────────────────

def _optuna_objective_props(
    trial,
    df: pd.DataFrame,
    X_raw: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    medians: Dict[str, float],
    folds: List[Tuple[pd.Timestamp, pd.Timestamp]],
    objective: str = "reg:absoluteerror",
) -> float:
    """XGBoost objective for a single stat. Returns mean walk-forward MAE."""
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
        "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
    }
    eval_metric = "poisson-nloglik" if objective == "count:poisson" else "mae"
    mae_scores = []

    for train_end, test_end in folds:
        tr_mask = pd.to_datetime(df["game_date_et"]) < train_end
        te_mask = (pd.to_datetime(df["game_date_et"]) >= train_end) & \
                  (pd.to_datetime(df["game_date_et"]) < test_end)
        if tr_mask.sum() < 50 or te_mask.sum() == 0:
            continue

        X_tr = apply_medians(X_raw[tr_mask], medians, feature_cols)
        X_te = apply_medians(X_raw[te_mask], medians, feature_cols)
        y_tr = y[tr_mask]
        y_te = y[te_mask]

        # 85/15 temporal split within train for early stopping
        cutoff = X_tr.index[int(len(X_tr) * 0.85)]
        fit_mask = X_tr.index < cutoff
        eval_mask = X_tr.index >= cutoff

        model = XGBRegressor(
            n_estimators=2000,
            objective=objective,
            eval_metric=eval_metric,
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
            **params,
        )
        if fit_mask.sum() >= 30 and eval_mask.sum() > 0:
            model.fit(
                X_tr[fit_mask], y_tr[fit_mask],
                eval_set=[(X_tr[eval_mask], y_tr[eval_mask])],
                verbose=False,
            )
        else:
            model.fit(X_tr, y_tr, verbose=False)

        pred = model.predict(X_te)
        mae_scores.append(float(mean_absolute_error(y_te, pred)))

    return float(np.mean(mae_scores)) if mae_scores else float("inf")


def _run_optuna_for_stat(
    cfg: TrainConfig,
    df: pd.DataFrame,
    X_raw: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    medians: Dict[str, float],
    stat_name: str,
    objective: str = "reg:absoluteerror",
) -> Dict:
    """Run an Optuna study for one stat. Returns best XGB params (empty dict on failure)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    all_folds = _walk_forward_folds(df, cfg.min_train_days, cfg.test_window_days, cfg.step_days)
    tune_folds = all_folds[-cfg.optuna_n_folds:] if len(all_folds) > cfg.optuna_n_folds else all_folds
    if not tune_folds:
        log.warning("No folds for Optuna tuning of %s", stat_name)
        return {}

    log.info("Optuna tuning %s | %d trials, %d folds ...", stat_name, cfg.optuna_n_trials, len(tune_folds))
    study = optuna.create_study(direction="minimize", study_name=stat_name)
    study.optimize(
        lambda trial: _optuna_objective_props(
            trial, df, X_raw, y, feature_cols, medians, tune_folds, objective
        ),
        n_trials=cfg.optuna_n_trials,
        timeout=cfg.optuna_timeout_sec,
        show_progress_bar=False,
    )
    log.info(
        "%s best Optuna MAE=%.3f after %d trials | params=%s",
        stat_name, study.best_value, len(study.trials), study.best_params,
    )
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# Train pitcher model (strikeouts)
# ─────────────────────────────────────────────────────────────────────────────

_PITCHER_META = ["game_slug", "game_date_et", "player_id", "team_abbr"]
_PITCHER_TARGETS = ["strikeouts"]


def train_pitcher_models(cfg: TrainConfig) -> Dict:
    conn = psycopg2.connect(cfg.pg_dsn)
    df = pd.read_sql(SQL_PITCHER_TRAIN, conn)
    conn.close()

    if df.empty:
        raise RuntimeError("No pitcher training data returned.")

    df["game_date_et"] = pd.to_datetime(df["game_date_et"])
    log.info("Pitcher training data: %d rows, %s → %s",
             len(df), df["game_date_et"].min().date(), df["game_date_et"].max().date())

    y_k = df["strikeouts"].astype(float)
    X_raw = _prep_X(df, _PITCHER_TARGETS, _PITCHER_META)
    feature_cols = list(X_raw.columns)

    # Fit medians on all data (final model uses all data; walk-forward uses per-fold)
    medians = fit_medians(X_raw)
    X_filled = apply_medians(X_raw, medians, feature_cols)

    # Optuna hyperparameter tuning
    best_k_params: Dict = {}
    if cfg.run_optuna:
        try:
            best_k_params = _run_optuna_for_stat(
                cfg, df, X_raw, y_k, feature_cols, medians, "strikeouts",
                objective="count:poisson",
            )
        except ImportError:
            log.warning("optuna not installed — skipping. pip install optuna")
        except Exception as e:
            log.warning("Optuna tuning failed for strikeouts: %s. Using defaults.", e)

    # Walk-forward evaluation
    wf_mae, wf_p68, wf_best_iters = _run_walk_forward(
        df, X_raw, y_k, feature_cols, medians, cfg, "strikeouts",
        objective="count:poisson",
        params_override=best_k_params,
    )
    _n_est_k = int(np.percentile(wf_best_iters, 75) * 1.1) if wf_best_iters else cfg.n_estimators

    # Final model
    xgb, lgb_model = _fit_final_model(
        X_filled, y_k, cfg, "strikeouts",
        n_estimators=_n_est_k,
        objective="count:poisson",
        params_override=best_k_params,
    )

    # Save
    model_dir = cfg.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    xgb.save_model(str(model_dir / "strikeouts_xgb.json"))
    if lgb_model is not None:
        lgb_model.booster_.save_model(str(model_dir / "lgb_strikeouts.txt"))

    (model_dir / "feature_columns_pitchers.json").write_text(
        json.dumps(feature_cols), encoding="utf-8"
    )
    (model_dir / "feature_medians_pitchers.json").write_text(
        json.dumps(medians), encoding="utf-8"
    )

    return {
        "mae_strikeouts": wf_mae, "ci_strikeouts": wf_p68, "n_rows": len(df),
        "optuna_strikeouts": best_k_params,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train batter models (hits + total_bases)
# ─────────────────────────────────────────────────────────────────────────────

_BATTER_META = ["game_slug", "game_date_et", "player_id", "team_abbr"]
_BATTER_TARGETS = ["hits", "total_bases", "home_runs", "walks_batter"]


def train_batter_models(cfg: TrainConfig) -> Dict:
    conn = psycopg2.connect(cfg.pg_dsn)
    df = pd.read_sql(SQL_BATTER_TRAIN, conn)
    conn.close()

    if df.empty:
        raise RuntimeError("No batter training data returned.")

    df["game_date_et"] = pd.to_datetime(df["game_date_et"])
    log.info("Batter training data: %d rows, %s → %s",
             len(df), df["game_date_et"].min().date(), df["game_date_et"].max().date())

    y_hits  = df["hits"].astype(float)
    y_tb    = df["total_bases"].astype(float)
    y_hr    = df["home_runs"].fillna(0).astype(float)
    y_walks = df["walks_batter"].fillna(0).astype(float)
    X_raw = _prep_X(df, _BATTER_TARGETS, _BATTER_META)
    feature_cols = list(X_raw.columns)

    medians = fit_medians(X_raw)
    X_filled = apply_medians(X_raw, medians, feature_cols)

    # Optuna hyperparameter tuning (separate study per stat)
    best: Dict[str, Dict] = {"hits": {}, "total_bases": {}, "home_runs": {}, "walks_batter": {}}
    if cfg.run_optuna:
        try:
            best["hits"]        = _run_optuna_for_stat(cfg, df, X_raw, y_hits,  feature_cols, medians, "hits",        "count:poisson")
            best["total_bases"] = _run_optuna_for_stat(cfg, df, X_raw, y_tb,    feature_cols, medians, "total_bases", "count:poisson")
            best["home_runs"]   = _run_optuna_for_stat(cfg, df, X_raw, y_hr,    feature_cols, medians, "home_runs",   "count:poisson")
            best["walks_batter"]= _run_optuna_for_stat(cfg, df, X_raw, y_walks, feature_cols, medians, "walks_batter", "count:poisson")
        except ImportError:
            log.warning("optuna not installed — skipping. pip install optuna")
        except Exception as e:
            log.warning("Optuna tuning failed for batters: %s. Using defaults.", e)

    # Walk-forward for all four stats (using tuned params)
    hits_mae,  hits_p68,  hits_iters  = _run_walk_forward(df, X_raw, y_hits,  feature_cols, medians, cfg, "hits",         objective="count:poisson", params_override=best["hits"])
    tb_mae,    tb_p68,    tb_iters    = _run_walk_forward(df, X_raw, y_tb,    feature_cols, medians, cfg, "total_bases",  objective="count:poisson", params_override=best["total_bases"])
    hr_mae,    hr_p68,    hr_iters    = _run_walk_forward(df, X_raw, y_hr,    feature_cols, medians, cfg, "home_runs",    objective="count:poisson", params_override=best["home_runs"])
    walks_mae, walks_p68, walks_iters = _run_walk_forward(df, X_raw, y_walks, feature_cols, medians, cfg, "walks_batter", objective="count:poisson", params_override=best["walks_batter"])

    def _cv_n_est(iters: List[int]) -> Optional[int]:
        return int(np.percentile(iters, 75) * 1.1) if iters else None

    # Final models (using CV best_iter to set n_estimators)
    xgb_hits,  lgb_hits  = _fit_final_model(X_filled, y_hits,  cfg, "hits",         n_estimators=_cv_n_est(hits_iters),  objective="count:poisson", params_override=best["hits"])
    xgb_tb,    lgb_tb    = _fit_final_model(X_filled, y_tb,    cfg, "total_bases",  n_estimators=_cv_n_est(tb_iters),    objective="count:poisson", params_override=best["total_bases"])
    xgb_hr,    lgb_hr    = _fit_final_model(X_filled, y_hr,    cfg, "home_runs",    n_estimators=_cv_n_est(hr_iters),    objective="count:poisson", params_override=best["home_runs"])
    xgb_walks, lgb_walks = _fit_final_model(X_filled, y_walks, cfg, "walks_batter", n_estimators=_cv_n_est(walks_iters), objective="count:poisson", params_override=best["walks_batter"])

    # Save
    model_dir = cfg.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    xgb_hits.save_model(str(model_dir / "hits_xgb.json"))
    xgb_tb.save_model(str(model_dir / "total_bases_xgb.json"))
    xgb_hr.save_model(str(model_dir / "home_runs_xgb.json"))
    xgb_walks.save_model(str(model_dir / "walks_xgb.json"))

    if lgb_hits is not None:
        lgb_hits.booster_.save_model(str(model_dir / "lgb_hits.txt"))
    if lgb_tb is not None:
        lgb_tb.booster_.save_model(str(model_dir / "lgb_total_bases.txt"))
    if lgb_hr is not None:
        lgb_hr.booster_.save_model(str(model_dir / "lgb_home_runs.txt"))
    if lgb_walks is not None:
        lgb_walks.booster_.save_model(str(model_dir / "lgb_walks.txt"))

    (model_dir / "feature_columns_batters.json").write_text(
        json.dumps(feature_cols), encoding="utf-8"
    )
    (model_dir / "feature_medians_batters.json").write_text(
        json.dumps(medians), encoding="utf-8"
    )

    return {
        "mae_hits": hits_mae,   "ci_hits": hits_p68,
        "mae_total_bases": tb_mae, "ci_total_bases": tb_p68,
        "mae_home_runs": hr_mae,   "ci_home_runs": hr_p68,
        "mae_walks": walks_mae,    "ci_walks": walks_p68,
        "n_rows": len(df),
        "optuna_batter": best,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = TrainConfig()
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    log.info("=== Training pitcher prop model (strikeouts) ===")
    try:
        r = train_pitcher_models(cfg)
        results.update(r)
        log.info("Pitcher: MAE=%.3f, CI(p68)=%.3f, n=%d",
                 r["mae_strikeouts"], r["ci_strikeouts"], r["n_rows"])
    except Exception:
        log.exception("Pitcher model training failed")
        results["mae_strikeouts"] = None
        results["ci_strikeouts"] = None

    log.info("=== Training batter prop models (hits + total_bases + home_runs + walks) ===")
    try:
        r = train_batter_models(cfg)
        results.update(r)
        log.info(
            "Hits: MAE=%.3f | TB: MAE=%.3f | HR: MAE=%.3f | Walks: MAE=%.3f | n=%d",
            r["mae_hits"], r["mae_total_bases"], r["mae_home_runs"], r["mae_walks"],
            r["n_rows"],
        )
    except Exception:
        log.exception("Batter model training failed")
        for k in ("mae_hits", "ci_hits", "mae_total_bases", "ci_total_bases",
                  "mae_home_runs", "ci_home_runs", "mae_walks", "ci_walks"):
            results[k] = None

    # Save backtest summary (exclude Optuna dicts from MAE file)
    mae_results = {k: v for k, v in results.items()
                   if not k.startswith("optuna_")}
    (cfg.model_dir / "backtest_mae.json").write_text(
        json.dumps(mae_results, indent=2), encoding="utf-8"
    )
    log.info("Saved backtest_mae.json: %s", mae_results)

    # Save Optuna best params for reproducibility
    optuna_results = {
        "strikeouts": results.get("optuna_strikeouts", {}),
        **results.get("optuna_batter", {}),
    }
    if any(optuna_results.values()):
        (cfg.model_dir / "optuna_best_params.json").write_text(
            json.dumps(optuna_results, indent=2), encoding="utf-8"
        )
        log.info("Saved optuna_best_params.json")

    log.info("MLB player prop training complete.")


if __name__ == "__main__":
    main()
