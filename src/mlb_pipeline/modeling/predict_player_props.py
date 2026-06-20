# src/mlb_pipeline/modeling/predict_player_props.py
"""
MLB player prop predictions for today's slate.

Loads pitcher/batter prop models and generates predictions for:
  - pitcher_strikeouts   (starting pitchers from raw.mlb_starting_pitchers)
  - batter_hits          (batters with ab_avg_10 >= 2.5 playing today)
  - batter_total_bases
  - batter_home_runs

Edge formula:  edge = pred - book_line
Bet signal:    |edge| >= threshold (K: 2.0, H: 0.75, TB: 1.5, HR: 0.05/0.45)

Discord output (DISCORD_FORMAT=1): compact grouped by game.
DB: bets.mlb_prop_predictions — one row per locked side-level prop offer.
"""
from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

# Ensure stdout is UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import poisson as _scipy_poisson
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from xgboost import XGBRegressor

try:
    import joblib
except ImportError:
    joblib = None

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from .bankroll_layers import BankrollAssessment, assess_bankroll_layer, bankroll_tag
from .bankroll_ledger import (
    insert_prop_bankroll_ledger,
    locked_bankroll_state,
    prop_bankroll_pick_key,
    prop_bankroll_risk_slot,
)
from .model_pick_ledger import insert_prop_model_pick_ledger
from .features import add_player_prop_derived_features, build_fd_parlay_url
from .prop_candidate_engine import (
    book_label as _prop_book_label,
    candidate_from_prediction_row,
    pick_score as _prop_pick_score,
)
from .prop_betting_layer import apply_prop_betting_layer, apply_prop_market_side_prior
from .prop_offer_links import (
    build_prop_line_map,
    build_prop_line_map_for_date,
    filter_prop_offers_for_game,
)
from .prop_offer_snapshots import minimum_american_price
from .prop_shadow_selector import (
    SelectorContext as PropSelectorContext,
    ShadowSelectorConfig as PropSelectorConfig,
    score_prediction_row as score_prop_shadow_row,
)
from .prop_real_money_kill_switch import load_prop_kill_switch_state
from .discord_record_summary import format_record_summary
from .side_recalibration import (
    apply_side_calibrator,
    price_bucket as _cal_price_bucket,
    prop_line_bucket as _cal_prop_line_bucket,
    prop_line_surface as _cal_prop_line_surface,
)

log = logging.getLogger("mlb_pipeline.modeling.predict_player_props")
_ET = ZoneInfo("America/New_York")

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_SQL_DIR = Path(__file__).resolve().parents[3] / "sql"
_BREAKEVEN_PROB = 0.524  # P(win) needed to break even at -110 juice


@dataclass
class PredictConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    et_date: date | None = None
    # Minimum samples to include a batter — must match training SQL filters
    # (train_player_prop_models.py: ab_avg_10 >= 2.5, n_games_prev_10 >= 3)
    # Lower values include part-time players the model never saw in training,
    # causing OVER TB predictions at 0.5 lines to win only 46% (below breakeven).
    min_ab_avg_10: float = 2.5
    min_n_games: int = 3
    # Bet thresholds (|edge| >=)
    # Raised from 1.0 → 2.0: scan_prop_thresholds shows edge only above 1.0; optimal 2.0 (ROI +18%)
    threshold_strikeouts: float = 2.0
    threshold_strikeouts_over: float | None = None
    threshold_strikeouts_under: float | None = None
    # Raised from 0.5 → 0.75: optimal threshold per scan (75-25, ROI +43%); UNDER side dominant
    threshold_hits: float = 0.75
    # Raised from 0.6 → 1.5: 2026-04-16 scan shows optimal 1.50 (15-5, ROI +43.2%, n=20);
    # OVER bets lose at all thresholds — UNDER side dominant
    threshold_total_bases: float = 1.5
    threshold_total_bases_over: float | None = None
    threshold_total_bases_under: float | None = None
    # Split OVER/UNDER thresholds for HR:
    # UNDER threshold kept at 0.45 (scan optimal, ROI +78%) but UNDER bets aren't bookable at -500+.
    # OVER threshold set to 0.05 — any positive edge above the 0.5 line qualifies.
    # E[HR] for top hitters peaks around 0.45-0.55 after bias correction, so threshold must be low.
    threshold_home_runs_over: float = 0.05
    threshold_home_runs_under: float = 0.45
    # Binary CLF threshold: P(over) - 0.524 >= this value to flag a bet.
    # 0.03 = P(over) > 55.4% — modest edge above breakeven.
    threshold_clf: float = 0.03
    # Minimum expected value per 1.0 unit stake to consider a bet.
    # Example: 0.02 means +2% EV.
    min_ev: float = 0.02
    # FanDuel does not offer UNDER for these batter props — suppress UNDER bets in output
    fd_over_only: frozenset = frozenset({"batter_home_runs"})
    # Maximum number of prop bets to output (sorted by |edge| desc).
    # Prevents flooding output with marginal picks — only the top N are shown.
    top_n_bets: int = 10
    # "All props" parlays include every model side regardless of edge. Keep this
    # opt-in so default output only contains filtered edge bets.
    include_all_props_parlay: bool = False
    # Optional per-stat threshold overrides loaded from model artifact JSON.
    thresholds_file: str = "prop_thresholds.json"
    # Optional classifier bucket controls file. By default this is advisory only;
    # side recalibration and price-aware EV decide bankroll eligibility.
    clf_controls_file: str = "clf_bucket_controls.json"
    honor_clf_bucket_controls: bool = False
    # Optional side/line/price recalibrators trained from historical model picks.
    side_recalibrators_file: str = "prop_side_recalibrators.json"
    # Optional separate betting-layer model trained from replay/graded market residuals.
    betting_layer_file: str = "prop_betting_layer.json"
    # Optional historical market/line/price prior. Kept off by default until shadow-tested.
    market_side_priors_file: str = "prop_market_side_priors.json"
    apply_market_side_priors: bool = False
    market_side_prior_max_blend: float = 0.35
    walk_forward_policy_file: str = "prop_walk_forward_accuracy_report.json"
    apply_walk_forward_policy: bool = True
    # Bankroll props reopen only after this policy approves the exact market/side/line/price bucket.
    bucket_reopen_policy_file: str = "prop_bucket_reopen_policy.json"
    enforce_prop_bucket_reopen: bool = True
    real_money_kill_switch_file: str = "prop_real_money_kill_switch.json"
    enforce_prop_real_money_kill_switch: bool = True
    real_money_kill_switch_max_age_hours: float = 36.0
    # Lottery parlay mode: HR/K/H/TB biased to plus-money lines with binary clf edge.
    lottery_mode: bool = False
    lottery_legs: int = 5
    lottery_min_american: int = 300   # floor: at least +300 per leg
    lottery_max_american: int = 900   # ceiling: cap at +900 to avoid true longshots
    lottery_max_per_game: int = 2
    # Shadow-bankroll layer: labels real-money readiness without hiding signals.
    bankroll_shadow_mode: bool = True
    bankroll_max_stake_pct: float = 0.005
    bankroll_max_daily_exposure_pct: float = 0.02
    bankroll_max_lay_price: int = -180
    bankroll_reference_usd: float = 1000.0
    bankroll_micro_stake_usd: float = 1.0
    bankroll_starter_stake_pct: float = 0.001
    # Discord research output: keep bankroll links/parlays first, then show
    # linked paper props by stat so research plays do not get mistaken for
    # bankroll bets. Set paper limit to 0 to show every priced row per section.
    discord_show_paper_links: bool = True
    discord_include_all_priced_props: bool = False
    discord_paper_limit: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot SQL
# ─────────────────────────────────────────────────────────────────────────────

SQL_PITCHER_SNAPSHOTS = """
WITH games_today AS (
    SELECT game_slug, home_team_abbr, away_team_abbr, start_ts_utc, game_date_et
    FROM raw.mlb_games
    WHERE game_date_et = %(game_date)s
      AND status = 'scheduled'
      AND (start_ts_utc IS NULL OR start_ts_utc > NOW())
),
today_starters AS (
    SELECT
        sp.game_slug,
        sp.player_id,
        sp.player_name,
        sp.team_abbr,
        CASE WHEN sp.team_abbr = gt.home_team_abbr THEN TRUE ELSE FALSE END AS is_home,
        CASE WHEN sp.team_abbr = gt.home_team_abbr
             THEN gt.away_team_abbr ELSE gt.home_team_abbr END AS opponent_abbr,
        gt.start_ts_utc,
        gt.game_date_et
    FROM raw.mlb_starting_pitchers sp
    JOIN games_today gt ON gt.game_slug = sp.game_slug
    WHERE sp.player_id IS NOT NULL
)
SELECT
    ts.game_slug,
    ts.game_date_et,
    ts.start_ts_utc,
    ts.player_id,
    ts.player_name,
    ts.team_abbr,
    ts.is_home,
    ts.opponent_abbr,
    -- Most recent rolling pitcher stats prior to today
    pr.season,
    pr.ip_avg_5,
    pr.k_pct_5,
    pr.k9_5,
    pr.era_5,
    pr.whip_5,
    pr.fip_5,
    pr.bb_pct_5,
    pr.hr9_5,
    pr.k_pct_10,
    pr.k9_10,
    pr.era_10,
    pr.whip_10,
    pr.fip_10,
    pr.starts_in_window_5,
    pr.starts_in_window_10,
    pr.last_start_k,
    pr.last_start_ip,
    pr.last_start_bb,
    -- Group B: SP rest + home/away splits (MLB003)
    pr.days_since_last_start AS sp_days_since_last_start,
    pr.is_short_rest,
    pr.era_home_10,
    pr.era_away_10,
    pr.k9_home_10,
    pr.k9_away_10,
    pr.fip_home_10,
    pr.fip_away_10,
    -- Opponent batting context
    ob.k_pct_avg_10     AS opp_k_pct_avg_10,
    ob.k_pct_avg_5      AS opp_k_pct_avg_5,
    ob.bb_pct_avg_10    AS opp_bb_pct_avg_10,
    ob.avg_avg_10       AS opp_avg_avg_10,
    ob.hr_avg_10        AS opp_hr_avg_10,
    ob.iso_avg_10       AS opp_iso_avg_10,
    ob.slg_avg_10       AS opp_slg_avg_10,
    -- Park factors
    bf.run_factor       AS park_run_factor,
    bf.hr_factor        AS park_hr_factor,
    -- Home plate umpire rolling stats (NULL for upcoming games → median-imputed)
    ur.ump_k9_10        AS ump_k9_avg_10,
    ur.ump_bb9_10       AS ump_bb9_avg_10,
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
         WHEN EXTRACT(HOUR FROM ts.start_ts_utc AT TIME ZONE 'America/New_York') < 17 THEN 1
         ELSE 0 END                                               AS is_day_game,
    -- Market total + same-day line movement (game-level run environment)
    mkt_odds.market_total                                        AS market_total,
    mkt_odds.line_move_total                                     AS line_move_total,
    -- Pitcher handedness
    ph.pitch_hand                                                AS pitcher_hand,
    -- Opponent lineup quality (NULL for upcoming games, median-imputed by model)
    lq_opp.lineup_avg_avg_10                                     AS opp_lineup_avg_avg_10,
    lq_opp.lineup_iso_avg_10                                     AS opp_lineup_iso_avg_10,
    lq_opp.top4_slg_avg_10                                       AS opp_top4_slg_avg_10,
    lq_opp.lineup_k_pct_std                                      AS opp_lineup_k_pct_std,
    lq_opp.lineup_k_pct_cv                                       AS opp_lineup_k_pct_cv,
    lq_opp.pct_lhb                                               AS opp_lineup_pct_lhb,
    -- Statcast: pitcher's own batted-ball-against profile
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
    -- Pitcher plate discipline (induced chase rate, whiff, zone pct)
    pd_p.oz_swing_pct             AS sc_sp_oz_swing_pct,
    pd_p.iz_contact_pct           AS sc_sp_iz_contact_pct,
    pd_p.oz_contact_pct           AS sc_sp_oz_contact_pct,
    pd_p.whiff_pct                AS sc_sp_disc_whiff_pct,
    -- Catcher framing (run value per 100 borderline pitches; + = better framer)
    cf.framing_rv_per_100         AS catcher_framing_rv,
    cf.framing_rate               AS catcher_framing_rate,
    -- SP per-start velocity rolling (MLB021) — pitcher's own fastball velocity trend
    sp_velo.fb_velo_avg_5    AS sp_fb_velo_avg_5,
    sp_velo.fb_velo_trend_5  AS sp_fb_velo_trend_5,
    -- SP venue stats (MLB017) — career ERA/K9/FIP at today's specific park
    -- NULL on venue debut; reliability-weighted in features.py
    sp_venue.n_starts_at_venue  AS venue_n_starts,
    sp_venue.venue_era          AS venue_era,
    sp_venue.venue_k9           AS venue_k9,
    sp_venue.venue_fip          AS venue_fip,
    -- SP K%% by batter handedness (MLB022)
    sp_hand_k.sp_k_pct_vs_lhb_25,
    sp_hand_k.sp_k_pct_vs_rhb_25,
    sp_hand_k.sp_k_pct_vs_lhb_10,
    sp_hand_k.sp_k_pct_vs_rhb_10,
    -- SP HR rate by batter handedness (MLB033)
    sp_hand_hr.sp_hr_rate_vs_lhb_25,
    sp_hand_hr.sp_hr_rate_vs_rhb_25,
    sp_hand_hr.sp_hr_rate_vs_lhb_10,
    sp_hand_hr.sp_hr_rate_vs_rhb_10,
    -- SP career stats vs today's opponent (MLB024)
    sp_vs_tm.svt_games,
    sp_vs_tm.svt_era,
    sp_vs_tm.svt_k9,
    sp_vs_tm.svt_k_pct,
    sp_vs_tm.svt_era_last3,
    sp_vs_tm.svt_k9_last3,
    -- SP strand rate (MLB027)
    sp_lob.sp_lob_pct_career,
    sp_lob.sp_lob_pct_10,
    -- Park BABIP factor (MLB028)
    pbf_babip.park_babip_avg,
    -- Opposing team offensive momentum (MLB029)
    opp_mom.team_runs_last3  AS opp_runs_last3,
    opp_mom.team_runs_avg3   AS opp_runs_avg3,
    opp_mom.team_runs_last5  AS opp_runs_last5,
    -- SP BABIP-against rolling (MLB031) — luck/regression signal for K predictions
    sp_babip.sp_babip_against_10,
    sp_babip.sp_babip_against_career,
    sp_babip.sp_babip_starts_10,
    -- SP K%% last 2 starts (item 9)
    sp_k_last2.sp_k_pct_last2,
    -- Opposing team DER (MLB032) — defensive quality
    opp_der.team_der_20  AS opp_team_der_20,
    opp_der.team_der_career AS opp_team_der_career,
    -- Market strikeout prop line (FanDuel; today's game)
    mkt_k.market_k_line
FROM today_starters ts
-- Most recent rolling stats
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_pitcher_rolling_mat pr
    WHERE pr.player_id = ts.player_id
      AND pr.game_date_et < %(game_date)s
    ORDER BY pr.game_date_et DESC, pr.game_slug DESC
    LIMIT 1
) pr ON TRUE
-- Opponent team batting (most recent for opponent)
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_team_batting_rolling_mat ob
    WHERE ob.team_abbr = ts.opponent_abbr
      AND ob.game_date_et < %(game_date)s
    ORDER BY ob.game_date_et DESC, ob.game_slug DESC
    LIMIT 1
) ob ON TRUE
LEFT JOIN features.mlb_ballpark_factors bf
    ON bf.team_abbr = (
        SELECT home_team_abbr FROM raw.mlb_games
        WHERE game_slug = ts.game_slug LIMIT 1
    )
LEFT JOIN raw.mlb_weather wx ON wx.game_slug = ts.game_slug
LEFT JOIN raw.mlb_venues  v  ON v.venue_id = (
    SELECT venue_id FROM raw.mlb_games WHERE game_slug = ts.game_slug LIMIT 1
)
LEFT JOIN raw.mlb_player_handedness ph ON ph.player_id = ts.player_id
-- Home plate umpire (NULL for upcoming games → median-imputed)
LEFT JOIN raw.mlb_game_umpires gu
    ON gu.game_slug = ts.game_slug AND gu.ump_position = 'Home Plate'
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_umpire_rolling_mat
    WHERE umpire_id = gu.umpire_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC LIMIT 1
) ur ON TRUE
-- Opponent batting lineup quality (NULL for upcoming games)
LEFT JOIN features.mlb_lineup_quality_mat lq_opp
    ON lq_opp.game_slug  = ts.game_slug
    AND lq_opp.team_abbr = ts.opponent_abbr
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
    WHERE player_id = ts.player_id
) sc_p ON TRUE
-- Extended Statcast: pitcher's own arsenal whiff/K profile
LEFT JOIN raw.mlb_statcast_pitcher_arsenal pa_self
    ON pa_self.player_id = ts.player_id
    AND pa_self.season_year = EXTRACT(YEAR FROM ts.game_date_et)::INT
-- Market total + same-day line movement (Feature 11)
LEFT JOIN LATERAL (
    SELECT
        MAX(o.total_points)                        AS market_total,
        MAX(o.total_points) - MIN(o.total_points)  AS line_move_total
    FROM odds.mlb_game_lines o
    JOIN raw.mlb_games mg ON mg.home_team_abbr = o.home_team
    WHERE mg.game_slug = ts.game_slug
      AND o.as_of_date = ts.game_date_et
      AND o.total_points IS NOT NULL
) mkt_odds ON TRUE
-- Pitcher plate discipline (own season-level discipline profile)
LEFT JOIN raw.mlb_statcast_pitcher_discipline pd_p
    ON pd_p.player_id = ts.player_id
    AND pd_p.season_year = EXTRACT(YEAR FROM ts.game_date_et)::INT
-- Catcher framing: find the most recent catcher from this pitcher's team, then join framing stats
LEFT JOIN LATERAL (
    SELECT bps.player_id AS catcher_id
    FROM raw.mlb_boxscore_player_stats bps
    JOIN raw.mlb_games mg ON mg.game_slug = bps.game_slug
    WHERE bps.primary_position = 'C'
      AND bps.team_abbr = ts.team_abbr
      AND mg.game_date_et < %(game_date)s
    ORDER BY mg.game_date_et DESC
    LIMIT 1
) cat_recent ON TRUE
LEFT JOIN raw.mlb_statcast_catcher_framing cf
    ON cf.player_id = cat_recent.catcher_id
    AND cf.season_year = EXTRACT(YEAR FROM ts.game_date_et)::INT
-- SP per-start velocity rolling (MLB021) — pitcher's own fastball velocity trend
-- NULL when no Savant data exists (pre-2024 or limited coverage)
LEFT JOIN LATERAL (
    SELECT fb_velo_avg_5, fb_velo_trend_5
    FROM features.mlb_sp_velocity_rolling
    WHERE player_id = ts.player_id
      AND game_date < %(game_date)s
    ORDER BY game_date DESC
    LIMIT 1
) sp_velo ON TRUE
-- SP venue stats (MLB017) — career ERA/K9/FIP at today's park (leakage-safe)
LEFT JOIN LATERAL (
    SELECT n_starts_at_venue, venue_era, venue_k9, venue_fip
    FROM features.mlb_sp_venue_stats
    WHERE player_id    = ts.player_id
      AND venue_id     = (SELECT venue_id FROM raw.mlb_games WHERE game_slug = ts.game_slug LIMIT 1)
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) sp_venue ON TRUE
-- SP's own K%% by batter handedness (MLB022)
LEFT JOIN LATERAL (
    SELECT sp_k_pct_vs_lhb_25, sp_k_pct_vs_rhb_25,
           sp_k_pct_vs_lhb_10, sp_k_pct_vs_rhb_10
    FROM features.mlb_sp_hand_k_pct
    WHERE pitcher_id   = ts.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) sp_hand_k ON TRUE
-- SP HR rate by batter handedness (MLB033)
LEFT JOIN LATERAL (
    SELECT sp_hr_rate_vs_lhb_25, sp_hr_rate_vs_rhb_25,
           sp_hr_rate_vs_lhb_10, sp_hr_rate_vs_rhb_10
    FROM features.mlb_sp_hand_hr_rate
    WHERE pitcher_id   = ts.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) sp_hand_hr ON TRUE
-- SP career stats vs today's specific opponent (MLB024)
LEFT JOIN LATERAL (
    SELECT svt_games, svt_era, svt_k9, svt_k_pct, svt_era_last3, svt_k9_last3
    FROM features.mlb_sp_vs_team_mat
    WHERE pitcher_id    = ts.player_id
      AND opp_team_abbr = ts.opponent_abbr
      AND game_date_et  < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) sp_vs_tm ON TRUE
-- SP strand rate (MLB027)
LEFT JOIN LATERAL (
    SELECT sp_lob_pct_career, sp_lob_pct_10
    FROM features.mlb_sp_lob_rate_mat
    WHERE player_id    = ts.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) sp_lob ON TRUE
-- Park BABIP factor (MLB028)
LEFT JOIN features.mlb_park_babip_factor pbf_babip
    ON pbf_babip.venue_id = (SELECT venue_id FROM raw.mlb_games WHERE game_slug = ts.game_slug LIMIT 1)
-- Opposing team offensive momentum (MLB029)
LEFT JOIN LATERAL (
    SELECT team_runs_last3, team_runs_avg3, team_runs_last5
    FROM features.mlb_team_offensive_momentum_mat
    WHERE team_abbr    = ts.opponent_abbr
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) opp_mom ON TRUE
-- Market strikeout prop line (FanDuel; highest available line = market ceiling for this SP)
LEFT JOIN LATERAL (
    SELECT MAX(pl.line) AS market_k_line
    FROM odds.mlb_player_prop_lines pl
    CROSS JOIN LATERAL (
        SELECT LOWER(REGEXP_REPLACE(
            UNACCENT(bx.first_name || ' ' || bx.last_name), '[^a-z ]', '', 'gi'
        )) AS name_norm
        FROM raw.mlb_boxscore_player_stats bx
        WHERE bx.player_id = ts.player_id
        LIMIT 1
    ) pn
    WHERE pl.player_name_norm = pn.name_norm
      AND pl.as_of_date       = %(game_date)s
      AND pl.stat             = 'pitcher_strikeouts'
      AND pl.bookmaker_key    = 'fanduel'
) mkt_k ON TRUE
-- SP BABIP-against rolling (MLB031) — luck/regression for K predictions
LEFT JOIN LATERAL (
    SELECT sp_babip_against_10, sp_babip_against_career, sp_babip_starts_10
    FROM features.mlb_sp_babip_rolling_mat
    WHERE player_id    = ts.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) sp_babip ON TRUE
-- SP K%% last 2 starts (item 9): recent K rate vs 5-start rolling (recency trend signal)
-- BF estimated as ROUND(IP*3) + H + BB
LEFT JOIN LATERAL (
    SELECT
        CASE WHEN SUM(bf_est) > 0
             THEN SUM(COALESCE(g2k.strikeouts_pitcher, 0))::float / SUM(bf_est)
             ELSE NULL END AS sp_k_pct_last2
    FROM (
        SELECT
            pgl2.strikeouts_pitcher,
            GREATEST(ROUND(COALESCE(pgl2.innings_pitched, 0) * 3)
                     + COALESCE(pgl2.hits_allowed, 0)
                     + COALESCE(pgl2.walks_allowed, 0), 1) AS bf_est
        FROM raw.mlb_player_gamelogs pgl2
        JOIN raw.mlb_games g2 ON g2.game_slug = pgl2.game_slug
        WHERE pgl2.player_id      = ts.player_id
          AND pgl2.is_starter     = TRUE
          AND pgl2.innings_pitched >= 1.0
          AND g2.status           = 'final'
          AND g2.game_date_et     < %(game_date)s
        ORDER BY g2.game_date_et DESC, g2.game_slug DESC
        LIMIT 2
    ) g2k
) sp_k_last2 ON TRUE
-- Opposing team DER (MLB032) — defensive quality of opposing lineup's fielders
LEFT JOIN LATERAL (
    SELECT team_der_20, team_der_career
    FROM features.mlb_team_der_rolling_mat
    WHERE team_abbr    = ts.opponent_abbr
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) opp_der ON TRUE
-- Injury exclusion (Feature 12): skip OUT/DOUBTFUL pitchers
-- Join on mlb_player_id (extracted from MSF image URL) which matches gamelog player_id
LEFT JOIN raw.mlb_injuries inj_p ON inj_p.mlb_player_id = ts.player_id
WHERE (inj_p.mlb_player_id IS NULL
       OR inj_p.playing_probability NOT IN ('OUT', 'DOUBTFUL'))
ORDER BY ts.start_ts_utc, ts.game_slug, ts.player_id
"""

SQL_BATTER_SNAPSHOTS = """
WITH games_today AS (
    SELECT game_slug, home_team_abbr, away_team_abbr, start_ts_utc, game_date_et
    FROM raw.mlb_games
    WHERE game_date_et = %(game_date)s
      AND status = 'scheduled'
      AND (start_ts_utc IS NULL OR start_ts_utc > NOW())
),
teams_today AS (
    SELECT home_team_abbr AS team_abbr, away_team_abbr AS opponent_abbr,
           TRUE AS is_home, game_slug, start_ts_utc, game_date_et
    FROM games_today
    UNION ALL
    SELECT away_team_abbr, home_team_abbr, FALSE, game_slug, start_ts_utc, game_date_et
    FROM games_today
),
-- Players who appeared for today's teams recently (last 14 days)
recent_players AS (
    SELECT DISTINCT gl.player_id, gl.team_abbr
    FROM raw.mlb_player_gamelogs gl
    JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
    WHERE gl.team_abbr IN (SELECT team_abbr FROM teams_today)
      AND g.game_date_et >= %(game_date)s::date - INTERVAL '14 days'
      AND gl.at_bats > 0
)
SELECT
    tt.game_slug,
    tt.start_ts_utc,
    tt.game_date_et,
    EXTRACT(MONTH FROM tt.game_date_et)::INT AS game_month,
    rp.player_id,
    tt.team_abbr,
    tt.opponent_abbr,
    tt.is_home,
    -- Latest rolling batter stats
    br.season,
    br.hits_avg_5,  br.hits_avg_10,  br.hits_avg_20,  br.hits_sd_10,
    br.hr_avg_5,    br.hr_avg_10,    br.hr_avg_20,
    br.tb_avg_5,    br.tb_avg_10,    br.tb_avg_20,    br.tb_sd_10,
    br.ab_avg_5,    br.ab_avg_10,
    br.avg_avg_10,  br.k_rate_avg_10, br.bb_rate_avg_10, br.iso_avg_10,
    br.hr_rate_avg_5, br.hr_rate_avg_10,
    -- AB-weighted cumulative HR rate (more reliable than per-game avg for rare HR events)
    br.hr_rate_cumul_5, br.hr_rate_cumul_10, br.hr_rate_cumul_20,
    br.n_games_prev_10,
    br.rest_days,
    -- Rolling OBP + HR recency (MLB008 additions: gaps C + E)
    br.obp_avg_10,
    br.hr_any_last1, br.hr_count_last3, br.hr_games_with_hr_last5,
    -- Absolute walk/K count rolling
    br.bb_avg_5,    br.bb_avg_10,    br.bb_avg_20,    br.bb_sd_10,
    br.k_avg_5,     br.k_avg_10,
    -- Home/away conditional rolling
    br.hits_home_avg_20, br.hits_away_avg_20,
    br.tb_home_avg_20,   br.tb_away_avg_20,
    br.hr_home_avg_20,   br.hr_away_avg_20,
    br.bb_home_avg_20,   br.bb_away_avg_20,
    -- Opponent SP stats
    sp_r.era_5     AS opp_sp_era_5,
    sp_r.fip_5     AS opp_sp_fip_5,
    sp_r.k_pct_5   AS opp_sp_k_pct_5,
    sp_r.k9_5      AS opp_sp_k9_5,
    sp_r.bb_pct_5  AS opp_sp_bb_pct_5,
    sp_r.whip_5    AS opp_sp_whip_5,
    sp_r.ip_avg_5  AS opp_sp_ip_avg_5,
    sp_r.ip_avg_5 * 16.5  AS opp_sp_pitches_est,   -- estimated pitches/start (workload proxy)
    sp_r.last_start_ip    AS opp_sp_last_start_ip, -- actual IP last start (pitch budget signal)
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
         WHEN EXTRACT(HOUR FROM tt.start_ts_utc AT TIME ZONE 'America/New_York') < 17 THEN 1
         ELSE 0 END                                               AS is_day_game,
    -- Market total + same-day line movement (game-level run environment)
    mkt_odds.market_total                                        AS market_total,
    mkt_odds.line_move_total                                     AS line_move_total,
    -- Lineup slot
    br.batting_order_avg_5,
    br.batting_order_avg_10,
    -- Batter + opponent SP handedness (OHE'd by _prep_features)
    bh.bat_side                AS batter_hand,
    CASE WHEN bh.bat_side = 'L' THEN 1 ELSE 0 END AS is_lhb_batter,
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
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.hr_avg_40_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.hr_avg_40_vs_rhp  END AS hr_avg_40_vs_hand,
    -- Split differential (positive = better vs LHP)
    COALESCE(bvh.hits_avg_40_vs_lhp, 0) - COALESCE(bvh.hits_avg_40_vs_rhp, 0) AS hits_hand_split_40,
    COALESCE(bvh.tb_avg_40_vs_lhp,   0) - COALESCE(bvh.tb_avg_40_vs_rhp,   0) AS tb_hand_split_40,
    COALESCE(bvh.hr_avg_40_vs_lhp,   0) - COALESCE(bvh.hr_avg_40_vs_rhp,   0) AS hr_hand_split_40,
    -- Sample sizes
    bvh.n_games_vs_lhp_40,
    bvh.n_games_vs_rhp_40,
    -- Platoon splits 10-game (MLB012 extended) — recent handedness form
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.hits_avg_10_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.hits_avg_10_vs_rhp END AS hits_avg_10_vs_hand,
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.tb_avg_10_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.tb_avg_10_vs_rhp END AS tb_avg_10_vs_hand,
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.hr_avg_10_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.hr_avg_10_vs_rhp END AS hr_avg_10_vs_hand,
    bvh.hits_avg_10_vs_lhp, bvh.hits_avg_10_vs_rhp,
    bvh.n_games_vs_lhp_10,  bvh.n_games_vs_rhp_10,
    -- Cross-season rolling features (MLB013)
    bcs.n_games_prev_10_cs,
    bcs.hits_avg_10_cs,  bcs.hits_avg_20_cs,
    bcs.tb_avg_10_cs,    bcs.tb_avg_20_cs,
    bcs.hr_avg_10_cs,    bcs.hr_rate_avg_10_cs,
    bcs.ab_avg_10_cs,
    bcs.k_rate_avg_10_cs, bcs.bb_rate_avg_10_cs, bcs.iso_avg_10_cs,
    -- Prior full-season stats (MLB014)
    pss.prev_games,
    pss.prev_hits_avg,  pss.prev_tb_avg,  pss.prev_hr_avg,
    pss.prev_ab_avg,    pss.prev_k_rate,  pss.prev_bb_rate,
    pss.prev_iso,       pss.prev_hr_rate,
    -- Career H2H stats vs today's SP (MLB015, most recent entry before game date)
    h2h.h2h_games,
    h2h.h2h_ba,
    h2h.h2h_obp,
    h2h.h2h_slg,
    h2h.h2h_k_rate,
    h2h.h2h_iso,
    h2h.h2h_hr,    -- career HRs vs this specific pitcher (gap A)
    h2h.h2h_ab,    -- career ABs vs this specific pitcher (for h2h_hr_rate in features.py)
    -- H2H last-3 recency (MLB015 extended)
    h2h.h2h_ba_last3,
    h2h.h2h_slg_last3,
    h2h.h2h_hr_rate_last3,
    h2h.h2h_ab_last3,
    -- Own-team lineup quality (NULL for upcoming games, median-imputed by model)
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
    -- Extended Statcast: spray angle + barrels/PA
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
    -- Opponent bullpen quality (team pitching rolling, BP split)
    opp_tp.bp_era_5          AS opp_bp_era_5,
    opp_tp.bp_era_10         AS opp_bp_era_10,
    opp_tp.bp_k9_5           AS opp_bp_k9_5,
    opp_tp.bullpen_ip_last_7 AS opp_bp_ip_last_7,
    opp_tp.bp_era_7d         AS opp_bp_era_7d,
    -- Reliever depth depletion (distinct arms used in past 1–2 days)
    opp_rl.bp_relievers_last_1d  AS opp_bp_relievers_last_1d,
    opp_rl.bp_relievers_last_2d  AS opp_bp_relievers_last_2d,
    -- Batter venue stats (MLB023)
    b_venue.batter_n_games_at_venue,
    b_venue.batter_venue_ba,
    b_venue.batter_venue_hr_rate,
    b_venue.batter_venue_slg,
    -- Opposing SP K%% by batter handedness (MLB022)
    opp_sp_hand_k.sp_k_pct_vs_lhb_25,
    opp_sp_hand_k.sp_k_pct_vs_rhb_25,
    opp_sp_hand_k.sp_k_pct_vs_lhb_10,
    opp_sp_hand_k.sp_k_pct_vs_rhb_10,
    -- Opposing SP HR rate by batter handedness (MLB033)
    opp_sp_hand_hr.sp_hr_rate_vs_lhb_25,
    opp_sp_hand_hr.sp_hr_rate_vs_rhb_25,
    opp_sp_hand_hr.sp_hr_rate_vs_lhb_10,
    opp_sp_hand_hr.sp_hr_rate_vs_rhb_10,
    -- Opposing catcher framing (MLB020) — good framing → more called strikes → fewer hits/TB for batter
    opp_cf.framing_rv_per_100  AS opp_catcher_framing_rv,
    opp_cf.framing_rate        AS opp_catcher_framing_rate,
    -- Confirmed batting order from pre-game lineup (raw.mlb_lineups)
    conf_lu.batting_order  AS confirmed_batting_order,
    conf_lu.lineup_source  AS confirmed_lineup_source,
    team_lu.lineup_slots   AS confirmed_team_lineup_slots,
    -- Batter career stats with today's home plate umpire (MLB025)
    btu.btu_games,
    btu.btu_ba,
    btu.btu_k_rate,
    btu.btu_bb_rate,
    -- Batter stats in bullpen games (MLB026)
    bvr.bvr_games_30,
    bvr.bvr_bp_games_30,
    bvr.bvr_ab_30,
    bvr.bvr_ba_30,
    bvr.bvr_hr_rate_30,
    bvr.bvr_slg_30,
    bvr.bvr_k_rate_30,
    -- Opposing SP strand rate (MLB027)
    opp_sp_lob.sp_lob_pct_career  AS opp_sp_lob_pct_career,
    opp_sp_lob.sp_lob_pct_10      AS opp_sp_lob_pct_10,
    -- Park BABIP factor (MLB028)
    pbf_babip.park_babip_avg,
    -- Own team offensive momentum (MLB029)
    own_mom.team_runs_last3  AS own_runs_last3,
    own_mom.team_runs_avg3   AS own_runs_avg3,
    -- Batter BABIP rolling (MLB030) — luck/regression signal for hits
    batter_babip.batter_babip_10,
    batter_babip.batter_babip_career,
    batter_babip.babip_games_10,
    -- Opposing team DER (MLB032) — defensive quality of opponent's fielders
    opp_def_der.team_der_20  AS opp_team_def_der_20,
    opp_def_der.team_der_career AS opp_team_def_der_career,
    -- Market over_price on canonical batter lines (FanDuel; today's game)
    -- American odds: -200 = 67%% implied prob; -130 = 57%%; -170 = 63%%; etc.
    mkt_props.market_hits_over_price,
    mkt_props.market_tb_over_price,
    mkt_props.market_hr_over_price,
    -- Opposing SP confirmation status (for leaderboard quality gate)
    sp.source                             AS opp_sp_source
FROM teams_today tt
JOIN recent_players rp ON rp.team_abbr = tt.team_abbr
-- Most recent rolling batter stats prior to today
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_player_batting_rolling_mat br
    WHERE br.player_id = rp.player_id
      AND br.game_date_et < %(game_date)s
    ORDER BY br.game_date_et DESC, br.game_slug DESC
    LIMIT 1
) br ON TRUE
-- Opposing starting pitcher
LEFT JOIN raw.mlb_starting_pitchers sp
    ON sp.game_slug = tt.game_slug
    AND sp.team_abbr = tt.opponent_abbr
-- Opposing SP rolling stats
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_pitcher_rolling_mat sp_r
    WHERE sp_r.player_id = sp.player_id
      AND sp_r.game_date_et < %(game_date)s
    ORDER BY sp_r.game_date_et DESC, sp_r.game_slug DESC
    LIMIT 1
) sp_r ON TRUE
-- Opponent SP handedness
LEFT JOIN raw.mlb_player_handedness opp_ph ON opp_ph.player_id = sp.player_id
-- Batter's own handedness
LEFT JOIN raw.mlb_player_handedness bh ON bh.player_id = rp.player_id
-- Most recent batter-vs-hand rolling stats prior to today (LATERAL for latest snapshot)
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_batting_vs_hand_mat
    WHERE player_id = rp.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC, game_slug DESC
    LIMIT 1
) bvh ON TRUE
-- Cross-season rolling stats (no season boundary) — gives prior-year data early in season
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_player_batting_rolling_cross_mat
    WHERE player_id = rp.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC, game_slug DESC
    LIMIT 1
) bcs ON TRUE
-- Prior full-season stats — stable 162-game prior for each player
LEFT JOIN features.mlb_player_prev_season_stats_mat pss
    ON pss.player_id = rp.player_id
    AND pss.season = %(prior_season)s
-- Career H2H stats vs today's SP (most recent entry before game date)
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_batter_vs_sp_mat
    WHERE batter_id    = rp.player_id
      AND pitcher_id   = sp.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) h2h ON TRUE
LEFT JOIN features.mlb_ballpark_factors bf
    ON bf.team_abbr = (
        SELECT home_team_abbr FROM games_today
        WHERE game_slug = tt.game_slug LIMIT 1
    )
-- Home plate umpire rolling stats
LEFT JOIN raw.mlb_game_umpires gu
    ON gu.game_slug = tt.game_slug AND gu.ump_position = 'Home Plate'
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_umpire_rolling_mat
    WHERE umpire_id = gu.umpire_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC LIMIT 1
) ur ON TRUE
-- Batter career stats with today's home plate umpire (MLB025)
LEFT JOIN LATERAL (
    SELECT btu_games, btu_ba, btu_k_rate, btu_bb_rate
    FROM features.mlb_batter_umpire_mat
    WHERE batter_id    = rp.player_id
      AND umpire_id    = gu.umpire_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) btu ON TRUE
-- Batter stats in bullpen games (MLB026)
LEFT JOIN LATERAL (
    SELECT bvr_games_30, bvr_bp_games_30, bvr_ab_30,
           bvr_ba_30, bvr_hr_rate_30, bvr_slg_30, bvr_k_rate_30
    FROM features.mlb_batter_vs_rp_mat
    WHERE batter_id    = rp.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) bvr ON TRUE
-- Opposing SP strand rate (MLB027)
LEFT JOIN LATERAL (
    SELECT sp_lob_pct_career, sp_lob_pct_10
    FROM features.mlb_sp_lob_rate_mat
    WHERE player_id    = sp.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) opp_sp_lob ON TRUE
-- Park BABIP factor (MLB028)
LEFT JOIN features.mlb_park_babip_factor pbf_babip
    ON pbf_babip.venue_id = (SELECT venue_id FROM raw.mlb_games WHERE game_slug = tt.game_slug LIMIT 1)
-- Own team offensive momentum (MLB029)
LEFT JOIN LATERAL (
    SELECT team_runs_last3, team_runs_avg3
    FROM features.mlb_team_offensive_momentum_mat
    WHERE team_abbr    = tt.team_abbr
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) own_mom ON TRUE
-- Weather
LEFT JOIN raw.mlb_weather wx ON wx.game_slug = tt.game_slug
LEFT JOIN raw.mlb_venues  v  ON v.venue_id = (
    SELECT venue_id FROM raw.mlb_games WHERE game_slug = tt.game_slug LIMIT 1
)
-- Own-team lineup quality (NULL for upcoming games)
LEFT JOIN features.mlb_lineup_quality_mat lq_own
    ON lq_own.game_slug  = tt.game_slug
    AND lq_own.team_abbr = tt.team_abbr
-- Statcast: batter's own batted-ball profile (BBE-weighted multi-year average)
-- Matches training SQL: weights all available seasons by batted_ball_events so
-- early-season small samples don't dominate; caps pct fields at 100.
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
    WHERE player_id = rp.player_id
) sc_b ON TRUE
-- Statcast: opposing SP's batted-ball-against profile (BBE-weighted multi-year average)
-- Matches training SQL; flyballs_percent/groundballs_percent capped at 100.
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
    ON ss.player_id = rp.player_id
    AND ss.season_year = EXTRACT(YEAR FROM tt.game_date_et)::INT
-- Extended Statcast: opposing SP fastball arsenal
LEFT JOIN raw.mlb_statcast_pitcher_arsenal pa
    ON pa.player_id = sp.player_id
    AND pa.season_year = EXTRACT(YEAR FROM tt.game_date_et)::INT
-- Market total + same-day line movement (Feature 11)
LEFT JOIN LATERAL (
    SELECT
        MAX(o.total_points)                        AS market_total,
        MAX(o.total_points) - MIN(o.total_points)  AS line_move_total
    FROM odds.mlb_game_lines o
    JOIN games_today gt ON gt.home_team_abbr = o.home_team
    WHERE gt.game_slug = tt.game_slug
      AND o.as_of_date = tt.game_date_et
      AND o.total_points IS NOT NULL
) mkt_odds ON TRUE
-- Batter plate discipline (chase rate, contact rates, swing-and-miss)
LEFT JOIN raw.mlb_statcast_batter_discipline pd_b
    ON pd_b.player_id = rp.player_id
    AND pd_b.season_year = EXTRACT(YEAR FROM tt.game_date_et)::INT
-- Opponent team bullpen rolling stats (most recent entry before today)
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_team_pitching_rolling_mat
    WHERE team_abbr = tt.opponent_abbr
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC, game_slug DESC
    LIMIT 1
) opp_tp ON TRUE
-- SP per-start velocity rolling (MLB021) — most recent start before today (gap B)
-- NULL when no velocity data exists for this pitcher
LEFT JOIN LATERAL (
    SELECT fb_velo_avg_5, fb_velo_trend_5
    FROM features.mlb_sp_velocity_rolling
    WHERE player_id = sp.player_id
      AND game_date < %(game_date)s
    ORDER BY game_date DESC
    LIMIT 1
) sp_velo ON TRUE
-- Opponent reliever depth: distinct arms used in prior 1–2 days (Tier 1B)
LEFT JOIN LATERAL (
    SELECT bp_relievers_last_1d, bp_relievers_last_2d
    FROM features.mlb_reliever_rolling
    WHERE team_abbr = tt.opponent_abbr
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) opp_rl ON TRUE
-- Batter career stats at this venue (MLB023)
LEFT JOIN LATERAL (
    SELECT batter_n_games_at_venue, batter_venue_ba, batter_venue_hr_rate,
           batter_venue_slg, batter_venue_bb_rate
    FROM features.mlb_batter_venue_stats
    WHERE player_id    = rp.player_id
      AND venue_id     = (SELECT venue_id FROM raw.mlb_games WHERE game_slug = tt.game_slug LIMIT 1)
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) b_venue ON TRUE
-- Opposing SP K%% by batter handedness (MLB022)
LEFT JOIN LATERAL (
    SELECT sp_k_pct_vs_lhb_25, sp_k_pct_vs_rhb_25,
           sp_k_pct_vs_lhb_10, sp_k_pct_vs_rhb_10,
           sp_n_ab_vs_lhb_25,  sp_n_ab_vs_rhb_25
    FROM features.mlb_sp_hand_k_pct
    WHERE pitcher_id   = sp.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) opp_sp_hand_k ON TRUE
-- Opposing SP HR rate by batter handedness (MLB033)
LEFT JOIN LATERAL (
    SELECT sp_hr_rate_vs_lhb_25, sp_hr_rate_vs_rhb_25,
           sp_hr_rate_vs_lhb_10, sp_hr_rate_vs_rhb_10
    FROM features.mlb_sp_hand_hr_rate
    WHERE pitcher_id   = sp.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) opp_sp_hand_hr ON TRUE
-- Opposing catcher framing (MLB020): most recent catcher seen for opponent team
LEFT JOIN LATERAL (
    SELECT bps.player_id AS opp_catcher_id
    FROM raw.mlb_boxscore_player_stats bps
    JOIN raw.mlb_games mg ON mg.game_slug = bps.game_slug
    WHERE bps.primary_position = 'C'
      AND bps.team_abbr        = tt.opponent_abbr
      AND mg.game_date_et      < %(game_date)s
    ORDER BY mg.game_date_et DESC, bps.batting_order
    LIMIT 1
) opp_cat_recent ON TRUE
LEFT JOIN raw.mlb_statcast_catcher_framing opp_cf
    ON opp_cf.player_id   = opp_cat_recent.opp_catcher_id
    AND opp_cf.season_year = EXTRACT(YEAR FROM %(game_date)s::date)::INT
-- Confirmed batting order from pre-game lineup.  MySportsFeeds lineup player
-- IDs are not always the same IDs used by gamelogs/boxscores, so prefer ID
-- matches but fall back to the normalized player name.
LEFT JOIN LATERAL (
    SELECT LOWER(REGEXP_REPLACE(
        UNACCENT(bx.first_name || ' ' || bx.last_name), '[^a-z ]', '', 'gi'
    )) AS player_name_norm
    FROM raw.mlb_boxscore_player_stats bx
    WHERE bx.player_id = rp.player_id
    ORDER BY bx.game_slug DESC
    LIMIT 1
) rp_name ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) FILTER (WHERE lu.batting_order IS NOT NULL) AS lineup_slots
    FROM raw.mlb_lineups lu
    WHERE lu.game_slug = tt.game_slug
      AND lu.team_abbr = tt.team_abbr
) team_lu ON TRUE
LEFT JOIN LATERAL (
    SELECT lu.batting_order, lu.lineup_source
    FROM raw.mlb_lineups lu
    WHERE lu.game_slug = tt.game_slug
      AND lu.team_abbr = tt.team_abbr
      AND (
          lu.player_id = rp.player_id
          OR (
              rp_name.player_name_norm IS NOT NULL
              AND lu.player_name_norm = rp_name.player_name_norm
          )
      )
    ORDER BY
      CASE WHEN lu.player_id = rp.player_id THEN 0 ELSE 1 END,
      lu.batting_order NULLS LAST
    LIMIT 1
) conf_lu ON TRUE
-- Market over_price on canonical batter lines (FanDuel; today's game)
-- Hits O/U 0.5: over_price encodes implied P(≥1 hit) — varies widely by player/matchup
-- TB   O/U 1.5: over_price encodes implied P(≥2 TB)
-- HR   O/U 0.5: over_price encodes implied P(≥1 HR)
LEFT JOIN LATERAL (
    SELECT
        MAX(CASE WHEN pl.stat = 'batter_hits'        AND pl.line = 0.5 THEN pl.over_price END) AS market_hits_over_price,
        MAX(CASE WHEN pl.stat = 'batter_total_bases'  AND pl.line = 1.5 THEN pl.over_price END) AS market_tb_over_price,
        MAX(CASE WHEN pl.stat = 'batter_home_runs'    AND pl.line = 0.5 THEN pl.over_price END) AS market_hr_over_price
    FROM odds.mlb_player_prop_lines pl
    CROSS JOIN LATERAL (
        SELECT LOWER(REGEXP_REPLACE(
            UNACCENT(bx.first_name || ' ' || bx.last_name), '[^a-z ]', '', 'gi'
        )) AS name_norm
        FROM raw.mlb_boxscore_player_stats bx
        WHERE bx.player_id = rp.player_id
        LIMIT 1
    ) pn
    WHERE pl.player_name_norm = pn.name_norm
      AND pl.as_of_date       = %(game_date)s
      AND pl.bookmaker_key    = 'fanduel'
) mkt_props ON TRUE
-- Batter BABIP rolling (MLB030) — luck/regression signal for hits
LEFT JOIN LATERAL (
    SELECT batter_babip_10, batter_babip_career, babip_games_10
    FROM features.mlb_batter_babip_rolling_mat
    WHERE player_id    = rp.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) batter_babip ON TRUE
-- Opposing team DER (MLB032) — defensive quality of opponent's fielders
LEFT JOIN LATERAL (
    SELECT team_der_20, team_der_career
    FROM features.mlb_team_der_rolling_mat
    WHERE team_abbr    = tt.opponent_abbr
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) opp_def_der ON TRUE
-- Injury exclusion (Feature 12): skip OUT/DOUBTFUL batters
-- Join on mlb_player_id (extracted from MSF image URL) which matches gamelog player_id
LEFT JOIN raw.mlb_injuries inj_b ON inj_b.mlb_player_id = rp.player_id
WHERE br.ab_avg_10 >= %(min_ab_avg_10)s
  AND br.n_games_prev_10 >= %(min_n_games)s
  AND (inj_b.mlb_player_id IS NULL
       OR inj_b.playing_probability NOT IN ('OUT', 'DOUBTFUL'))
ORDER BY tt.start_ts_utc, tt.game_slug, rp.player_id
"""

SQL_PROP_LINES = """
SELECT
    player_name_norm,
    stat,
    bookmaker_key,
    line,
    over_price,
    under_price,
    over_link,
    under_link
FROM odds.mlb_player_prop_lines
WHERE as_of_date = %(game_date)s
  AND bookmaker_key IN ('fanduel', 'draftkings')
ORDER BY player_name_norm, stat, bookmaker_key
"""

SQL_PLAYER_NAMES = """
SELECT DISTINCT player_id,
       COALESCE(player_name, 'id=' || player_id::text) AS player_name
FROM raw.mlb_starting_pitchers
UNION ALL
SELECT DISTINCT player_id,
       player_name
FROM raw.mlb_starting_pitchers
WHERE player_name IS NOT NULL
"""


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_pitcher_artifacts(model_dir: Path):
    """Returns (xgb_k, lgb_k, feature_cols, medians, backtest, k_meta)."""
    xgb_k = XGBRegressor()
    xgb_k.load_model(str(model_dir / "strikeouts_xgb.json"))

    lgb_k = None
    lgb_path = model_dir / "lgb_strikeouts.txt"
    if _HAS_LGB and lgb_path.exists():
        lgb_k = lgb.Booster(model_file=str(lgb_path))

    feat = json.loads((model_dir / "feature_columns_pitchers.json").read_text())
    meds = json.loads((model_dir / "feature_medians_pitchers.json").read_text())

    bt_path = model_dir / "backtest_mae.json"
    bt = json.loads(bt_path.read_text()) if bt_path.exists() else {}

    k_meta_path = model_dir / "k_model_meta.json"
    k_meta = json.loads(k_meta_path.read_text()) if k_meta_path.exists() else {}

    return xgb_k, lgb_k, feat, meds, bt, k_meta


def _load_batter_artifacts(model_dir: Path):
    """Returns (xgb_hits, lgb_hits, xgb_tb, lgb_tb, xgb_hr, lgb_hr, feature_cols, medians, backtest)."""
    xgb_h = XGBRegressor()
    xgb_h.load_model(str(model_dir / "hits_xgb.json"))

    xgb_tb = XGBRegressor()
    xgb_tb.load_model(str(model_dir / "total_bases_xgb.json"))

    xgb_hr = XGBRegressor()
    xgb_hr.load_model(str(model_dir / "home_runs_xgb.json"))

    lgb_h = lgb_tb_m = lgb_hr = None
    if _HAS_LGB:
        h_path  = model_dir / "lgb_hits.txt"
        tb_path = model_dir / "lgb_total_bases.txt"
        hr_path = model_dir / "lgb_home_runs.txt"
        if h_path.exists():
            lgb_h = lgb.Booster(model_file=str(h_path))
        if tb_path.exists():
            lgb_tb_m = lgb.Booster(model_file=str(tb_path))
        if hr_path.exists():
            lgb_hr = lgb.Booster(model_file=str(hr_path))

    feat = json.loads((model_dir / "feature_columns_batters.json").read_text())
    meds = json.loads((model_dir / "feature_medians_batters.json").read_text())

    bt_path = model_dir / "backtest_mae.json"
    bt = json.loads(bt_path.read_text()) if bt_path.exists() else {}

    return xgb_h, lgb_h, xgb_tb, lgb_tb_m, xgb_hr, lgb_hr, feat, meds, bt


def _load_batter_clf_artifacts(model_dir: Path):
    """Load binary classifier models for batter props.
    Returns (models_dict, feat_clf, meds_clf, bt_clf, cal_map) where models_dict
    maps stat_key → (xgb_model, lgb_booster_or_None).
    Returns None if artifacts are missing.
    """
    feat_path = model_dir / "feature_columns_clf_batters.json"
    if not feat_path.exists():
        return None

    feat_clf = json.loads(feat_path.read_text())
    meds_clf = json.loads((model_dir / "feature_medians_clf_batters.json").read_text())
    bt_clf_path = model_dir / "backtest_clf.json"
    bt_clf = json.loads(bt_clf_path.read_text()) if bt_clf_path.exists() else {}
    cal_path = model_dir / "clf_calibration_batters.json"
    cal_map = json.loads(cal_path.read_text()) if cal_path.exists() else {}

    models: Dict[str, Tuple] = {}
    stat_files = {
        "batter_hits":        ("hits_clf_xgb.json",        "lgb_hits_clf.txt"),
        "batter_total_bases": ("total_bases_clf_xgb.json", "lgb_total_bases_clf.txt"),
        "batter_home_runs":   ("home_runs_clf_xgb.json",   "lgb_home_runs_clf.txt"),
    }
    for stat_key, (xgb_file, lgb_file) in stat_files.items():
        xgb_path = model_dir / xgb_file
        if not xgb_path.exists():
            continue
        xgb_m = XGBRegressor()
        xgb_m.load_model(str(xgb_path))
        lgb_m = None
        lgb_path = model_dir / lgb_file
        if _HAS_LGB and lgb_path.exists():
            lgb_m = lgb.Booster(model_file=str(lgb_path))
        models[stat_key] = (xgb_m, lgb_m)

    if not models:
        return None
    return models, feat_clf, meds_clf, bt_clf, cal_map


def _load_batter_alt_clf_artifacts(model_dir: Path):
    """Load FanDuel alt-line binary classifiers for batters (hits, total_bases).

    These are trained on all FD alt lines (0.5/1.5/2.5/3.5 for hits, etc.) so
    book_line is a proper feature spanning the full odds spectrum.
    Returns (models_dict, feat_clf, meds_clf, cal_map) or None.
    """
    feat_path = model_dir / "feature_columns_alt_clf_batters.json"
    if not feat_path.exists():
        return None

    feat_clf = json.loads(feat_path.read_text())
    meds_clf = json.loads((model_dir / "feature_medians_alt_clf_batters.json").read_text())
    cal_path = model_dir / "clf_calibration_alt_batters.json"
    cal_map = json.loads(cal_path.read_text()) if cal_path.exists() else {}

    models: Dict[str, Tuple] = {}
    stat_files = {
        "batter_hits":        ("hits_alt_clf_xgb.json",        "lgb_hits_alt_clf.txt"),
        "batter_total_bases": ("total_bases_alt_clf_xgb.json", "lgb_total_bases_alt_clf.txt"),
    }
    for stat_key, (xgb_file, lgb_file) in stat_files.items():
        xgb_path = model_dir / xgb_file
        if not xgb_path.exists():
            continue
        xgb_m = XGBRegressor()
        xgb_m.load_model(str(xgb_path))
        lgb_m = None
        lgb_path = model_dir / lgb_file
        if _HAS_LGB and lgb_path.exists():
            lgb_m = lgb.Booster(model_file=str(lgb_path))
        models[stat_key] = (xgb_m, lgb_m)

    if not models:
        return None
    return models, feat_clf, meds_clf, cal_map


def _load_pitcher_alt_clf_artifacts(model_dir: Path):
    """Load FanDuel alt-line binary classifier for pitcher strikeouts.

    Trained on all FD K lines (3.5–7.5) so the model captures P(over|book_line).
    Returns (xgb_clf, lgb_clf, feat_clf, meds_clf, cal_map) or None.
    """
    feat_path = model_dir / "feature_columns_alt_clf_pitchers.json"
    xgb_path  = model_dir / "strikeouts_alt_clf_xgb.json"
    if not feat_path.exists() or not xgb_path.exists():
        return None

    feat_clf = json.loads(feat_path.read_text())
    meds_clf = json.loads((model_dir / "feature_medians_alt_clf_pitchers.json").read_text())
    cal_path = model_dir / "clf_calibration_alt_pitchers.json"
    cal_map = json.loads(cal_path.read_text()) if cal_path.exists() else {}

    xgb_clf = XGBRegressor()
    xgb_clf.load_model(str(xgb_path))
    lgb_clf = None
    lgb_path = model_dir / "lgb_strikeouts_alt_clf.txt"
    if _HAS_LGB and lgb_path.exists():
        lgb_clf = lgb.Booster(model_file=str(lgb_path))

    return xgb_clf, lgb_clf, feat_clf, meds_clf, cal_map


def _compute_alt_clf_probs(
    X_base: "pd.DataFrame",
    norm_names: List[str],
    all_alt_lines: Dict,
    stats: List[str],
    meta_cols: List[str],
    alt_clf_arts,
) -> Dict:
    """Batch-compute alt CLF P(over) for all (player, stat, alt_line) combos today.

    X_base is the pre-processed (derived features applied, medians filled) feature
    matrix for all players.  alt_clf_arts is the return value of
    _load_batter_alt_clf_artifacts or _load_pitcher_alt_clf_artifacts.

    Returns dict mapping (norm_name, stat, line_val_float) → calibrated P(over).
    """
    if alt_clf_arts is None or X_base is None or X_base.empty:
        return {}

    # Unpack based on whether it's batter (models_dict) or pitcher (single model)
    if isinstance(alt_clf_arts[0], dict):
        # Batter: (models_dict, feat_clf, meds_clf, cal_map)
        models_dict, feat_clf, meds_clf, cal_map = alt_clf_arts
    else:
        # Pitcher: (xgb_clf, lgb_clf, feat_clf, meds_clf, cal_map)
        xgb_clf, lgb_clf, feat_clf, meds_clf, cal_map = alt_clf_arts
        models_dict = {"_pitcher_k": (xgb_clf, lgb_clf)}

    result: Dict = {}
    orig_stats = stats  # save for key mapping

    for stat_idx, stat_key in enumerate(orig_stats):
        internal_key = "_pitcher_k" if stat_key == "pitcher_strikeouts" else stat_key
        if internal_key not in models_dict:
            continue
        xgb_m, lgb_m = models_dict[internal_key]
        cal = cal_map.get(stat_key)

        # Collect (row_idx, line_val, norm_name) triples
        pairs: List[Tuple[int, float, str]] = []
        for row_idx, norm in enumerate(norm_names):
            for ld in all_alt_lines.get((norm, stat_key), []):
                line_val = ld.get("line")
                if line_val is not None:
                    pairs.append((row_idx, float(line_val), norm))

        if not pairs:
            continue

        indices   = [p[0] for p in pairs]
        line_vals = np.array([p[1] for p in pairs], dtype=float)

        # Build per-(player,line) matrix — X_base rows repeated for each alt line
        X_sub = X_base.iloc[indices].reset_index(drop=True)
        X_clf = _prep_features_clf(X_sub, line_vals, feat_clf, meds_clf)

        raw_p = np.clip(_predict_ensemble(X_clf, xgb_m, lgb_m), 0.01, 0.99)
        cal_p = _apply_platt_calibration(raw_p, cal)

        for j, (row_idx, line_val, norm) in enumerate(pairs):
            result[(norm, stat_key, line_val)] = float(cal_p[j])

    return result


def _lookup_alt_clf_prob(
    alt_probs: Optional[Dict],
    norm_name: str,
    stat: str,
    line: Optional[float],
) -> Optional[float]:
    if not alt_probs or line is None:
        return None
    try:
        line_v = float(line)
    except Exception:
        return None
    direct = alt_probs.get((norm_name, stat, line_v))
    if direct is not None:
        return float(direct)
    for key, value in alt_probs.items():
        try:
            nm, st, ln = key
        except Exception:
            continue
        if nm == norm_name and st == stat and abs(float(ln) - line_v) < 1e-9:
            return float(value)
    return None


def _apply_threshold_overrides(cfg: PredictConfig) -> None:
    """Load threshold overrides from models/player_props/prop_thresholds.json.

    Supported keys (either style):
      - direct config names:
          threshold_strikeouts, threshold_hits, threshold_total_bases,
          threshold_home_runs_over, threshold_home_runs_under,
          threshold_clf
      - stat-group names:
          pitcher_strikeouts, batter_hits, batter_total_bases,
          batter_home_runs (with over/under or abs_edge),
          clf_prob_edge
    """
    path = cfg.model_dir / cfg.thresholds_file
    if not path.exists():
        return

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not parse threshold overrides at %s: %s", path, exc)
        return

    def _set(attr: str, val) -> None:
        try:
            f = float(val)
        except Exception:
            return
        if f >= 0.0:
            setattr(cfg, attr, f)

    # Direct names
    for key in (
        "threshold_strikeouts",
        "threshold_strikeouts_over",
        "threshold_strikeouts_under",
        "threshold_hits",
        "threshold_total_bases",
        "threshold_total_bases_over",
        "threshold_total_bases_under",
        "threshold_home_runs_over",
        "threshold_home_runs_under",
        "threshold_clf",
        "min_ev",
    ):
        if key in raw:
            _set(key, raw[key])

    # Stat-group names
    if "pitcher_strikeouts" in raw:
        v = raw["pitcher_strikeouts"]
        if isinstance(v, dict):
            _set("threshold_strikeouts", v.get("abs_edge"))
            _set("threshold_strikeouts_over", v.get("over"))
            _set("threshold_strikeouts_under", v.get("under"))
        else:
            _set("threshold_strikeouts", v)
    if "batter_hits" in raw:
        v = raw["batter_hits"]
        _set("threshold_hits", v.get("abs_edge") if isinstance(v, dict) else v)
    if "batter_total_bases" in raw:
        v = raw["batter_total_bases"]
        if isinstance(v, dict):
            _set("threshold_total_bases", v.get("abs_edge"))
            _set("threshold_total_bases_over", v.get("over"))
            _set("threshold_total_bases_under", v.get("under"))
        else:
            _set("threshold_total_bases", v)
    if "batter_home_runs" in raw:
        v = raw["batter_home_runs"]
        if isinstance(v, dict):
            if "over" in v:
                _set("threshold_home_runs_over", v["over"])
            if "under" in v:
                _set("threshold_home_runs_under", v["under"])
            if "abs_edge" in v:
                _set("threshold_home_runs_over", v["abs_edge"])
                _set("threshold_home_runs_under", v["abs_edge"])
        else:
            _set("threshold_home_runs_over", v)
            _set("threshold_home_runs_under", v)
    if "clf_prob_edge" in raw:
        v = raw["clf_prob_edge"]
        _set("threshold_clf", v.get("abs_edge") if isinstance(v, dict) else v)

    log.info(
        "Loaded threshold overrides from %s | K=%.2f(o=%s/u=%s) H=%.2f TB=%.2f(o=%s/u=%s) HR(o/u)=%.2f/%.2f CLF=%.3f EV=%.3f",
        path,
        cfg.threshold_strikeouts,
        "-" if cfg.threshold_strikeouts_over is None else f"{cfg.threshold_strikeouts_over:.2f}",
        "-" if cfg.threshold_strikeouts_under is None else f"{cfg.threshold_strikeouts_under:.2f}",
        cfg.threshold_hits,
        cfg.threshold_total_bases,
        "-" if cfg.threshold_total_bases_over is None else f"{cfg.threshold_total_bases_over:.2f}",
        "-" if cfg.threshold_total_bases_under is None else f"{cfg.threshold_total_bases_under:.2f}",
        cfg.threshold_home_runs_over,
        cfg.threshold_home_runs_under,
        cfg.threshold_clf,
        cfg.min_ev,
    )


def _load_pitcher_clf_artifacts(model_dir: Path):
    """Load binary classifier model for pitcher strikeouts.
    Returns (xgb_clf, lgb_clf, feat_clf, meds_clf, bt_clf, cal_map) or None.
    """
    feat_path = model_dir / "feature_columns_clf_pitchers.json"
    xgb_path  = model_dir / "strikeouts_clf_xgb.json"
    if not feat_path.exists() or not xgb_path.exists():
        return None

    feat_clf = json.loads(feat_path.read_text())
    meds_clf = json.loads((model_dir / "feature_medians_clf_pitchers.json").read_text())
    bt_clf_path = model_dir / "backtest_clf.json"
    bt_clf = json.loads(bt_clf_path.read_text()) if bt_clf_path.exists() else {}
    cal_path = model_dir / "clf_calibration_pitchers.json"
    cal_map = json.loads(cal_path.read_text()) if cal_path.exists() else {}

    xgb_clf = XGBRegressor()
    xgb_clf.load_model(str(xgb_path))
    lgb_clf = None
    lgb_path = model_dir / "lgb_strikeouts_clf.txt"
    if _HAS_LGB and lgb_path.exists():
        lgb_clf = lgb.Booster(model_file=str(lgb_path))

    return xgb_clf, lgb_clf, feat_clf, meds_clf, bt_clf, cal_map


def _apply_platt_calibration(p: np.ndarray, cal: Optional[Dict]) -> np.ndarray:
    """Apply persisted probability calibration (platt or isotonic)."""
    p_safe = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if not cal:
        return p_safe
    method = str(cal.get("method") or "").lower()
    if method == "platt":
        try:
            a = float(cal.get("a"))
            b = float(cal.get("b"))
        except Exception:
            return p_safe
        z = np.log(p_safe / (1.0 - p_safe))
        out = 1.0 / (1.0 + np.exp(-(a * z + b)))
        return np.clip(out, 1e-6, 1 - 1e-6)
    if method == "isotonic":
        try:
            x = np.asarray(cal.get("x"), dtype=float)
            y = np.asarray(cal.get("y"), dtype=float)
        except Exception:
            return p_safe
        if x.size < 2 or y.size < 2:
            return p_safe
        out = np.interp(p_safe, x, y)
        return np.clip(out, 1e-6, 1 - 1e-6)
    return p_safe


def _clf_line_bucket(stat: str, line: Optional[float]) -> str:
    if line is None:
        return "__no_line__"
    try:
        l = float(line)
    except Exception:
        return "__no_line__"
    if stat == "pitcher_strikeouts":
        return _line_bucket_for_over_penalty(stat, l)
    if stat == "batter_total_bases":
        return _line_bucket_for_over_penalty(stat, l)
    if stat == "batter_hits":
        if l < 1.0:
            return "H 0.5"
        if l < 2.0:
            return "H 1.5"
        return "H 2.5+"
    if stat == "batter_home_runs":
        if l < 1.0:
            return "HR 0.5"
        return "HR 1.5+"
    return "other"


def _load_clf_bucket_controls(model_dir: Path, file_name: str) -> dict[tuple[str, str], bool]:
    """Load optional bucket disable map for CLF usage."""
    path = model_dir / file_name
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not parse CLF controls at %s: %s", path, exc)
        return {}
    disabled = raw.get("disabled_buckets") or {}
    out: dict[tuple[str, str], bool] = {}
    if isinstance(disabled, dict):
        for stat, buckets in disabled.items():
            if not isinstance(buckets, list):
                continue
            for b in buckets:
                out[(str(stat), str(b))] = True
    if out:
        log.info("Loaded CLF bucket controls from %s (%d disabled buckets)", path, len(out))
    return out


def _clf_bucket_is_disabled(
    controls: dict[tuple[str, str], bool],
    stat: str,
    line: Optional[float],
    *,
    honor_controls: bool = False,
) -> bool:
    if not honor_controls:
        return False
    if not controls:
        return False
    b = _clf_line_bucket(stat, line)
    return bool(controls.get((stat, b), False))


def _prep_features_clf(
    X_base: pd.DataFrame,
    book_lines: np.ndarray,
    feat_clf: List[str],
    meds_clf: Dict[str, float],
) -> pd.DataFrame:
    """Add per-player book_line to base feature matrix and align to clf schema.

    X_base must already be processed by _prep_features() (derived features applied).
    book_lines is a float array of length len(X_base); NaN where no line available.
    """
    X = X_base.copy()
    X["book_line"] = book_lines
    X = X.reindex(columns=feat_clf)
    for c, m in meds_clf.items():
        if c in X.columns:
            X[c] = X[c].fillna(m)
    return X.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Feature prep
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_numeric(X: pd.DataFrame) -> pd.DataFrame:
    for c in list(X.columns):
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]) or is_datetime64_any_dtype(X[c]):
            continue
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _prep_features(
    df: pd.DataFrame,
    meta_cols: List[str],
    feature_cols: List[str],
    medians: Dict[str, float],
) -> pd.DataFrame:
    """Drop meta, OHE season, coerce, add derived, fill medians, align to feature_cols."""
    X = df.drop(columns=[c for c in meta_cols if c in df.columns]).copy()

    if "season" in X.columns:
        X = pd.get_dummies(X, columns=["season"], drop_first=False, dummy_na=False)
    if "is_home" in X.columns:
        X["is_home"] = X["is_home"].astype("boolean").fillna(False).astype(int)

    X = _coerce_numeric(X)

    bad = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if bad:
        X = pd.get_dummies(X, columns=bad, drop_first=False, dummy_na=True)

    X = add_player_prop_derived_features(X)

    # Align to training schema
    X = X.reindex(columns=feature_cols)
    for c, m in medians.items():
        if c in X.columns:
            X[c] = X[c].fillna(m)
    return X.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Name normalization (for matching prop lines)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Lower-case, strip accents, remove punctuation, collapse whitespace."""
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    ascii_str = re.sub(r"[^a-z0-9\s]", "", ascii_str.lower())
    return re.sub(r"\s+", " ", ascii_str).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Kelly sizing
# ─────────────────────────────────────────────────────────────────────────────

def _kelly_prop(edge: float, sigma: float, max_kelly: float = 0.05) -> float:
    """Simple Kelly fraction for a prop bet.
    edge = pred - line (positive = over), sigma = CI p68.
    Approximates win probability from edge / sigma ratio.
    """
    if sigma <= 0:
        return 0.0
    z = edge / sigma
    # Use logistic approximation: p ≈ 1 / (1 + exp(-z))
    p = 1.0 / (1.0 + math.exp(-z))
    # Kelly fraction = (p * 2 - 1) for 1:1 odds (simplified for -110 juice)
    k = max(0.0, (p - 0.5238) / 0.4762)  # break-even is 52.38% at -110
    return min(k, max_kelly)


def _american_to_profit_mult(price: Optional[float]) -> Optional[float]:
    """American odds -> net profit multiplier on 1 unit stake."""
    if price is None:
        return None
    try:
        p = float(price)
    except Exception:
        return None
    if p == 0:
        return None
    if p > 0:
        return p / 100.0
    return 100.0 / abs(p)


def _american_to_implied_prob(price: Optional[float]) -> Optional[float]:
    """American odds -> break-even probability before vig removal."""
    if price is None:
        return None
    try:
        p = float(price)
    except Exception:
        return None
    if p == 0:
        return None
    if p > 0:
        return 100.0 / (p + 100.0)
    return abs(p) / (abs(p) + 100.0)


def _ev_per_unit(p_win: Optional[float], price: Optional[float]) -> Optional[float]:
    """Expected net return for 1.0 unit stake."""
    if p_win is None:
        return None
    b = _american_to_profit_mult(price)
    if b is None:
        return None
    p = max(0.0, min(1.0, float(p_win)))
    return p * b - (1.0 - p)


def _prob_over_from_regression(pred: Optional[float], line: Optional[float], sigma: Optional[float] = None) -> Optional[float]:
    """Estimate P(stat > line) using a Poisson CDF with mu=pred.

    All MLB prop stats (hits, TB, HR, strikeouts) are non-negative integer
    counts.  P(X > line) = 1 - Poisson.cdf(floor(line), mu=pred) is far more
    accurate than the Gaussian sigmoid approximation, especially at high alt lines
    where the Gaussian overestimates P(over) by 10-13 percentage points.

    sigma is accepted for backward compatibility but is not used.
    """
    if pred is None or line is None:
        return None
    mu = float(pred)
    if mu <= 0.0:
        return 0.0
    k = int(math.floor(float(line)))
    return float(1.0 - _scipy_poisson.cdf(k, mu=mu))


def _signed_probability_edge(p_over: Optional[float], breakeven: float = _BREAKEVEN_PROB) -> Optional[float]:
    """Return signed side edge in probability space.

    Positive means the over clears break-even; negative means the under clears
    break-even. Probabilities in the no-edge band return 0.0 instead of being
    misread as an under signal.
    """
    if p_over is None:
        return None
    try:
        p = max(0.0, min(1.0, float(p_over)))
    except Exception:
        return None
    over_edge = p - breakeven
    under_edge = (1.0 - p) - breakeven
    if over_edge > 0 and over_edge >= under_edge:
        return over_edge
    if under_edge > 0:
        return -under_edge
    return 0.0


def _load_prop_side_recalibrators(model_dir: Path, file_name: str) -> dict:
    path = model_dir / file_name
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not parse prop side recalibrators at %s: %s", path, exc)
        return {}
    n = len(raw.get("calibrators") or {})
    if n:
        log.info("Loaded prop side recalibrators from %s (%d buckets)", path, n)
    return raw


def _load_prop_betting_layer(model_dir: Path, file_name: str) -> dict:
    path = model_dir / file_name
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not parse prop betting layer at %s: %s", path, exc)
        return {}
    n = len(raw.get("models") or {})
    if n:
        log.info("Loaded prop betting layer from %s (%d models)", path, n)
    return raw


def _load_prop_market_side_priors(model_dir: Path, file_name: str) -> dict:
    path = model_dir / file_name
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not parse prop market-side priors at %s: %s", path, exc)
        return {}
    n = len(raw.get("models") or {})
    if n:
        log.info("Loaded prop market-side priors from %s (%d models)", path, n)
    return raw


def _load_prop_walk_forward_policy(model_dir: Path, file_name: str) -> dict:
    path = model_dir / file_name
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not parse prop walk-forward policy at %s: %s", path, exc)
        return {}
    policy = raw.get("live_policy") or {}
    n = (
        len(policy.get("exact_bucket") or {})
        + len(policy.get("line_surface") or {})
        + len(policy.get("market_side") or {})
    )
    if n:
        log.info("Loaded prop walk-forward live policy from %s (%d buckets)", path, n)
    return policy


def _load_prop_bucket_reopen_policy(model_dir: Path, file_name: str) -> dict:
    path = model_dir / file_name
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not parse prop bucket reopen policy at %s: %s", path, exc)
        return {}
    n = len(raw.get("reopen_buckets") or {})
    log.info("Loaded prop bucket reopen policy from %s (%d reopened buckets)", path, n)
    return raw


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _slot_pa_prior(slot: Optional[float]) -> Optional[float]:
    if slot is None:
        return None
    try:
        return max(3.2, min(4.85, 4.75 - 0.16 * (float(slot) - 1.0)))
    except Exception:
        return None


def _rough_hitter_projected_pa(row: pd.Series, effective_order: Optional[float]) -> Optional[float]:
    ab_per_game = _float_or_none(row.get("ab_per_game_played"))
    if ab_per_game is None:
        ab_per_game = _float_or_none(row.get("ab_avg_10"))
    bb_avg_10 = _float_or_none(row.get("bb_avg_10")) or 0.0
    if ab_per_game is not None:
        return max(1.0, min(5.8, ab_per_game + max(0.0, bb_avg_10)))
    return _slot_pa_prior(effective_order)


def _load_hitter_pa_artifact(model_dir: Path) -> Optional[dict]:
    if joblib is None:
        return None
    path = model_dir / "hitter_player_game_outcome_models.joblib"
    if not path.exists():
        return None
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        log.warning("Could not load hitter player-game outcome artifact at %s: %s", path, exc)
        return None
    models = artifact.get("models") or {}
    pa_model = models.get("pa_model")
    if pa_model is None:
        return None
    rec = artifact.get("recommendation") or {}
    pa_gain = _float_or_none(rec.get("pa_mae_gain_vs_slot_prior"))
    if pa_gain is not None and pa_gain <= 0.0:
        log.info("Hitter PA artifact present but disabled; PA gain vs slot prior is %.4f", pa_gain)
        return None
    log.info(
        "Loaded validated hitter PA model from %s (PA MAE gain vs slot prior=%s)",
        path,
        "unknown" if pa_gain is None else f"{pa_gain:.4f}",
    )
    return artifact


def _pa_feature_value(row: pd.Series, name: str, projected_pa: Optional[float], effective_order: Optional[float]) -> Any:
    if name == "lineup_slot":
        return effective_order
    if name == "confirmed_starter_num":
        return 1.0 if _float_or_none(row.get("confirmed_batting_order")) is not None else 0.0
    if name == "projected_pa":
        return projected_pa
    if name == "pa_games":
        return row.get("n_games_prev_10")
    if name == "team_implied_runs":
        # Live hitter snapshots usually have game total but not side implied
        # runs; half-total is a neutral fallback for the PA model.
        implied = row.get("team_implied_runs")
        if _float_or_none(implied) is not None:
            return implied
        total = _float_or_none(row.get("game_total_line")) or _float_or_none(row.get("market_total"))
        return None if total is None else total / 2.0
    if name == "opponent_implied_runs":
        implied = row.get("opponent_implied_runs")
        if _float_or_none(implied) is not None:
            return implied
        total = _float_or_none(row.get("game_total_line")) or _float_or_none(row.get("market_total"))
        return None if total is None else total / 2.0
    if name == "game_total_line":
        return row.get("game_total_line") if "game_total_line" in row else row.get("market_total")
    if name == "lineup_confirmed_flag":
        return 1.0 if _float_or_none(row.get("confirmed_batting_order")) is not None else 0.0
    if name == "confirmed_team_lineup_slots":
        return row.get("confirmed_team_lineup_slots")
    if name == "team_lineup_confirmed_flag":
        slots = _float_or_none(row.get("confirmed_team_lineup_slots"))
        return 1.0 if slots is not None and slots >= 7.0 else 0.0
    if name == "lineup_boxscore_proxy_flag":
        return 0.0 if _float_or_none(row.get("confirmed_batting_order")) is not None else 1.0
    if name == "lineup_slot_x_team_implied_runs":
        runs = _pa_feature_value(row, "team_implied_runs", projected_pa, effective_order)
        return None if effective_order is None or runs is None else float(effective_order) * float(runs)
    live_aliases = {
        "own_lineup_xslg_avg": ("own_lineup_slg_avg_10",),
        "batter_sc_barrel_rate": ("sc_barrel_rate",),
        "batter_sc_hard_hit_pct": ("sc_hard_hit_pct",),
        "batter_sc_avg_exit_velo": ("sc_avg_exit_velo",),
        "batter_sc_avg_launch_angle": ("sc_avg_launch_angle",),
        "batter_sc_sweet_spot_pct": ("sc_sweet_spot_pct",),
        "batter_sc_fb_pct": ("sc_fb_pct",),
        "batter_sc_gb_pct": ("sc_gb_pct",),
        "batter_sc_ld_pct": ("sc_ld_pct",),
        "batter_sc_xba": ("sc_xba",),
        "batter_sc_xslg": ("sc_xslg",),
        "batter_sc_xwoba": ("sc_xwoba",),
        "batter_sc_xiso": ("sc_xiso",),
        "batter_sc_pull_pct": ("sc_pull_pct",),
        "batter_sc_opposite_pct": ("sc_opposite_pct",),
        "batter_sc_popup_pct": ("sc_popup_pct",),
        "batter_sc_brl_pa": ("sc_brl_pa",),
        "batter_sprint_speed": ("sprint_speed",),
        "batter_disc_oz_swing_pct": ("sc_b_oz_swing_pct",),
        "batter_disc_iz_contact_pct": ("sc_b_iz_contact_pct",),
        "batter_disc_oz_contact_pct": ("sc_b_oz_contact_pct",),
        "batter_disc_whiff_pct": ("sc_b_disc_whiff_pct",),
        "batter_disc_out_zone_pct": ("sc_b_out_zone_pct",),
        "batter_disc_k_pct": ("sc_b_k_pct",),
        "batter_disc_bb_pct": ("sc_b_bb_pct",),
        "opp_sp_sc_barrel_rate": ("opp_sp_sc_barrel_rate",),
        "opp_sp_sc_hard_hit_pct": ("opp_sp_sc_hard_hit_pct",),
        "opp_sp_sc_avg_exit_velo": ("opp_sp_sc_avg_exit_velo",),
        "opp_sp_sc_avg_launch_angle": ("opp_sp_sc_avg_launch_angle",),
        "opp_sp_sc_xba": ("opp_sp_sc_xba",),
        "opp_sp_sc_xslg": ("opp_sp_sc_xslg",),
        "opp_sp_sc_xwoba": ("opp_sp_sc_xwoba",),
        "opp_sp_sc_xiso": ("opp_sp_sc_xiso",),
    }
    for alias in live_aliases.get(name, ()):
        if alias in row and _float_or_none(row.get(alias)) is not None:
            return row.get(alias)
    if name == "opp_sp_hand_l":
        hand = str(row.get("opp_sp_hand") or "").upper()
        return 1.0 if hand == "L" else (0.0 if hand in {"R", "S"} else None)
    if name == "opp_sp_k_pct_10":
        return row.get("opp_sp_k_pct_10") if "opp_sp_k_pct_10" in row else row.get("opp_sp_k_pct_5")
    if name == "opp_sp_bb_pct":
        return row.get("opp_sp_bb_pct") if "opp_sp_bb_pct" in row else row.get("opp_sp_bb_pct_5")
    if name == "opp_sp_xwoba":
        return row.get("opp_sp_xwoba") if "opp_sp_xwoba" in row else row.get("opp_sp_sc_xwoba")
    if name == "opp_sp_hard_hit_pct":
        return row.get("opp_sp_hard_hit_pct") if "opp_sp_hard_hit_pct" in row else row.get("opp_sp_sc_hard_hit_pct")
    if name == "opp_sp_whiff_pct":
        return row.get("opp_sp_whiff_pct") if "opp_sp_whiff_pct" in row else row.get("opp_sp_fb_whiff_pct")
    if name == "opp_bp_k9_10":
        return row.get("opp_bp_k9_10") if "opp_bp_k9_10" in row else row.get("opp_bp_k9_5")
    if name == "batter_vs_hand_hits_avg_10":
        return row.get("batter_vs_hand_hits_avg_10") if "batter_vs_hand_hits_avg_10" in row else row.get("hits_avg_10_vs_hand")
    if name == "batter_vs_hand_tb_avg_10":
        return row.get("batter_vs_hand_tb_avg_10") if "batter_vs_hand_tb_avg_10" in row else row.get("tb_avg_10_vs_hand")
    if name == "batter_vs_hand_hr_avg_10":
        return row.get("batter_vs_hand_hr_avg_10") if "batter_vs_hand_hr_avg_10" in row else row.get("hr_avg_10_vs_hand")
    if name == "batter_vs_hand_iso_avg_10":
        return row.get("batter_vs_hand_iso_avg_10") if "batter_vs_hand_iso_avg_10" in row else row.get("iso_avg_10_vs_hand")
    if name == "batter_vs_hand_k_rate_10":
        return row.get("batter_vs_hand_k_rate_10") if "batter_vs_hand_k_rate_10" in row else row.get("k_rate_avg_40_vs_hand")
    if name == "batter_vs_hand_games_10":
        if "batter_vs_hand_games_10" in row:
            return row.get("batter_vs_hand_games_10")
        hand = str(row.get("opp_sp_hand") or "").upper()
        return row.get("n_games_vs_lhp_10") if hand == "L" else row.get("n_games_vs_rhp_10")
    if name == "batter_vs_rp_ba_30":
        return row.get("batter_vs_rp_ba_30") if "batter_vs_rp_ba_30" in row else row.get("bvr_ba_30")
    if name == "batter_vs_rp_slg_30":
        return row.get("batter_vs_rp_slg_30") if "batter_vs_rp_slg_30" in row else row.get("bvr_slg_30")
    if name == "batter_vs_rp_hr_rate_30":
        return row.get("batter_vs_rp_hr_rate_30") if "batter_vs_rp_hr_rate_30" in row else row.get("bvr_hr_rate_30")
    if name == "batter_vs_rp_k_rate_30":
        return row.get("batter_vs_rp_k_rate_30") if "batter_vs_rp_k_rate_30" in row else row.get("bvr_k_rate_30")
    return row.get(name)


def _pa_feature_category(row: pd.Series, name: str) -> Any:
    if name == "lineup_source":
        return row.get("confirmed_lineup_source") or "rolling_batting_order"
    if name == "starter_status_source":
        return "confirmed_lineup" if _float_or_none(row.get("confirmed_batting_order")) is not None else "rolling_batting_order"
    if name == "primary_position":
        return row.get("primary_position") or "unknown"
    return row.get(name) or "unknown"


def _predict_validated_hitter_pa(df: pd.DataFrame, artifact: Optional[dict]) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    feature_rows: list[Optional[dict[str, Any]]] = []
    numeric_features = list((artifact or {}).get("numeric_features") or [])
    categorical_features = list((artifact or {}).get("categorical_features") or [])

    for _, row in df.iterrows():
        conf_order = _float_or_none(row.get("confirmed_batting_order"))
        avg_order = _float_or_none(row.get("batting_order_avg_10"))
        team_lineup_slots = _float_or_none(row.get("confirmed_team_lineup_slots"))
        team_lineup_available = team_lineup_slots is not None and team_lineup_slots >= 7
        effective_order = conf_order if conf_order is not None else avg_order
        baseline_pa = _rough_hitter_projected_pa(row, effective_order)
        allow_pa_model = conf_order is not None or team_lineup_available
        info = {
            "confirmed_batting_order": conf_order,
            "confirmed_team_lineup_slots": team_lineup_slots,
            "effective_batting_order": effective_order,
            "baseline_projected_pa": baseline_pa,
            "projected_pa": baseline_pa,
            "pa_scale": 1.0,
            "pa_model_source": "baseline" if allow_pa_model else "baseline_no_team_lineup",
        }
        infos.append(info)

        if artifact and allow_pa_model:
            feat = {
                name: _pa_feature_value(row, name, baseline_pa, effective_order)
                for name in numeric_features
            }
            try:
                player_key = str(int(row.get("player_id")))
            except (TypeError, ValueError):
                player_key = str(row.get("player_id") or "")
            player_prior = ((artifact.get("player_prior_state") or {}).get(player_key) or {})
            for name, value in player_prior.items():
                if name in feat:
                    feat[name] = value
            feat.update({name: _pa_feature_category(row, name) for name in categorical_features})
            feature_rows.append(feat)
        elif artifact:
            feature_rows.append(None)

    if not artifact or not feature_rows:
        return infos

    artifact_models = artifact.get("models") or {}
    model = artifact_models.get("pa_model")
    low_model = artifact_models.get("pa_low_model")
    normal_model = artifact_models.get("pa_normal_model")
    use_two_part = bool(artifact_models.get("pa_two_part_use") and low_model is not None and normal_model is not None)
    if model is None and not use_two_part:
        return infos
    try:
        valid_indices = [i for i, feat in enumerate(feature_rows) if feat is not None]
        if not valid_indices:
            return infos
        X = pd.DataFrame([feature_rows[i] for i in valid_indices])
        for col in numeric_features:
            if col not in X:
                X[col] = np.nan
            X[col] = pd.to_numeric(X[col], errors="coerce")
        for col in categorical_features:
            if col not in X:
                X[col] = "unknown"
            X[col] = X[col].fillna("unknown").astype(str)
        model_input = X[numeric_features + categorical_features]
        if use_two_part:
            low_prob = np.clip(np.asarray(low_model.predict_proba(model_input)[:, 1], dtype=float), 1e-5, 1.0 - 1e-5)
            normal_pred = np.clip(np.asarray(normal_model.predict(model_input), dtype=float), 3.0, 7.0)
            low_states = ((((artifact.get("pa_uncertainty") or {}).get("global") or {}).get("low_pa_state_probs"))
                          or {"0": 0.05, "1": 0.20, "2": 0.75})
            low_total = sum(float(low_states.get(str(n), 0.0)) for n in range(3)) or 1.0
            low_mean = sum(n * float(low_states.get(str(n), 0.0)) for n in range(3)) / low_total
            pred = np.clip(low_prob * low_mean + (1.0 - low_prob) * normal_pred, 0.4, 6.4)
        else:
            low_prob = np.full(len(model_input), np.nan, dtype=float)
            normal_pred = np.full(len(model_input), np.nan, dtype=float)
            pred = np.clip(np.asarray(model.predict(model_input), dtype=float), 0.4, 6.4)
    except Exception:
        log.warning("Validated hitter PA model failed at prediction time; using baseline PA", exc_info=True)
        return infos

    for pos, (i, pa) in enumerate(zip(valid_indices, pred)):
        if not math.isfinite(float(pa)):
            continue
        base = infos[i].get("baseline_projected_pa")
        scale = 1.0
        if base is not None and base > 0:
            scale = max(0.75, min(1.25, float(pa) / float(base)))
        infos[i].update({
            "projected_pa": float(pa),
            "validated_projected_pa": float(pa),
            "pa_scale": scale,
            "pa_model_source": "validated_two_part_pa" if use_two_part else "validated_pa_model",
            "low_pa_probability": float(low_prob[pos]) if math.isfinite(float(low_prob[pos])) else None,
            "normal_projected_pa": float(normal_pred[pos]) if math.isfinite(float(normal_pred[pos])) else None,
        })
    return infos


def _prop_reopen_bucket_key(
    stat: str,
    side: str,
    line: Optional[float],
    price,
    bookmaker_key: Optional[str] = None,
) -> str:
    return "|".join([
        str(stat or "*"),
        str(side or "*"),
        _cal_prop_line_surface(str(stat or ""), side, line),
        _cal_prop_line_bucket(str(stat or ""), line),
        _cal_price_bucket(price),
        str(bookmaker_key or "*"),
    ])


def _prop_reopen_bucket_key_legacy(
    stat: str,
    side: str,
    line: Optional[float],
    price,
    bookmaker_key: Optional[str] = None,
) -> str:
    return "|".join([
        str(stat or "*"),
        str(side or "*"),
        _cal_prop_line_bucket(str(stat or ""), line),
        _cal_price_bucket(price),
        str(bookmaker_key or "*"),
    ])


def _prop_bucket_is_reopened(
    policy: Optional[dict],
    *,
    stat: str,
    side: str | None,
    line: Optional[float],
    price,
    bookmaker_key: Optional[str] = None,
) -> tuple[bool, str | None]:
    tier, key, _record = _prop_bucket_ladder_tier(
        policy,
        stat=stat,
        side=side,
        line=line,
        price=price,
        bookmaker_key=bookmaker_key,
    )
    return tier in {"micro", "starter", "bankroll"}, key


def _prop_bucket_ladder_tier(
    policy: Optional[dict],
    *,
    stat: str,
    side: str | None,
    line: Optional[float],
    price,
    bookmaker_key: Optional[str] = None,
) -> tuple[str, str | None, dict]:
    if not side:
        return "watch", None, {}
    if not policy or not policy.get("source"):
        return "watch", None, {}
    key = _prop_reopen_bucket_key(stat, side, line, price, bookmaker_key)
    if policy.get("force_reopen_all"):
        return ("watch" if policy.get("research_only") else "bankroll"), key, {}
    ladder = policy.get("ladder_buckets") or {}
    record = ladder.get(key)
    if record:
        return str(record.get("ladder_tier") or "watch"), key, record
    reopened = policy.get("reopen_buckets") or {}
    if key in reopened:
        record = reopened[key] or {}
        return str(record.get("ladder_tier") or "bankroll"), key, record
    legacy_key = _prop_reopen_bucket_key_legacy(stat, side, line, price, bookmaker_key)
    if legacy_key in reopened:
        record = reopened[legacy_key] or {}
        return str(record.get("ladder_tier") or "bankroll"), key, record
    return "watch", key, {}


def _is_tail_alt_over(stat: str, side: str | None, line: Optional[float]) -> bool:
    if side != "over" or line is None:
        return False
    try:
        line_v = float(line)
    except (TypeError, ValueError):
        return False
    if stat == "batter_hits":
        return line_v >= 2.5
    if stat == "batter_total_bases":
        return line_v >= 3.5
    if stat == "batter_home_runs":
        return line_v >= 1.5
    return False


def _signed_probability_edge_for_side(
    p_over: Optional[float],
    side: str,
    ld: Optional[Dict],
) -> Optional[float]:
    if p_over is None:
        return None
    try:
        p = max(0.0, min(1.0, float(p_over)))
    except Exception:
        return None
    line_data = ld or {}
    if side == "over":
        be = _american_to_implied_prob(line_data.get("over_price")) or _BREAKEVEN_PROB
        return max(0.0, p - be)
    if side == "under":
        be = _american_to_implied_prob(line_data.get("under_price")) or _BREAKEVEN_PROB
        return -max(0.0, (1.0 - p) - be)
    return None


def _clean_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clip_probability(value) -> Optional[float]:
    v = _clean_float_or_none(value)
    if v is None:
        return None
    return max(1e-6, min(1.0 - 1e-6, v))


def _no_vig_side_probability(*, side: str, price, over_price, under_price) -> Optional[float]:
    raw = _american_to_implied_prob(price)
    over_raw = _american_to_implied_prob(over_price)
    under_raw = _american_to_implied_prob(under_price)
    if over_raw is not None and under_raw is not None and over_raw + under_raw > 0:
        if side == "over":
            return over_raw / (over_raw + under_raw)
        if side == "under":
            return under_raw / (over_raw + under_raw)
    return raw


def _walk_forward_policy_record(
    policy: dict | None,
    *,
    market: str,
    side: str,
    line: Optional[float],
    price=None,
    bookmaker_key: Optional[str] = None,
) -> tuple[dict | None, str | None]:
    if not policy or side not in {"over", "under"}:
        return None, None
    line_surface = _cal_prop_line_surface(market, side, line)
    line_bucket = _cal_prop_line_bucket(market, line)
    price_bucket_value = _cal_price_bucket(price)
    book = str(bookmaker_key or "*").lower()
    lookups = [
        ("exact_bucket", f"{market}|{side}|{line_surface}|{line_bucket}|{price_bucket_value}|{book}"),
        ("line_surface", f"{market}|{side}|{line_surface}"),
        ("market_side", f"{market}|{side}"),
    ]
    for level, key in lookups:
        rec = (policy.get(level) or {}).get(key)
        if rec:
            return rec, f"{level}:{key}"
    return None, None


def _apply_walk_forward_probability_policy(
    p_side: Optional[float],
    policy: dict | None,
    *,
    market: str,
    side: str,
    line: Optional[float],
    price,
    over_price,
    under_price,
    bookmaker_key: Optional[str] = None,
) -> tuple[Optional[float], Optional[str]]:
    model_prob = _clip_probability(p_side)
    if model_prob is None:
        return p_side, None
    rec, key = _walk_forward_policy_record(
        policy,
        market=market,
        side=side,
        line=line,
        price=price,
        bookmaker_key=bookmaker_key,
    )
    if not rec:
        return model_prob, None
    variant = str(rec.get("variant") or "model_only")
    market_prob = _clip_probability(_no_vig_side_probability(
        side=side,
        price=price,
        over_price=over_price,
        under_price=under_price,
    ))
    if market_prob is None:
        return model_prob, None
    if variant == "market_no_vig":
        return market_prob, key
    if variant == "walk_forward_blend":
        weight = _clean_float_or_none(rec.get("model_weight"))
        if weight is None:
            return model_prob, None
        weight = max(0.0, min(1.0, weight))
        return _clip_probability(weight * model_prob + (1.0 - weight) * market_prob), key
    return model_prob, None


def _apply_prop_side_recalibration(
    *,
    stat: str,
    line: Optional[float],
    raw_p_over: Optional[float],
    model_family: str,
    ld: Optional[Dict],
    recalibrators: dict,
    betting_layer: dict | None = None,
    market_side_priors: dict | None = None,
    apply_market_side_priors: bool = False,
    market_side_prior_max_blend: float = 0.35,
    walk_forward_policy: dict | None = None,
    opportunity_features: Optional[Dict[str, float]] = None,
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """Return calibrated P(over), signed price-aware edge, and calibrator key."""
    if raw_p_over is None:
        return None, None, None
    try:
        raw_p = max(1e-6, min(1.0 - 1e-6, float(raw_p_over)))
    except Exception:
        return raw_p_over, _signed_probability_edge(raw_p_over), None

    line_data = ld or {}
    line_bucket = _cal_prop_line_bucket(stat, line)
    family = str(model_family or "unknown")

    p_over_side, key_over = apply_side_calibrator(
        raw_p,
        recalibrators,
        market=stat,
        side="over",
        line_bucket=line_bucket,
        price_bucket_value=_cal_price_bucket(line_data.get("over_price")),
        model_family=family,
    )
    if apply_market_side_priors:
        p_over_side, prior_key_over = apply_prop_market_side_prior(
            p_over_side,
            market_side_priors,
            market=stat,
            side="over",
            line=line,
            price=line_data.get("over_price"),
            over_price=line_data.get("over_price"),
            under_price=line_data.get("under_price"),
            bookmaker_key=line_data.get("over_bookmaker_key") or line_data.get("bookmaker_key"),
            max_blend_weight=market_side_prior_max_blend,
        )
        key_over = prior_key_over or key_over
    p_over_side, bet_key_over = apply_prop_betting_layer(
        p_over_side,
        betting_layer,
        market=stat,
        side="over",
        line=line,
        price=line_data.get("over_price"),
        over_price=line_data.get("over_price"),
        under_price=line_data.get("under_price"),
        model_family=family,
        bookmaker_key=line_data.get("over_bookmaker_key") or line_data.get("bookmaker_key"),
        opportunity=opportunity_features,
    )
    p_over_side, wf_key_over = _apply_walk_forward_probability_policy(
        p_over_side,
        walk_forward_policy,
        market=stat,
        side="over",
        line=line,
        price=line_data.get("over_price"),
        over_price=line_data.get("over_price"),
        under_price=line_data.get("under_price"),
        bookmaker_key=line_data.get("over_bookmaker_key") or line_data.get("bookmaker_key"),
    )
    p_under_side, key_under = apply_side_calibrator(
        1.0 - raw_p,
        recalibrators,
        market=stat,
        side="under",
        line_bucket=line_bucket,
        price_bucket_value=_cal_price_bucket(line_data.get("under_price")),
        model_family=family,
    )
    if apply_market_side_priors:
        p_under_side, prior_key_under = apply_prop_market_side_prior(
            p_under_side,
            market_side_priors,
            market=stat,
            side="under",
            line=line,
            price=line_data.get("under_price"),
            over_price=line_data.get("over_price"),
            under_price=line_data.get("under_price"),
            bookmaker_key=(
                line_data.get("under_bookmaker_key")
                or line_data.get("under_link_book")
                or line_data.get("bookmaker_key")
            ),
            max_blend_weight=market_side_prior_max_blend,
        )
        key_under = prior_key_under or key_under
    p_under_side, bet_key_under = apply_prop_betting_layer(
        p_under_side,
        betting_layer,
        market=stat,
        side="under",
        line=line,
        price=line_data.get("under_price"),
        over_price=line_data.get("over_price"),
        under_price=line_data.get("under_price"),
        model_family=family,
        bookmaker_key=(
            line_data.get("under_bookmaker_key")
            or line_data.get("under_link_book")
            or line_data.get("bookmaker_key")
        ),
        opportunity=opportunity_features,
    )
    p_under_side, wf_key_under = _apply_walk_forward_probability_policy(
        p_under_side,
        walk_forward_policy,
        market=stat,
        side="under",
        line=line,
        price=line_data.get("under_price"),
        over_price=line_data.get("over_price"),
        under_price=line_data.get("under_price"),
        bookmaker_key=(
            line_data.get("under_bookmaker_key")
            or line_data.get("under_link_book")
            or line_data.get("bookmaker_key")
        ),
    )
    key_over = wf_key_over or bet_key_over or key_over
    key_under = wf_key_under or bet_key_under or key_under

    cands: list[tuple[float, str, float, str | None]] = []
    ev_over = _ev_per_unit(p_over_side, line_data.get("over_price"))
    if ev_over is not None and line_data.get("over_link"):
        cands.append((float(ev_over), "over", float(p_over_side), key_over))
    ev_under = _ev_per_unit(p_under_side, line_data.get("under_price"))
    if ev_under is not None and line_data.get("under_link"):
        cands.append((float(ev_under), "under", float(p_under_side), key_under))

    if not cands:
        return raw_p, _signed_probability_edge(raw_p), None

    best_ev, side, p_side, key = max(cands, key=lambda item: item[0])
    p_over = p_side if side == "over" else 1.0 - p_side
    if best_ev <= 0:
        return p_over, 0.0, key
    return p_over, _signed_probability_edge_for_side(p_over, side, line_data), key


# ─────────────────────────────────────────────────────────────────────────────
# Regression / CLF agreement gate
# ─────────────────────────────────────────────────────────────────────────────

# Approximate walk-forward MAE per stat (used to scale disagreement).
# When regression disagrees with CLF direction by > 1 MAE we start blending CLF
# toward 0.5; full blend (CLF → 0.5) occurs at 2 MAE disagreement.
_PROP_MAE: dict[str, float] = {
    "pitcher_strikeouts": 1.73,
    "batter_hits":        0.68,
    "batter_total_bases": 1.30,
    "batter_home_runs":   0.20,
}


def _apply_regression_gate(
    clf_prob: float,
    pred_regression: float,
    book_line: float,
    stat: str,
    blend_mae_threshold: float = 1.0,
) -> float:
    """Blend CLF prob toward 0.5 when regression strongly disagrees on direction.

    If CLF says OVER (>0.5) but regression predicts meaningfully under the line
    (or vice versa), we dampen the CLF signal proportional to how strongly the
    regression disagrees, measured in units of per-stat MAE.

    No effect when CLF and regression agree on direction.
    """
    mae = _PROP_MAE.get(stat, 1.0)
    regression_edge = pred_regression - book_line
    clf_edge = clf_prob - 0.5
    if clf_edge * regression_edge >= 0:
        return clf_prob  # same direction — no modification
    disagree_n_mae = abs(regression_edge) / max(mae, 0.01)
    blend = min(max((disagree_n_mae - blend_mae_threshold) / blend_mae_threshold, 0.0), 1.0)
    return clf_prob + blend * (0.5 - clf_prob)


# ─────────────────────────────────────────────────────────────────────────────
# Bias correction
# ─────────────────────────────────────────────────────────────────────────────

# Stats where additive bias correction is applied.
# K and walks are excluded: near-zero bias and insufficient sample respectively.
_BIAS_CORRECT_STATS = frozenset({"batter_hits", "batter_total_bases", "batter_home_runs"})


def _load_bias_corrections(conn, lookback_days: int = 90) -> dict[str, float]:
    """Return per-stat additive correction = mean(actual - pred) over recent history.

    Queries graded predictions (actual_value filled) for *_BIAS_CORRECT_STATS only.
    Uses regression-mode rows only: classifier rows store P(over) in pred_value,
    which is not comparable to actual count outcomes.
    Falls back to 0.0 for any stat with fewer than 50 graded rows (too noisy).
    Corrections are capped at ±0.5 to guard against data anomalies.
    """
    from datetime import timedelta
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    corrections: dict[str, float] = {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT stat,
                       COUNT(*)                          AS n,
                       AVG(actual_value::float - COALESCE(pred_count, pred_value)::float) AS mean_bias
                FROM bets.mlb_prop_predictions
                WHERE over_hit IS NOT NULL
                  AND COALESCE(pred_count, pred_value) IS NOT NULL
                  AND actual_value IS NOT NULL
                  AND book_line IS NOT NULL
                  AND edge IS NOT NULL
                  AND (
                      edge_type = 'count'
                      OR (
                          edge_type IS NULL
                          AND ABS((pred_value::float - book_line::float) - edge::float) <= 0.02
                      )
                  )
                  AND game_date_et >= %s
                  AND stat = ANY(%s)
                GROUP BY stat
                """,
                (cutoff, list(_BIAS_CORRECT_STATS)),
            )
            for stat, n, mean_bias in cur.fetchall():
                if n >= 50 and mean_bias is not None:
                    corrections[stat] = max(-0.5, min(0.5, float(mean_bias)))
                    log.info(
                        "Bias correction %s: n=%d, offset=%+.3f",
                        stat, n, corrections[stat],
                    )
    except Exception:
        log.warning("Could not load bias corrections — using 0.0 for all stats")
    return corrections


def _line_bucket_for_over_penalty(stat: str, line: float) -> str:
    if stat == "batter_total_bases":
        if line < 1.0:
            return "TB 0.5"
        if line < 2.0:
            return "TB 1.5"
        return "TB 2.5+"
    if stat == "pitcher_strikeouts":
        if line < 4.5:
            return "K <4.5"
        if line < 6.5:
            return "K 4.5-6.0"
        if line < 8.5:
            return "K 6.5-8.0"
        return "K 8.5+"
    return "other"


def _load_side_penalties(
    conn,
    lookback_days: int = 180,
    min_samples: int = 40,
    shrinkage: float = 0.5,
) -> dict[tuple[str, str, str], float]:
    """Load dynamic side-specific penalties for weak buckets with shrinkage.

    Keys are (stat, bucket, side) with side in {"over","under"} plus stat-level
    fallback (stat, "__all__", side).
    """
    from datetime import timedelta

    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    penalties: dict[tuple[str, str, str], float] = {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH base AS (
                    SELECT
                        stat,
                        book_line::float AS line,
                        CASE WHEN edge >= 0 THEN 'over' ELSE 'under' END AS side,
                        over_hit,
                        COALESCE(pred_count, pred_value)::float AS pred_value,
                        actual_value::float AS actual_value,
                        (COALESCE(pred_count, pred_value)::float - actual_value::float) AS pred_minus_actual
                    FROM bets.mlb_prop_predictions
                    WHERE game_date_et >= %s
                      AND over_hit IS NOT NULL
                      AND stat IN ('pitcher_strikeouts', 'batter_total_bases')
                      AND book_line IS NOT NULL
                      AND edge IS NOT NULL
                      AND COALESCE(pred_count, pred_value) IS NOT NULL
                      AND actual_value IS NOT NULL
                      AND (
                          edge_type = 'count'
                          OR (
                              edge_type IS NULL
                              AND ABS((pred_value::float - book_line::float) - edge::float) <= 0.02
                          )
                      )
                )
                SELECT
                    stat,
                    CASE
                        WHEN stat = 'batter_total_bases' AND line < 1.0 THEN 'TB 0.5'
                        WHEN stat = 'batter_total_bases' AND line < 2.0 THEN 'TB 1.5'
                        WHEN stat = 'batter_total_bases' THEN 'TB 2.5+'
                        WHEN stat = 'pitcher_strikeouts' AND line < 4.5 THEN 'K <4.5'
                        WHEN stat = 'pitcher_strikeouts' AND line < 6.5 THEN 'K 4.5-6.0'
                        WHEN stat = 'pitcher_strikeouts' AND line < 8.5 THEN 'K 6.5-8.0'
                        ELSE 'K 8.5+'
                    END AS bucket,
                    side,
                    COUNT(*) AS n,
                    AVG(CASE WHEN (side = 'over' AND over_hit) OR (side = 'under' AND NOT over_hit) THEN 1.0 ELSE 0.0 END) AS side_win_rate,
                    AVG(pred_minus_actual) AS mean_pred_minus_actual
                FROM base
                GROUP BY 1, 2, 3
                """,
                (cutoff,),
            )
            rows = cur.fetchall()

            # Stat-level fallback
            cur.execute(
                """
                SELECT
                    stat,
                    CASE WHEN edge >= 0 THEN 'over' ELSE 'under' END AS side,
                    COUNT(*) AS n,
                    AVG(CASE WHEN (edge >= 0 AND over_hit) OR (edge < 0 AND NOT over_hit) THEN 1.0 ELSE 0.0 END) AS side_win_rate,
                    AVG(COALESCE(pred_count, pred_value)::float - actual_value::float) AS mean_pred_minus_actual
                FROM bets.mlb_prop_predictions
                WHERE game_date_et >= %s
                  AND over_hit IS NOT NULL
                  AND stat IN ('pitcher_strikeouts', 'batter_total_bases')
                  AND COALESCE(pred_count, pred_value) IS NOT NULL
                  AND actual_value IS NOT NULL
                  AND book_line IS NOT NULL
                  AND edge IS NOT NULL
                  AND (
                      edge_type = 'count'
                      OR (
                          edge_type IS NULL
                          AND ABS((pred_value::float - book_line::float) - edge::float) <= 0.02
                      )
                  )
                GROUP BY 1, 2
                """,
                (cutoff,),
            )
            fallback_rows = cur.fetchall()

        def _make_penalty(side: str, n: int, win_rate: float, pred_minus_actual: float) -> float:
            if n < min_samples or win_rate is None:
                return 0.0
            w = float(win_rate)
            pma = float(pred_minus_actual or 0.0)
            if w >= 0.52:
                return 0.0
            # over: pred too high (pma > 0) hurts over; under: pred too low (pma < 0) hurts under
            directional = pma if side == "over" else -pma
            raw = max(0.0, directional, (0.52 - w) * 2.0)
            return float(max(0.03, min(0.75, shrinkage * raw)))

        for stat, bucket, side, n, wr, pma in rows:
            pen = _make_penalty(str(side), int(n or 0), float(wr or 0.0), float(pma or 0.0))
            if pen > 0:
                penalties[(stat, bucket, str(side))] = pen
                log.info(
                    "Side penalty %s/%s/%s: n=%d wr=%.3f pma=%+.3f pen=%.3f",
                    stat, bucket, side, n, wr, pma, pen
                )
        for stat, side, n, wr, pma in fallback_rows:
            pen = _make_penalty(str(side), int(n or 0), float(wr or 0.0), float(pma or 0.0))
            if pen > 0:
                penalties[(stat, "__all__", str(side))] = pen
                log.info("Side penalty %s/all/%s: n=%d wr=%.3f pma=%+.3f pen=%.3f", stat, side, n, wr, pma, pen)
    except Exception:
        log.warning("Could not load side penalties — using none", exc_info=True)

    return penalties


def _side_penalty_for_line(
    stat: str,
    line: float | None,
    side: str,
    penalties: dict[tuple[str, str, str], float],
) -> float:
    if line is None:
        return 0.0
    bucket = _line_bucket_for_over_penalty(stat, float(line))
    return float(
        penalties.get((stat, bucket, side), penalties.get((stat, "__all__", side), 0.0))
    )


def _apply_count_side_penalty(
    stat: str,
    pred: float,
    line: float | None,
    penalties: dict[tuple[str, str, str], float],
) -> tuple[float, float]:
    """Move weak side predictions toward the line without flipping sides."""
    if line is None:
        return float(pred), 0.0
    edge = float(pred) - float(line)
    side = "over" if edge >= 0 else "under"
    penalty = _side_penalty_for_line(stat, line, side, penalties)
    if penalty <= 0:
        return float(pred), 0.0
    if side == "over":
        return max(float(line), float(pred) - penalty), penalty
    return min(float(line), float(pred) + penalty), penalty


def _weak_sides_for_line(
    stat: str,
    line: float | None,
    penalties: dict[tuple[str, str, str], float],
) -> dict[str, bool]:
    return {
        "over": _side_penalty_for_line(stat, line, "over", penalties) > 0,
        "under": _side_penalty_for_line(stat, line, "under", penalties) > 0,
    }


def _blocked_sides_for_row(row: Dict, stat: str) -> set[str]:
    sides = ((row.get("weak_prop_sides") or {}).get(stat) or {})
    return {side for side, blocked in sides.items() if blocked}


def _kelly_from_price(
    p_win: Optional[float],
    price: Optional[float],
    *,
    max_kelly: float = 0.05,
) -> float:
    if p_win is None or price is None:
        return 0.0
    try:
        p = float(p_win)
        odds = float(price)
    except (TypeError, ValueError):
        return 0.0
    if not 0 < p < 1:
        return 0.0
    b = odds / 100.0 if odds > 0 else 100.0 / abs(odds)
    if b <= 0:
        return 0.0
    q = 1.0 - p
    return max(0.0, min(float(max_kelly), ((b * p) - q) / b))


def _prop_count_threshold(cfg: PredictConfig, stat: str, side: str | None) -> float:
    """Return side-specific count threshold when configured."""
    side = (side or "").lower()
    if stat == "pitcher_strikeouts":
        if side == "over" and cfg.threshold_strikeouts_over is not None:
            return float(cfg.threshold_strikeouts_over)
        if side == "under" and cfg.threshold_strikeouts_under is not None:
            return float(cfg.threshold_strikeouts_under)
        return float(cfg.threshold_strikeouts)
    if stat == "batter_total_bases":
        if side == "over" and cfg.threshold_total_bases_over is not None:
            return float(cfg.threshold_total_bases_over)
        if side == "under" and cfg.threshold_total_bases_under is not None:
            return float(cfg.threshold_total_bases_under)
        return float(cfg.threshold_total_bases)
    if stat == "batter_hits":
        return float(cfg.threshold_hits)
    if stat == "batter_home_runs":
        return (
            float(cfg.threshold_home_runs_over)
            if side == "over"
            else float(cfg.threshold_home_runs_under)
        )
    return float("inf")


def _prop_signal_block_reason(
    *,
    stat: str,
    side: str | None,
    edge: Optional[float],
    edge_type: Optional[str],
    ev: Optional[float] = None,
    cfg: PredictConfig,
) -> str:
    if side is None:
        return "missing_side"
    if edge is None:
        return "missing_edge"
    try:
        if ev is not None and not pd.isna(ev) and float(ev) >= float(cfg.min_ev):
            return ""
    except Exception:
        pass
    try:
        e = float(edge)
    except (TypeError, ValueError):
        return "missing_edge"
    if edge_type == "probability":
        threshold = float(cfg.threshold_clf)
        name = "threshold_clf"
        passes = abs(e) >= threshold
    else:
        threshold = _prop_count_threshold(cfg, stat, side)
        if stat == "pitcher_strikeouts":
            name = f"threshold_strikeouts_{side}"
        elif stat == "batter_total_bases":
            name = f"threshold_total_bases_{side}"
        elif stat == "batter_hits":
            name = "threshold_hits"
        elif stat == "batter_home_runs":
            name = f"threshold_home_runs_{side}"
        else:
            name = "threshold_unknown"
        passes = e >= threshold if stat == "batter_home_runs" and side == "over" else abs(e) >= threshold
    if threshold >= 900.0:
        return f"threshold_disabled:{name}"
    if not passes:
        return f"below_edge_threshold:{name}"
    return ""


def _prop_db_signal_allowed(
    *,
    stat: str,
    side: str | None,
    edge: Optional[float],
    edge_type: Optional[str],
    ev: Optional[float],
    cfg: PredictConfig,
) -> bool:
    if side is None:
        return False
    if edge is None:
        return False
    try:
        e = float(edge)
    except (TypeError, ValueError):
        return False
    try:
        if ev is not None and not pd.isna(ev) and float(ev) >= float(cfg.min_ev):
            return True
    except Exception:
        pass
    if edge_type == "probability":
        return abs(e) >= cfg.threshold_clf
    threshold = _prop_count_threshold(cfg, stat, side)
    if stat == "batter_home_runs" and side == "over":
        return e >= threshold
    return abs(e) >= threshold


def _prop_bankroll_assessment(
    *,
    stat: str,
    side: str | None,
    cfg: PredictConfig,
    line: Optional[float] = None,
    bet_price: Optional[float],
    ev: Optional[float],
    kelly_fraction: Optional[float],
    bookmaker_key: Optional[str] = None,
    blocked_sides: Optional[set[str]] = None,
    bucket_reopen_policy: Optional[dict] = None,
    has_signal: Optional[bool] = None,
    no_signal_reason: str = "no_model_edge",
) -> BankrollAssessment:
    """Label a prop signal for shadow-bankroll readiness."""
    hard: list[str] = []
    soft: list[str] = []
    ladder_tier, _ladder_key, _ladder_record = _prop_bucket_ladder_tier(
        bucket_reopen_policy,
        stat=stat,
        side=side,
        line=line,
        price=bet_price,
        bookmaker_key=bookmaker_key,
    )

    if side and side in (blocked_sides or set()):
        soft.append(f"weak_{side}_bucket")
    if _is_tail_alt_over(stat, side, line):
        tail_reopened, _tail_key = _prop_bucket_is_reopened(
            bucket_reopen_policy,
            stat=stat,
            side=side,
            line=line,
            price=bet_price,
            bookmaker_key=bookmaker_key,
        )
        if not tail_reopened:
            hard.append("tail_alt_not_reopened")
    if cfg.enforce_prop_bucket_reopen:
        reopened, _bucket_key = _prop_bucket_is_reopened(
            bucket_reopen_policy,
            stat=stat,
            side=side,
            line=line,
            price=bet_price,
            bookmaker_key=bookmaker_key,
        )
        if not reopened:
            hard.append("bucket_not_reopened")
    if bucket_reopen_policy and bucket_reopen_policy.get("research_only"):
        hard.append("research_reopen_only")
    if side == "under" and stat in cfg.fd_over_only:
        soft.append("unbookable_under")
    if stat == "batter_home_runs":
        soft.append("hr_longshot_variance")

    clean_price = None
    if bet_price is not None:
        try:
            clean_price = None if pd.isna(bet_price) else float(bet_price)
        except Exception:
            clean_price = None

    if clean_price is None:
        hard.append("missing_price")
    elif clean_price < cfg.bankroll_max_lay_price:
        hard.append("heavy_juice")

    if ev is None:
        soft.append("missing_ev")
    elif ev < cfg.min_ev:
        soft.append("ev_below_min")

    if kelly_fraction is None or kelly_fraction <= 0:
        soft.append("zero_kelly")

    assessment = assess_bankroll_layer(
        has_signal=(side is not None if has_signal is None else has_signal),
        hard_blocks=hard,
        soft_warnings=soft,
        kelly_fraction=kelly_fraction,
        max_stake_pct=cfg.bankroll_max_stake_pct,
        no_signal_reason=no_signal_reason,
    )
    if not assessment.candidate:
        if ladder_tier == "watch" and side is not None:
            return BankrollAssessment(
                tier="watch",
                candidate=False,
                reasons=assessment.reasons,
                stake_pct=0.0,
                stake_usd=0.0,
            )
        return assessment
    if ladder_tier == "micro":
        stake_usd = max(0.0, float(cfg.bankroll_micro_stake_usd))
        reference = max(1.0, float(cfg.bankroll_reference_usd))
        return BankrollAssessment(
            tier="micro",
            candidate=True,
            reasons="",
            stake_pct=stake_usd / reference,
            stake_usd=stake_usd,
        )
    if ladder_tier == "starter":
        return BankrollAssessment(
            tier="starter",
            candidate=True,
            reasons="",
            stake_pct=max(0.0, float(cfg.bankroll_starter_stake_pct)),
            stake_usd=0.0,
        )
    return BankrollAssessment(
        tier="bankroll",
        candidate=True,
        reasons="",
        stake_pct=assessment.stake_pct,
        stake_usd=0.0,
    )


def _prop_bankroll_from_pick(
    *,
    stat: str,
    side_label: str,
    cfg: PredictConfig,
    ld: Dict,
    p_over: Optional[float],
    edge: Optional[float],
    sigma: Optional[float],
    blocked_sides: Optional[set[str]] = None,
    bucket_reopen_policy: Optional[dict] = None,
) -> BankrollAssessment:
    side = "over" if side_label == "O" else "under"
    price = ld.get("over_price") if side == "over" else ld.get("under_price")
    line = ld.get("line")
    p_side = None
    if p_over is not None:
        p_side = float(p_over) if side == "over" else 1.0 - float(p_over)
    ev = _ev_per_unit(p_side, price)
    kelly = _kelly_from_price(p_side, price)
    if kelly <= 0 and edge is not None:
        kelly = _kelly_prop(abs(float(edge)), sigma or 0.5)
    no_signal_reason = _prop_signal_block_reason(
        stat=stat,
        side=side,
        edge=edge,
        edge_type="count",
        ev=ev,
        cfg=cfg,
    )
    has_signal = not no_signal_reason
    return _prop_bankroll_assessment(
        stat=stat,
        side=side,
        cfg=cfg,
        line=line,
        bet_price=price,
        ev=ev,
        kelly_fraction=kelly,
        bookmaker_key=(
            ld.get("over_bookmaker_key")
            if side == "over"
            else (ld.get("under_bookmaker_key") or ld.get("under_link_book"))
        ) or ld.get("bookmaker_key"),
        blocked_sides=blocked_sides,
        bucket_reopen_policy=bucket_reopen_policy,
        has_signal=has_signal,
        no_signal_reason=no_signal_reason,
    )


def _append_bankroll_reason(existing: str, reason: str) -> str:
    parts = [p.strip() for p in (existing or "").split(";") if p.strip()]
    if reason not in parts:
        parts.append(reason)
    return "; ".join(parts)


def _downgrade_for_daily_cap(assessment: BankrollAssessment) -> BankrollAssessment:
    return BankrollAssessment(
        tier="paper",
        candidate=False,
        reasons=_append_bankroll_reason(assessment.reasons, "daily_exposure_cap"),
        stake_pct=0.0,
        stake_usd=0.0,
    )


def _apply_assessment_daily_cap(
    assessment: BankrollAssessment,
    *,
    used_exposure_pct: float,
    cfg: PredictConfig,
) -> tuple[BankrollAssessment, float]:
    if not assessment.candidate:
        return assessment, used_exposure_pct
    stake = float(assessment.stake_pct or 0.0)
    if used_exposure_pct + stake <= cfg.bankroll_max_daily_exposure_pct + 1e-12:
        return assessment, used_exposure_pct + stake
    return _downgrade_for_daily_cap(assessment), used_exposure_pct


def _cap_bankroll_output(items: List[Dict], cfg: PredictConfig) -> None:
    used = 0.0
    for item in items:
        assessment = item.get("bankroll")
        if not isinstance(assessment, BankrollAssessment):
            continue
        capped, used = _apply_assessment_daily_cap(
            assessment,
            used_exposure_pct=used,
            cfg=cfg,
        )
        item["bankroll"] = capped


def _cap_prop_db_rows(
    rows: List[Dict],
    cfg: PredictConfig,
    *,
    existing_exposure_pct: float = 0.0,
    existing_pick_keys: set[str] | None = None,
    existing_risk_slots: set[str] | None = None,
    prop_lines=None,
    require_locked: bool = False,
) -> List[Dict]:
    """Apply daily bankroll exposure cap to persisted prop candidates."""
    candidate_indexes = [
        i for i, row in enumerate(rows)
        if bool(row.get("bankroll_candidate"))
    ]
    pick_keys = set(existing_pick_keys or set())
    risk_slots = set(existing_risk_slots or set())
    initially_occupied_slots = set(risk_slots)

    def _score(idx: int) -> tuple[bool, float, float]:
        row = rows[idx]
        ev = row.get("ev")
        edge = row.get("edge")
        try:
            ev_score = float(ev) if ev is not None else -999.0
        except (TypeError, ValueError):
            ev_score = -999.0
        try:
            edge_score = abs(float(edge)) if edge is not None else 0.0
        except (TypeError, ValueError):
            edge_score = 0.0
        return prop_bankroll_pick_key(row, prop_lines) in pick_keys, ev_score, edge_score

    used = max(float(existing_exposure_pct or 0.0), 0.0)
    seen_player_stat: set[tuple] = set()
    for idx in sorted(candidate_indexes, key=_score, reverse=True):
        row = rows[idx]
        player_stat_key = (
            row.get("game_date_et"),
            row.get("game_slug"),
            row.get("player_id"),
            row.get("stat"),
        )
        pick_key = prop_bankroll_pick_key(row, prop_lines)
        risk_slot = prop_bankroll_risk_slot(row)
        if pick_key and pick_key in pick_keys:
            seen_player_stat.add(player_stat_key)
            continue
        block_reason = None
        if not pick_key or not risk_slot:
            block_reason = "missing_locked_pick_key"
        elif risk_slot in initially_occupied_slots:
            block_reason = "already_locked_player_stat"
        elif require_locked:
            block_reason = "not_in_locked_bankroll_ledger"
        if player_stat_key in seen_player_stat:
            block_reason = block_reason or "duplicate_offer_lower_rank"
        seen_player_stat.add(player_stat_key)
        stake = float(row.get("stake_pct") or 0.0)
        if (
            block_reason is None
            and used + stake <= cfg.bankroll_max_daily_exposure_pct + 1e-12
        ):
            used += stake
            pick_keys.add(pick_key)
            risk_slots.add(risk_slot)
            continue
        row["bankroll_candidate"] = False
        row["bankroll_tier"] = "paper"
        row["bankroll_reasons"] = _append_bankroll_reason(
            row.get("bankroll_reasons") or "",
            block_reason or "daily_exposure_cap",
        )
        row["stake_pct"] = 0.0
        row["stake_usd"] = 0.0
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# DB table
# ─────────────────────────────────────────────────────────────────────────────

def _apply_prop_shadow_selector_gate(rows: List[Dict], cfg: PredictConfig) -> List[Dict]:
    """Attach residual/CLV-aware selector fields and fail real-money props closed."""
    if not rows:
        return rows
    try:
        selector_ctx = PropSelectorContext(cfg.model_dir)
    except Exception:
        log.exception("Could not load prop shadow selector artifacts; blocking prop bankroll rows")
        selector_ctx = None
    selector_cfg = PropSelectorConfig(
        pg_dsn=cfg.pg_dsn,
        report_date=cfg.et_date,
        model_dir=cfg.model_dir,
        min_ev=cfg.min_ev,
    )
    for row in rows:
        selector = None
        if selector_ctx is not None:
            try:
                selector = score_prop_shadow_row(row, ctx=selector_ctx, cfg=selector_cfg)
            except Exception:
                log.exception("Could not score prop selector row %s", row.get("prediction_key"))
        if selector:
            row["selector_score"] = selector.get("selector_score")
            row["selector_tier"] = selector.get("selector_tier")
            row["selector_real_candidate"] = selector.get("selector_real_candidate")
            row["selector_prob_side"] = selector.get("selector_prob_side")
            row["selector_ev"] = selector.get("selector_ev")
            row["selector_reasons"] = "; ".join(selector.get("selector_reasons") or [])
        if bool(row.get("bankroll_candidate")) and not (selector or {}).get("selector_real_candidate"):
            row["bankroll_candidate"] = False
            row["bankroll_tier"] = "watch"
            row["bankroll_reasons"] = _append_bankroll_reason(
                row.get("bankroll_reasons") or "",
                "selector_clv_residual_block",
            )
            row["stake_pct"] = 0.0
            row["stake_usd"] = 0.0
    return rows


def _downgrade_prop_row(row: Dict, reason: str, *, tier: str = "watch") -> None:
    row["bankroll_candidate"] = False
    row["bankroll_tier"] = tier
    row["bankroll_reasons"] = _append_bankroll_reason(
        row.get("bankroll_reasons") or "",
        reason,
    )
    row["stake_pct"] = 0.0
    row["stake_usd"] = 0.0


def _apply_prop_real_money_kill_switch(rows: List[Dict], cfg: PredictConfig) -> List[Dict]:
    """Apply global and row-level real-money kill switches to prop candidates."""
    if not rows or not getattr(cfg, "enforce_prop_real_money_kill_switch", True):
        return rows
    state = load_prop_kill_switch_state(
        cfg.model_dir,
        file_name=cfg.real_money_kill_switch_file,
        max_age_hours=cfg.real_money_kill_switch_max_age_hours,
    )
    blockers = [str(reason) for reason in (state.get("blockers") or []) if str(reason)]
    active = bool(state.get("active")) or str(state.get("status") or "").lower() == "disabled"
    global_reasons = ["real_money_kill_switch_active", *blockers[:6]] if active else []
    for row in rows:
        if not bool(row.get("bankroll_candidate")):
            continue
        reasons = list(global_reasons)
        if row.get("minimum_acceptable_price") is None:
            reasons.append("minimum_acceptable_price_missing")
        if not row.get("bet_link"):
            reasons.append("bet_link_missing")
        if not reasons:
            continue
        row["kill_switch_status"] = state.get("status")
        row["kill_switch_blockers"] = "; ".join(blockers)
        for reason in dict.fromkeys(reasons):
            _downgrade_prop_row(row, reason)
    return rows


def _round_or_none(value: Optional[float], ndigits: int) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return round(float(value), ndigits)


def _clean_int_or_none(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _prop_prediction_key(
    *,
    game_date_et: date,
    game_slug: str,
    player_id: int,
    stat: str,
    side: Optional[str],
    book_line: Optional[float],
    bookmaker_key: Optional[str],
    prop_offer_id: Optional[int],
) -> str:
    parts = [
        "mlb",
        "prop_prediction",
        game_date_et,
        game_slug,
        player_id,
        stat,
        side or "",
        _round_or_none(book_line, 3),
        (bookmaker_key or "").lower(),
        prop_offer_id or "",
    ]
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def _prop_offers_for_key(
    prop_lines: Dict,
    norm: str,
    stat: str,
    game_row: Optional[Dict] = None,
) -> List[Dict]:
    ld = prop_lines.get((norm, stat)) or {}
    offers = ld.get("offers") or []
    out = [dict(o) for o in offers if o.get("side") in {"over", "under"} and o.get("line") is not None]
    if out and game_row is not None:
        out = filter_prop_offers_for_game(
            out,
            team_abbr=game_row.get("team_abbr"),
            opponent_abbr=game_row.get("opponent_abbr"),
            start_ts_utc=game_row.get("start_ts_utc"),
        )
    if offers or ld.get("line") is None:
        return out
    line = ld.get("line")
    for side in ("over", "under"):
        price = ld.get(f"{side}_price")
        link = ld.get(f"{side}_link")
        if price is None and not link:
            continue
        out.append({
            "id": ld.get(f"{side}_offer_id"),
            "source_row_id": ld.get(f"{side}_offer_source_row_id"),
            "player_name_norm": norm,
            "stat": stat,
            "side": side,
            "line": line,
            "price": price,
            "link": link,
            "bookmaker_key": (
                ld.get(f"{side}_bookmaker_key")
                or (ld.get("under_link_book") if side == "under" else None)
                or ld.get("bookmaker_key")
            ),
        })
    return out


def _prop_line_data_for_row(
    prop_lines: Dict,
    norm: str,
    stat: str,
    game_row: Dict,
) -> tuple[Optional[Dict], List[Dict]]:
    offers = _prop_offers_for_key(prop_lines, norm, stat, game_row)
    if not offers:
        return None, []
    line_map = build_prop_line_map(offers)
    return line_map.get((norm, stat)), offers


def _offer_line_data(offer: Dict, sibling_offers: Optional[List[Dict]] = None) -> Dict:
    """Return legacy line-data shape for one locked side-level offer.

    The selected side gets a link; the opposite side may keep its price for
    calibration/no-vig context but not its link, so side selection stays locked
    to the offer being evaluated.
    """
    side = (offer.get("side") or "").lower()
    line = _round_or_none(offer.get("line"), 3)
    book = (offer.get("bookmaker_key") or "").lower() or None
    ld = {
        "line": line,
        "bookmaker_key": book,
        "selected_offer_side": side,
        "selected_offer_id": _clean_int_or_none(offer.get("id")),
        "selected_offer_source_row_id": _clean_int_or_none(offer.get("source_row_id")),
        "selected_offer_link": offer.get("link"),
        "selected_offer_price": offer.get("price"),
        "offer_source": "features.mlb_prop_offer_links",
    }
    for candidate in sibling_offers or [offer]:
        candidate_side = (candidate.get("side") or "").lower()
        candidate_book = (candidate.get("bookmaker_key") or "").lower()
        candidate_line = _round_or_none(candidate.get("line"), 3)
        if candidate_side not in {"over", "under"}:
            continue
        if candidate_book != book or candidate_line != line:
            continue
        prefix = "over" if candidate_side == "over" else "under"
        ld[f"{prefix}_price"] = candidate.get("price")
        ld[f"{prefix}_bookmaker_key"] = candidate_book
        ld[f"{prefix}_offer_id"] = _clean_int_or_none(candidate.get("id"))
        ld[f"{prefix}_offer_source_row_id"] = _clean_int_or_none(candidate.get("source_row_id"))
        if candidate_side == side:
            ld[f"{prefix}_link"] = candidate.get("link")
            if candidate_side == "under":
                ld["under_link_book"] = candidate_book
    if side == "over":
        ld.setdefault("over_price", offer.get("price"))
        ld.setdefault("over_link", offer.get("link"))
        ld.setdefault("over_bookmaker_key", book)
        ld.setdefault("over_offer_id", _clean_int_or_none(offer.get("id")))
        ld.setdefault("over_offer_source_row_id", _clean_int_or_none(offer.get("source_row_id")))
    elif side == "under":
        ld.setdefault("under_price", offer.get("price"))
        ld.setdefault("under_link", offer.get("link"))
        ld.setdefault("under_bookmaker_key", book)
        ld.setdefault("under_link_book", book)
        ld.setdefault("under_offer_id", _clean_int_or_none(offer.get("id")))
        ld.setdefault("under_offer_source_row_id", _clean_int_or_none(offer.get("source_row_id")))
    exact_side_offers = [
        candidate for candidate in (sibling_offers or [offer])
        if str(candidate.get("side") or "").lower() == side
        and _same_line(candidate.get("line"), line)
        and candidate.get("price") is not None
    ]
    implied = [
        value for value in (_american_to_implied_prob(candidate.get("price")) for candidate in exact_side_offers)
        if value is not None
    ]
    selected_implied = _american_to_implied_prob(offer.get("price"))
    open_implied = _american_to_implied_prob(offer.get("open_price"))
    if open_implied is not None and bool(offer.get("open_exact_line")):
        ld["open_price_implied"] = open_implied
        ld["open_to_lock_prob_move"] = (
            selected_implied - open_implied if selected_implied is not None else None
        )
    open_line = _round_or_none(offer.get("open_line"), 3)
    if open_line is not None and line is not None:
        line_move = float(line) - float(open_line)
        ld["open_to_lock_line_move_side"] = line_move if side == "over" else -line_move
    if implied:
        consensus = float(np.mean(implied))
        ld.update({
            "lock_price_implied": selected_implied,
            "consensus_prob_at_lock": consensus,
            "consensus_price_dispersion": float(np.std(implied, ddof=0)),
            "consensus_book_count": float(len({str(candidate.get('bookmaker_key') or '').lower() for candidate in exact_side_offers})),
            "book_lead_lag_prob": (selected_implied - consensus) if selected_implied is not None else None,
            "best_consensus_prob": float(min(implied)),
            "worst_consensus_prob": float(max(implied)),
        })
    ld["lock_offer_available"] = 1.0 if offer.get("price") is not None else 0.0
    ld["lock_same_book_pair_available"] = 1.0 if ld.get("over_price") is not None and ld.get("under_price") is not None else 0.0
    fetched = pd.to_datetime(offer.get("fetched_at_utc"), utc=True, errors="coerce")
    commence = pd.to_datetime(offer.get("commence_time_utc"), utc=True, errors="coerce")
    now_utc = pd.Timestamp.now(tz="UTC")
    if pd.notna(fetched):
        ld["lock_price_age_minutes"] = max(0.0, float((now_utc - fetched).total_seconds() / 60.0))
    if pd.notna(fetched) and pd.notna(commence):
        ld["minutes_to_first_pitch_at_lock"] = float((commence - fetched).total_seconds() / 60.0)
    return ld


def _same_line(a, b) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _prop_db_row(
    *,
    game_date_et: date,
    game_slug: str,
    player_id: int,
    player_name: str,
    team_abbr,
    stat: str,
    pred_value: Optional[float],
    pred_count: Optional[float],
    pred_prob_over: Optional[float],
    book_line: Optional[float],
    edge: Optional[float],
    edge_type: Optional[str],
    model_family: Optional[str],
    kelly_fraction: Optional[float],
    ld: Optional[Dict],
    forced_side: Optional[str] = None,
    cfg: Optional[PredictConfig] = None,
    blocked_sides: Optional[set[str]] = None,
    bucket_reopen_policy: Optional[dict] = None,
) -> Dict:
    """Build the persisted prop row with explicit count/probability/price fields."""
    line = float(book_line) if book_line is not None else None
    side = (forced_side or "").lower() or None
    if edge is not None and abs(float(edge)) > 1e-12:
        side = side or ("over" if float(edge) > 0 else "under")

    p_over = pred_prob_over
    if p_over is None and pred_count is not None and line is not None:
        p_over = _prob_over_from_regression(pred_count, line)
    p_side = None
    if side == "over" and p_over is not None:
        p_side = float(p_over)
    elif side == "under" and p_over is not None:
        p_side = 1.0 - float(p_over)

    line_data = ld or {}
    over_price = line_data.get("over_price")
    under_price = line_data.get("under_price")
    bet_price = over_price if side == "over" else (under_price if side == "under" else None)
    bookmaker_key = None
    if side == "over":
        bookmaker_key = line_data.get("over_bookmaker_key") or line_data.get("bookmaker_key")
    elif side == "under":
        bookmaker_key = (
            line_data.get("under_bookmaker_key")
            or line_data.get("under_link_book")
            or line_data.get("bookmaker_key")
        )
    else:
        bookmaker_key = line_data.get("bookmaker_key")
    prop_offer_id = None
    prop_offer_source_row_id = None
    bet_link = None
    if side == "over":
        prop_offer_id = line_data.get("over_offer_id") or line_data.get("selected_offer_id")
        prop_offer_source_row_id = (
            line_data.get("over_offer_source_row_id")
            or line_data.get("selected_offer_source_row_id")
        )
        bet_link = line_data.get("over_link") or line_data.get("selected_offer_link")
    elif side == "under":
        prop_offer_id = line_data.get("under_offer_id") or line_data.get("selected_offer_id")
        prop_offer_source_row_id = (
            line_data.get("under_offer_source_row_id")
            or line_data.get("selected_offer_source_row_id")
        )
        bet_link = line_data.get("under_link") or line_data.get("selected_offer_link")
    prop_offer_id = _clean_int_or_none(prop_offer_id)
    prop_offer_source_row_id = _clean_int_or_none(prop_offer_source_row_id)
    breakeven = _american_to_implied_prob(bet_price)
    ev = _ev_per_unit(p_side, bet_price)
    minimum_acceptable_price = minimum_american_price(
        p_side,
        cfg.min_ev if cfg is not None else 0.0,
    )
    kelly_for_bankroll = kelly_fraction
    if (kelly_for_bankroll is None or kelly_for_bankroll <= 0) and p_side is not None:
        kelly_for_bankroll = _kelly_from_price(p_side, bet_price)
    no_signal_reason = (
        _prop_signal_block_reason(
            stat=stat,
            side=side,
            edge=edge,
            edge_type=edge_type,
            ev=ev,
            cfg=cfg,
        )
        if cfg is not None
        else ("" if side is not None else "missing_side")
    )
    bankroll = (
        _prop_bankroll_assessment(
            stat=stat,
            side=side,
            cfg=cfg,
            line=line,
            bet_price=bet_price,
            ev=ev,
            kelly_fraction=kelly_for_bankroll,
            bookmaker_key=bookmaker_key,
            blocked_sides=blocked_sides,
            bucket_reopen_policy=bucket_reopen_policy,
            has_signal=not no_signal_reason,
            no_signal_reason=no_signal_reason,
        )
        if cfg is not None
        else assess_bankroll_layer(has_signal=side is not None, kelly_fraction=kelly_for_bankroll)
    )

    return {
        "game_date_et": game_date_et,
        "game_slug": game_slug,
        "player_id": int(player_id),
        "player_name": player_name,
        "team_abbr": team_abbr,
        "stat": stat,
        "prediction_key": _prop_prediction_key(
            game_date_et=game_date_et,
            game_slug=game_slug,
            player_id=int(player_id),
            stat=stat,
            side=side,
            book_line=line,
            bookmaker_key=bookmaker_key,
            prop_offer_id=prop_offer_id,
        ),
        "prop_offer_id": prop_offer_id,
        "prop_offer_source_row_id": prop_offer_source_row_id,
        # Backward compatibility: pred_value remains what older reports expect.
        "pred_value": _round_or_none(pred_value, 3),
        "pred_count": _round_or_none(pred_count, 3),
        "pred_prob_over": _round_or_none(p_over, 4),
        "book_line": line,
        "edge": _round_or_none(edge, 4),
        "edge_type": edge_type,
        "model_family": model_family,
        "bet_side": side,
        "line_bucket": _clf_line_bucket(stat, line),
        "over_price": _round_or_none(over_price, 0),
        "under_price": _round_or_none(under_price, 0),
        "bet_price": _round_or_none(bet_price, 0),
        "minimum_acceptable_price": minimum_acceptable_price,
        "breakeven_prob": _round_or_none(breakeven, 4),
        "ev": _round_or_none(ev, 4),
        "bookmaker_key": bookmaker_key,
        "bet_link": bet_link,
        "kelly_fraction": _round_or_none(kelly_for_bankroll, 4),
        "bankroll_tier": bankroll.tier,
        "bankroll_candidate": bankroll.candidate,
        "bankroll_reasons": bankroll.reasons,
        "stake_pct": _round_or_none(bankroll.stake_pct, 4),
        "stake_usd": _round_or_none(bankroll.stake_usd, 2),
        "lock_price_implied": line_data.get("lock_price_implied"),
        "consensus_prob_at_lock": line_data.get("consensus_prob_at_lock"),
        "consensus_price_dispersion": line_data.get("consensus_price_dispersion"),
        "consensus_book_count": line_data.get("consensus_book_count"),
        "book_lead_lag_prob": line_data.get("book_lead_lag_prob"),
        "best_consensus_prob": line_data.get("best_consensus_prob"),
        "worst_consensus_prob": line_data.get("worst_consensus_prob"),
        "lock_offer_available": line_data.get("lock_offer_available"),
        "lock_same_book_pair_available": line_data.get("lock_same_book_pair_available"),
        "minutes_to_first_pitch_at_lock": line_data.get("minutes_to_first_pitch_at_lock"),
        "lock_price_age_minutes": line_data.get("lock_price_age_minutes"),
    }


_ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bets.mlb_prop_predictions (
    id               SERIAL PRIMARY KEY,
    game_date_et     DATE        NOT NULL,
    game_slug        TEXT        NOT NULL,
    player_id        BIGINT      NOT NULL,
    player_name      TEXT,
    team_abbr        TEXT,
    stat             TEXT        NOT NULL,
    prediction_key   TEXT,
    prop_offer_id    BIGINT,
    prop_offer_source_row_id INTEGER,
    pred_value       NUMERIC,
    book_line        NUMERIC,
    edge             NUMERIC,
    kelly_fraction   NUMERIC,
    actual_value     NUMERIC,
    over_hit         BOOLEAN,
    closing_line     NUMERIC,
    closing_price    NUMERIC,
    clv_line         NUMERIC,
    clv_price        NUMERIC,
    beat_clv_line    BOOLEAN,
    beat_clv_price   BOOLEAN,
    run_id           TEXT,
    is_active        BOOLEAN NOT NULL DEFAULT TRUE,
    superseded_at    TIMESTAMPTZ,
    stale_reason     TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (game_date_et, game_slug, player_id, stat)
);

ALTER TABLE bets.mlb_prop_predictions
    DROP CONSTRAINT IF EXISTS mlb_prop_predictions_game_date_et_game_slug_player_id_stat_key,
    DROP CONSTRAINT IF EXISTS mlb_prop_predictions_game_slug_player_id_stat_key;

ALTER TABLE bets.mlb_prop_predictions
    ADD COLUMN IF NOT EXISTS prediction_key   TEXT,
    ADD COLUMN IF NOT EXISTS prop_offer_id    BIGINT,
    ADD COLUMN IF NOT EXISTS prop_offer_source_row_id INTEGER,
    ADD COLUMN IF NOT EXISTS closing_line     NUMERIC,
    ADD COLUMN IF NOT EXISTS closing_price    NUMERIC,
    ADD COLUMN IF NOT EXISTS clv_line         NUMERIC,
    ADD COLUMN IF NOT EXISTS clv_price        NUMERIC,
    ADD COLUMN IF NOT EXISTS beat_clv_line    BOOLEAN,
    ADD COLUMN IF NOT EXISTS beat_clv_price   BOOLEAN,
    ADD COLUMN IF NOT EXISTS pred_count      NUMERIC,
    ADD COLUMN IF NOT EXISTS pred_prob_over  NUMERIC,
    ADD COLUMN IF NOT EXISTS edge_type       TEXT,
    ADD COLUMN IF NOT EXISTS model_family    TEXT,
    ADD COLUMN IF NOT EXISTS bet_side        TEXT,
    ADD COLUMN IF NOT EXISTS line_bucket     TEXT,
    ADD COLUMN IF NOT EXISTS over_price      NUMERIC,
    ADD COLUMN IF NOT EXISTS under_price     NUMERIC,
    ADD COLUMN IF NOT EXISTS bet_price       NUMERIC,
    ADD COLUMN IF NOT EXISTS minimum_acceptable_price NUMERIC,
    ADD COLUMN IF NOT EXISTS breakeven_prob  NUMERIC,
    ADD COLUMN IF NOT EXISTS ev              NUMERIC,
    ADD COLUMN IF NOT EXISTS bookmaker_key   TEXT,
    ADD COLUMN IF NOT EXISTS bet_link        TEXT,
    ADD COLUMN IF NOT EXISTS bankroll_tier   TEXT,
    ADD COLUMN IF NOT EXISTS bankroll_candidate BOOLEAN,
    ADD COLUMN IF NOT EXISTS bankroll_reasons TEXT,
    ADD COLUMN IF NOT EXISTS stake_pct       NUMERIC,
    ADD COLUMN IF NOT EXISTS stake_usd       NUMERIC,
    ADD COLUMN IF NOT EXISTS lock_snapshot_id BIGINT,
    ADD COLUMN IF NOT EXISTS locked_at_utc   TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS closing_snapshot_id BIGINT,
    ADD COLUMN IF NOT EXISTS closing_source_row_id BIGINT,
    ADD COLUMN IF NOT EXISTS closing_fetched_at_utc TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS clv_match_method TEXT,
    ADD COLUMN IF NOT EXISTS clv_valid BOOLEAN,
    ADD COLUMN IF NOT EXISTS clv_status TEXT,
    ADD COLUMN IF NOT EXISTS clv_unknown_reason TEXT,
    ADD COLUMN IF NOT EXISTS run_id          TEXT,
    ADD COLUMN IF NOT EXISTS is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS superseded_at   TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS stale_reason    TEXT,
    ADD COLUMN IF NOT EXISTS updated_at      TIMESTAMPTZ DEFAULT NOW();

CREATE UNIQUE INDEX IF NOT EXISTS uq_mlb_prop_predictions_prediction_key
    ON bets.mlb_prop_predictions (prediction_key);
CREATE INDEX IF NOT EXISTS idx_mlb_prop_predictions_offer
    ON bets.mlb_prop_predictions (prop_offer_id);
CREATE INDEX IF NOT EXISTS idx_mlb_prop_predictions_date_market
    ON bets.mlb_prop_predictions (game_date_et, stat, bet_side);
"""

_UPSERT_SQL = """
INSERT INTO bets.mlb_prop_predictions
    (game_date_et, game_slug, player_id, player_name, team_abbr, stat,
     prediction_key, prop_offer_id, prop_offer_source_row_id,
     pred_value, pred_count, pred_prob_over, book_line, edge, edge_type,
     model_family, bet_side, line_bucket, over_price, under_price, bet_price,
     minimum_acceptable_price, breakeven_prob, ev, bookmaker_key, bet_link, kelly_fraction,
     bankroll_tier, bankroll_candidate, bankroll_reasons, stake_pct, stake_usd,
     run_id, is_active, stale_reason)
VALUES
    (%(game_date_et)s, %(game_slug)s, %(player_id)s, %(player_name)s, %(team_abbr)s,
     %(stat)s, %(prediction_key)s, %(prop_offer_id)s, %(prop_offer_source_row_id)s,
     %(pred_value)s, %(pred_count)s, %(pred_prob_over)s, %(book_line)s,
     %(edge)s, %(edge_type)s, %(model_family)s, %(bet_side)s, %(line_bucket)s,
     %(over_price)s, %(under_price)s, %(bet_price)s, %(minimum_acceptable_price)s, %(breakeven_prob)s,
     %(ev)s, %(bookmaker_key)s, %(bet_link)s, %(kelly_fraction)s,
     %(bankroll_tier)s, %(bankroll_candidate)s, %(bankroll_reasons)s, %(stake_pct)s, %(stake_usd)s,
     %(run_id)s, TRUE, NULL)
ON CONFLICT (prediction_key) DO UPDATE SET
    player_name     = EXCLUDED.player_name,
    team_abbr       = EXCLUDED.team_abbr,
    prop_offer_id   = EXCLUDED.prop_offer_id,
    prop_offer_source_row_id = EXCLUDED.prop_offer_source_row_id,
    pred_value      = EXCLUDED.pred_value,
    pred_count      = EXCLUDED.pred_count,
    pred_prob_over  = EXCLUDED.pred_prob_over,
    book_line       = EXCLUDED.book_line,
    edge            = EXCLUDED.edge,
    edge_type       = EXCLUDED.edge_type,
    model_family    = EXCLUDED.model_family,
    bet_side        = EXCLUDED.bet_side,
    line_bucket     = EXCLUDED.line_bucket,
    over_price      = EXCLUDED.over_price,
    under_price     = EXCLUDED.under_price,
    bet_price       = EXCLUDED.bet_price,
    minimum_acceptable_price = EXCLUDED.minimum_acceptable_price,
    breakeven_prob  = EXCLUDED.breakeven_prob,
    ev              = EXCLUDED.ev,
    bookmaker_key   = EXCLUDED.bookmaker_key,
    bet_link        = EXCLUDED.bet_link,
    kelly_fraction  = EXCLUDED.kelly_fraction,
    bankroll_tier   = EXCLUDED.bankroll_tier,
    bankroll_candidate = EXCLUDED.bankroll_candidate,
    bankroll_reasons = EXCLUDED.bankroll_reasons,
    stake_pct       = EXCLUDED.stake_pct,
    stake_usd       = EXCLUDED.stake_usd,
    run_id          = EXCLUDED.run_id,
    is_active       = TRUE,
    stale_reason    = NULL,
    superseded_at   = NULL,
    lock_snapshot_id = NULL,
    locked_at_utc   = NULL,
    actual_value    = NULL,
    over_hit        = NULL,
    closing_line    = NULL,
    closing_price   = NULL,
    clv_line        = NULL,
    clv_price       = NULL,
    beat_clv_line   = NULL,
    beat_clv_price  = NULL,
    closing_source_row_id = NULL,
    closing_snapshot_id = NULL,
    closing_fetched_at_utc = NULL,
    clv_match_method = NULL,
    clv_valid       = NULL,
    clv_status      = NULL,
    clv_unknown_reason = NULL,
    updated_at      = NOW()
"""


def _ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_ENSURE_TABLE_SQL)
    conn.commit()
    _ensure_lineup_quality_dependency(conn)


def _regclass_exists(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (name,))
        return bool(cur.fetchone()[0])


def _ensure_player_batting_rolling_mat(conn) -> None:
    if _regclass_exists(conn, "features.mlb_player_batting_rolling_mat"):
        return
    if not _regclass_exists(conn, "features.mlb_player_batting_rolling"):
        log.warning("features.mlb_player_batting_rolling missing; lineup quality will use empty fallback")
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_player_batting_rolling_mat AS
            SELECT * FROM features.mlb_player_batting_rolling
            WITH DATA;

            CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_batting_player_mat_pk
                ON features.mlb_player_batting_rolling_mat (game_slug, player_id);
            CREATE INDEX IF NOT EXISTS idx_mlb_batting_player_mat_player_date
                ON features.mlb_player_batting_rolling_mat (player_id, game_date_et DESC, game_slug DESC);
            """
        )
    conn.commit()
    log.info("Created compatibility matview features.mlb_player_batting_rolling_mat")


def _create_empty_lineup_quality_mat(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE SCHEMA IF NOT EXISTS features;
            DROP MATERIALIZED VIEW IF EXISTS features.mlb_lineup_quality_mat;
            CREATE MATERIALIZED VIEW features.mlb_lineup_quality_mat AS
            SELECT
                NULL::text AS game_slug,
                NULL::text AS team_abbr,
                NULL::boolean AS is_home,
                NULL::numeric AS lineup_avg_avg_10,
                NULL::numeric AS lineup_slg_avg_10,
                NULL::numeric AS lineup_iso_avg_10,
                NULL::numeric AS top4_slg_avg_10,
                NULL::double precision AS lineup_data_completeness,
                NULL::double precision AS lineup_xwoba_avg,
                NULL::double precision AS lineup_xslg_avg,
                NULL::double precision AS lineup_barrel_avg,
                NULL::double precision AS lineup_hard_hit_avg,
                NULL::numeric AS lineup_k_pct_std,
                NULL::numeric AS lineup_k_pct_cv,
                NULL::numeric AS pct_lhb
            WHERE false;
            CREATE UNIQUE INDEX IF NOT EXISTS mlb_lineup_quality_mat_pk
                ON features.mlb_lineup_quality_mat (game_slug, team_abbr);
            """
        )
    conn.commit()
    log.warning("Created empty fallback features.mlb_lineup_quality_mat; lineup quality fields will be median-imputed")


def _ensure_lineup_quality_dependency(conn) -> None:
    if _regclass_exists(conn, "features.mlb_lineup_quality_mat"):
        return
    try:
        _ensure_player_batting_rolling_mat(conn)
        lineup_sql = _SQL_DIR / "MLB011_mlb_lineup_quality.sql"
        mat_sql = _SQL_DIR / "MLB011b_mlb_lineup_quality_mat.sql"
        if not lineup_sql.exists() or not mat_sql.exists():
            raise FileNotFoundError("MLB011 lineup quality SQL files are missing")
        with conn.cursor() as cur:
            cur.execute(lineup_sql.read_text(encoding="utf-8"))
            cur.execute(mat_sql.read_text(encoding="utf-8"))
        conn.commit()
        log.info("Created features.mlb_lineup_quality_mat for player prop prediction")
    except Exception:
        conn.rollback()
        log.exception("Could not create full lineup quality matview; using empty fallback")
        _create_empty_lineup_quality_mat(conn)


def _save_predictions(conn, rows: List[Dict]) -> None:
    if not rows:
        return
    run_id = datetime.now(timezone.utc).strftime("prop-predict-%Y%m%dT%H%M%SZ")
    with conn.cursor() as cur:
        run_dates = sorted({row.get("game_date_et") for row in rows if row.get("game_date_et")})
        current_keys = sorted({
            str(row.get("prediction_key"))
            for row in rows
            if row.get("prediction_key")
        })
        if run_dates:
            cur.execute(
                """
                UPDATE bets.mlb_prop_predictions
                SET is_active = FALSE,
                    stale_reason = 'not_in_latest_prediction_run',
                    superseded_at = COALESCE(superseded_at, NOW()),
                    updated_at = NOW()
                WHERE game_date_et = ANY(%s::date[])
                  AND COALESCE(is_active, TRUE) IS TRUE
                  AND actual_value IS NULL
                  AND (
                    prediction_key IS NULL
                    OR NOT (prediction_key = ANY(%s::text[]))
                  )
                """,
                (run_dates, current_keys),
            )
            if cur.rowcount:
                log.info("_save_predictions: marked %d stale prop rows inactive for %s", cur.rowcount, run_dates)
        # Remove stale predictions for players now marked OUT/DOUBTFUL who were
        # predicted in earlier runs today (upsert alone won't hide them).
        cur.execute("""
            UPDATE bets.mlb_prop_predictions pp
            SET is_active = FALSE,
                bankroll_candidate = FALSE,
                bankroll_tier = 'paper',
                stake_pct = 0,
                stake_usd = 0,
                bankroll_reasons = CASE
                    WHEN COALESCE(bankroll_reasons, '') = '' THEN 'inactive_player'
                    WHEN bankroll_reasons NOT LIKE '%inactive_player%' THEN bankroll_reasons || '; inactive_player'
                    ELSE bankroll_reasons
                END,
                stale_reason = 'inactive_player',
                superseded_at = COALESCE(superseded_at, NOW()),
                updated_at = NOW()
            FROM raw.mlb_injuries inj
            WHERE inj.mlb_player_id = pp.player_id
              AND inj.playing_probability IN ('OUT', 'DOUBTFUL')
              AND pp.game_date_et = CURRENT_DATE
              AND COALESCE(pp.is_active, TRUE) IS TRUE
              AND pp.actual_value IS NULL
        """)
        deleted = cur.rowcount
        if deleted:
            log.info("_save_predictions: marked %d rows inactive for OUT/DOUBTFUL players", deleted)
        for row in rows:
            for key in (
                "pred_count", "pred_prob_over", "edge_type", "model_family",
                "bet_side", "line_bucket", "over_price", "under_price",
                "bet_price", "minimum_acceptable_price", "breakeven_prob", "ev", "bookmaker_key",
                "prediction_key", "prop_offer_id", "prop_offer_source_row_id", "bet_link",
                "bankroll_tier", "bankroll_candidate", "bankroll_reasons", "stake_pct", "stake_usd",
            ):
                row.setdefault(key, None)
            row["run_id"] = run_id
            cur.execute(_UPSERT_SQL, row)
    conn.commit()
    log.info("Saved %d prop prediction rows", len(rows))


# ─────────────────────────────────────────────────────────────────────────────
# Prop line loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_prop_lines(conn, game_date: date) -> Dict[Tuple[str, str], Dict]:
    try:
        line_map = build_prop_line_map_for_date(conn, game_date)
        if line_map:
            return line_map
    except Exception:
        log.warning("Normalized prop offer map failed; falling back to legacy loader", exc_info=True)
    return _load_prop_lines_legacy(conn, game_date)


def _load_prop_lines_legacy(conn, game_date: date) -> Dict[Tuple[str, str], Dict]:
    """
    Returns {(player_name_norm, stat): {line, over_link, under_link, bookmaker_key}}.

    FanDuel is the primary source (line + over_link). DraftKings supplements
    under_link/under_price for batter stats where FD has none — FD batter props
    come from alternate markets (Over-only), while DK uses standard markets
    (full Over + Under).
    """
    df = pd.read_sql(SQL_PROP_LINES, conn, params={"game_date": game_date})
    if df.empty:
        return {}

    def _clean(v):
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return str(v) if v else None

    # FD now stores multiple lines per player per stat (alt market rows).
    # Build fd_by_line[(player, stat)][line_val] = entry so we can pick correctly.
    fd_by_line: Dict[Tuple[str, str], Dict[float, Dict]] = {}
    dk_rows: Dict[Tuple[str, str], Dict] = {}

    for _, row in df.iterrows():
        key = (str(row["player_name_norm"]), str(row["stat"]))
        entry = {
            "bookmaker_key": str(row["bookmaker_key"]),
            "line": float(row["line"]) if row["line"] is not None else None,
            "over_price": float(row["over_price"]) if row.get("over_price") is not None else None,
            "under_price": float(row["under_price"]) if row.get("under_price") is not None else None,
            "over_link":  _clean(row.get("over_link")),
            "under_link": _clean(row.get("under_link")),
        }
        entry["over_bookmaker_key"] = entry["bookmaker_key"] if entry.get("over_price") is not None else None
        entry["under_bookmaker_key"] = entry["bookmaker_key"] if entry.get("under_price") is not None else None
        if str(row["bookmaker_key"]) == "fanduel":
            line_val = entry["line"]
            if line_val is not None:
                fd_by_line.setdefault(key, {})[line_val] = entry
        else:
            # DK standard market: one row per player per stat (lowest line).
            # If multiple DK lines exist, prefer the lowest (standard market line).
            if key not in dk_rows or (
                entry["line"] is not None
                and dk_rows[key]["line"] is not None
                and entry["line"] < dk_rows[key]["line"]
            ):
                dk_rows[key] = entry

    # Merge: DK standard line is primary for line value (1.5 hits, 1.5 TB, 0.5 HR).
    # FD supplies over_link + over_price at the matching line.  For FD-only stats
    # (HR where DK doesn't have a line) pick the FD line closest to even-money.
    result: Dict[Tuple[str, str], Dict] = {}
    for key in set(fd_by_line) | set(dk_rows):
        fd_lines = fd_by_line.get(key, {})

        if key in dk_rows:
            entry = dict(dk_rows[key])
            dk_line = entry.get("line")
            # Attach FD over_link for the matching line; fall back to closest.
            fd_at_match = fd_lines.get(dk_line) if dk_line is not None else None
            if fd_at_match is None and fd_lines:
                # Pick FD line closest to DK line value.
                fd_at_match = fd_lines[min(fd_lines, key=lambda x: abs(x - (dk_line or 0)))]
            if fd_at_match:
                entry["over_link"]  = fd_at_match.get("over_link") or entry.get("over_link")
                entry["over_price"] = fd_at_match.get("over_price") if fd_at_match.get("over_price") is not None else entry.get("over_price")
                if fd_at_match.get("over_price") is not None:
                    entry["over_bookmaker_key"] = fd_at_match.get("over_bookmaker_key") or "fanduel"
            if not entry.get("under_link") and key in dk_rows:
                pass  # already have DK under_link in entry
            entry["under_link_book"] = "draftkings"
            entry["under_bookmaker_key"] = entry.get("under_bookmaker_key") or "draftkings"
            result[key] = entry
        else:
            # FD-only stat (e.g., HR with no DK line): pick most even-money FD line.
            best_line = min(fd_lines, key=lambda x: abs(fd_lines[x].get("over_price") or 99999))
            entry = dict(fd_lines[best_line])
            entry["under_link_book"] = "fanduel"
            entry["over_bookmaker_key"] = entry.get("over_bookmaker_key") or "fanduel"
            entry["under_bookmaker_key"] = entry.get("under_bookmaker_key") or "fanduel"
            result[key] = entry

    # Force batter_hits to always use line=0.5 (1+ hit) from FanDuel if available.
    # DK's standard market is 1.5 (2+ hits); we always want the 1-hit line.
    for key in list(result.keys()):
        if key[1] == "batter_hits":
            fd_lines = fd_by_line.get(key, {})
            if 0.5 in fd_lines:
                fd_entry = fd_lines[0.5]
                result[key]["line"] = 0.5
                result[key]["over_link"] = fd_entry.get("over_link") or result[key].get("over_link")
                if fd_entry.get("over_price") is not None:
                    result[key]["over_price"] = fd_entry["over_price"]
                    result[key]["over_bookmaker_key"] = fd_entry.get("over_bookmaker_key") or "fanduel"

    return result


def _load_all_alt_lines(conn, game_date: date) -> Dict[Tuple[str, str], List[Dict]]:
    """Return ALL FD alternate prop lines per (player_name_norm, stat), sorted low→high.

    Used by the lottery parlay to pick the highest line where the model has edge.
    Excludes HR (handled by the dedicated HR parlay).
    """
    df = pd.read_sql(SQL_PROP_LINES, conn, params={"game_date": game_date})
    if df.empty:
        return {}

    result: Dict[Tuple[str, str], List[Dict]] = {}

    def _clean(v):
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return str(v) if v else None

    for _, row in df.iterrows():
        if str(row["bookmaker_key"]) != "fanduel":
            continue
        stat = str(row["stat"])
        if stat == "batter_home_runs":
            continue  # HR has its own parlay
        line_val = float(row["line"]) if row["line"] is not None else None
        over_price = float(row["over_price"]) if row.get("over_price") is not None else None
        over_link = _clean(row.get("over_link"))
        if line_val is None or over_price is None or not over_link:
            continue
        key = (str(row["player_name_norm"]), stat)
        result.setdefault(key, []).append({
            "line": line_val,
            "over_price": over_price,
            "over_link": over_link,
        })

    # Sort each player's lines from lowest to highest.
    for key in result:
        result[key].sort(key=lambda x: x["line"])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _predict_ensemble(
    X: pd.DataFrame,
    xgb_model: XGBRegressor,
    lgb_model,
) -> np.ndarray:
    preds = xgb_model.predict(X)
    if lgb_model is not None:
        lgb_preds = lgb_model.predict(X.values)
        preds = (preds + lgb_preds) / 2.0
    return preds


_PITCHER_META = ["game_slug", "game_date_et", "start_ts_utc", "player_id",
                 "player_name", "team_abbr", "opponent_abbr"]
_BATTER_META = ["game_slug", "game_date_et", "start_ts_utc", "player_id",
                "team_abbr", "opponent_abbr"]


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_edge(edge: Optional[float], threshold: float, direction: str = "OVER") -> str:
    if edge is None or math.isnan(edge):
        return "[no line]"
    abs_e = abs(edge)
    direction = "OVER" if edge > 0 else "UNDER"
    if abs_e >= threshold:
        return f"★ {direction} +{abs_e:.2f}"
    return f"[{direction} +{abs_e:.2f}]" if abs_e > 0 else "[no edge]"


def _best_side_from_ev(
    ld: Dict,
    p_over: Optional[float],
    min_ev: float,
    blocked_sides: set[str] | None = None,
) -> Optional[Dict]:
    """Return best EV side dict when EV is computable and above threshold."""
    if p_over is None:
        return None
    ev_over = _ev_per_unit(p_over, ld.get("over_price"))
    ev_under = _ev_per_unit(1.0 - p_over, ld.get("under_price"))
    cands = []
    # blocked_sides are now warnings only. The model still picks the best priced side.
    if ev_over is not None and ld.get("over_link"):
        cands.append(("over", ev_over, ld.get("over_price"), ld.get("over_link")))
    if ev_under is not None and ld.get("under_link"):
        cands.append(("under", ev_under, ld.get("under_price"), ld.get("under_link")))
    if not cands:
        return None
    side, ev, price, link = max(cands, key=lambda t: t[1])
    if ev < min_ev:
        return None
    return {"side": side, "ev": float(ev), "price": price, "link": link}


def _collect_all_prop_links(
    all_pitcher_rows: List[Dict],
    all_batter_rows: List[Dict],
    prop_lines: Dict,
) -> List[str]:
    """Collect one FD bet link per prop, taking the model's predicted direction.

    Used to build the All Props Parlay (every player, model's side, regardless of edge).
    """
    links: List[str] = []

    for row in all_pitcher_rows:
        pred_k = row.get("pred_strikeouts")
        if pred_k is None:
            continue
        norm = _normalize_name(row.get("player_name", f"id={row['player_id']}"))
        ld = prop_lines.get((norm, "pitcher_strikeouts"))
        if not ld or ld.get("line") is None:
            continue
        edge = pred_k - ld["line"]
        link = ld.get("over_link") if edge >= 0 else ld.get("under_link")
        if link:
            links.append(link)

    for row in all_batter_rows:
        norm = _normalize_name(row.get("player_name", f"id={row['player_id']}"))
        for pred_col, stat_key in [
            ("pred_hits",        "batter_hits"),
            ("pred_total_bases", "batter_total_bases"),
            ("pred_home_runs",   "batter_home_runs"),
        ]:
            pred_v = row.get(pred_col)
            if pred_v is None:
                continue
            ld = prop_lines.get((norm, stat_key))
            if not ld or ld.get("line") is None:
                continue
            edge = pred_v - ld["line"]
            link = ld.get("over_link") if edge >= 0 else ld.get("under_link")
            if link:
                links.append(link)

    return links


def _collect_prop_links_by_stat(
    all_pitcher_rows: List[Dict],
    all_batter_rows: List[Dict],
    prop_lines: Dict,
) -> Dict[str, List[str]]:
    """Collect model-direction prop links grouped by stat key."""
    out: Dict[str, List[str]] = {
        "pitcher_strikeouts": [],
        "batter_hits": [],
        "batter_total_bases": [],
        "batter_home_runs": [],
    }
    seen_pitcher: set[tuple] = set()
    seen_batter: set[tuple] = set()

    for row in all_pitcher_rows:
        pkey = (row.get("game_slug"), row.get("player_id"), "pitcher_strikeouts")
        if pkey in seen_pitcher:
            continue
        seen_pitcher.add(pkey)
        pred_k = row.get("pred_strikeouts")
        if pred_k is None:
            continue
        norm = _normalize_name(row.get("player_name", f"id={row['player_id']}"))
        ld = prop_lines.get((norm, "pitcher_strikeouts"))
        if not ld or ld.get("line") is None:
            continue
        edge = pred_k - ld["line"]
        link = ld.get("over_link") if edge >= 0 else ld.get("under_link")
        if link:
            out["pitcher_strikeouts"].append(link)

    for row in all_batter_rows:
        norm = _normalize_name(row.get("player_name", f"id={row['player_id']}"))
        for pred_col, stat_key in [
            ("pred_hits", "batter_hits"),
            ("pred_total_bases", "batter_total_bases"),
            ("pred_home_runs", "batter_home_runs"),
        ]:
            bkey = (row.get("game_slug"), row.get("player_id"), stat_key)
            if bkey in seen_batter:
                continue
            seen_batter.add(bkey)
            pred_v = row.get(pred_col)
            if pred_v is None:
                continue
            ld = prop_lines.get((norm, stat_key))
            if not ld or ld.get("line") is None:
                continue
            edge = pred_v - ld["line"]
            link = ld.get("over_link") if edge >= 0 else ld.get("under_link")
            if link:
                out[stat_key].append(link)

    return out


def _collect_top_hr_parlay_links(
    all_batter_rows: List[Dict],
    prop_lines: Dict,
    top_n: int = 7,
    max_per_game: int = 2,
    min_pred: float = 0.17,
) -> List[str]:
    """Collect top-N HR prediction links (deduped) for a single HR-focused parlay.

    min_pred=0.17: below this, predicted HR rate ≈ base rate (~10-11%). Historical
    data shows real signal only starts around pred>=0.17 (10% HR rate at 0.17, rising
    to 14-20% at 0.19-0.23 and 25-33% at 0.26-0.31). Rank 8+ of daily top picks
    drops to 9% HR rate — parlay quality degrades sharply after top 7.
    """
    rows = [r for r in all_batter_rows
            if r.get("pred_home_runs") is not None
            and float(r.get("pred_home_runs") or 0.0) >= min_pred]
    rows.sort(key=lambda r: float(r.get("pred_home_runs") or 0.0), reverse=True)
    links: List[str] = []
    seen_links: set[str] = set()
    seen_players: set[tuple] = set()
    game_counts: Dict[str, int] = {}

    for r in rows:
        pkey = (r.get("game_slug"), r.get("player_id"))
        if pkey in seen_players:
            continue
        slug = str(r.get("game_slug") or "")
        if slug and game_counts.get(slug, 0) >= max(int(max_per_game), 1):
            continue
        seen_players.add(pkey)
        norm = _normalize_name(r.get("player_name", f"id={r['player_id']}"))
        ld = prop_lines.get((norm, "batter_home_runs"))
        if not ld:
            continue
        # HR parlay is intended as top HR picks, so prioritize OVER links.
        lnk = ld.get("over_link")
        if not lnk or lnk in seen_links:
            continue
        seen_links.add(lnk)
        links.append(lnk)
        if slug:
            game_counts[slug] = game_counts.get(slug, 0) + 1
        if len(links) >= max(int(top_n), 1):
            break
    return links


def _streak_mult(short_avg: Optional[float], long_avg: Optional[float],
                 sensitivity: float = 0.4, floor: float = 0.5, cap: float = 2.0) -> float:
    """Return an EV ranking multiplier based on recent vs baseline performance.

    mult > 1.0  →  player is running hot  (boost in lottery ranking)
    mult < 1.0  →  player is running cold (penalty in lottery ranking)
    mult = 1.0  →  at baseline or insufficient data

    sensitivity=0.4 means a player 2x their baseline gets mult=1.4,
    and a player at half their baseline gets mult=0.8.
    The multiplier does NOT change P(over) — only affects sort order.
    """
    try:
        s = float(short_avg)
        l = float(long_avg)
    except (TypeError, ValueError):
        return 1.0
    if l <= 0.05:   # too close to zero; ratio would be unreliable
        return 1.0
    ratio = s / l
    return max(floor, min(cap, 1.0 + sensitivity * (ratio - 1.0)))


def _collect_lottery_parlay_links(
    all_pitcher_rows: List[Dict],
    all_batter_rows: List[Dict],
    prop_lines: Dict,
    cfg: PredictConfig,
    all_alt_lines: Optional[Dict] = None,
    alt_clf_probs: Optional[Dict] = None,
    min_p_over: float = 0.15,
) -> List[str]:
    """Collect matchup-based lottery legs using the highest alt line where model has edge.

    For each pitcher/batter we look at every available FD alt line (low→high) and
    find the HIGHEST line value where P(OVER) is still positive-EV at the offered
    odds.  This surfaces bets like "Skubal K OVER 8.5 at +350" or
    "Freeman TB OVER 3.5 at +500" when the model's regression prediction supports it.
    No HR — those have a dedicated parlay.

    When alt_clf_probs is available, P(over) is blended 60/40 (regression Poisson /
    alt CLF) for each (player, stat, line) triple.  This adds the alt-line CLF signal
    which was specifically trained on high-threshold binary outcomes.

    Candidates are ranked by EV × streak_mult so hot-streak players rise to the top
    and cold-streak players fall even if their raw EV is marginally positive.
    min_p_over: legs with blended P(over) below this floor are skipped regardless of EV.
    """
    lo = int(cfg.lottery_min_american)
    hi = int(cfg.lottery_max_american)
    candidates: List[Dict] = []

    def _in_range(odds: Optional[float]) -> bool:
        return odds is not None and lo <= float(odds) <= hi

    # ── Pitcher strikeouts: highest alt K line with model edge ──────────────
    for r in all_pitcher_rows:
        name = r.get("player_name", f"id={r.get('player_id')}")
        norm = _normalize_name(name)
        pred_k = r.get("pred_strikeouts")
        if pred_k is None:
            continue
        sigma_k = r.get("sigma_strikeouts")
        # Streak signal: K/9 last 5 starts vs last 10 starts
        smult = _streak_mult(r.get("k9_5"), r.get("k9_10"))
        # All available K lines for this pitcher (already sorted low→high)
        k_lines = (all_alt_lines or {}).get((norm, "pitcher_strikeouts"), [])
        if not k_lines:
            # Fall back to main line dict
            ld = prop_lines.get((norm, "pitcher_strikeouts"))
            if ld and ld.get("line") is not None:
                k_lines = [{"line": ld["line"], "over_price": ld.get("over_price"),
                             "over_link": ld.get("over_link")}]
        # Walk from highest to lowest; pick the highest qualifying line
        best = None
        for ld in reversed(k_lines):
            line_val = ld.get("line")
            over_odds = ld.get("over_price")
            if not _in_range(over_odds):
                continue
            # Poisson probability from regression — well-calibrated for count data;
            # monotonic across all alt-line thresholds (P(K≥k+1) < P(K≥k) guaranteed).
            p_over = _prob_over_from_regression(pred_k, line_val)
            if p_over is None:
                continue
            # Blend with alt CLF when available (60% regression / 40% CLF).
            # alt_clf_probs keys: (norm_name, stat, line_val_float).
            clf_p = (alt_clf_probs or {}).get((norm, "pitcher_strikeouts", float(line_val)))
            if clf_p is not None:
                p_over = 0.6 * p_over + 0.4 * float(clf_p)
            if p_over < min_p_over:
                continue
            ev = _ev_per_unit(float(p_over), over_odds)
            if ev is not None and ev >= cfg.min_ev:
                best = {"ev": float(ev), "p_over": float(p_over), "odds": float(over_odds),
                        "line": line_val, "link": ld.get("over_link"), "streak_mult": smult}
                break  # highest qualifying line found
        if best and best.get("link"):
            candidates.append({
                **best,
                "ranked_ev": best["ev"] * smult,
                "game_slug": r.get("game_slug"),
                "player_id": r.get("player_id"),
                "player_name": name,
                "team_abbr": r.get("team_abbr"),
                "stat": "pitcher_strikeouts",
                "pred_value": pred_k,
                "player_key": (r.get("game_slug"), r.get("player_id"), "pitcher_strikeouts"),
                "label": f"{name} K O{best['line']} ({best['odds']:+.0f})",
            })

    # ── Batters: highest alt TB / hits line with model edge ─────────────────
    for r in all_batter_rows:
        name = r.get("player_name", f"id={r.get('player_id')}")
        norm = _normalize_name(name)
        sigma_map = r.get("sigma_map") or {}

        for stat_key, pred_col, short_col, long_col in [
            ("batter_total_bases", "pred_total_bases", "tb_avg_5",   "tb_avg_20"),
            ("batter_hits",        "pred_hits",        "hits_avg_5", "hits_avg_20"),
        ]:
            pred_v = r.get(pred_col)
            if pred_v is None:
                continue
            sigma_v = sigma_map.get(stat_key)
            # Streak signal: last-5 average vs last-20 baseline
            smult = _streak_mult(r.get(short_col), r.get(long_col))
            # All available alt lines for this player/stat (sorted low→high)
            alt_lines = (all_alt_lines or {}).get((norm, stat_key), [])
            if not alt_lines:
                ld = prop_lines.get((norm, stat_key))
                if ld and ld.get("line") is not None:
                    alt_lines = [{"line": ld["line"], "over_price": ld.get("over_price"),
                                  "over_link": ld.get("over_link")}]
            # Walk highest→lowest, pick the highest where model still shows edge
            best = None
            for ld in reversed(alt_lines):
                line_val = ld.get("line")
                over_odds = ld.get("over_price")
                if not _in_range(over_odds):
                    continue
                # Poisson probability from regression — well-calibrated for count data;
                # monotonic across all alt-line thresholds (P(stat≥k+1) < P(stat≥k) guaranteed).
                p_over = _prob_over_from_regression(pred_v, line_val)
                if p_over is None:
                    continue
                # Blend with alt CLF when available (60% regression / 40% CLF).
                clf_p = (alt_clf_probs or {}).get((norm, stat_key, float(line_val)))
                if clf_p is not None:
                    p_over = 0.6 * p_over + 0.4 * float(clf_p)
                if p_over < min_p_over:
                    continue
                ev = _ev_per_unit(float(p_over), over_odds)
                if ev is not None and ev >= cfg.min_ev:
                    best = {"ev": float(ev), "p_over": float(p_over),
                            "odds": float(over_odds), "line": line_val,
                            "link": ld.get("over_link"), "streak_mult": smult}
                    break
            if best and best.get("link"):
                candidates.append({
                    **best,
                    "ranked_ev": best["ev"] * smult,
                    "game_slug": r.get("game_slug"),
                    "player_id": r.get("player_id"),
                    "player_name": name,
                    "team_abbr": r.get("team_abbr"),
                    "stat": stat_key,
                    "pred_value": pred_v,
                    "player_key": (r.get("game_slug"), r.get("player_id"), stat_key),
                    "label": f"{name} {stat_key.split('_')[-1].upper()} O{best['line']} ({best['odds']:+.0f})",
                })

    # ── Sort by streak-adjusted EV desc, dedupe, per-game cap, pick top N ──
    candidates.sort(key=lambda c: c["ranked_ev"], reverse=True)
    out: List[Dict] = []
    seen_links: set[str] = set()
    seen_player_stat: set[tuple] = set()
    game_counts: Dict[str, int] = {}
    for c in candidates:
        link = c.get("link")
        if not link or link in seen_links:
            continue
        pkey = c["player_key"]
        if pkey in seen_player_stat:
            continue
        slug = str(c.get("game_slug") or "")
        if slug and game_counts.get(slug, 0) >= max(int(cfg.lottery_max_per_game), 1):
            continue
        seen_links.add(link)
        seen_player_stat.add(pkey)
        out.append(c)
        if slug:
            game_counts[slug] = game_counts.get(slug, 0) + 1
        if len(out) >= max(int(cfg.lottery_legs), 1):
            break
    return out


# ---------------------------------------------------------------------------
# Lottery persistence helpers
# ---------------------------------------------------------------------------

_LOTTERY_UPSERT_SQL = """
    INSERT INTO bets.mlb_lottery_picks
        (game_date_et, game_slug, player_id, player_name, team_abbr,
         stat, pred_value, book_line, p_over, ev, streak_mult, ranked_ev, over_odds)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (game_date_et, game_slug, player_id, stat) DO UPDATE SET
        pred_value  = EXCLUDED.pred_value,
        book_line   = EXCLUDED.book_line,
        p_over      = EXCLUDED.p_over,
        ev          = EXCLUDED.ev,
        streak_mult = EXCLUDED.streak_mult,
        ranked_ev   = EXCLUDED.ranked_ev,
        over_odds   = EXCLUDED.over_odds
"""

_LOTTERY_GRADE_SQL = """
    UPDATE bets.mlb_lottery_picks lp
    SET actual_value = CASE lp.stat
            WHEN 'pitcher_strikeouts' THEN gl.strikeouts_pitcher::numeric
            WHEN 'batter_hits'        THEN gl.hits::numeric
            WHEN 'batter_total_bases' THEN gl.total_bases::numeric
            WHEN 'batter_home_runs'   THEN gl.home_runs::numeric
        END,
        over_hit = CASE lp.stat
            WHEN 'pitcher_strikeouts' THEN gl.strikeouts_pitcher > lp.book_line
            WHEN 'batter_hits'        THEN gl.hits > lp.book_line
            WHEN 'batter_total_bases' THEN gl.total_bases > lp.book_line
            WHEN 'batter_home_runs'   THEN gl.home_runs > lp.book_line
        END
    FROM raw.mlb_player_gamelogs gl
    WHERE gl.game_slug = lp.game_slug
      AND gl.player_id = lp.player_id
      AND lp.actual_value IS NULL
"""


def _save_lottery_picks(conn, legs: List[Dict], game_date: date) -> None:
    """Upsert today's lottery leg selections into bets.mlb_lottery_picks."""
    if not legs:
        return
    cur = conn.cursor()
    rows = []
    for leg in legs:
        pid = leg.get("player_id")
        slug = leg.get("game_slug")
        if not pid or not slug:
            continue
        rows.append((
            game_date,
            slug,
            int(pid),
            leg.get("player_name", ""),
            leg.get("team_abbr"),
            leg.get("stat", ""),
            leg.get("pred_value"),
            leg["line"],
            leg.get("p_over"),
            leg["ev"],
            leg.get("streak_mult", 1.0),
            leg.get("ranked_ev"),
            int(leg["odds"]) if leg.get("odds") is not None else None,
        ))
    if not rows:
        return
    cur.executemany(_LOTTERY_UPSERT_SQL, rows)
    conn.commit()
    log.info("Saved %d lottery legs to bets.mlb_lottery_picks for %s", len(rows), game_date)


def _grade_lottery_picks(conn) -> int:
    """Fill actual_value / over_hit for any ungraded lottery picks.

    Joins on raw.mlb_player_gamelogs so grading happens automatically once
    the day's boxscores are parsed.  Returns the number of rows updated.
    """
    cur = conn.cursor()
    cur.execute(_LOTTERY_GRADE_SQL)
    n = cur.rowcount
    conn.commit()
    if n:
        log.info("Graded %d lottery pick(s) in bets.mlb_lottery_picks", n)
    return n


def _prop_offer_health(prop_lines: Dict) -> Dict[str, int]:
    entries = len(prop_lines or {})
    offer_rows = 0
    priced_sides = 0
    linked_sides = 0
    for line_data in (prop_lines or {}).values():
        offers = list(line_data.get("offers") or [])
        offer_rows += len(offers)
        priced_sides += sum(1 for offer in offers if offer.get("price") is not None)
        linked_sides += sum(1 for offer in offers if offer.get("link"))
        if not offers:
            for side in ("over", "under"):
                if line_data.get(f"{side}_price") is not None:
                    priced_sides += 1
                if line_data.get(f"{side}_link"):
                    linked_sides += 1
    return {
        "entries": entries,
        "offer_rows": offer_rows,
        "priced_sides": priced_sides,
        "linked_sides": linked_sides,
    }


def _prop_offers_missing(prop_lines: Dict) -> bool:
    health = _prop_offer_health(prop_lines)
    return health["entries"] == 0 or (health["offer_rows"] == 0 and health["priced_sides"] == 0)


def _print_discord(
    all_pitcher_rows: List[Dict],
    all_batter_rows: List[Dict],
    prop_lines: Dict,
    game_map: Dict[str, Dict],  # game_slug -> {home, away, start_ts_utc}
    cfg: PredictConfig,
    all_alt_lines: Optional[Dict] = None,
    lottery_legs: Optional[List[Dict]] = None,
    db_rows: Optional[List[Dict]] = None,
    bucket_reopen_policy: Optional[Dict] = None,
) -> List[str]:
    """Print per-game prop output. Returns edge-play links for parlay.

    DISCORD_FORMAT=1  →  compact mode: prints Top-10 props and chunked (25-leg max)
                         stat-specific FD parlay links (K / H / TB / HR / BB).
    (no env var)      →  full table mode: all players in aligned columns.
    """
    is_discord = os.getenv("DISCORD_FORMAT") == "1"
    bucket_reopen_policy = bucket_reopen_policy or {}
    fd_links: List[str] = []
    prop_selector_ctx: Optional[PropSelectorContext] = None
    prop_selector_cfg = PropSelectorConfig(
        pg_dsn=cfg.pg_dsn,
        report_date=cfg.et_date,
        model_dir=cfg.model_dir,
        min_ev=cfg.min_ev,
    )
    if db_rows is not None:
        try:
            prop_selector_ctx = PropSelectorContext(cfg.model_dir)
        except Exception:
            log.exception("Failed to load prop shadow selector artifacts")

    # Discord mode: Top-10 leaderboards per stat + HR parlay + lottery parlay.
    if is_discord:
        printed_any = False
        used_bankroll_exposure = 0.0

        def _budget_assessment(assessment: BankrollAssessment) -> BankrollAssessment:
            nonlocal used_bankroll_exposure
            capped, used_bankroll_exposure = _apply_assessment_daily_cap(
                assessment,
                used_exposure_pct=used_bankroll_exposure,
                cfg=cfg,
            )
            return capped

        def _bankroll_label(assessment: BankrollAssessment) -> str:
            capped = _budget_assessment(assessment)
            return bankroll_tag(capped)

        def _market_implied_prob(over_price) -> Optional[float]:
            """Convert American over_price → implied probability (no vig removal)."""
            if over_price is None:
                return None
            try:
                v = float(over_price)
            except (TypeError, ValueError):
                return None
            if v < 0:
                return abs(v) / (abs(v) + 100.0)
            if v > 0:
                return 100.0 / (v + 100.0)
            return None

        def _leaderboard_rows(
            rows, pred_key, stat_key,
            pred_gate=True, min_pred=None,
            min_p_over=0.55,
            min_edge_vs_market=0.03, over_price_key=None,
            max_batting_order=None,
            require_confirmed_sp=False,
            sp_k_ceiling=None, sp_k_lookup=None,
            skip_clf=False,
        ):
            """Return (p_over, pred, line, name, team, opp, link, bankroll) for rows that
            pass all quality gates, sorted by P(over) descending.

            Gates (all optional):
            - pred_gate: pred_val >= min_pred (or line); skip rare-event stats like HR
            - min_p_over: model P(over) floor (default 55%)
            - min_edge_vs_market: model P(over) must exceed market implied P by this margin
            - over_price_key: row key for the market over price (to compute implied P)
            - max_batting_order: skip players batting above this slot (7th+ = excluded)
            - require_confirmed_sp: skip rows where opposing SP is not 'confirmed'
            - sp_k_ceiling: skip rows where opposing SP pred_k > this value
            - sp_k_lookup: dict {(game_slug, opp_team_abbr): pred_k}
            - skip_clf: if True, bypass CLF and derive P(over) from regression alone
            """
            entries = []
            for r in rows:
                pred_val = r.get(pred_key)
                if pred_val is None:
                    continue

                # ── Batting order gate ───────────────────────────────────────
                if max_batting_order is not None:
                    eff_order = r.get("effective_batting_order")
                    if eff_order is not None and eff_order > max_batting_order:
                        continue

                # ── SP confirmation gate ─────────────────────────────────────
                if require_confirmed_sp and r.get("opp_sp_source") != "confirmed":
                    continue

                # ── SP quality ceiling ───────────────────────────────────────
                if sp_k_ceiling is not None and sp_k_lookup is not None:
                    opp_pred_k = sp_k_lookup.get(
                        (r.get("game_slug"), r.get("opponent_abbr"))
                    )
                    if opp_pred_k is not None and opp_pred_k > sp_k_ceiling:
                        continue

                name = r.get("player_name", f"id={r['player_id']}")
                norm = _normalize_name(name)
                ld = prop_lines.get((norm, stat_key))
                if not ld or ld.get("line") is None:
                    continue
                line = ld["line"]

                # ── Prediction gate ──────────────────────────────────────────
                if pred_gate and pred_val < (min_pred if min_pred is not None else line):
                    continue

                clf_p = None if skip_clf else (r.get("clf_p_over") or {}).get(stat_key)
                if clf_p is not None:
                    p_over = _apply_regression_gate(clf_p, pred_val, line, stat_key)
                else:
                    p_over = _prob_over_from_regression(pred_val, line, None)
                if p_over is None:
                    continue

                # ── Minimum P(over) floor ────────────────────────────────────
                if p_over < min_p_over:
                    continue

                # ── Edge vs market implied probability ───────────────────────
                if over_price_key is not None and min_edge_vs_market > 0:
                    mkt_p = _market_implied_prob(r.get(over_price_key))
                    if mkt_p is not None and (p_over - mkt_p) < min_edge_vs_market:
                        continue

                bankroll = _prop_bankroll_from_pick(
                    stat=stat_key,
                    side_label="O",
                    cfg=cfg,
                    ld=ld,
                    p_over=p_over,
                    edge=float(pred_val) - float(line),
                    sigma=(
                        r.get("sigma_strikeouts")
                        if stat_key == "pitcher_strikeouts"
                        else (r.get("sigma_map") or {}).get(stat_key)
                    ),
                    blocked_sides=_blocked_sides_for_row(r, stat_key),
                    bucket_reopen_policy=bucket_reopen_policy,
                )
                entries.append((p_over, pred_val, line, name,
                                 r.get("team_abbr", "?"), r.get("opponent_abbr", "?"),
                                 ld.get("over_link"), bankroll))
            return sorted(entries, key=lambda x: x[0], reverse=True)

        # ── SP K lookup: {(game_slug, pitching_team_abbr): pred_k} ──────────────
        # Used to gate batters facing an elite SP (sp_k_ceiling)
        _sp_k_lookup = {
            (r["game_slug"], r["team_abbr"]): r["pred_strikeouts"]
            for r in all_pitcher_rows
            if r.get("pred_strikeouts") is not None and r.get("team_abbr")
        }

        bankroll_rows: List[Dict] = []
        paper_rows: List[Dict] = []
        model_pick_rows: List[Dict] = []
        research_rows: List[Dict] = []
        bankroll_links: List[str] = []

        def _message_game_date() -> Optional[date]:
            for row in all_pitcher_rows + all_batter_rows:
                val = row.get("game_date_et")
                if val is not None:
                    try:
                        return pd.Timestamp(val).date()
                    except Exception:
                        return val
            return cfg.et_date

        def _locked_bankroll_exposure() -> float:
            game_date = _message_game_date()
            if not game_date:
                return 0.0
            try:
                with psycopg2.connect(cfg.pg_dsn) as conn:
                    exposure, _keys, _slots = locked_bankroll_state(conn, game_date)
                    return exposure
            except Exception:
                log.exception("Could not load locked MLB bankroll exposure")
                return cfg.bankroll_max_daily_exposure_pct

        def _record_prop_item(
            *,
            market: str,
            side: str = "O",
            name: str,
            team: str,
            opp: str,
            pred_val: float,
            pred_fmt: str,
            line: float,
            p_over: Optional[float],
            link: Optional[str],
            bankroll: BankrollAssessment,
        ) -> None:
            capped = bankroll if db_rows is not None else _budget_assessment(bankroll)
            item = {
                "market": market,
                "side": side,
                "name": name,
                "team": team,
                "opp": opp,
                "pred_val": pred_val,
                "pred_fmt": pred_fmt,
                "line": line,
                "p_over": p_over,
                "link": link,
                "bankroll": capped,
            }
            if capped.candidate:
                if db_rows is None:
                    bankroll_rows.append(item)
                    if link:
                        bankroll_links.append(link)
            else:
                paper_rows.append(item)

        def _book_label(link: Optional[str], fallback: Optional[str] = None) -> str:
            return _prop_book_label(link, fallback)

        def _print_prop_item(item: Dict, *, include_link: bool) -> None:
            pred_s = item["pred_fmt"].format(item["pred_val"])
            display_prob = _as_float(item.get("selector_prob_side"))
            if display_prob is None:
                display_prob = item.get("p_over")
            display_ev = _as_float(item.get("selector_ev"))
            if display_ev is None:
                display_ev = item.get("ev")
            p_s = f" P={display_prob:.1%}" if display_prob is not None else ""
            ev_s = f" EV={display_ev:+.1%}" if display_ev is not None else ""
            clv_prob = _as_float(item.get("clv_beat_prob"))
            clv_s = f" CLV={clv_prob:.0%}" if clv_prob is not None else ""
            current_price = _as_float(item.get("current_price"))
            minimum_price = _as_float(item.get("minimum_acceptable_price"))
            price_s = f" Price={current_price:+.0f}" if current_price is not None else " Price=?"
            min_s = f" Min={minimum_price:+.0f}" if minimum_price is not None else " Min=?"
            drift_s = "" if item.get("price_drift_ok") else " DRIFT BLOCK"
            tags_s = _tag_text(item)
            book = item.get("book") or _book_label(item.get("link"))
            link_s = f" [Bet {book}](<{item['link']}>)" if include_link and item.get("link") else ""
            print(
                f"- {item['name']} ({item['team']} vs {item['opp']}) "
                f"{item['market']} {item.get('side', 'O')}{item['line']:.1f} -> {pred_s}{p_s}{ev_s}"
                f"{clv_s}{price_s}{min_s}{drift_s}{tags_s} "
                f"[{bankroll_tag(item['bankroll'])}]{link_s}"
            )

        def _as_float(value) -> Optional[float]:
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _opponent_for_row(row: Dict) -> str:
            gm = game_map.get(row.get("game_slug"), {})
            team = row.get("team_abbr")
            home = gm.get("home")
            away = gm.get("away")
            if team == home:
                return away or "?"
            if team == away:
                return home or "?"
            return row.get("opponent_abbr") or "?"

        def _db_row_to_prop_item(row: Dict) -> Optional[Dict]:
            item = candidate_from_prediction_row(
                row,
                prop_lines,
                opponent=_opponent_for_row(row),
            )
            if item is None:
                return None
            if prop_selector_ctx is not None:
                selector_row = dict(row)
                selector_row["price_drift_ok"] = item.get("price_drift_ok")
                if item.get("current_price") is not None:
                    selector_row["bet_price"] = item.get("current_price")
                try:
                    selector = score_prop_shadow_row(
                        selector_row,
                        ctx=prop_selector_ctx,
                        cfg=prop_selector_cfg,
                    )
                    item.update({
                        "selector_score": selector.get("selector_score"),
                        "selector_tier": selector.get("selector_tier"),
                        "selector_real_candidate": selector.get("selector_real_candidate"),
                        "selector_prob_side": selector.get("selector_prob_side"),
                        "selector_ev": selector.get("selector_ev"),
                        "clv_beat_prob": selector.get("clv_beat_prob"),
                        "bookable_prob": selector.get("bookable_prob"),
                        "policy_variant": selector.get("policy_variant"),
                        "pair_quality": selector.get("pair_quality"),
                        "market_prob_source": selector.get("market_prob_source"),
                        "event_side_line_prob_side": selector.get("event_side_line_prob_side"),
                        "bucket_trust_status": selector.get("bucket_trust_status"),
                        "no_bet_decision": selector.get("no_bet_decision"),
                        "selector_reasons": selector.get("selector_reasons") or [],
                    })
                    if item["bankroll"].candidate and not selector.get("selector_real_candidate"):
                        item["bankroll"] = BankrollAssessment(
                            tier="watch",
                            candidate=False,
                            reasons=_append_bankroll_reason(
                                item["bankroll"].reasons,
                                "selector_clv_residual_block",
                            ),
                            stake_pct=0.0,
                            stake_usd=0.0,
                        )
                except Exception:
                    log.exception("Failed to apply prop shadow selector to %s", row.get("prediction_key"))
            return item

        def _pick_score(item: Dict) -> tuple[float, float]:
            selector_score = _as_float(item.get("selector_score"))
            if selector_score is not None:
                selector_ev = _as_float(item.get("selector_ev"))
                return selector_score, selector_ev if selector_ev is not None else -999.0
            return _prop_pick_score(item)

        def _print_prop_parlay(title: str, links: List[str]) -> None:
            dedup = [
                link for link in list(dict.fromkeys([l for l in links if l]))
                if "fanduel.com" in link and "marketId=" in link and "selectionId=" in link
            ]
            if len(dedup) < 2:
                return
            parlay_url = build_fd_parlay_url(dedup[:25])
            if parlay_url:
                print(f"- {title}: [FD]({parlay_url})")

        def _is_tail_alt_item(item: Dict) -> bool:
            stat_key = str(item.get("stat_key") or "")
            side = str(item.get("side") or "").upper()
            line = _as_float(item.get("line"))
            if side != "O" or line is None:
                return False
            return (
                (stat_key == "batter_hits" and line >= 2.5)
                or (stat_key == "batter_total_bases" and line >= 3.5)
                or (stat_key == "batter_home_runs" and line >= 1.5)
            )

        def _compact_reason(reason: str) -> Optional[str]:
            r = str(reason or "").lower()
            if not r:
                return None
            if "synthetic" in r:
                return "synthetic"
            if "one_sided" in r or ("unknown" in r and "pair_quality" in r):
                return "one-sided"
            if "cross_book" in r:
                return "cross-book"
            if "clv" in r:
                return "CLV-weak"
            if "bookability" in r:
                return "bookability"
            if "opportunity" in r:
                return "opportunity"
            if "bucket_closed" in r or r.startswith("bucket_watch"):
                return "bucket-closed"
            if "bucket_roi_negative" in r or "roi" in r:
                return "ROI-weak"
            if "market_beats" in r or "market_residual" in r or "market_disagreement" in r:
                return "market-better"
            if "alt_line" in r or "lottery" in r:
                return "lottery"
            if "drift" in r:
                return "drift"
            if "no_bet" in r:
                return "no-bet"
            return None

        def _selector_tags(item: Dict) -> List[str]:
            tags: List[str] = []
            tier = str(item.get("selector_tier") or "").lower()
            if tier in {"paper", "lottery", "no_bet"}:
                tags.append("paper" if tier == "paper" else ("lottery" if tier == "lottery" else "no-bet"))
            pair = str(item.get("pair_quality") or "").lower()
            if pair == "same_book":
                tags.append("clean-pair")
            elif pair == "cross_book":
                tags.append("cross-book")
            elif pair in {"synthetic", "one_sided", "unknown"}:
                tags.append("synthetic" if pair == "synthetic" else "one-sided")
            clv_prob = _as_float(item.get("clv_beat_prob"))
            if clv_prob is not None:
                tags.append("CLV+" if clv_prob >= prop_selector_cfg.min_clv_beat_prob else "CLV-weak")
            if not item.get("price_drift_ok", True):
                tags.append("drift")
            for reason in item.get("selector_reasons") or []:
                tag = _compact_reason(reason)
                if tag:
                    tags.append(tag)
            out: List[str] = []
            for tag in tags:
                if tag not in out:
                    out.append(tag)
            return out[:5]

        def _tag_text(item: Dict) -> str:
            tags = _selector_tags(item)
            return f" tags={','.join(tags)}" if tags else ""

        def _reason_counts(rows: List[Dict]) -> Counter:
            counts: Counter = Counter()
            for item in rows:
                row_tags: List[str] = []
                for reason in item.get("selector_reasons") or []:
                    tag = _compact_reason(reason)
                    if tag and tag not in row_tags:
                        row_tags.append(tag)
                if not row_tags:
                    row_tags = _selector_tags(item)
                counts.update(row_tags)
            return counts

        def _print_no_bet_summary(rows: List[Dict]) -> None:
            print("")
            print("**NO-BET SUMMARY**")
            if not rows:
                print("- No selector no-bet rows in the displayed model-pick pool")
                return
            counts = _reason_counts(rows)
            top = ", ".join(f"{reason} {count}" for reason, count in counts.most_common(6))
            print(f"- No-bet rows: {len(rows)}")
            print(f"- Top blockers: {top or 'none'}")

        def _print_data_health(rows: List[Dict], *, offers_missing: bool) -> None:
            health = _prop_offer_health(prop_lines)
            clean_pairs = sum(1 for item in rows if item.get("pair_quality") in {"same_book", "cross_book"})
            synthetic = sum(1 for item in rows if item.get("pair_quality") == "synthetic")
            clv_known = sum(1 for item in rows if _as_float(item.get("clv_beat_prob")) is not None)
            total = len(rows)
            clean_pair_s = f"{(clean_pairs / total):.0%}" if total else "-"
            synthetic_s = f"{(synthetic / total):.0%}" if total else "-"
            clv_s = f"{(clv_known / total):.0%}" if total else "-"
            print("")
            print("**DATA HEALTH**")
            if offers_missing:
                print("- Prop odds not loaded yet - no links/rankings generated")
            else:
                print(
                    f"- Offers: {health['entries']} markets, {health['offer_rows']} normalized offers, "
                    f"{health['linked_sides']} linked sides"
                )
            print(f"- Display pool: {total} rows; clean pair {clean_pair_s}; synthetic {synthetic_s}; CLV model coverage {clv_s}")

        def _print_paper_sections(
            rows: List[Dict],
            *,
            source_label: str,
            title: str = "PAPER PLAYER PROPS",
            include_empty_stats: bool = True,
            heading_kind: str = "Paper",
        ) -> None:
            configured_limit = int(cfg.discord_paper_limit or 10)
            per_section_limit = min(max(configured_limit, 1), 10)
            printed_section = False
            total_shown = 0
            print("")
            print(f"**{title} ({len(rows)} {source_label})**")
            paper_stat_sections = [
                ("pitcher_strikeouts", "Strikeouts"),
                ("batter_total_bases", "Total Bases"),
                ("batter_hits", "Hits"),
                ("batter_home_runs", "Home Runs"),
            ]
            for stat_key, label in paper_stat_sections:
                stat_rows = [item for item in rows if item.get("stat_key") == stat_key]
                stat_rows.sort(key=_pick_score, reverse=True)
                shown = stat_rows[:per_section_limit]
                print("")
                print(f"**Top {per_section_limit} {heading_kind} {label}**")
                if not shown:
                    if include_empty_stats:
                        print("- No qualifying rows")
                    continue
                printed_section = True
                total_shown += len(shown)
                for item in shown:
                    _print_prop_item(item, include_link=cfg.discord_show_paper_links)
            if not printed_section and not include_empty_stats:
                print("- No qualifying player props to show")
            elif total_shown < len(rows):
                print("")
                print(f"- Showing {total_shown} of {len(rows)} paper/research rows")

        if db_rows is not None:
            for row in db_rows:
                item = _db_row_to_prop_item(row)
                if item is None:
                    continue
                if item.get("link") or item.get("ev") is not None:
                    research_rows.append(item)
                ev = item.get("ev")
                if ev is not None and ev >= cfg.min_ev:
                    model_pick_rows.append(item)
                if item["bankroll"].candidate:
                    bankroll_rows.append(item)
                    used_bankroll_exposure += item["bankroll"].stake_pct
                    if item.get("link"):
                        bankroll_links.append(item["link"])

            model_pick_rows.sort(key=_pick_score, reverse=True)
            bankroll_rows.sort(
                key=lambda item: (item["bankroll"].stake_pct, item.get("p_over") or 0.0),
                reverse=True,
            )
            display_source = research_rows if cfg.discord_include_all_priced_props else model_pick_rows
            display_source.sort(key=_pick_score, reverse=True)
            non_bankroll_rows = [item for item in display_source if not item["bankroll"].candidate]
            paper_rows: List[Dict] = []
            no_bet_rows: List[Dict] = []
            watch_rows: List[Dict] = []
            for item in non_bankroll_rows:
                tier = str(item.get("selector_tier") or "").lower()
                if tier == "no_bet" or item.get("no_bet_decision"):
                    no_bet_rows.append(item)
                elif tier == "lottery" or _is_tail_alt_item(item):
                    continue
                elif tier == "paper":
                    paper_rows.append(item)
                elif item in model_pick_rows and not _is_tail_alt_item(item):
                    watch_rows.append(item)
                else:
                    no_bet_rows.append(item)

            prop_record = format_record_summary(
                pg_dsn=cfg.pg_dsn,
                end_date=cfg.et_date,
                lookback_days=30,
                include_game_bankroll=False,
                include_game_model=False,
                include_prop_shadow=True,
            )
            if prop_record:
                print(prop_record)
            offers_missing = _prop_offers_missing(prop_lines)
            _print_data_health(display_source, offers_missing=offers_missing)
            if bankroll_rows:
                print("")
                print(f"**BANKROLL PROP BETS ({len(bankroll_rows)})**")
                for item in bankroll_rows:
                    _print_prop_item(item, include_link=True)
                _print_prop_parlay("Bankroll Props Parlay", bankroll_links)
            else:
                print("")
                print("**BANKROLL PROP BETS**")
                print("- No bankroll-qualified player props today")
                blockers = _reason_counts(non_bankroll_rows)
                if blockers:
                    top_blockers = ", ".join(
                        f"{reason} {count}" for reason, count in blockers.most_common(6)
                    )
                    print(f"- Real-money blockers: {top_blockers}")

            combined_exposure = _locked_bankroll_exposure()
            cap = cfg.bankroll_max_daily_exposure_pct
            if combined_exposure > cap + 1e-12:
                print(
                    f"- GLOBAL CAP WARNING: games + props total {combined_exposure:.2%} "
                    f"vs daily cap {cap:.2%} (over by {combined_exposure - cap:.2%})"
                )

            source_label = "priced props" if cfg.discord_include_all_priced_props else "model picks"
            _print_paper_sections(
                paper_rows,
                source_label=source_label,
                title="PAPER PLAYER PROPS",
                include_empty_stats=True,
            )
            if watch_rows:
                _print_paper_sections(
                    watch_rows,
                    source_label="watch model picks",
                    title="WATCHLIST - NOT SELECTOR APPROVED",
                    include_empty_stats=False,
                    heading_kind="Watch",
                )
            _print_no_bet_summary(no_bet_rows)

            # ── Projection Leaderboards ──────────────────────────────────────────
            # Top 10 Strikeouts (by highest projection)
            k_proj_rows = sorted(
                [r for r in all_pitcher_rows if r.get("pred_strikeouts") is not None],
                key=lambda r: r["pred_strikeouts"], reverse=True,
            )
            if k_proj_rows:
                print("")
                print("**Top 10 Strikeout Projections Today**")
                for i, r in enumerate(k_proj_rows[:10], start=1):
                    _name = r.get("player_name", f"id={r['player_id']}")
                    _team = r.get("team_abbr", "?")
                    _opp = r.get("opponent_abbr", "?")
                    _pred_k = r["pred_strikeouts"]
                    _ld = prop_lines.get((_normalize_name(_name), "pitcher_strikeouts"))
                    if _ld and _ld.get("line") is not None:
                        _lnk = _ld.get("over_link")
                        _link_str = f" [Bet](<{_lnk}>)" if _lnk else ""
                        print(f"{i:>2}. {_name} ({_team} vs {_opp}) — {_pred_k:.1f} · O{_ld['line']:.1f}{_link_str}")
                    else:
                        print(f"{i:>2}. {_name} ({_team} vs {_opp}) — {_pred_k:.1f}")

            # Top 2 Total Bases (by highest projection)
            tb_proj_rows = sorted(
                [r for r in all_batter_rows if r.get("pred_total_bases") is not None],
                key=lambda r: r["pred_total_bases"], reverse=True,
            )
            if tb_proj_rows:
                print("")
                print("**Top 2 Total Bases Projections Today**")
                for i, r in enumerate(tb_proj_rows[:2], start=1):
                    _name = r.get("player_name", f"id={r['player_id']}")
                    _team = r.get("team_abbr", "?")
                    _opp = r.get("opponent_abbr", "?")
                    _pred_tb = r["pred_total_bases"]
                    _ld = prop_lines.get((_normalize_name(_name), "batter_total_bases"))
                    if _ld and _ld.get("line") is not None:
                        _lnk = _ld.get("over_link")
                        _link_str = f" [Bet](<{_lnk}>)" if _lnk else ""
                        print(f"{i:>2}. {_name} ({_team} vs {_opp}) — {_pred_tb:.2f} · O{_ld['line']:.1f}{_link_str}")
                    else:
                        print(f"{i:>2}. {_name} ({_team} vs {_opp}) — {_pred_tb:.2f}")

            # Top 2 Hits (by highest projection)
            h_proj_rows = sorted(
                [r for r in all_batter_rows if r.get("pred_hits") is not None],
                key=lambda r: r["pred_hits"], reverse=True,
            )
            if h_proj_rows:
                print("")
                print("**Top 2 Hits Projections Today**")
                for i, r in enumerate(h_proj_rows[:2], start=1):
                    _name = r.get("player_name", f"id={r['player_id']}")
                    _team = r.get("team_abbr", "?")
                    _opp = r.get("opponent_abbr", "?")
                    _pred_h = r["pred_hits"]
                    _ld = prop_lines.get((_normalize_name(_name), "batter_hits"))
                    if _ld and _ld.get("line") is not None:
                        _lnk = _ld.get("over_link")
                        _link_str = f" [Bet](<{_lnk}>)" if _lnk else ""
                        print(f"{i:>2}. {_name} ({_team} vs {_opp}) — {_pred_h:.2f} · O{_ld['line']:.1f}{_link_str}")
                    else:
                        print(f"{i:>2}. {_name} ({_team} vs {_opp}) — {_pred_h:.2f}")

            # Top 2 Home Runs (by highest projection)
            hr_proj_rows = sorted(
                [r for r in all_batter_rows if r.get("pred_home_runs") is not None],
                key=lambda r: r["pred_home_runs"], reverse=True,
            )
            if hr_proj_rows:
                print("")
                print("**Top 2 Home Run Projections Today**")
                for i, r in enumerate(hr_proj_rows[:2], start=1):
                    _name = r.get("player_name", f"id={r['player_id']}")
                    _team = r.get("team_abbr", "?")
                    _opp = r.get("opponent_abbr", "?")
                    _pred_hr = r["pred_home_runs"]
                    _ld = prop_lines.get((_normalize_name(_name), "batter_home_runs"))
                    if _ld and _ld.get("line") is not None:
                        _lnk = _ld.get("over_link")
                        _link_str = f" [Bet](<{_lnk}>)" if _lnk else ""
                        print(f"{i:>2}. {_name} ({_team} vs {_opp}) — {_pred_hr:.3f} · O{_ld['line']:.1f}{_link_str}")
                    else:
                        print(f"{i:>2}. {_name} ({_team} vs {_opp}) — {_pred_hr:.3f}")

            # ── Lottery picks ────────────────────────────────────────────────────
            if cfg.lottery_mode:
                _legs = lottery_legs if lottery_legs is not None else _collect_lottery_parlay_links(
                    all_pitcher_rows, all_batter_rows, prop_lines, cfg,
                    all_alt_lines=all_alt_lines,
                )
                if _legs:
                    lottery_url = build_fd_parlay_url([c["link"] for c in _legs[:25]])
                    if lottery_url:
                        print("")
                        print(f"• Lottery Parlay ({cfg.lottery_legs} legs, +{cfg.lottery_min_american}–+{cfg.lottery_max_american}): [FD]({lottery_url})")
                else:
                    print("")
                    print("• Lottery Parlay: no qualifying lottery legs today")

            print("")
            return []

        k_entries = _leaderboard_rows(
            all_pitcher_rows, "pred_strikeouts", "pitcher_strikeouts",
            min_p_over=0.52,
        )
        for p_over, pred_val, line, name, team, opp, lnk, bankroll in k_entries[:10]:
            _record_prop_item(
                market="K",
                name=name,
                team=team,
                opp=opp,
                pred_val=pred_val,
                pred_fmt="{:.1f}",
                line=line,
                p_over=p_over,
                link=lnk,
                bankroll=bankroll,
            )

        h_entries = _leaderboard_rows(
            all_batter_rows, "pred_hits", "batter_hits",
            min_pred=1.0,
            min_p_over=0.63,
            min_edge_vs_market=0, over_price_key="market_hits_over_price",
            max_batting_order=7,
            require_confirmed_sp=False,
            sp_k_ceiling=9.0, sp_k_lookup=_sp_k_lookup,
            skip_clf=True,
        )
        for p_over, pred_val, line, name, team, opp, lnk, bankroll in h_entries[:10]:
            _record_prop_item(
                market="H",
                name=name,
                team=team,
                opp=opp,
                pred_val=pred_val,
                pred_fmt="{:.3f}",
                line=line,
                p_over=p_over,
                link=lnk,
                bankroll=bankroll,
            )

        tb_entries = _leaderboard_rows(
            all_batter_rows, "pred_total_bases", "batter_total_bases",
            min_pred=2.0, min_p_over=0.45,
            min_edge_vs_market=0, over_price_key="market_tb_over_price",
            max_batting_order=7,
            require_confirmed_sp=False,
            sp_k_ceiling=9.0, sp_k_lookup=_sp_k_lookup,
            skip_clf=True,
        )
        for p_over, pred_val, line, name, team, opp, lnk, bankroll in tb_entries[:10]:
            _record_prop_item(
                market="TB",
                name=name,
                team=team,
                opp=opp,
                pred_val=pred_val,
                pred_fmt="{:.3f}",
                line=line,
                p_over=p_over,
                link=lnk,
                bankroll=bankroll,
            )

        hr_top_rows_d = sorted(
            [r for r in all_batter_rows
             if r.get("pred_home_runs") is not None
             and float(r.get("pred_home_runs") or 0.0) >= 0.15],
            key=lambda r: r["pred_home_runs"], reverse=True,
        )
        for r in hr_top_rows_d[:10]:
            name = r.get("player_name", f"id={r['player_id']}")
            norm = _normalize_name(name)
            ld = prop_lines.get((norm, "batter_home_runs"))
            if not ld or ld.get("line") is None:
                continue
            line = ld["line"]
            pred_hr = float(r["pred_home_runs"])
            clf_p = (r.get("clf_p_over") or {}).get("batter_home_runs")
            if clf_p is not None:
                p_over = _apply_regression_gate(clf_p, pred_hr, line, "batter_home_runs")
            else:
                p_over = _prob_over_from_regression(pred_hr, line, None)
            bankroll = _prop_bankroll_from_pick(
                stat="batter_home_runs",
                side_label="O",
                cfg=cfg,
                ld=ld,
                p_over=p_over,
                edge=pred_hr - float(line),
                sigma=(r.get("sigma_map") or {}).get("batter_home_runs"),
                blocked_sides=_blocked_sides_for_row(r, "batter_home_runs"),
                bucket_reopen_policy=bucket_reopen_policy,
            )
            _record_prop_item(
                market="HR",
                name=name,
                team=r.get("team_abbr", "?"),
                opp=r.get("opponent_abbr", "?"),
                pred_val=pred_hr,
                pred_fmt="{:.3f}",
                line=line,
                p_over=p_over,
                link=ld.get("over_link"),
                bankroll=bankroll,
            )

        bankroll_rows.sort(
            key=lambda item: (item["bankroll"].stake_pct, item.get("p_over") or 0.0),
            reverse=True,
        )

        if bankroll_rows:
            print(f"**BANKROLL PROP BETS ({len(bankroll_rows)})**")
            for item in bankroll_rows:
                _print_prop_item(item, include_link=True)
        else:
            print("**BANKROLL PROP BETS**")
            print("- No bankroll-qualified player props today")
            blockers = _reason_counts(paper_rows)
            if blockers:
                top_blockers = ", ".join(
                    f"{reason} {count}" for reason, count in blockers.most_common(6)
                )
                print(f"- Real-money blockers: {top_blockers}")

        fd_bankroll_links = [
            link for link in list(dict.fromkeys(bankroll_links))
            if "fanduel.com" in link and "marketId=" in link and "selectionId=" in link
        ]
        parlay_url = build_fd_parlay_url(fd_bankroll_links[:25]) if len(fd_bankroll_links) >= 2 else None
        if parlay_url:
            print(f"- Bankroll Props Parlay: [FD]({parlay_url})")

        combined_exposure = _locked_bankroll_exposure()
        cap = cfg.bankroll_max_daily_exposure_pct
        if combined_exposure > cap + 1e-12:
            print(
                f"- GLOBAL CAP WARNING: games + props total {combined_exposure:.2%} "
                f"vs daily cap {cap:.2%} (over by {combined_exposure - cap:.2%})"
            )

        if paper_rows:
            main_paper_rows = [item for item in paper_rows if not _is_tail_alt_item(item)]
            if main_paper_rows:
                _print_paper_sections(main_paper_rows, source_label="model picks")

        print("")
        return []

        # ── Top 10 Strikeouts ─────────────────────────────────────────────────
        k_entries = _leaderboard_rows(
            all_pitcher_rows, "pred_strikeouts", "pitcher_strikeouts",
            min_p_over=0.52,
        )
        if k_entries:
            print("**Top 10 Strikeouts Today**")
            for i, (p_over, pred_val, line, name, team, opp, lnk, bankroll) in enumerate(k_entries[:10], start=1):
                link_str = f" [Bet](<{lnk}>)" if lnk else ""
                print(f"{i:>2}. {name} ({team} vs {opp}) — {pred_val:.1f} · O{line:.1f} · P={p_over:.1%} [{_bankroll_label(bankroll)}]{link_str}")
            k_links = [lnk for _, _, _, _, _, _, lnk, _bankroll in k_entries[:10] if lnk]
            k_parlay_url = build_fd_parlay_url(k_links[:25]) if k_links else None
            if k_parlay_url:
                print(f"• Top 10 K Parlay: [FD]({k_parlay_url})")
            printed_any = True

        # ── Top 10 Hits ───────────────────────────────────────────────────────
        # skip_clf=True: batter_hits CLF has abs_cal_error=0.18-0.45 (disabled
        # in clf_bucket_controls). Its ~60-70% predictions compress the range
        # and block most batters. Use regression-only Poisson P(hits >= 1).
        # min_pred=1.0: pred>=1.0 vs O0.5 line = edge>=0.5; historical 58.9% WR
        #   (vs 54.6% at pred>=0.75). Higher pred shows stronger discrimination:
        #   pred 1.0-1.1 → 59-62%, pred 1.2 → 68%, pred 1.3 → 78%.
        # min_p_over=0.63: Poisson P(hits>=1 | λ=1.0) ≈ 0.63.
        # min_edge_vs_market=0: market edge gate disabled (book lines ~-250).
        h_entries = _leaderboard_rows(
            all_batter_rows, "pred_hits", "batter_hits",
            min_pred=1.0,
            min_p_over=0.63,
            min_edge_vs_market=0, over_price_key="market_hits_over_price",
            max_batting_order=7,
            require_confirmed_sp=False,
            sp_k_ceiling=9.0, sp_k_lookup=_sp_k_lookup,
            skip_clf=True,
        )
        if h_entries:
            print("")
            print("**Top 10 Hits Today**")
            for i, (p_over, pred_val, line, name, team, opp, lnk, bankroll) in enumerate(h_entries[:10], start=1):
                link_str = f" [Bet](<{lnk}>)" if lnk else ""
                print(f"{i:>2}. {name} ({team} vs {opp}) — {pred_val:.3f} · O{line:.1f} · P={p_over:.1%} [{_bankroll_label(bankroll)}]{link_str}")
            h_links = [lnk for _, _, _, _, _, _, lnk, _bankroll in h_entries[:10] if lnk]
            h_parlay_url = build_fd_parlay_url(h_links[:25]) if h_links else None
            if h_parlay_url:
                print(f"• Top 10 H Parlay: [FD]({h_parlay_url})")
            printed_any = True

        # ── Top 10 Total Bases ────────────────────────────────────────────────
        # skip_clf=True: TB CLF disabled (abs_cal_error=0.32).
        # min_pred=2.0: pred>=2.0 vs O1.5 line = edge>=0.5. Historical win rates:
        #   edge>=0.5 → 54.2% WR on 59 bets (~5/day). Below this threshold
        #   (edge 0-0.5) the model generates negative value (39-44% WR).
        # min_edge_vs_market=0: disabled — market_tb_over_price prices are near
        #   even money (-120 to +110), implying mkt_p ≈ 0.50-0.55.  With Poisson
        #   p_over ≈ 0.65-0.70 for pred=2.0, the probability gap is only 0.10-0.22
        #   and can never reach 0.5.  min_pred=2.0 already enforces edge >= 0.5.
        # min_p_over=0.45: regression Poisson P(TB>=2 | λ=2.0) ≈ 0.59.
        tb_entries = _leaderboard_rows(
            all_batter_rows, "pred_total_bases", "batter_total_bases",
            min_pred=2.0, min_p_over=0.45,
            min_edge_vs_market=0, over_price_key="market_tb_over_price",
            max_batting_order=7,
            require_confirmed_sp=False,
            sp_k_ceiling=9.0, sp_k_lookup=_sp_k_lookup,
            skip_clf=True,
        )
        if tb_entries:
            print("")
            print("**Top TB Today (high confidence only)**")
            for i, (p_over, pred_val, line, name, team, opp, lnk, bankroll) in enumerate(tb_entries[:10], start=1):
                link_str = f" [Bet](<{lnk}>)" if lnk else ""
                print(f"{i:>2}. {name} ({team} vs {opp}) — {pred_val:.3f} · O{line:.1f} · P={p_over:.1%} [{_bankroll_label(bankroll)}]{link_str}")
            tb_links = [lnk for _, _, _, _, _, _, lnk, _bankroll in tb_entries[:10] if lnk]
            tb_parlay_url = build_fd_parlay_url(tb_links[:25]) if tb_links else None
            if tb_parlay_url:
                print(f"• Top TB Parlay: [FD]({tb_parlay_url})")
            printed_any = True

        print("")
        # Dedicated Top-10 HR parlay (single slip unless links are missing).
        top_hr_links = _collect_top_hr_parlay_links(
            all_batter_rows, prop_lines, top_n=7, min_pred=0.17
        )
        if top_hr_links:
            top_hr_url = build_fd_parlay_url(top_hr_links[:25])
            if top_hr_url:
                printed_any = True
                print(f"• Top 7 HR Parlay (pred>=0.17): [FD]({top_hr_url})")

        # ── HR top-10 leaderboard with line, P(HR), and bet link ──────────────
        # Only show players with pred >= 0.15 — below this is base rate territory.
        # Historical: pred 0.00-0.13 → 8-12% HR rate (= base rate, no signal).
        #             pred 0.15+ → 10.7%+ and rising; pred 0.19+ → 14-21%.
        hr_top_rows_d = sorted(
            [r for r in all_batter_rows
             if r.get("pred_home_runs") is not None
             and float(r.get("pred_home_runs") or 0.0) >= 0.15],
            key=lambda r: r["pred_home_runs"], reverse=True,
        )
        _HR_PARLAY_LEGS = 4
        hr_parlay_legs: List[Dict] = []
        if hr_top_rows_d:
            print("")
            print("**Top HR Hitters Today**")
            for i, r in enumerate(hr_top_rows_d[:10], start=1):
                name = r.get("player_name", f"id={r['player_id']}")
                team = r.get("team_abbr", "?")
                opp  = r.get("opponent_abbr", "?")
                pred_hr = r["pred_home_runs"]
                norm = _normalize_name(name)
                ld = prop_lines.get((norm, "batter_home_runs"))
                if ld and ld.get("line") is not None:
                    line = ld["line"]
                    clf_p = (r.get("clf_p_over") or {}).get("batter_home_runs")
                    if clf_p is not None:
                        clf_p = _apply_regression_gate(clf_p, pred_hr, line, "batter_home_runs")
                        p_over = clf_p
                    else:
                        p_over = _prob_over_from_regression(pred_hr, line, None)
                    p_str = f"P={p_over:.1%}" if p_over is not None else ""
                    lnk = ld.get("over_link")
                    link_str = f" [Bet](<{lnk}>)" if lnk else ""
                    bankroll = _prop_bankroll_from_pick(
                        stat="batter_home_runs",
                        side_label="O",
                        cfg=cfg,
                        ld=ld,
                        p_over=p_over,
                        edge=float(pred_hr) - float(line),
                        sigma=(r.get("sigma_map") or {}).get("batter_home_runs"),
                        blocked_sides=_blocked_sides_for_row(r, "batter_home_runs"),
                        bucket_reopen_policy=bucket_reopen_policy,
                    )
                    print(f"{i:>2}. {name} ({team} vs {opp}) — {pred_hr:.3f} · O{line:.1f} · {p_str} [{_bankroll_label(bankroll)}]{link_str}")
                    if lnk and p_over and len(hr_parlay_legs) < _HR_PARLAY_LEGS:
                        hr_parlay_legs.append({
                            "name": name, "team": team, "opp": opp,
                            "pred_hr": pred_hr, "line": line, "p_over": p_over, "lnk": lnk,
                        })
                else:
                    print(f"{i:>2}. {name} ({team} vs {opp}) — {pred_hr:.3f}")
            printed_any = True

        # ── HR lottery parlay (top 4 by pred HR who have a FD line) ───────────
        if hr_parlay_legs:
            combined_p = 1.0
            for leg in hr_parlay_legs:
                combined_p *= leg["p_over"]
            fair_odds = int((1.0 / combined_p - 1.0) * 100) if combined_p > 0 else 0
            print("")
            print(f"**HR Lottery Parlay ({len(hr_parlay_legs)}-leg)**")
            print(f"Combined P: {combined_p:.3%} · Fair value: +{fair_odds:,}")
            for leg in hr_parlay_legs:
                last = leg["name"].split()[-1]
                print(f"• {last} ({leg['team']} vs {leg['opp']}) O{leg['line']:.1f} HR · P={leg['p_over']:.1%} · [Bet FD](<{leg['lnk']}>)")
            printed_any = True

        if cfg.lottery_mode:
            _legs = lottery_legs if lottery_legs is not None else _collect_lottery_parlay_links(
                all_pitcher_rows, all_batter_rows, prop_lines, cfg,
                all_alt_lines=all_alt_lines,
            )
            if _legs:
                lottery_url = build_fd_parlay_url([c["link"] for c in _legs[:25]])
                if lottery_url:
                    printed_any = True
                    print(f"• Lottery Parlay ({cfg.lottery_legs} legs, +{cfg.lottery_min_american}–+{cfg.lottery_max_american}): [FD]({lottery_url})")
            else:
                printed_any = True
                print("• Lottery Parlay: no qualifying lottery legs today")

        if not printed_any:
            print("**No player-prop parlay links for today**")
        print("")
        return []

    def _link_name(full_name: str) -> str:
        """'Fernando Tatis Jr.' → 'Tatis Jr.',  'Michael King' → 'King'."""
        parts = full_name.split()
        if len(parts) >= 2 and parts[-1].rstrip(".").lower() in ("jr", "sr", "ii", "iii", "iv"):
            return f"{parts[-2]} {parts[-1]}"
        return parts[-1] if parts else full_name

    # ── Table-mode helpers (used only when not is_discord) ────────────────────
    NW, PW, LW, EW = 22, 5, 6, 6
    SEP = "─" * (NW + PW + LW + EW)

    def _hdr(section: str, stat: str) -> str:
        return f"{section:<{NW}}{stat:>{PW}}{'LINE':>{LW}}{'EDGE':>{EW}}"

    def _row(name: str, pred: float, pred_fmt: str, line, edge, starred: bool) -> str:
        pfx = "* " if starred else "  "
        nm = (pfx + name[:20]).ljust(NW)
        ps = pred_fmt.format(pred)
        if line is not None:
            ls = ("O" if edge > 0 else "U") + f"{line:.1f}"
            es = f"{edge:+.2f}"
        else:
            ls = es = ""
        return f"{nm}{ps:>{PW}}{ls:>{LW}}{es:>{EW}}"

    # ── Game loop ──────────────────────────────────────────────────────────────
    games_seen = sorted(
        set(r["game_slug"] for r in all_pitcher_rows + all_batter_rows),
        key=lambda s: (game_map.get(s, {}).get("start_ts_utc", ""), s),
    )

    for slug in games_seen:
        gm = game_map.get(slug, {})
        home, away = gm.get("home", "???"), gm.get("away", "???")
        start_ts = gm.get("start_ts_utc")
        time_str = ""
        if start_ts:
            try:
                dt_et = pd.Timestamp(start_ts).tz_convert(_ET)
                hour = dt_et.hour % 12 or 12
                time_str = f"{hour}:{dt_et.strftime('%M %p')} ET"
            except Exception:
                pass

        sp_rows = [r for r in all_pitcher_rows if r["game_slug"] == slug]
        bat_rows = [r for r in all_batter_rows if r["game_slug"] == slug]
        game_header = f"**{away} @ {home}**" + (f" · {time_str}" if time_str else "")

        if is_discord:
            pass  # collected below — Discord mode does one pass over all games

        else:
            # ── Full table mode ────────────────────────────────────────────────
            print(game_header)
            tbl: List[str] = []

            if sp_rows:
                tbl += [_hdr("PITCHER", "K"), SEP]
                for row in sp_rows:
                    name = row.get("player_name", f"id={row['player_id']}")
                    pred_k = row.get("pred_strikeouts")
                    if pred_k is None:
                        continue
                    norm = _normalize_name(name)
                    ld = prop_lines.get((norm, "pitcher_strikeouts"))
                    line = ld["line"] if ld else None
                    clf_p = (row.get("clf_p_over") or {}).get("pitcher_strikeouts")
                    p_over_reg = _prob_over_from_regression(pred_k, line, row.get("sigma_strikeouts"))
                    if clf_p is not None and line is not None:
                        clf_p = _apply_regression_gate(clf_p, pred_k, line, "pitcher_strikeouts")
                    p_over = clf_p if clf_p is not None else p_over_reg
                    ev_pick = _best_side_from_ev(
                        ld or {},
                        p_over,
                        cfg.min_ev,
                        _blocked_sides_for_row(row, "pitcher_strikeouts"),
                    ) if ld and line is not None else None
                    if ev_pick is not None:
                        edge = ev_pick["ev"]
                        has_edge = True
                        display_pred, fmt = (p_over if clf_p is not None else pred_k), "{:.3f}" if clf_p is not None else "{:.1f}"
                        lnk = ev_pick["link"]
                    else:
                        edge = (pred_k - line) if line is not None else None
                        side_key = "over" if (edge or 0.0) > 0 else "under"
                        has_edge = edge is not None and abs(edge) >= _prop_count_threshold(
                            cfg,
                            "pitcher_strikeouts",
                            side_key,
                        )
                        display_pred, fmt = pred_k, "{:.1f}"
                        lnk = ld.get("over_link") if (ld and edge is not None and edge > 0) else (ld.get("under_link") if ld else None)
                    tbl.append(_row(name, display_pred, fmt, line, edge, has_edge))
                    if has_edge and ld:
                        if lnk:
                            fd_links.append(lnk)

            if bat_rows:
                if tbl:
                    tbl.append("")
                tbl += [_hdr("HITS", "H"), SEP]
                for row in bat_rows:
                    name = row.get("player_name", f"id={row['player_id']}")
                    norm = _normalize_name(name)
                    pred_h = row.get("pred_hits")
                    if pred_h is None:
                        continue
                    _ci = math.sqrt(10.0 / max(row.get("n_games_prev_10") or 1, 1))
                    ld = prop_lines.get((norm, "batter_hits"))
                    line = ld["line"] if ld else None
                    edge = (pred_h - line) if line is not None else None
                    p_over = _prob_over_from_regression(pred_h, line, (row.get("sigma_map") or {}).get("batter_hits"))
                    ev_pick = _best_side_from_ev(
                        ld or {},
                        p_over,
                        cfg.min_ev,
                        _blocked_sides_for_row(row, "batter_hits"),
                    ) if ld and line is not None else None
                    if ev_pick is not None and ev_pick["side"] == "under" and "batter_hits" in cfg.fd_over_only:
                        ev_pick = None
                    has_edge = (ev_pick is not None) or (edge is not None and abs(edge) >= cfg.threshold_hits * _ci)
                    if edge is not None and edge < 0 and "batter_hits" in cfg.fd_over_only:
                        has_edge = False
                    tbl.append(_row(name, pred_h, "{:.2f}", line, edge, has_edge))
                    if has_edge and ld:
                        lnk = ev_pick["link"] if ev_pick is not None else (ld.get("over_link") if edge > 0 else ld.get("under_link"))
                        if lnk:
                            fd_links.append(lnk)

            for stat_lbl, hdr_lbl, pred_col, stat_key, thresh, fmt in [
                ("TB", "TOTAL BASES", "pred_total_bases", "batter_total_bases", cfg.threshold_total_bases, "{:.2f}"),
                ("HR", "HOME RUNS",   "pred_home_runs",   "batter_home_runs",   None,                      "{:.3f}"),
            ]:
                stat_edges: List[tuple] = []
                for row in bat_rows:
                    name = row.get("player_name", f"id={row['player_id']}")
                    norm = _normalize_name(name)
                    pred_v = row.get(pred_col)
                    if pred_v is None:
                        continue
                    _ci = math.sqrt(10.0 / max(row.get("n_games_prev_10") or 1, 1))
                    ld = prop_lines.get((norm, stat_key))
                    if not ld or ld.get("line") is None:
                        continue
                    line = ld["line"]
                    # Use binary CLF P(over) when available
                    clf_p = (row.get("clf_p_over") or {}).get(stat_key)
                    if clf_p is not None:
                        clf_p = _apply_regression_gate(clf_p, pred_v, line, stat_key)
                        edge = _signed_probability_edge(clf_p)
                        eff_thresh = cfg.threshold_clf
                        display_pred = clf_p
                        p_over = clf_p
                    else:
                        edge = pred_v - line
                        eff_thresh = (
                            (cfg.threshold_home_runs_over if edge >= 0 else cfg.threshold_home_runs_under)
                            if thresh is None else _prop_count_threshold(
                                cfg,
                                stat_key,
                                "over" if edge >= 0 else "under",
                            )
                        )
                        display_pred = pred_v
                        p_over = _prob_over_from_regression(pred_v, line, (row.get("sigma_map") or {}).get(stat_key))
                    ev_pick = _best_side_from_ev(
                        ld,
                        p_over,
                        cfg.min_ev,
                        _blocked_sides_for_row(row, stat_key),
                    )
                    if ev_pick is not None and ev_pick["side"] == "under" and stat_key in cfg.fd_over_only:
                        ev_pick = None
                    if ev_pick is None and edge < 0 and stat_key in cfg.fd_over_only:
                        continue
                    if ev_pick is not None or abs(edge) >= eff_thresh * _ci:
                        stat_edges.append((name, display_pred, line, edge, ld))
                if stat_edges:
                    tbl += ["", _hdr(hdr_lbl, stat_lbl), SEP]
                    for name, pred_v, line, edge, ld in stat_edges:
                        tbl.append(_row(name, pred_v, fmt, line, edge, True))
                        lnk = ld.get("over_link") if edge > 0 else ld.get("under_link")
                        if lnk:
                            fd_links.append(lnk)

            for line_txt in tbl:
                print(line_txt)
            print("")

    if not is_discord:
        return fd_links

    # ── Discord mode: collect ALL edge plays and show as one ranked list ────────
    k_plays: List[Dict] = []
    for row in all_pitcher_rows:
        name = row.get("player_name", f"id={row['player_id']}")
        pred_k = row.get("pred_strikeouts")
        if pred_k is None:
            continue
        norm = _normalize_name(name)
        ld = prop_lines.get((norm, "pitcher_strikeouts"))
        if not ld or ld.get("line") is None:
            continue
        line = ld["line"]
        p_over_reg = _prob_over_from_regression(pred_k, line, row.get("sigma_strikeouts"))
        clf_p = (row.get("clf_p_over") or {}).get("pitcher_strikeouts")
        if clf_p is not None:
            clf_p = _apply_regression_gate(clf_p, pred_k, line, "pitcher_strikeouts")
        p_over = clf_p if clf_p is not None else p_over_reg
        ev_pick = _best_side_from_ev(
            ld,
            p_over,
            cfg.min_ev,
            _blocked_sides_for_row(row, "pitcher_strikeouts"),
        )
        if ev_pick is not None:
            edge = ev_pick["ev"]
            display_pred = p_over if clf_p is not None else pred_k
            lnk = ev_pick["link"]
            side = "O" if ev_pick["side"] == "over" else "U"
        else:
            edge = _signed_probability_edge(clf_p) if clf_p is not None else (pred_k - line)
            side_key = "over" if edge > 0 else "under"
            min_thr = cfg.threshold_clf if clf_p is not None else _prop_count_threshold(
                cfg,
                "pitcher_strikeouts",
                side_key,
            )
            if abs(edge) < min_thr:
                continue
            display_pred = p_over if clf_p is not None else pred_k
            lnk = ld.get("over_link") if edge > 0 else ld.get("under_link")
            side = "O" if edge > 0 else "U"
        bankroll = _prop_bankroll_from_pick(
            stat="pitcher_strikeouts",
            side_label=side,
            cfg=cfg,
            ld=ld,
            p_over=p_over,
            edge=edge,
            sigma=row.get("sigma_strikeouts"),
            blocked_sides=_blocked_sides_for_row(row, "pitcher_strikeouts"),
            bucket_reopen_policy=bucket_reopen_policy,
        )
        k_plays.append({
            "name": name, "team": row.get("team_abbr", ""), "stat": "K",
            "pred": display_pred, "line": line, "edge": edge, "lnk": lnk, "book": "FD",
            "is_ev": ev_pick is not None,
            "side": side,
            "game_slug": row.get("game_slug"),
            "bankroll": bankroll,
        })

    batter_edge_plays: List[Dict] = []
    for row in all_batter_rows:
        name = row.get("player_name", f"id={row['player_id']}")
        norm = _normalize_name(name)
        team = row.get("team_abbr", "")
        _n_g = row.get("n_games_prev_10") or 0
        _ci = math.sqrt(10.0 / max(_n_g, 1))
        _clf_pover = row.get("clf_p_over", {})
        for pred_col, stat_key, thresh, stat_lbl in [
            ("pred_hits",        "batter_hits",        cfg.threshold_hits,        "H"),
            ("pred_total_bases", "batter_total_bases",  cfg.threshold_total_bases, "TB"),
            ("pred_home_runs",   "batter_home_runs",    None,                      "HR"),
        ]:
            pred_v = row.get(pred_col)
            if pred_v is None:
                continue
            ld = prop_lines.get((norm, stat_key))
            if not ld or ld.get("line") is None:
                continue
            line = ld["line"]

            # Use binary CLF P(over) when available — edge is in probability space
            clf_p = _clf_pover.get(stat_key)
            if clf_p is not None:
                clf_p = _apply_regression_gate(clf_p, pred_v, line, stat_key)
                edge = _signed_probability_edge(clf_p)
                eff_thresh = cfg.threshold_clf
                display_pred = clf_p  # show P(over) directly
                p_over = clf_p
            else:
                edge = pred_v - line
                eff_thresh = (
                    (cfg.threshold_home_runs_over if edge >= 0 else cfg.threshold_home_runs_under)
                    if thresh is None else _prop_count_threshold(
                        cfg,
                        stat_key,
                        "over" if edge >= 0 else "under",
                    )
                )
                display_pred = pred_v
                p_over = _prob_over_from_regression(pred_v, line, (row.get("sigma_map") or {}).get(stat_key))

            ev_pick = _best_side_from_ev(
                ld,
                p_over,
                cfg.min_ev,
                _blocked_sides_for_row(row, stat_key),
            )
            if ev_pick is not None:
                side_over = ev_pick["side"] == "over"
                if (not side_over) and stat_key in cfg.fd_over_only:
                    continue
                edge_for_rank = ev_pick["ev"]
                lnk = ev_pick["link"]
                book = "FD" if (side_over or ld.get("under_link_book", "fanduel") == "fanduel") else "DK"
                side = "O" if side_over else "U"
                is_pick = True
            else:
                if abs(edge) < eff_thresh * _ci:
                    continue
                if edge < 0 and stat_key in cfg.fd_over_only:
                    continue  # FD doesn't offer UNDER for this stat
                lnk = ld.get("over_link") if edge > 0 else ld.get("under_link")
                book = "FD" if (edge > 0 or ld.get("under_link_book", "fanduel") == "fanduel") else "DK"
                edge_for_rank = edge
                side = "O" if edge > 0 else "U"
                is_pick = True

            if is_pick:
                bankroll = _prop_bankroll_from_pick(
                    stat=stat_key,
                    side_label=side,
                    cfg=cfg,
                    ld=ld,
                    p_over=p_over,
                    edge=edge_for_rank,
                    sigma=(row.get("sigma_map") or {}).get(stat_key),
                    blocked_sides=_blocked_sides_for_row(row, stat_key),
                    bucket_reopen_policy=bucket_reopen_policy,
                )
                batter_edge_plays.append({
                    "name": name, "team": team, "stat": stat_lbl,
                    "pred": display_pred, "line": line, "edge": edge_for_rank, "lnk": lnk, "book": book,
                    "is_clf": clf_p is not None,
                    "is_ev": ev_pick is not None,
                    "side": side,
                    "game_slug": row.get("game_slug"),
                    "bankroll": bankroll,
                })

    all_prop_bets: List[Dict] = k_plays + batter_edge_plays
    all_prop_bets.sort(key=lambda x: abs(x["edge"]), reverse=True)
    all_prop_bets = all_prop_bets[:cfg.top_n_bets]
    _cap_bankroll_output(all_prop_bets, cfg)

    if all_prop_bets:
        print(f"**PROP BETS TODAY ({len(all_prop_bets)})**")
        grouped: Dict[str, List[Dict]] = {}
        for b in all_prop_bets:
            grouped.setdefault(b.get("game_slug") or "unknown", []).append(b)

        ordered_slugs = sorted(
            grouped.keys(),
            key=lambda s: (game_map.get(s, {}).get("start_ts_utc", ""), s),
        )
        for slug in ordered_slugs:
            gm = game_map.get(slug, {})
            home, away = gm.get("home", "???"), gm.get("away", "???")
            start_ts = gm.get("start_ts_utc")
            time_str = ""
            if start_ts:
                try:
                    dt_et = pd.Timestamp(start_ts).tz_convert(_ET)
                    hour = dt_et.hour % 12 or 12
                    time_str = f"{hour}:{dt_et.strftime('%M %p')} ET"
                except Exception:
                    pass
            hdr = f"**{away} @ {home}**" + (f" · {time_str}" if time_str else "")
            print(hdr)
            for b in grouped[slug]:
                short = _link_name(b["name"])
                d = b.get("side", ("O" if b["edge"] > 0 else "U"))
                ls = f"{d}{b['line']:.1f}"
                ps = "{:.1f}".format(b["pred"]) if b["stat"] == "K" else "{:.2f}".format(b["pred"])
                link_txt = f"  [Bet {b.get('book', 'FD')}](<{b['lnk']}>)" if b.get("lnk") else ""
                bankroll = b.get("bankroll")
                bankroll_str = f" [{bankroll_tag(bankroll)}]" if bankroll else ""
                print(f"• {short} ({b['team']}) {b['stat']} {ls} → {ps}  +{abs(b['edge']):.2f}{bankroll_str}{link_txt}")
        print("")
    else:
        print("**No prop edge bets today**\n")

    return []  # Discord path exited early above; this covers the non-Discord path


def _print_best_bets(
    all_pitcher_rows: List[Dict],
    all_batter_rows: List[Dict],
    prop_lines: Dict,
    cfg: PredictConfig,
    bucket_reopen_policy: Optional[dict] = None,
) -> List[str]:
    """Print best bets ranked by |edge| as a table. Returns FD links for parlay."""
    is_discord = os.getenv("DISCORD_FORMAT") == "1"
    fd_links: List[str] = []
    best: List[Dict] = []

    for row in all_pitcher_rows:
        name = row.get("player_name", f"id={row['player_id']}")
        pred_k = row.get("pred_strikeouts")
        if pred_k is None:
            continue
        norm = _normalize_name(name)
        ld = prop_lines.get((norm, "pitcher_strikeouts"))
        if not ld or ld["line"] is None:
            continue
        line = ld["line"]
        p_over_reg = _prob_over_from_regression(pred_k, line, row.get("sigma_strikeouts"))
        clf_p = (row.get("clf_p_over") or {}).get("pitcher_strikeouts")
        p_over = clf_p if clf_p is not None else p_over_reg
        ev_pick = _best_side_from_ev(
            ld,
            p_over,
            cfg.min_ev,
            _blocked_sides_for_row(row, "pitcher_strikeouts"),
        )
        if ev_pick is not None:
            edge = ev_pick["ev"]
            is_edge = True
            display_pred = p_over if clf_p is not None else pred_k
            bet_link = ev_pick["link"]
            side = "O" if ev_pick["side"] == "over" else "U"
        else:
            if clf_p is not None:
                edge = _signed_probability_edge(clf_p)
                is_edge = abs(edge) >= cfg.threshold_clf
                display_pred = clf_p
            else:
                edge = pred_k - line
                side_key = "over" if edge > 0 else "under"
                is_edge = abs(edge) >= _prop_count_threshold(
                    cfg,
                    "pitcher_strikeouts",
                    side_key,
                )
                display_pred = pred_k
            bet_link = ld.get("over_link") if edge > 0 else ld.get("under_link")
            side = "O" if edge > 0 else "U"
        if is_edge:
            bankroll = _prop_bankroll_from_pick(
                stat="pitcher_strikeouts",
                side_label=side,
                cfg=cfg,
                ld=ld,
                p_over=p_over,
                edge=edge,
                sigma=row.get("sigma_strikeouts"),
                blocked_sides=_blocked_sides_for_row(row, "pitcher_strikeouts"),
                bucket_reopen_policy=bucket_reopen_policy,
            )
            best.append({
                "name": name, "stat": "K", "pred": display_pred,
                "line": line, "edge": edge,
                "bet_link": bet_link,
                "team": row.get("team_abbr", ""),
                "game_slug": row.get("game_slug", ""),
                "is_ev": ev_pick is not None,
                "side": side,
                "bankroll": bankroll,
            })

    for row in all_batter_rows:
        name = row.get("player_name", f"id={row['player_id']}")
        norm = _normalize_name(name)
        _n_g = row.get("n_games_prev_10") or 0
        _ci_scale = math.sqrt(10.0 / max(_n_g, 1))
        for stat, pred_col, stat_key, threshold in [
            ("H",  "pred_hits",        "batter_hits",        cfg.threshold_hits),
            ("TB", "pred_total_bases",  "batter_total_bases", cfg.threshold_total_bases),
            ("HR", "pred_home_runs",    "batter_home_runs",   None),
        ]:
            pred_v = row.get(pred_col)
            if pred_v is None:
                continue
            ld = prop_lines.get((norm, stat_key))
            if not ld or ld["line"] is None:
                continue
            line = ld["line"]
            edge = pred_v - line
            eff_thr = (
                (cfg.threshold_home_runs_over if edge >= 0 else cfg.threshold_home_runs_under)
                if threshold is None else _prop_count_threshold(
                    cfg,
                    stat_key,
                    "over" if edge >= 0 else "under",
                )
            )
            clf_p = (row.get("clf_p_over") or {}).get(stat_key)
            p_over = clf_p if clf_p is not None else _prob_over_from_regression(
                pred_v, line, (row.get("sigma_map") or {}).get(stat_key)
            )
            ev_pick = _best_side_from_ev(
                ld,
                p_over,
                cfg.min_ev,
                _blocked_sides_for_row(row, stat_key),
            )

            if ev_pick is not None:
                side_over = ev_pick["side"] == "over"
                if (not side_over) and stat_key in cfg.fd_over_only:
                    continue
                edge_pick = ev_pick["ev"]
                bet_link = ev_pick["link"]
                side = "O" if side_over else "U"
            else:
                if abs(edge) < eff_thr * _ci_scale:
                    continue
                if edge < 0 and stat_key in cfg.fd_over_only:
                    continue  # FD doesn't offer UNDER for this stat
                edge_pick = edge
                bet_link = ld.get("over_link") if edge > 0 else ld.get("under_link")
                side = "O" if edge > 0 else "U"

            if ev_pick is not None or abs(edge) >= eff_thr * _ci_scale:
                bankroll = _prop_bankroll_from_pick(
                    stat=stat_key,
                    side_label=side,
                    cfg=cfg,
                    ld=ld,
                    p_over=p_over,
                    edge=edge_pick,
                    sigma=(row.get("sigma_map") or {}).get(stat_key),
                    blocked_sides=_blocked_sides_for_row(row, stat_key),
                    bucket_reopen_policy=bucket_reopen_policy,
                )
                best.append({
                    "name": name, "stat": stat, "pred": pred_v,
                    "line": line, "edge": edge_pick,
                    "bet_link": bet_link,
                    "bookmaker_key": ld.get("bookmaker_key", ""),
                    "team": row.get("team_abbr", ""),
                    "game_slug": row.get("game_slug", ""),
                    "is_ev": ev_pick is not None,
                    "side": side,
                    "bankroll": bankroll,
                })

    best.sort(key=lambda r: abs(r["edge"]), reverse=True)
    _cap_bankroll_output(best, cfg)

    if best:
        print("─" * 40)
        print("**Best Props (ranked by |edge|)**")

        for b in best[:cfg.top_n_bets]:
            parts = b["name"].split()
            if len(parts) >= 2 and parts[-1].rstrip(".").lower() in ("jr", "sr", "ii", "iii", "iv"):
                short = f"{parts[-2]} {parts[-1]} ({b['team']})"
            else:
                short = f"{parts[-1]} ({b['team']})"
            ps = f"{b['pred']:.1f}" if b["stat"] == "K" else f"{b['pred']:.2f}"
            d = b.get("side", ("O" if b["edge"] > 0 else "U"))
            ls = f"{d}{b['line']:.1f}"
            es = f"{b['edge']:+.2f}"
            lnk = b.get("bet_link")
            link_str = f" [Bet FD](<{lnk}>)" if lnk else " [FD]"
            bankroll = b.get("bankroll")
            bankroll_str = f" [{bankroll_tag(bankroll)}]" if bankroll else ""
            print(f"★ {short} {b['stat']} {ls} → {ps}{bankroll_str}{link_str}")

    # Build parlay links with correlation filtering applied to the full ranked list:
    # max 1 prop per player (HR + H + TB from the same player are correlated),
    # max 2 props per game (multiple props from a high-scoring game are correlated).
    _seen_players: set[str] = set()
    _game_counts: dict[str, int] = {}
    for b in best:  # already sorted by |edge| desc
        lnk = b.get("bet_link")
        if not lnk:
            continue
        if b["name"] in _seen_players:
            continue
        _slug = b.get("game_slug", "")
        if _slug and _game_counts.get(_slug, 0) >= 2:
            continue
        fd_links.append(lnk)
        _seen_players.add(b["name"])
        if _slug:
            _game_counts[_slug] = _game_counts.get(_slug, 0) + 1

    return fd_links


def _print_top_hr_hitters(
    all_batter_rows: List[Dict],
    prop_lines: Dict,
    top_n: int = 10,
) -> None:
    """Print the top predicted HR hitters for the day, sorted by pred_home_runs desc."""
    hr_rows = [r for r in all_batter_rows if r.get("pred_home_runs") is not None]
    if not hr_rows:
        return
    hr_rows.sort(key=lambda r: r["pred_home_runs"], reverse=True)

    print("─" * 40)
    print(f"**Top HR Hitters Today (top {min(top_n, len(hr_rows))})**")
    for r in hr_rows[:top_n]:
        name  = r.get("player_name", f"id={r['player_id']}")
        norm  = _normalize_name(name)
        team  = r.get("team_abbr", "?")
        opp   = r.get("opponent_abbr", "?")
        phr   = r["pred_home_runs"]

        ld    = prop_lines.get((norm, "batter_home_runs"))
        line  = ld["line"] if ld else None

        if line is not None:
            edge = phr - line
            dir_str = "O" if edge > 0 else "U"
            bet_link = ld.get("over_link") if edge > 0 else ld.get("under_link")
            link_str = f"  [Bet FD](<{bet_link}>)" if bet_link else ""
            print(
                f"  {name} ({team} vs {opp}) — pred {phr:.3f} | "
                f"{dir_str}{line:.1f} edge {edge:+.2f}{link_str}"
            )
        else:
            print(f"  {name} ({team} vs {opp}) — pred {phr:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main prediction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def predict_props(cfg: PredictConfig) -> None:
    et_date = cfg.et_date or datetime.now(tz=_ET).date()

    # ── Load models ───────────────────────────────────────────────────────────
    model_dir = cfg.model_dir
    pitcher_artifacts_ok = (
        (model_dir / "strikeouts_xgb.json").exists() and
        (model_dir / "feature_columns_pitchers.json").exists()
    )
    batter_artifacts_ok = (
        (model_dir / "hits_xgb.json").exists() and
        (model_dir / "total_bases_xgb.json").exists() and
        (model_dir / "home_runs_xgb.json").exists() and
        (model_dir / "feature_columns_batters.json").exists()
    )

    xgb_k = lgb_k = feat_p = meds_p = bt = None
    xgb_h = lgb_h = xgb_tb_m = lgb_tb_m = None
    xgb_hr = lgb_hr = feat_b = meds_b = None
    # Binary classifier models (optional — loaded if training has been run)
    pitcher_clf_arts = None   # (xgb_clf, lgb_clf, feat_clf, meds_clf, bt_clf, cal_map)
    batter_clf_arts  = None   # (models_dict, feat_clf, meds_clf, bt_clf, cal_map)
    # Alt-line binary CLF (FD all-lines) — used in lottery parlay scoring
    pitcher_alt_clf_arts = None
    batter_alt_clf_arts  = None
    hitter_pa_artifact = None

    if pitcher_artifacts_ok:
        try:
            xgb_k, lgb_k, feat_p, meds_p, bt, k_meta = _load_pitcher_artifacts(model_dir)
        except Exception:
            log.exception("Failed to load pitcher artifacts")
            pitcher_artifacts_ok = False
        try:
            pitcher_clf_arts = _load_pitcher_clf_artifacts(model_dir)
            if pitcher_clf_arts:
                log.info("Loaded pitcher binary CLF model")
        except Exception:
            log.warning("Could not load pitcher CLF artifacts — using regression only")
        try:
            pitcher_alt_clf_arts = _load_pitcher_alt_clf_artifacts(model_dir)
            if pitcher_alt_clf_arts:
                log.info("Loaded pitcher alt-line CLF model")
        except Exception:
            log.warning("Could not load pitcher alt CLF artifacts")

    if batter_artifacts_ok:
        try:
            (xgb_h, lgb_h, xgb_tb_m, lgb_tb_m,
             xgb_hr, lgb_hr,
             feat_b, meds_b, bt) = _load_batter_artifacts(model_dir)
        except Exception:
            log.exception("Failed to load batter artifacts")
            batter_artifacts_ok = False
        try:
            batter_clf_arts = _load_batter_clf_artifacts(model_dir)
            if batter_clf_arts:
                log.info("Loaded batter binary CLF models: %s",
                         list(batter_clf_arts[0].keys()))
        except Exception:
            log.warning("Could not load batter CLF artifacts — using regression only")
        try:
            batter_alt_clf_arts = _load_batter_alt_clf_artifacts(model_dir)
            if batter_alt_clf_arts:
                log.info("Loaded batter alt-line CLF models: %s",
                         list(batter_alt_clf_arts[0].keys()))
        except Exception:
            log.warning("Could not load batter alt CLF artifacts")
        try:
            hitter_pa_artifact = _load_hitter_pa_artifact(model_dir)
        except Exception:
            log.warning("Could not load hitter PA artifact", exc_info=True)

    if not pitcher_artifacts_ok and not batter_artifacts_ok:
        log.warning("No prop models found. Run train_player_prop_models first.")
        print("_(No prop models — run train_player_prop_models first)_")
        return

    # ── Connect and fetch data ─────────────────────────────────────────────
    conn = psycopg2.connect(cfg.pg_dsn)
    _ensure_schema(conn)

    prop_lines = _load_prop_lines(conn, et_date)
    log.info("Loaded %d prop line entries for %s", len(prop_lines), et_date)
    all_alt_lines = _load_all_alt_lines(conn, et_date)
    log.info("Loaded alt lines for %d (player, stat) keys", len(all_alt_lines))

    bias = _load_bias_corrections(conn)
    side_penalties = _load_side_penalties(conn)
    clf_controls = _load_clf_bucket_controls(cfg.model_dir, cfg.clf_controls_file)
    side_recalibrators = _load_prop_side_recalibrators(cfg.model_dir, cfg.side_recalibrators_file)
    betting_layer = _load_prop_betting_layer(cfg.model_dir, cfg.betting_layer_file)
    market_side_priors = _load_prop_market_side_priors(cfg.model_dir, cfg.market_side_priors_file)
    walk_forward_policy = (
        _load_prop_walk_forward_policy(cfg.model_dir, cfg.walk_forward_policy_file)
        if cfg.apply_walk_forward_policy
        else {}
    )
    bucket_reopen_policy = _load_prop_bucket_reopen_policy(cfg.model_dir, cfg.bucket_reopen_policy_file)

    # ── Pitcher predictions ────────────────────────────────────────────────
    all_pitcher_rows: List[Dict] = []
    db_rows: List[Dict] = []
    # Alt-CLF probability dicts — populated inside pitcher/batter blocks, used by lottery
    _pitcher_alt_clf_probs: Dict = {}
    _batter_alt_clf_probs: Dict = {}

    if pitcher_artifacts_ok:
        df_p = pd.read_sql(SQL_PITCHER_SNAPSHOTS, conn, params={"game_date": et_date})
        log.info("Pitcher snapshots: %d rows", len(df_p))

        if not df_p.empty:
            X_p = _prep_features(df_p, _PITCHER_META, feat_p, meds_p)
            pred_k_raw = _predict_ensemble(X_p, xgb_k, lgb_k)
            if k_meta.get("is_residual"):
                # Residual model: output is delta vs market K line. Reconstruct.
                _modal_k = float(k_meta.get("modal_k_line", 4.5))
                _mkt_k_arr = pd.to_numeric(
                    df_p["market_k_line"], errors="coerce"
                ).fillna(_modal_k).to_numpy()
                pred_k = _mkt_k_arr + pred_k_raw
            else:
                pred_k = pred_k_raw
            sigma_k = bt.get("ci_strikeouts") if bt else None
            norms_p = np.array([
                _normalize_name(row.get("player_name", f"id={row['player_id']}"))
                for _, row in df_p.iterrows()
            ])
            pitcher_offer_contexts = [
                _prop_line_data_for_row(prop_lines, norm, "pitcher_strikeouts", row)
                for norm, (_, row) in zip(norms_p, df_p.iterrows())
            ]
            clf_pover_k: Optional[np.ndarray] = None
            if pitcher_clf_arts is not None:
                xgb_clf, lgb_clf, feat_p_clf, meds_p_clf, _bt_p_clf, cal_p_map = pitcher_clf_arts
                lines_p = np.array([
                    (line_data or {}).get("line", np.nan)
                    for line_data, _offers in pitcher_offer_contexts
                ], dtype=float)
                X_p_clf = _prep_features_clf(X_p, lines_p, feat_p_clf, meds_p_clf)
                raw_p = np.clip(_predict_ensemble(X_p_clf, xgb_clf, lgb_clf), 0.01, 0.99)
                clf_pover_k = _apply_platt_calibration(raw_p, cal_p_map.get("pitcher_strikeouts"))
                clf_pover_k[np.isnan(lines_p)] = np.nan

            if pitcher_alt_clf_arts is not None:
                _pitcher_alt_clf_probs = _compute_alt_clf_probs(
                    X_p,
                    list(norms_p),
                    all_alt_lines,
                    ["pitcher_strikeouts"],
                    _PITCHER_META,
                    pitcher_alt_clf_arts,
                )
                if _pitcher_alt_clf_probs:
                    log.info("Alt CLF pitcher strikeouts: %d line-level probabilities", len(_pitcher_alt_clf_probs))

            for i, (_, row) in enumerate(df_p.iterrows()):
                pk = max(0.0, float(pred_k[i]))
                name = row.get("player_name", f"id={row['player_id']}")
                norm = _normalize_name(name)
                ld, offer_rows = pitcher_offer_contexts[i]
                line = ld["line"] if ld else None
                try:
                    _ip_avg_5 = float(row.get("ip_avg_5")) if row.get("ip_avg_5") is not None else None
                except Exception:
                    _ip_avg_5 = None
                pitcher_opportunity = {
                    "projected_bf": (_ip_avg_5 * 4.25) if _ip_avg_5 is not None else None,
                    "projected_pitch_count": (_ip_avg_5 * 16.5) if _ip_avg_5 is not None else None,
                }
                # Dynamic side-penalty (with shrinkage) for weak directional buckets.
                if line is not None:
                    pk, _pen_k = _apply_count_side_penalty(
                        "pitcher_strikeouts",
                        pk,
                        line,
                        side_penalties,
                    )
                p_over_k = None
                clf_family_k = "clf"
                alt_p_over_k = _lookup_alt_clf_prob(
                    _pitcher_alt_clf_probs,
                    norm,
                    "pitcher_strikeouts",
                    line,
                )
                if alt_p_over_k is not None:
                    p_over_k = alt_p_over_k
                    clf_family_k = "alt_clf"
                if clf_pover_k is not None:
                    v = float(clf_pover_k[i])
                    if p_over_k is None:
                        p_over_k = None if np.isnan(v) else v
                clf_k_disabled = (
                    p_over_k is not None
                    and line is not None
                    and _clf_bucket_is_disabled(
                        clf_controls,
                        "pitcher_strikeouts",
                        line,
                        honor_controls=cfg.honor_clf_bucket_controls,
                    )
                )

                if (
                    p_over_k is not None
                    and line is not None
                    and not clf_k_disabled
                ):
                    p_over_k = _apply_regression_gate(p_over_k, pk, line, "pitcher_strikeouts")
                    p_over_k, edge, _cal_key = _apply_prop_side_recalibration(
                        stat="pitcher_strikeouts",
                        line=line,
                        raw_p_over=p_over_k,
                        model_family=clf_family_k,
                        ld=ld,
                        recalibrators=side_recalibrators,
                        betting_layer=betting_layer,
                        market_side_priors=market_side_priors,
                        apply_market_side_priors=cfg.apply_market_side_priors,
                        market_side_prior_max_blend=cfg.market_side_prior_max_blend,
                        walk_forward_policy=walk_forward_policy,
                        opportunity_features=pitcher_opportunity,
                    )
                    kel = 0.0
                    pred_for_db = p_over_k
                    pred_prob_for_db = p_over_k
                    edge_type = "probability"
                    model_family = clf_family_k
                else:
                    pred_for_db = pk
                    pred_prob_for_db = _prob_over_from_regression(pk, line, sigma_k) if line is not None else None
                    pred_prob_for_db, edge, _cal_key = _apply_prop_side_recalibration(
                        stat="pitcher_strikeouts",
                        line=line,
                        raw_p_over=pred_prob_for_db,
                        model_family="regression",
                        ld=ld,
                        recalibrators=side_recalibrators,
                        betting_layer=betting_layer,
                        market_side_priors=market_side_priors,
                        apply_market_side_priors=cfg.apply_market_side_priors,
                        market_side_prior_max_blend=cfg.market_side_prior_max_blend,
                        walk_forward_policy=walk_forward_policy,
                        opportunity_features=pitcher_opportunity,
                    )
                    kel = 0.0
                    edge_type = "probability" if pred_prob_for_db is not None else "count"
                    model_family = "regression"

                r = {
                    "game_slug": row["game_slug"],
                    "game_date_et": et_date,
                    "player_id": int(row["player_id"]),
                    "player_name": name,
                    "team_abbr": row.get("team_abbr"),
                    "is_home": row.get("is_home"),
                    "opponent_abbr": row.get("opponent_abbr"),
                    "start_ts_utc": row.get("start_ts_utc"),
                    "pred_strikeouts": pk,
                    "projected_bf": pitcher_opportunity.get("projected_bf"),
                    "projected_pitch_count": pitcher_opportunity.get("projected_pitch_count"),
                    "clf_p_over": {"pitcher_strikeouts": None if clf_k_disabled else p_over_k},
                    "sigma_strikeouts": sigma_k,
                    "weak_prop_sides": {
                        "pitcher_strikeouts": _weak_sides_for_line(
                            "pitcher_strikeouts",
                            line,
                            side_penalties,
                        ),
                    },
                }
                all_pitcher_rows.append(r)
                for offer in offer_rows:
                    offer_ld = _offer_line_data(offer, offer_rows)
                    offer_line = offer_ld.get("line")
                    offer_side = offer_ld.get("selected_offer_side")
                    offer_raw_p = None
                    offer_family = "regression"
                    alt_offer_p = _lookup_alt_clf_prob(
                        _pitcher_alt_clf_probs,
                        norm,
                        "pitcher_strikeouts",
                        offer_line,
                    )
                    if alt_offer_p is not None:
                        offer_raw_p = _apply_regression_gate(
                            alt_offer_p,
                            pk,
                            offer_line,
                            "pitcher_strikeouts",
                        )
                        offer_family = "alt_clf"
                    elif (
                        clf_pover_k is not None
                        and _same_line(offer_line, line)
                        and not clf_k_disabled
                    ):
                        v = float(clf_pover_k[i])
                        if not np.isnan(v):
                            offer_raw_p = _apply_regression_gate(
                                v,
                                pk,
                                offer_line,
                                "pitcher_strikeouts",
                            )
                            offer_family = "clf"
                    if offer_raw_p is None and offer_line is not None:
                        offer_raw_p = _prob_over_from_regression(pk, offer_line, sigma_k)
                        offer_family = "regression"
                    offer_p_over, offer_edge, _offer_cal_key = _apply_prop_side_recalibration(
                        stat="pitcher_strikeouts",
                        line=offer_line,
                        raw_p_over=offer_raw_p,
                        model_family=offer_family,
                        ld=offer_ld,
                        recalibrators=side_recalibrators,
                        betting_layer=betting_layer,
                        market_side_priors=market_side_priors,
                        apply_market_side_priors=cfg.apply_market_side_priors,
                        market_side_prior_max_blend=cfg.market_side_prior_max_blend,
                        walk_forward_policy=walk_forward_policy,
                        opportunity_features=pitcher_opportunity,
                    )
                    db_rows.append(_prop_db_row(
                        game_date_et=et_date,
                        game_slug=row["game_slug"],
                        player_id=int(row["player_id"]),
                        player_name=name,
                        team_abbr=row.get("team_abbr"),
                        stat="pitcher_strikeouts",
                        pred_value=offer_p_over if offer_family != "regression" else pk,
                        pred_count=pk,
                        pred_prob_over=offer_p_over,
                        book_line=offer_line,
                        edge=offer_edge,
                        edge_type="probability" if offer_p_over is not None else "count",
                        model_family=offer_family,
                        kelly_fraction=0.0,
                        ld=offer_ld,
                        forced_side=offer_side,
                        cfg=cfg,
                        blocked_sides=_blocked_sides_for_row(r, "pitcher_strikeouts"),
                        bucket_reopen_policy=bucket_reopen_policy,
                    ))

    # ── Batter predictions ─────────────────────────────────────────────────
    all_batter_rows: List[Dict] = []

    if batter_artifacts_ok:
        prior_season = f"{et_date.year - 1}-regular"
        df_b = pd.read_sql(
            SQL_BATTER_SNAPSHOTS, conn,
            params={
                "game_date": et_date,
                "prior_season": prior_season,
                "min_ab_avg_10": cfg.min_ab_avg_10,
                "min_n_games": cfg.min_n_games,
            }
        )
        log.info("Batter snapshots: %d rows", len(df_b))

        # Fetch player names from boxscore (first + last name) and SP table as fallback
        name_df = pd.read_sql(
            """SELECT player_id,
                  TRIM(first_name || ' ' || last_name) AS player_name
               FROM (
                   SELECT DISTINCT ON (player_id)
                       player_id,
                       first_name,
                       last_name
                   FROM raw.mlb_boxscore_player_stats
                   WHERE first_name IS NOT NULL AND last_name IS NOT NULL
                   ORDER BY player_id, game_slug DESC
               ) sub
               UNION ALL
               SELECT player_id, player_name
               FROM raw.mlb_starting_pitchers
               WHERE player_name IS NOT NULL""",
            conn
        )
        # Deduplicate: keep first occurrence
        name_df = name_df.drop_duplicates(subset=["player_id"], keep="first")
        name_map = dict(zip(name_df["player_id"], name_df["player_name"]))

        if not df_b.empty:
            X_b = _prep_features(df_b, _BATTER_META, feat_b, meds_b)
            pred_h     = _predict_ensemble(X_b, xgb_h,    lgb_h)
            pred_tb    = _predict_ensemble(X_b, xgb_tb_m, lgb_tb_m)
            pred_hr    = _predict_ensemble(X_b, xgb_hr,   lgb_hr)
            sigma_h  = bt.get("ci_hits")        if bt else None
            sigma_tb = bt.get("ci_total_bases") if bt else None
            sigma_hr = bt.get("ci_home_runs")   if bt else None
            hitter_pa_infos = _predict_validated_hitter_pa(df_b, hitter_pa_artifact)
            _pa_scales = [
                float(info.get("pa_scale") or 1.0)
                for info in hitter_pa_infos
                if info.get("pa_model_source") in {"validated_pa_model", "validated_two_part_pa"}
            ]
            if _pa_scales:
                log.info(
                    "Applied validated hitter PA model to %d batter rows (avg count scale %.3f)",
                    len(_pa_scales),
                    float(np.mean(_pa_scales)),
                )


            # Binary classifier predictions (if models trained)
            # One P(over) array per stat — indexed same as df_b rows
            clf_pover: Dict[str, Optional[np.ndarray]] = {
                "batter_hits": None, "batter_total_bases": None,
                "batter_home_runs": None,
            }
            pids_arr = df_b["player_id"].astype(int).values
            norms_arr = np.array([_normalize_name(name_map.get(p, "")) for p in pids_arr])
            batter_offer_contexts = {
                stat_key: [
                    _prop_line_data_for_row(prop_lines, norm, stat_key, row)
                    for norm, (_, row) in zip(norms_arr, df_b.iterrows())
                ]
                for stat_key in clf_pover
            }
            if batter_clf_arts is not None:
                clf_models, feat_clf, meds_clf, _bt_clf, cal_map_b = batter_clf_arts
                for stat_key in clf_pover:
                    if stat_key not in clf_models:
                        continue
                    lines_arr = np.array([
                        (line_data or {}).get("line", np.nan)
                        for line_data, _offers in batter_offer_contexts[stat_key]
                    ], dtype=float)
                    X_clf = _prep_features_clf(X_b, lines_arr, feat_clf, meds_clf)
                    xgb_c, lgb_c = clf_models[stat_key]
                    raw_p = np.clip(_predict_ensemble(X_clf, xgb_c, lgb_c), 0.01, 0.99)
                    pover = _apply_platt_calibration(raw_p, cal_map_b.get(stat_key))
                    # Mask rows with no line — use NaN so they fall back to regression
                    pover[np.isnan(lines_arr)] = np.nan
                    clf_pover[stat_key] = pover
                    over_pct = float(np.nanmean(pover > 0.524)) * 100
                    log.info("CLF %s: %d preds, %.1f%% positive P(over)>breakeven",
                             stat_key, int(np.sum(~np.isnan(pover))), over_pct)

            if batter_alt_clf_arts is not None:
                _batter_alt_clf_probs = _compute_alt_clf_probs(
                    X_b,
                    list(norms_arr),
                    all_alt_lines,
                    ["batter_hits", "batter_total_bases"],
                    _BATTER_META,
                    batter_alt_clf_arts,
                )
                if _batter_alt_clf_probs:
                    log.info("Alt CLF batters: %d line-level probabilities", len(_batter_alt_clf_probs))

            seen = set()
            for i, (_, row) in enumerate(df_b.iterrows()):
                pid = int(row["player_id"])
                slug = row["game_slug"]
                key = (slug, pid)
                if key in seen:
                    continue
                seen.add(key)

                name = name_map.get(pid, f"id={pid}")
                norm = _normalize_name(name)
                pa_info = hitter_pa_infos[i] if i < len(hitter_pa_infos) else {}
                pa_scale = float(pa_info.get("pa_scale") or 1.0)
                # Regression predictions with additive bias correction
                raw_ph  = max(0.0, float(pred_h[i])  + bias.get("batter_hits",        0.0)) * pa_scale
                raw_ptb = max(0.0, float(pred_tb[i]) + bias.get("batter_total_bases", 0.0)) * pa_scale
                raw_phr = max(0.0, float(pred_hr[i]) + bias.get("batter_home_runs",   0.0)) * pa_scale

                row_offer_contexts = {
                    stat_key: batter_offer_contexts[stat_key][i]
                    for stat_key in clf_pover
                }
                line_h = (row_offer_contexts["batter_hits"][0] or {}).get("line")
                line_tb = (row_offer_contexts["batter_total_bases"][0] or {}).get("line")
                line_hr = (row_offer_contexts["batter_home_runs"][0] or {}).get("line")
                ph, _pen_h = _apply_count_side_penalty("batter_hits", raw_ph, line_h, side_penalties)
                ptb, _pen_tb = _apply_count_side_penalty("batter_total_bases", raw_ptb, line_tb, side_penalties)
                phr, _pen_hr = _apply_count_side_penalty("batter_home_runs", raw_phr, line_hr, side_penalties)

                # Collect binary CLF P(over) for each stat (None when clf unavailable or no line).
                def _safe_clf_with_family(stat_key: str) -> tuple[Optional[float], Optional[str]]:
                    ld_for_stat = row_offer_contexts[stat_key][0]
                    line_for_stat = ld_for_stat["line"] if ld_for_stat else None
                    alt_p = _lookup_alt_clf_prob(
                        _batter_alt_clf_probs,
                        norm,
                        stat_key,
                        line_for_stat,
                    )
                    if alt_p is not None:
                        return alt_p, "alt_clf"
                    arr = clf_pover.get(stat_key)
                    if arr is None:
                        return None, None
                    if _clf_bucket_is_disabled(
                        clf_controls,
                        stat_key,
                        line_for_stat,
                        honor_controls=cfg.honor_clf_bucket_controls,
                    ):
                        return None, None
                    v = float(arr[i])
                    return (None, None) if np.isnan(v) else (v, "clf")

                _clf_h, _clf_h_family = _safe_clf_with_family("batter_hits")
                _clf_tb, _clf_tb_family = _safe_clf_with_family("batter_total_bases")
                _clf_hr, _clf_hr_family = _safe_clf_with_family("batter_home_runs")

                _conf_order = pa_info.get("confirmed_batting_order")
                _effective_order = pa_info.get("effective_batting_order")
                _projected_pa = pa_info.get("projected_pa")
                batter_opportunity = {
                    "confirmed_batting_order": _conf_order,
                    "projected_pa": _projected_pa,
                    "baseline_projected_pa": pa_info.get("baseline_projected_pa"),
                    "validated_projected_pa": pa_info.get("validated_projected_pa"),
                    "pa_model_scale": pa_scale,
                    "pa_model_source": pa_info.get("pa_model_source"),
                    "opp_model_low_pa": pa_info.get("low_pa_probability"),
                    "opp_model_normal_pa": pa_info.get("normal_projected_pa"),
                }
                r = {
                    "game_slug": slug,
                    "game_date_et": et_date,
                    "player_id": pid,
                    "player_name": name,
                    "team_abbr": row.get("team_abbr"),
                    "is_home": row.get("is_home"),
                    "opponent_abbr": row.get("opponent_abbr"),
                    "start_ts_utc": row.get("start_ts_utc"),
                    "n_games_prev_10": int(row.get("n_games_prev_10") or 0),
                    "pred_hits": ph,
                    "pred_total_bases": ptb,
                    "pred_home_runs": phr,
                    "raw_pred_hits": raw_ph,
                    "raw_pred_total_bases": raw_ptb,
                    "raw_pred_home_runs": raw_phr,
                    "weak_prop_sides": {
                        "batter_hits": _weak_sides_for_line("batter_hits", line_h, side_penalties),
                        "batter_total_bases": _weak_sides_for_line("batter_total_bases", line_tb, side_penalties),
                        "batter_home_runs": _weak_sides_for_line("batter_home_runs", line_hr, side_penalties),
                    },
                    # Binary CLF P(over) per stat — None when clf not trained or line unavailable
                    "clf_p_over": {
                        "batter_hits":        _clf_h,
                        "batter_total_bases": _clf_tb,
                        "batter_home_runs":   _clf_hr,
                    },
                    "clf_model_family": {
                        "batter_hits":        _clf_h_family,
                        "batter_total_bases": _clf_tb_family,
                        "batter_home_runs":   _clf_hr_family,
                    },
                    "sigma_map": {
                        "batter_hits": sigma_h,
                        "batter_total_bases": sigma_tb,
                        "batter_home_runs": sigma_hr,
                    },
                    # Leaderboard quality filters
                    "market_hits_over_price": row.get("market_hits_over_price"),
                    "market_tb_over_price":   row.get("market_tb_over_price"),
                    "opp_sp_source":          row.get("opp_sp_source"),
                    "projected_pa":           _projected_pa,
                    "baseline_projected_pa":  pa_info.get("baseline_projected_pa"),
                    "validated_projected_pa": pa_info.get("validated_projected_pa"),
                    "pa_model_scale":         pa_scale,
                    "pa_model_source":        pa_info.get("pa_model_source"),
                    "opp_model_low_pa":       pa_info.get("low_pa_probability"),
                    "opp_model_normal_pa":    pa_info.get("normal_projected_pa"),
                    # Effective batting order: confirmed if available, else rolling avg
                    "effective_batting_order": _effective_order,
                }
                all_batter_rows.append(r)

                for stat_key, stat_label, reg_pred, sigma in [
                    ("batter_hits",        "batter_hits",        ph,  sigma_h),
                    ("batter_total_bases", "batter_total_bases", ptb, sigma_tb),
                    ("batter_home_runs",   "batter_home_runs",   phr, sigma_hr),
                ]:
                    ld, offer_rows = row_offer_contexts[stat_key]
                    line = ld["line"] if ld else None
                    canonical_clf_p = (r.get("clf_p_over") or {}).get(stat_key)
                    canonical_clf_family = (r.get("clf_model_family") or {}).get(stat_key) or "clf"
                    for offer in offer_rows:
                        offer_ld = _offer_line_data(offer, offer_rows)
                        offer_line = offer_ld.get("line")
                        offer_side = offer_ld.get("selected_offer_side")
                        offer_raw_p = None
                        offer_family = "regression"
                        alt_offer_p = _lookup_alt_clf_prob(
                            _batter_alt_clf_probs,
                            norm,
                            stat_key,
                            offer_line,
                        )
                        if alt_offer_p is not None:
                            offer_raw_p = _apply_regression_gate(
                                alt_offer_p,
                                reg_pred,
                                offer_line,
                                stat_key,
                            )
                            offer_family = "alt_clf"
                        elif (
                            canonical_clf_p is not None
                            and _same_line(offer_line, line)
                            and not _clf_bucket_is_disabled(
                                clf_controls,
                                stat_key,
                                offer_line,
                                honor_controls=cfg.honor_clf_bucket_controls,
                            )
                        ):
                            offer_raw_p = _apply_regression_gate(
                                canonical_clf_p,
                                reg_pred,
                                offer_line,
                                stat_key,
                            )
                            offer_family = canonical_clf_family
                        if offer_raw_p is None and offer_line is not None:
                            offer_raw_p = _prob_over_from_regression(reg_pred, offer_line, sigma)
                            offer_family = "regression"
                        offer_p_over, offer_edge, _offer_cal_key = _apply_prop_side_recalibration(
                            stat=stat_key,
                            line=offer_line,
                            raw_p_over=offer_raw_p,
                            model_family=offer_family,
                            ld=offer_ld,
                            recalibrators=side_recalibrators,
                            betting_layer=betting_layer,
                            market_side_priors=market_side_priors,
                            apply_market_side_priors=cfg.apply_market_side_priors,
                            market_side_prior_max_blend=cfg.market_side_prior_max_blend,
                            walk_forward_policy=walk_forward_policy,
                            opportunity_features=batter_opportunity,
                        )
                        db_rows.append(_prop_db_row(
                            game_date_et=et_date,
                            game_slug=slug,
                            player_id=pid,
                            player_name=name,
                            team_abbr=row.get("team_abbr"),
                            stat=stat_label,
                            pred_value=offer_p_over if offer_family != "regression" else reg_pred,
                            pred_count=reg_pred,
                            pred_prob_over=offer_p_over,
                            book_line=offer_line,
                            edge=offer_edge,
                            edge_type="probability" if offer_p_over is not None else "count",
                            model_family=offer_family,
                            kelly_fraction=0.0,
                            ld=offer_ld,
                            forced_side=offer_side,
                            cfg=cfg,
                            blocked_sides=_blocked_sides_for_row(r, stat_label),
                            bucket_reopen_policy=bucket_reopen_policy,
                        ))

    # ── Save to DB ────────────────────────────────────────────────────────
    db_rows = _apply_prop_shadow_selector_gate(db_rows, cfg)
    db_rows = _apply_prop_real_money_kill_switch(db_rows, cfg)

    try:
        existing_exposure, existing_keys, existing_slots = locked_bankroll_state(
            conn,
            et_date,
        )
    except Exception:
        conn.rollback()
        log.exception("Could not load locked MLB bankroll state; failing prop cap closed")
        existing_exposure = cfg.bankroll_max_daily_exposure_pct
        existing_keys = set()
        existing_slots = set()
    db_rows = _cap_prop_db_rows(
        db_rows,
        cfg,
        existing_exposure_pct=existing_exposure,
        existing_pick_keys=existing_keys,
        existing_risk_slots=existing_slots,
        prop_lines=prop_lines,
    )
    try:
        locked = insert_prop_bankroll_ledger(conn, db_rows, prop_lines=prop_lines, cfg=cfg)
        log.info("Locked %d MLB prop bankroll ledger rows", locked)
    except Exception:
        conn.rollback()
        log.exception("Failed to lock MLB prop bankroll ledger rows")
    try:
        locked_exposure, locked_keys, locked_slots = locked_bankroll_state(conn, et_date)
    except Exception:
        conn.rollback()
        log.exception("Could not verify locked MLB prop bankroll rows; failing output closed")
        locked_exposure = cfg.bankroll_max_daily_exposure_pct
        locked_keys = set()
        locked_slots = set()
    db_rows = _cap_prop_db_rows(
        db_rows,
        cfg,
        existing_exposure_pct=locked_exposure,
        existing_pick_keys=locked_keys,
        existing_risk_slots=locked_slots,
        prop_lines=prop_lines,
        require_locked=True,
    )
    _save_predictions(conn, db_rows)
    try:
        locked = insert_prop_model_pick_ledger(conn, db_rows, prop_lines=prop_lines, cfg=cfg)
        log.info("Locked %d MLB prop model-pick ledger rows", locked)
    except Exception:
        conn.rollback()
        log.exception("Failed to lock MLB prop model-pick ledger rows")

    # ── Collect lottery legs once (no conn needed) then persist ───────────
    lottery_legs_collected: List[Dict] = []
    if cfg.lottery_mode:
        _alt_clf_probs = {**_pitcher_alt_clf_probs, **_batter_alt_clf_probs}
        lottery_legs_collected = _collect_lottery_parlay_links(
            all_pitcher_rows, all_batter_rows, prop_lines, cfg,
            all_alt_lines=all_alt_lines,
            alt_clf_probs=_alt_clf_probs if _alt_clf_probs else None,
        )
        _save_lottery_picks(conn, lottery_legs_collected, et_date)
    _grade_lottery_picks(conn)

    conn.close()

    if not all_pitcher_rows and not all_batter_rows:
        print(f"_(No prop data for {et_date})_")
        return

    # ── Build game map for grouping ───────────────────────────────────────
    game_map: Dict[str, Dict] = {}
    for row in all_pitcher_rows + all_batter_rows:
        slug = row["game_slug"]
        if slug not in game_map:
            parts = slug.split("-")
            away = parts[1] if len(parts) > 2 else "?"
            home = parts[2] if len(parts) > 2 else "?"
            game_map[slug] = {
                "home": home,
                "away": away,
                "start_ts_utc": row.get("start_ts_utc"),
            }

    # ── Print output ──────────────────────────────────────────────────────
    header = f"⚾ **MLB Props — {et_date.strftime('%b')} {et_date.day}**"
    print(header)
    print("")

    is_discord = os.getenv("DISCORD_FORMAT") == "1"

    _print_discord(all_pitcher_rows, all_batter_rows, prop_lines, game_map, cfg,
                   all_alt_lines=all_alt_lines, lottery_legs=lottery_legs_collected,
                   db_rows=db_rows, bucket_reopen_policy=bucket_reopen_policy)

    if not is_discord:
        fd_links = _print_best_bets(
            all_pitcher_rows,
            all_batter_rows,
            prop_lines,
            cfg,
            bucket_reopen_policy=bucket_reopen_policy,
        )

        # Top HR hitters leaderboard
        _print_top_hr_hitters(all_batter_rows, prop_lines)

        # Best props parlay (high-edge bets only) — chunked at 25 legs
        seen_bp: set[str] = set()
        fd_links_dedup = [l for l in fd_links if l not in seen_bp and not seen_bp.add(l)]  # type: ignore[func-returns-value]
        n_bp = len(fd_links_dedup)
        n_bp_chunks = math.ceil(n_bp / 25) if n_bp else 0
        for i in range(0, n_bp, 25):
            chunk = fd_links_dedup[i:i + 25]
            parlay_url = build_fd_parlay_url(chunk)
            if parlay_url:
                suffix = f" {i // 25 + 1}/{n_bp_chunks}" if n_bp_chunks > 1 else ""
                print(f"\n**Best Props Parlay{suffix}** [FD]({parlay_url})")

        # All props parlay — every player, model's predicted direction, chunked at 25 legs
        if cfg.include_all_props_parlay:
            all_links = _collect_all_prop_links(all_pitcher_rows, all_batter_rows, prop_lines)
            seen: set[str] = set()
            all_links_dedup = [l for l in all_links if l not in seen and not seen.add(l)]  # type: ignore[func-returns-value]
            n_all = len(all_links_dedup)
            n_chunks = math.ceil(n_all / 25) if n_all else 0
            for i in range(0, n_all, 25):
                chunk = all_links_dedup[i:i + 25]
                ap = build_fd_parlay_url(chunk)
                if ap:
                    suffix = f" {i // 25 + 1}/{n_chunks}" if n_chunks > 1 else ""
                    print(f"\n**All Props Parlay{suffix}** [FD]({ap})")

        # Lottery parlay — use pre-collected legs (already saved to DB above)
        if cfg.lottery_mode:
            if lottery_legs_collected:
                lottery_url = build_fd_parlay_url([c["link"] for c in lottery_legs_collected[:25]])
                if lottery_url:
                    print(f"\n**Lottery Parlay** ({cfg.lottery_legs} legs, +{cfg.lottery_min_american}–+{cfg.lottery_max_american}) [FD]({lottery_url})")
            else:
                print("\n**Lottery Parlay**: no qualifying legs today")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--lottery-mode", action="store_true")
    parser.add_argument("--lottery-legs", type=int, default=None)
    parser.add_argument("--lottery-min-american", type=int, default=None)
    parser.add_argument("--lottery-max-american", type=int, default=None)
    parser.add_argument("--lottery-max-per-game", type=int, default=None)
    parser.add_argument("--bucket-reopen-policy-file", type=str, default=None)
    parser.add_argument("--fun-reopen-props", action="store_true")
    parser.add_argument("--walk-forward-policy-file", type=str, default=None)
    parser.add_argument("--disable-walk-forward-policy", action="store_true")
    parser.add_argument("--discord-paper-limit", type=int, default=None)
    parser.add_argument("--hide-discord-paper-links", action="store_true")
    parser.add_argument(
        "--discord-model-picks-only",
        action="store_true",
        help="Keep Discord paper sections limited to positive-EV model picks.",
    )
    args = parser.parse_args()

    et_date = None
    if args.date:
        from datetime import date as _date
        et_date = _date.fromisoformat(args.date)
    elif os.getenv("MLB_ET_DATE"):
        from datetime import date as _date
        et_date = _date.fromisoformat(os.getenv("MLB_ET_DATE"))

    lottery_mode_env = os.getenv("MLB_LOTTERY_MODE")
    if args.lottery_mode:
        lottery_mode = True
    elif lottery_mode_env is not None:
        lottery_mode = lottery_mode_env.strip().lower() in {"1", "true", "yes", "on"}
    else:
        # Discord compact output defaults to showing a lottery section.
        lottery_mode = os.getenv("DISCORD_FORMAT") == "1"
    lottery_legs = args.lottery_legs if args.lottery_legs is not None else int(os.getenv("MLB_LOTTERY_LEGS", "5"))
    lottery_min_american = (
        args.lottery_min_american
        if args.lottery_min_american is not None
        else int(os.getenv("MLB_LOTTERY_MIN_AMERICAN", "300"))
    )
    lottery_max_american = (
        args.lottery_max_american
        if args.lottery_max_american is not None
        else int(os.getenv("MLB_LOTTERY_MAX_AMERICAN", "900"))
    )
    lottery_max_per_game = (
        args.lottery_max_per_game
        if args.lottery_max_per_game is not None
        else int(os.getenv("MLB_LOTTERY_MAX_PER_GAME", "2"))
    )
    fun_reopen_env = os.getenv("MLB_PROP_FUN_REOPEN")
    fun_reopen_props = args.fun_reopen_props or (
        fun_reopen_env is not None
        and fun_reopen_env.strip().lower() in {"1", "true", "yes", "on"}
    )
    bucket_reopen_policy_file = (
        args.bucket_reopen_policy_file
        or os.getenv("MLB_PROP_BUCKET_REOPEN_POLICY_FILE")
        or ("prop_bucket_reopen_policy_fun.json" if fun_reopen_props else "prop_bucket_reopen_policy.json")
    )
    walk_forward_policy_file = (
        args.walk_forward_policy_file
        or os.getenv("MLB_PROP_WALK_FORWARD_POLICY_FILE")
        or "prop_walk_forward_accuracy_report.json"
    )
    walk_forward_env = os.getenv("MLB_PROP_APPLY_WALK_FORWARD_POLICY")
    apply_walk_forward_policy = not args.disable_walk_forward_policy
    if walk_forward_env is not None and not args.disable_walk_forward_policy:
        apply_walk_forward_policy = walk_forward_env.strip().lower() in {"1", "true", "yes", "on"}
    paper_links_env = os.getenv("MLB_DISCORD_PAPER_LINKS")
    discord_show_paper_links = not args.hide_discord_paper_links
    if paper_links_env is not None and not args.hide_discord_paper_links:
        discord_show_paper_links = paper_links_env.strip().lower() in {"1", "true", "yes", "on"}
    all_priced_env = os.getenv("MLB_DISCORD_ALL_PRICED_PROPS")
    discord_include_all_priced_props = False
    if all_priced_env is not None and not args.discord_model_picks_only:
        discord_include_all_priced_props = all_priced_env.strip().lower() in {"1", "true", "yes", "on"}
    discord_paper_limit = (
        args.discord_paper_limit
        if args.discord_paper_limit is not None
        else int(os.getenv("MLB_DISCORD_PAPER_LIMIT", "10"))
    )
    bankroll_reference_usd = float(os.getenv("MLB_BANKROLL_REFERENCE_USD", "1000"))
    bankroll_micro_stake_usd = float(os.getenv("MLB_PROP_MICRO_STAKE_USD", "1"))
    bankroll_starter_stake_pct = float(os.getenv("MLB_PROP_STARTER_STAKE_PCT", "0.001"))

    cfg = PredictConfig(
        et_date=et_date,
        lottery_mode=lottery_mode,
        lottery_legs=lottery_legs,
        lottery_min_american=lottery_min_american,
        lottery_max_american=lottery_max_american,
        lottery_max_per_game=lottery_max_per_game,
        bucket_reopen_policy_file=bucket_reopen_policy_file,
        walk_forward_policy_file=walk_forward_policy_file,
        apply_walk_forward_policy=apply_walk_forward_policy,
        discord_show_paper_links=discord_show_paper_links,
        discord_include_all_priced_props=discord_include_all_priced_props,
        discord_paper_limit=discord_paper_limit,
        bankroll_reference_usd=bankroll_reference_usd,
        bankroll_micro_stake_usd=bankroll_micro_stake_usd,
        bankroll_starter_stake_pct=bankroll_starter_stake_pct,
    )
    _apply_threshold_overrides(cfg)
    predict_props(cfg)


if __name__ == "__main__":
    main()
