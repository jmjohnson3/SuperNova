# src/mlb_pipeline/modeling/predict_player_props.py
"""
MLB player prop predictions for today's slate.

Loads pitcher/batter prop models and generates predictions for:
  - pitcher_strikeouts   (starting pitchers from raw.mlb_starting_pitchers)
  - batter_hits          (batters with ab_avg_10 >= 2.5 playing today)
  - batter_total_bases
  - batter_home_runs
  - batter_walks

Edge formula:  edge = pred - book_line
Bet signal:    |edge| >= threshold (K: 2.0, H: 0.75, TB: 0.6, HR: 0.45, BB: 0.3)

Discord output (DISCORD_FORMAT=1): compact grouped by game.
DB: bets.mlb_prop_predictions — one row per (game_date_et, game_slug, player_id, stat).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from .features import add_player_prop_derived_features, build_fd_parlay_url

log = logging.getLogger("mlb_pipeline.modeling.predict_player_props")
_ET = ZoneInfo("America/New_York")

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
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
    # Raised from 0.5 → 0.75: optimal threshold per scan (75-25, ROI +43%); UNDER side dominant
    threshold_hits: float = 0.75
    # Raised from 0.6 → 1.5: 2026-04-16 scan shows optimal 1.50 (15-5, ROI +43.2%, n=20);
    # OVER bets lose at all thresholds — UNDER side dominant
    threshold_total_bases: float = 1.5
    # Split OVER/UNDER thresholds for HR:
    # UNDER threshold kept at 0.45 (scan optimal, ROI +78%) but UNDER bets aren't bookable at -500+.
    # OVER threshold set to 0.05 — any positive edge above the 0.5 line qualifies.
    # E[HR] for top hitters peaks around 0.45-0.55 after bias correction, so threshold must be low.
    threshold_home_runs_over: float = 0.05
    threshold_home_runs_under: float = 0.45
    # Lowered from 0.30 → 0.05: 2026-04-16 scan shows optimal 0.05 (12-1, ROI +76.2%, n=13)
    threshold_walks: float = 0.05
    # Binary CLF threshold: P(over) - 0.524 >= this value to flag a bet.
    # 0.03 = P(over) > 55.4% — modest edge above breakeven.
    threshold_clf: float = 0.03
    # Minimum expected value per 1.0 unit stake to consider a bet.
    # Example: 0.02 means +2% EV.
    min_ev: float = 0.02
    # FanDuel does not offer UNDER for these batter props — suppress UNDER bets in output
    fd_over_only: frozenset = frozenset({"batter_hits", "batter_home_runs", "batter_walks"})
    # Optional per-stat threshold overrides loaded from model artifact JSON.
    thresholds_file: str = "prop_thresholds.json"
    # Optional classifier bucket controls file; used to disable drifted CLF buckets.
    clf_controls_file: str = "clf_bucket_controls.json"
    # Lottery parlay mode: HR/K/H/TB biased to plus-money lines with binary clf edge.
    lottery_mode: bool = False
    lottery_legs: int = 5
    lottery_min_american: int = 300   # floor: at least +300 per leg
    lottery_max_american: int = 900   # ceiling: cap at +900 to avoid true longshots
    lottery_max_per_game: int = 2


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot SQL
# ─────────────────────────────────────────────────────────────────────────────

SQL_PITCHER_SNAPSHOTS = """
WITH games_today AS (
    SELECT game_slug, home_team_abbr, away_team_abbr, start_ts_utc, game_date_et
    FROM raw.mlb_games
    WHERE game_date_et = %(game_date)s
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
    -- Market total (game-level run environment)
    mkt_odds.market_total                                        AS market_total,
    -- Pitcher handedness
    ph.pitch_hand                                                AS pitcher_hand,
    -- Opponent lineup quality (NULL for upcoming games, median-imputed by model)
    lq_opp.lineup_avg_avg_10                                     AS opp_lineup_avg_avg_10,
    lq_opp.lineup_iso_avg_10                                     AS opp_lineup_iso_avg_10,
    lq_opp.top4_slg_avg_10                                       AS opp_top4_slg_avg_10,
    lq_opp.lineup_k_pct_std                                      AS opp_lineup_k_pct_std,
    lq_opp.lineup_k_pct_cv                                       AS opp_lineup_k_pct_cv,
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
    cf.framing_rate               AS catcher_framing_rate
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
LEFT JOIN features.mlb_lineup_quality lq_opp
    ON lq_opp.game_slug  = ts.game_slug
    AND lq_opp.team_abbr = ts.opponent_abbr
-- Statcast pitcher profile (season-level)
LEFT JOIN raw.mlb_statcast_pitching sc_p
    ON sc_p.player_id = ts.player_id
    AND sc_p.season_year = EXTRACT(YEAR FROM ts.game_date_et)::INT
-- Extended Statcast: pitcher's own arsenal whiff/K profile
LEFT JOIN raw.mlb_statcast_pitcher_arsenal pa_self
    ON pa_self.player_id = ts.player_id
    AND pa_self.season_year = EXTRACT(YEAR FROM ts.game_date_et)::INT
-- Market total (game-level run environment signal)
LEFT JOIN LATERAL (
    SELECT o.total_points AS market_total
    FROM odds.mlb_game_lines o
    JOIN raw.mlb_games mg ON mg.home_team_abbr = o.home_team
    WHERE mg.game_slug = ts.game_slug
      AND o.as_of_date = ts.game_date_et
      AND o.total_points IS NOT NULL
    ORDER BY o.fetched_at_utc DESC
    LIMIT 1
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
ORDER BY ts.start_ts_utc, ts.game_slug, ts.player_id
"""

SQL_BATTER_SNAPSHOTS = """
WITH games_today AS (
    SELECT game_slug, home_team_abbr, away_team_abbr, start_ts_utc, game_date_et
    FROM raw.mlb_games
    WHERE game_date_et = %(game_date)s
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
    br.n_games_prev_10,
    br.rest_days,
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
    -- Market total (game-level run environment)
    mkt_odds.market_total                                        AS market_total,
    -- Lineup slot
    br.batting_order_avg_5,
    br.batting_order_avg_10,
    -- Batter + opponent SP handedness (OHE'd by _prep_features)
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
    -- Sample sizes
    bvh.n_games_vs_lhp_40,
    bvh.n_games_vs_rhp_40,
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
    -- Statcast: opposing SP's batted-ball-against profile
    sc_opp_p.barrel_batted_rate  AS opp_sp_sc_barrel_rate,
    sc_opp_p.hard_hit_percent    AS opp_sp_sc_hard_hit_pct,
    sc_opp_p.avg_exit_velocity   AS opp_sp_sc_avg_exit_velo,
    sc_opp_p.groundballs_percent AS opp_sp_sc_gb_pct,
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
    opp_tp.bp_era_7d         AS opp_bp_era_7d
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
-- Weather
LEFT JOIN raw.mlb_weather wx ON wx.game_slug = tt.game_slug
LEFT JOIN raw.mlb_venues  v  ON v.venue_id = (
    SELECT venue_id FROM raw.mlb_games WHERE game_slug = tt.game_slug LIMIT 1
)
-- Own-team lineup quality (NULL for upcoming games)
LEFT JOIN features.mlb_lineup_quality lq_own
    ON lq_own.game_slug  = tt.game_slug
    AND lq_own.team_abbr = tt.team_abbr
-- Statcast: batter's own batted-ball profile (season-level)
LEFT JOIN raw.mlb_statcast_batting sc_b
    ON sc_b.player_id = rp.player_id
    AND sc_b.season_year = EXTRACT(YEAR FROM tt.game_date_et)::INT
-- Statcast: opposing SP's batted-ball-against profile
LEFT JOIN raw.mlb_statcast_pitching sc_opp_p
    ON sc_opp_p.player_id = sp.player_id
    AND sc_opp_p.season_year = EXTRACT(YEAR FROM tt.game_date_et)::INT
-- Extended Statcast: batter sprint speed
LEFT JOIN raw.mlb_statcast_sprint_speed ss
    ON ss.player_id = rp.player_id
    AND ss.season_year = EXTRACT(YEAR FROM tt.game_date_et)::INT
-- Extended Statcast: opposing SP fastball arsenal
LEFT JOIN raw.mlb_statcast_pitcher_arsenal pa
    ON pa.player_id = sp.player_id
    AND pa.season_year = EXTRACT(YEAR FROM tt.game_date_et)::INT
-- Market total (game-level run environment signal)
LEFT JOIN LATERAL (
    SELECT o.total_points AS market_total
    FROM odds.mlb_game_lines o
    JOIN games_today gt ON gt.home_team_abbr = o.home_team
    WHERE gt.game_slug = tt.game_slug
      AND o.as_of_date = tt.game_date_et
      AND o.total_points IS NOT NULL
    ORDER BY o.fetched_at_utc DESC
    LIMIT 1
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
WHERE br.ab_avg_10 >= %(min_ab_avg_10)s
  AND br.n_games_prev_10 >= %(min_n_games)s
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
    """Returns (xgb_k, lgb_k, feature_cols, medians, backtest)."""
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

    return xgb_k, lgb_k, feat, meds, bt


def _load_batter_artifacts(model_dir: Path):
    """Returns (xgb_hits, lgb_hits, xgb_tb, lgb_tb, xgb_hr, lgb_hr,
                xgb_walks, lgb_walks, feature_cols, medians, backtest)."""
    xgb_h = XGBRegressor()
    xgb_h.load_model(str(model_dir / "hits_xgb.json"))

    xgb_tb = XGBRegressor()
    xgb_tb.load_model(str(model_dir / "total_bases_xgb.json"))

    xgb_hr = XGBRegressor()
    xgb_hr.load_model(str(model_dir / "home_runs_xgb.json"))

    xgb_walks = XGBRegressor()
    xgb_walks.load_model(str(model_dir / "walks_xgb.json"))

    lgb_h = lgb_tb_m = lgb_hr = lgb_walks = None
    if _HAS_LGB:
        h_path    = model_dir / "lgb_hits.txt"
        tb_path   = model_dir / "lgb_total_bases.txt"
        hr_path   = model_dir / "lgb_home_runs.txt"
        walk_path = model_dir / "lgb_walks.txt"
        if h_path.exists():
            lgb_h = lgb.Booster(model_file=str(h_path))
        if tb_path.exists():
            lgb_tb_m = lgb.Booster(model_file=str(tb_path))
        if hr_path.exists():
            lgb_hr = lgb.Booster(model_file=str(hr_path))
        if walk_path.exists():
            lgb_walks = lgb.Booster(model_file=str(walk_path))

    feat = json.loads((model_dir / "feature_columns_batters.json").read_text())
    meds = json.loads((model_dir / "feature_medians_batters.json").read_text())

    bt_path = model_dir / "backtest_mae.json"
    bt = json.loads(bt_path.read_text()) if bt_path.exists() else {}

    return xgb_h, lgb_h, xgb_tb, lgb_tb_m, xgb_hr, lgb_hr, xgb_walks, lgb_walks, feat, meds, bt


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
        "batter_walks":       ("walks_batter_clf_xgb.json", "lgb_walks_batter_clf.txt"),
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
        stats = ["_pitcher_k"]  # internal key; caller maps to "pitcher_strikeouts"

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


def _apply_threshold_overrides(cfg: PredictConfig) -> None:
    """Load threshold overrides from models/player_props/prop_thresholds.json.

    Supported keys (either style):
      - direct config names:
          threshold_strikeouts, threshold_hits, threshold_total_bases,
          threshold_home_runs_over, threshold_home_runs_under,
          threshold_walks, threshold_clf
      - stat-group names:
          pitcher_strikeouts, batter_hits, batter_total_bases, batter_walks,
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
        "threshold_hits",
        "threshold_total_bases",
        "threshold_home_runs_over",
        "threshold_home_runs_under",
        "threshold_walks",
        "threshold_clf",
        "min_ev",
    ):
        if key in raw:
            _set(key, raw[key])

    # Stat-group names
    if "pitcher_strikeouts" in raw:
        v = raw["pitcher_strikeouts"]
        _set("threshold_strikeouts", v.get("abs_edge") if isinstance(v, dict) else v)
    if "batter_hits" in raw:
        v = raw["batter_hits"]
        _set("threshold_hits", v.get("abs_edge") if isinstance(v, dict) else v)
    if "batter_total_bases" in raw:
        v = raw["batter_total_bases"]
        _set("threshold_total_bases", v.get("abs_edge") if isinstance(v, dict) else v)
    if "batter_walks" in raw:
        v = raw["batter_walks"]
        _set("threshold_walks", v.get("abs_edge") if isinstance(v, dict) else v)
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
        "Loaded threshold overrides from %s | K=%.2f H=%.2f TB=%.2f HR(o/u)=%.2f/%.2f BB=%.2f CLF=%.3f EV=%.3f",
        path,
        cfg.threshold_strikeouts,
        cfg.threshold_hits,
        cfg.threshold_total_bases,
        cfg.threshold_home_runs_over,
        cfg.threshold_home_runs_under,
        cfg.threshold_walks,
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
    if stat == "batter_walks":
        if l < 1.0:
            return "BB 0.5"
        return "BB 1.5+"
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
) -> bool:
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

    All MLB prop stats (hits, TB, HR, strikeouts, walks) are non-negative integer
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
    "batter_walks":       0.44,
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
                       AVG(actual_value - pred_value)   AS mean_bias
                FROM bets.mlb_prop_predictions
                WHERE over_hit IS NOT NULL
                  AND pred_value IS NOT NULL
                  AND actual_value IS NOT NULL
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
                        pred_value::float AS pred_value,
                        actual_value::float AS actual_value,
                        (pred_value::float - actual_value::float) AS pred_minus_actual
                    FROM bets.mlb_prop_predictions
                    WHERE game_date_et >= %s
                      AND over_hit IS NOT NULL
                      AND stat IN ('pitcher_strikeouts', 'batter_total_bases')
                      AND book_line IS NOT NULL
                      AND pred_value IS NOT NULL
                      AND actual_value IS NOT NULL
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
                    AVG(pred_value::float - actual_value::float) AS mean_pred_minus_actual
                FROM bets.mlb_prop_predictions
                WHERE game_date_et >= %s
                  AND over_hit IS NOT NULL
                  AND stat IN ('pitcher_strikeouts', 'batter_total_bases')
                  AND pred_value IS NOT NULL
                  AND actual_value IS NOT NULL
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


# ─────────────────────────────────────────────────────────────────────────────
# DB table
# ─────────────────────────────────────────────────────────────────────────────

_ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bets.mlb_prop_predictions (
    id               SERIAL PRIMARY KEY,
    game_date_et     DATE        NOT NULL,
    game_slug        TEXT        NOT NULL,
    player_id        BIGINT      NOT NULL,
    player_name      TEXT,
    team_abbr        TEXT,
    stat             TEXT        NOT NULL,
    pred_value       NUMERIC,
    book_line        NUMERIC,
    edge             NUMERIC,
    kelly_fraction   NUMERIC,
    actual_value     NUMERIC,
    over_hit         BOOLEAN,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (game_date_et, game_slug, player_id, stat)
);
"""

_UPSERT_SQL = """
INSERT INTO bets.mlb_prop_predictions
    (game_date_et, game_slug, player_id, player_name, team_abbr, stat,
     pred_value, book_line, edge, kelly_fraction)
VALUES
    (%(game_date_et)s, %(game_slug)s, %(player_id)s, %(player_name)s, %(team_abbr)s,
     %(stat)s, %(pred_value)s, %(book_line)s, %(edge)s, %(kelly_fraction)s)
ON CONFLICT (game_date_et, game_slug, player_id, stat) DO UPDATE SET
    player_name    = EXCLUDED.player_name,
    pred_value     = EXCLUDED.pred_value,
    book_line      = EXCLUDED.book_line,
    edge           = EXCLUDED.edge,
    kelly_fraction = EXCLUDED.kelly_fraction
"""


def _ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_ENSURE_TABLE_SQL)
    conn.commit()


def _save_predictions(conn, rows: List[Dict]) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(_UPSERT_SQL, row)
    conn.commit()
    log.info("Saved %d prop prediction rows", len(rows))


# ─────────────────────────────────────────────────────────────────────────────
# Prop line loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_prop_lines(conn, game_date: date) -> Dict[Tuple[str, str], Dict]:
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
            if not entry.get("under_link") and key in dk_rows:
                pass  # already have DK under_link in entry
            entry["under_link_book"] = "draftkings"
            result[key] = entry
        else:
            # FD-only stat (e.g., HR with no DK line): pick most even-money FD line.
            best_line = min(fd_lines, key=lambda x: abs(fd_lines[x].get("over_price") or 99999))
            entry = dict(fd_lines[best_line])
            entry["under_link_book"] = "fanduel"
            result[key] = entry

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


def _best_side_from_ev(ld: Dict, p_over: Optional[float], min_ev: float) -> Optional[Dict]:
    """Return best EV side dict when EV is computable and above threshold."""
    if p_over is None:
        return None
    ev_over = _ev_per_unit(p_over, ld.get("over_price"))
    ev_under = _ev_per_unit(1.0 - p_over, ld.get("under_price"))
    cands = []
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
            ("pred_walks",       "batter_walks"),
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
        "batter_walks": [],
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
            ("pred_walks", "batter_walks"),
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
    top_n: int = 10,
    max_per_game: int = 2,
) -> List[str]:
    """Collect top-N HR prediction links (deduped) for a single HR-focused parlay."""
    rows = [r for r in all_batter_rows if r.get("pred_home_runs") is not None]
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
) -> List[str]:
    """Collect matchup-based lottery legs using the highest alt line where model has edge.

    For each pitcher/batter we look at every available FD alt line (low→high) and
    find the HIGHEST line value where P(OVER) is still positive-EV at the offered
    odds.  This surfaces bets like "Skubal K OVER 8.5 at +350" or
    "Freeman TB OVER 3.5 at +500" when the model's regression prediction supports it.
    No HR — those have a dedicated parlay.

    Candidates are ranked by EV × streak_mult so hot-streak players rise to the top
    and cold-streak players fall even if their raw EV is marginally positive.
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
            # Prefer alt CLF probability; fall back to Poisson regression estimate
            if alt_clf_probs:
                p_over = alt_clf_probs.get((norm, "pitcher_strikeouts", float(line_val)))
            else:
                p_over = None
            if p_over is None:
                p_over = _prob_over_from_regression(pred_k, line_val, sigma_k)
            if p_over is None:
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
                # Prefer alt CLF probability; fall back to Poisson regression estimate
                if alt_clf_probs:
                    p_over = alt_clf_probs.get((norm, stat_key, float(line_val)))
                else:
                    p_over = None
                if p_over is None:
                    p_over = _prob_over_from_regression(pred_v, line_val, sigma_v)
                if p_over is None:
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
            WHEN 'batter_walks'       THEN gl.walks_batter::numeric
        END,
        over_hit = CASE lp.stat
            WHEN 'pitcher_strikeouts' THEN gl.strikeouts_pitcher > lp.book_line
            WHEN 'batter_hits'        THEN gl.hits > lp.book_line
            WHEN 'batter_total_bases' THEN gl.total_bases > lp.book_line
            WHEN 'batter_home_runs'   THEN gl.home_runs > lp.book_line
            WHEN 'batter_walks'       THEN gl.walks_batter > lp.book_line
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


def _print_discord(
    all_pitcher_rows: List[Dict],
    all_batter_rows: List[Dict],
    prop_lines: Dict,
    game_map: Dict[str, Dict],  # game_slug -> {home, away, start_ts_utc}
    cfg: PredictConfig,
    all_alt_lines: Optional[Dict] = None,
    lottery_legs: Optional[List[Dict]] = None,
) -> List[str]:
    """Print per-game prop output. Returns edge-play links for parlay.

    DISCORD_FORMAT=1  →  compact mode: prints Top-10 props and chunked (25-leg max)
                         stat-specific FD parlay links (K / H / TB / HR / BB).
    (no env var)      →  full table mode: all players in aligned columns.
    """
    is_discord = os.getenv("DISCORD_FORMAT") == "1"
    fd_links: List[str] = []

    # Discord mode: print concise Top-10 props + chunked all-props parlays.
    if is_discord:
        by_stat = _collect_prop_links_by_stat(all_pitcher_rows, all_batter_rows, prop_lines)
        printed_any = False
        for stat_key, label in [
            ("pitcher_strikeouts", "Strikeout"),
            ("batter_hits", "Hits"),
            ("batter_total_bases", "2 Total Bases"),
            ("batter_home_runs", "Home Runs"),
            ("batter_walks", "Walks"),
        ]:
            links = by_stat.get(stat_key, [])
            seen: set[str] = set()
            dedup = [l for l in links if l and l not in seen and not seen.add(l)]  # type: ignore[func-returns-value]
            if not dedup:
                continue
            printed_any = True
            n_chunks = math.ceil(len(dedup) / 25)
            for i in range(0, len(dedup), 25):
                chunk = dedup[i:i + 25]
                parlay_url = build_fd_parlay_url(chunk)
                if not parlay_url:
                    continue
                idx = i // 25 + 1
                print(f"• {label} Parlay {idx}/{n_chunks}: [FD]({parlay_url})")

        # Dedicated Top-10 HR parlay (single slip unless links are missing).
        top_hr_links = _collect_top_hr_parlay_links(all_batter_rows, prop_lines, top_n=10)
        if top_hr_links:
            top_hr_url = build_fd_parlay_url(top_hr_links[:25])
            if top_hr_url:
                printed_any = True
                print(f"• Top 10 HR Parlay: [FD]({top_hr_url})")

        # ── HR top-10 leaderboard with line, P(HR), and bet link ──────────────
        hr_top_rows_d = sorted(
            [r for r in all_batter_rows if r.get("pred_home_runs") is not None],
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
                    print(f"{i:>2}. {name} ({team} vs {opp}) — {pred_hr:.3f} · O{line:.1f} · {p_str}{link_str}")
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
                    ev_pick = _best_side_from_ev(ld or {}, p_over, cfg.min_ev) if ld and line is not None else None
                    if ev_pick is not None:
                        edge = ev_pick["ev"]
                        has_edge = True
                        display_pred, fmt = (p_over if clf_p is not None else pred_k), "{:.3f}" if clf_p is not None else "{:.1f}"
                        lnk = ev_pick["link"]
                    else:
                        edge = (pred_k - line) if line is not None else None
                        has_edge = edge is not None and abs(edge) >= cfg.threshold_strikeouts
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
                    ev_pick = _best_side_from_ev(ld or {}, p_over, cfg.min_ev) if ld and line is not None else None
                    has_edge = (ev_pick is not None) or (edge is not None and abs(edge) >= cfg.threshold_hits * _ci)
                    tbl.append(_row(name, pred_h, "{:.2f}", line, edge, has_edge))
                    if has_edge and ld:
                        lnk = ev_pick["link"] if ev_pick is not None else (ld.get("over_link") if edge > 0 else ld.get("under_link"))
                        if lnk:
                            fd_links.append(lnk)

            for stat_lbl, hdr_lbl, pred_col, stat_key, thresh, fmt in [
                ("TB", "TOTAL BASES", "pred_total_bases", "batter_total_bases", cfg.threshold_total_bases, "{:.2f}"),
                ("HR", "HOME RUNS",   "pred_home_runs",   "batter_home_runs",   None,                      "{:.3f}"),
                ("BB", "WALKS",       "pred_walks",       "batter_walks",       cfg.threshold_walks,        "{:.2f}"),
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
                        edge = clf_p - _BREAKEVEN_PROB
                        eff_thresh = cfg.threshold_clf
                        display_pred = clf_p
                        p_over = clf_p
                    else:
                        edge = pred_v - line
                        eff_thresh = (
                            (cfg.threshold_home_runs_over if edge >= 0 else cfg.threshold_home_runs_under)
                            if thresh is None else thresh
                        )
                        display_pred = pred_v
                        p_over = _prob_over_from_regression(pred_v, line, (row.get("sigma_map") or {}).get(stat_key))
                    ev_pick = _best_side_from_ev(ld, p_over, cfg.min_ev)
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
        ev_pick = _best_side_from_ev(ld, p_over, cfg.min_ev)
        if ev_pick is not None:
            edge = ev_pick["ev"]
            display_pred = p_over if clf_p is not None else pred_k
            lnk = ev_pick["link"]
            side = "O" if ev_pick["side"] == "over" else "U"
        else:
            edge = (clf_p - _BREAKEVEN_PROB) if clf_p is not None else (pred_k - line)
            min_thr = cfg.threshold_clf if clf_p is not None else cfg.threshold_strikeouts
            if abs(edge) < min_thr:
                continue
            display_pred = p_over if clf_p is not None else pred_k
            lnk = ld.get("over_link") if edge > 0 else ld.get("under_link")
            side = "O" if edge > 0 else "U"
        k_plays.append({
            "name": name, "team": row.get("team_abbr", ""), "stat": "K",
            "pred": display_pred, "line": line, "edge": edge, "lnk": lnk, "book": "FD",
            "is_ev": ev_pick is not None,
            "side": side,
            "game_slug": row.get("game_slug"),
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
            ("pred_walks",       "batter_walks",        cfg.threshold_walks,       "BB"),
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
                edge = clf_p - _BREAKEVEN_PROB
                eff_thresh = cfg.threshold_clf
                display_pred = clf_p  # show P(over) directly
                p_over = clf_p
            else:
                edge = pred_v - line
                eff_thresh = (
                    (cfg.threshold_home_runs_over if edge >= 0 else cfg.threshold_home_runs_under)
                    if thresh is None else thresh
                )
                display_pred = pred_v
                p_over = _prob_over_from_regression(pred_v, line, (row.get("sigma_map") or {}).get(stat_key))

            ev_pick = _best_side_from_ev(ld, p_over, cfg.min_ev)
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
                batter_edge_plays.append({
                    "name": name, "team": team, "stat": stat_lbl,
                    "pred": display_pred, "line": line, "edge": edge_for_rank, "lnk": lnk, "book": book,
                    "is_clf": clf_p is not None,
                    "is_ev": ev_pick is not None,
                    "side": side,
                    "game_slug": row.get("game_slug"),
                })

    all_prop_bets: List[Dict] = k_plays + batter_edge_plays
    all_prop_bets.sort(key=lambda x: abs(x["edge"]), reverse=True)

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
                print(f"• {short} ({b['team']}) {b['stat']} {ls} → {ps}  +{abs(b['edge']):.2f}{link_txt}")
        print("")
    else:
        print("**No prop edge bets today**\n")

    return []  # Discord path exited early above; this covers the non-Discord path


def _print_best_bets(
    all_pitcher_rows: List[Dict],
    all_batter_rows: List[Dict],
    prop_lines: Dict,
    cfg: PredictConfig,
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
        ev_pick = _best_side_from_ev(ld, p_over, cfg.min_ev)
        if ev_pick is not None:
            edge = ev_pick["ev"]
            is_edge = True
            display_pred = p_over if clf_p is not None else pred_k
            bet_link = ev_pick["link"]
            side = "O" if ev_pick["side"] == "over" else "U"
        else:
            if clf_p is not None:
                edge = clf_p - _BREAKEVEN_PROB
                is_edge = abs(edge) >= cfg.threshold_clf
                display_pred = clf_p
            else:
                edge = pred_k - line
                is_edge = abs(edge) >= cfg.threshold_strikeouts
                display_pred = pred_k
            bet_link = ld.get("over_link") if edge > 0 else ld.get("under_link")
            side = "O" if edge > 0 else "U"
        if is_edge:
            best.append({
                "name": name, "stat": "K", "pred": display_pred,
                "line": line, "edge": edge,
                "bet_link": bet_link,
                "team": row.get("team_abbr", ""),
                "game_slug": row.get("game_slug", ""),
                "is_ev": ev_pick is not None,
                "side": side,
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
            ("BB", "pred_walks",        "batter_walks",       cfg.threshold_walks),
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
                if threshold is None else threshold
            )
            clf_p = (row.get("clf_p_over") or {}).get(stat_key)
            p_over = clf_p if clf_p is not None else _prob_over_from_regression(
                pred_v, line, (row.get("sigma_map") or {}).get(stat_key)
            )
            ev_pick = _best_side_from_ev(ld, p_over, cfg.min_ev)

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
                best.append({
                    "name": name, "stat": stat, "pred": pred_v,
                    "line": line, "edge": edge_pick,
                    "bet_link": bet_link,
                    "bookmaker_key": ld.get("bookmaker_key", ""),
                    "team": row.get("team_abbr", ""),
                    "game_slug": row.get("game_slug", ""),
                    "is_ev": ev_pick is not None,
                    "side": side,
                })

    best.sort(key=lambda r: abs(r["edge"]), reverse=True)

    if best:
        print("─" * 40)
        print("**Best Props (ranked by |edge|)**")

        for b in best[:10]:
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
            print(f"★ {short} {b['stat']} {ls} → {ps}{link_str}")

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
        (model_dir / "walks_xgb.json").exists() and
        (model_dir / "feature_columns_batters.json").exists()
    )

    xgb_k = lgb_k = feat_p = meds_p = bt = None
    xgb_h = lgb_h = xgb_tb_m = lgb_tb_m = None
    xgb_hr = lgb_hr = xgb_walks_m = lgb_walks_m = feat_b = meds_b = None
    # Binary classifier models (optional — loaded if training has been run)
    pitcher_clf_arts = None   # (xgb_clf, lgb_clf, feat_clf, meds_clf, bt_clf, cal_map)
    batter_clf_arts  = None   # (models_dict, feat_clf, meds_clf, bt_clf, cal_map)
    # Alt-line binary CLF (FD all-lines) — used in lottery parlay scoring
    pitcher_alt_clf_arts = None
    batter_alt_clf_arts  = None

    if pitcher_artifacts_ok:
        try:
            xgb_k, lgb_k, feat_p, meds_p, bt = _load_pitcher_artifacts(model_dir)
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
             xgb_hr, lgb_hr, xgb_walks_m, lgb_walks_m,
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
            pred_k = _predict_ensemble(X_p, xgb_k, lgb_k)
            sigma_k = bt.get("ci_strikeouts") if bt else None
            clf_pover_k: Optional[np.ndarray] = None
            if pitcher_clf_arts is not None:
                xgb_clf, lgb_clf, feat_p_clf, meds_p_clf, _bt_p_clf, cal_p_map = pitcher_clf_arts
                norms_p = np.array([
                    _normalize_name(row.get("player_name", f"id={row['player_id']}"))
                    for _, row in df_p.iterrows()
                ])
                lines_p = np.array([
                    (prop_lines.get((nm, "pitcher_strikeouts")) or {}).get("line", np.nan)
                    for nm in norms_p
                ], dtype=float)
                X_p_clf = _prep_features_clf(X_p, lines_p, feat_p_clf, meds_p_clf)
                raw_p = np.clip(_predict_ensemble(X_p_clf, xgb_clf, lgb_clf), 0.01, 0.99)
                clf_pover_k = _apply_platt_calibration(raw_p, cal_p_map.get("pitcher_strikeouts"))
                clf_pover_k[np.isnan(lines_p)] = np.nan

            # Alt-line CLF probs for lottery (all FD K lines per pitcher)
            if cfg.lottery_mode and pitcher_alt_clf_arts is not None:
                _norms_p_list = [
                    _normalize_name(row.get("player_name", f"id={row['player_id']}"))
                    for _, row in df_p.iterrows()
                ]
                try:
                    _pitcher_alt_clf_probs = _compute_alt_clf_probs(
                        X_p, _norms_p_list, all_alt_lines,
                        ["pitcher_strikeouts"], _PITCHER_META, pitcher_alt_clf_arts,
                    )
                    log.info("Alt CLF: %d pitcher K probs computed", len(_pitcher_alt_clf_probs))
                except Exception:
                    log.warning("Could not compute pitcher alt CLF probs", exc_info=True)

            for i, (_, row) in enumerate(df_p.iterrows()):
                pk = max(0.0, float(pred_k[i]))
                name = row.get("player_name", f"id={row['player_id']}")
                norm = _normalize_name(name)
                ld = prop_lines.get((norm, "pitcher_strikeouts"))
                line = ld["line"] if ld else None
                # Dynamic side-penalty (with shrinkage) for weak directional buckets.
                if line is not None:
                    raw_edge_k = pk - line
                    side_k = "over" if raw_edge_k >= 0 else "under"
                    pen_k = _side_penalty_for_line("pitcher_strikeouts", line, side_k, side_penalties)
                    if side_k == "over":
                        pk = max(0.0, pk - pen_k)
                    else:
                        pk = max(0.0, pk + pen_k)
                p_over_k = None
                if clf_pover_k is not None:
                    v = float(clf_pover_k[i])
                    p_over_k = None if np.isnan(v) else v

                if (
                    p_over_k is not None
                    and line is not None
                    and not _clf_bucket_is_disabled(clf_controls, "pitcher_strikeouts", line)
                ):
                    edge = p_over_k - _BREAKEVEN_PROB
                    kel = min(max(0.0, edge / (1.0 - _BREAKEVEN_PROB)), 0.05)
                    pred_for_db = p_over_k
                else:
                    edge = (pk - line) if line is not None else None
                    kel = _kelly_prop(abs(edge), sigma_k or cfg.threshold_strikeouts * 2) \
                          if edge is not None else 0.0
                    pred_for_db = pk

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
                    "clf_p_over": {"pitcher_strikeouts": p_over_k},
                    "sigma_strikeouts": sigma_k,
                }
                all_pitcher_rows.append(r)
                db_rows.append({
                    "game_date_et": et_date,
                    "game_slug": row["game_slug"],
                    "player_id": int(row["player_id"]),
                    "player_name": name,
                    "team_abbr": row.get("team_abbr"),
                    "stat": "pitcher_strikeouts",
                    "pred_value": round(pred_for_db, 3),
                    "book_line": line,
                    "edge": round(edge, 3) if edge is not None else None,
                    "kelly_fraction": round(kel, 4),
                })

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
            pred_h     = _predict_ensemble(X_b, xgb_h,      lgb_h)
            pred_tb    = _predict_ensemble(X_b, xgb_tb_m,   lgb_tb_m)
            pred_hr    = _predict_ensemble(X_b, xgb_hr,     lgb_hr)
            pred_walks = _predict_ensemble(X_b, xgb_walks_m, lgb_walks_m)
            sigma_h     = bt.get("ci_hits")        if bt else None
            sigma_tb    = bt.get("ci_total_bases") if bt else None
            sigma_hr    = bt.get("ci_home_runs")   if bt else None
            sigma_walks = bt.get("ci_walks")       if bt else None

            # Alt-line CLF probs for lottery (all FD hits/TB lines per batter)
            if cfg.lottery_mode and batter_alt_clf_arts is not None:
                pids_b = df_b["player_id"].astype(int).values
                _norms_b_list = [_normalize_name(name_map.get(p, "")) for p in pids_b]
                try:
                    _batter_alt_clf_probs = _compute_alt_clf_probs(
                        X_b, _norms_b_list, all_alt_lines,
                        ["batter_hits", "batter_total_bases"], _BATTER_META, batter_alt_clf_arts,
                    )
                    log.info("Alt CLF: %d batter probs computed", len(_batter_alt_clf_probs))
                except Exception:
                    log.warning("Could not compute batter alt CLF probs", exc_info=True)

            # Binary classifier predictions (if models trained)
            # One P(over) array per stat — indexed same as df_b rows
            clf_pover: Dict[str, Optional[np.ndarray]] = {
                "batter_hits": None, "batter_total_bases": None,
                "batter_home_runs": None, "batter_walks": None,
            }
            if batter_clf_arts is not None:
                clf_models, feat_clf, meds_clf, _bt_clf, cal_map_b = batter_clf_arts
                # Build normalized name array for vectorized line lookup
                pids_arr = df_b["player_id"].astype(int).values
                norms_arr = np.array([_normalize_name(name_map.get(p, "")) for p in pids_arr])

                for stat_key in clf_pover:
                    if stat_key not in clf_models:
                        continue
                    lines_arr = np.array([
                        (prop_lines.get((nm, stat_key)) or {}).get("line", np.nan)
                        for nm in norms_arr
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
                # Regression predictions with additive bias correction
                ph    = max(0.0, float(pred_h[i])     + bias.get("batter_hits",        0.0))
                ptb   = max(0.0, float(pred_tb[i])    + bias.get("batter_total_bases", 0.0))
                phr   = max(0.0, float(pred_hr[i])    + bias.get("batter_home_runs",   0.0))
                pwalk = max(0.0, float(pred_walks[i]))

                # Collect binary CLF P(over) for each stat (None when clf unavailable or no line)
                def _safe_clf(stat_key: str) -> Optional[float]:
                    arr = clf_pover.get(stat_key)
                    if arr is None:
                        return None
                    v = float(arr[i])
                    return None if np.isnan(v) else v

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
                    "pred_walks": pwalk,
                    # Binary CLF P(over) per stat — None when clf not trained or line unavailable
                    "clf_p_over": {
                        "batter_hits":        _safe_clf("batter_hits"),
                        "batter_total_bases": _safe_clf("batter_total_bases"),
                        "batter_home_runs":   _safe_clf("batter_home_runs"),
                        "batter_walks":       _safe_clf("batter_walks"),
                    },
                    "sigma_map": {
                        "batter_hits": sigma_h,
                        "batter_total_bases": sigma_tb,
                        "batter_home_runs": sigma_hr,
                        "batter_walks": sigma_walks,
                    },
                }
                all_batter_rows.append(r)

                for stat_key, stat_label, reg_pred, sigma in [
                    ("batter_hits",        "batter_hits",        ph,    sigma_h),
                    ("batter_total_bases", "batter_total_bases", ptb,   sigma_tb),
                    ("batter_home_runs",   "batter_home_runs",   phr,   sigma_hr),
                    ("batter_walks",       "batter_walks",       pwalk, sigma_walks),
                ]:
                    ld = prop_lines.get((norm, stat_key))
                    line = ld["line"] if ld else None

                    # Use binary CLF edge when model exists and line is available
                    clf_arr = clf_pover.get(stat_key)
                    p_over_i = float(clf_arr[i]) if clf_arr is not None else np.nan
                    use_clf = (
                        clf_arr is not None
                        and not np.isnan(p_over_i)
                        and line is not None
                        and not _clf_bucket_is_disabled(clf_controls, stat_key, line)
                    )

                    if use_clf:
                        # CLF: pred_value = P(over), edge = P(over) - breakeven
                        pred_v = p_over_i
                        edge   = p_over_i - _BREAKEVEN_PROB
                        # Kelly: full Kelly capped at 5%
                        kel = min(max(0.0, edge / (1.0 - _BREAKEVEN_PROB)), 0.05)
                    elif line is not None:
                        # Regression fallback
                        pred_v = reg_pred
                        edge_raw = reg_pred - line
                        side = "over" if edge_raw >= 0 else "under"
                        pen = _side_penalty_for_line(stat_key, line, side, side_penalties)
                        if side == "over":
                            pred_v = max(0.0, reg_pred - pen)
                        else:
                            pred_v = max(0.0, reg_pred + pen)
                        edge   = pred_v - line
                        kel    = _kelly_prop(abs(edge), sigma or 0.5)
                    else:
                        pred_v = reg_pred
                        edge   = None
                        kel    = 0.0

                    db_rows.append({
                        "game_date_et": et_date,
                        "game_slug": slug,
                        "player_id": pid,
                        "player_name": name,
                        "team_abbr": row.get("team_abbr"),
                        "stat": stat_label,
                        "pred_value": round(pred_v, 3),
                        "book_line": line,
                        "edge": round(edge, 3) if edge is not None else None,
                        "kelly_fraction": round(kel, 4),
                    })

    # ── Save to DB ────────────────────────────────────────────────────────
    _save_predictions(conn, db_rows)

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
                   all_alt_lines=all_alt_lines, lottery_legs=lottery_legs_collected)

    if not is_discord:
        fd_links = _print_best_bets(all_pitcher_rows, all_batter_rows, prop_lines, cfg)

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

    cfg = PredictConfig(
        et_date=et_date,
        lottery_mode=lottery_mode,
        lottery_legs=lottery_legs,
        lottery_min_american=lottery_min_american,
        lottery_max_american=lottery_max_american,
        lottery_max_per_game=lottery_max_per_game,
    )
    _apply_threshold_overrides(cfg)
    predict_props(cfg)


if __name__ == "__main__":
    main()
