"""Train shadow opportunity models for MLB player props.

The prop side models are only as good as their opportunity inputs.  This
script learns the opportunity layer directly from locked historical rows:

* hitters: actual PA and low-PA/removal risk
* pitchers: actual BF, IP, and pitch-count proxy

The artifacts are shadow-only.  They feed distribution/model-selection audits
and provide a cleaner way to measure whether opportunity modeling is improving
predictive power before any bankroll promotion.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import joblib
from lightgbm import LGBMRegressor
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"


HITTER_NUMERIC = [
    "confirmed_batting_order",
    "projected_pa",
    "pa_games",
    "is_home",
    "team_implied_runs",
    "opponent_implied_runs",
    "game_total_line",
    "opp_sp_hand_l",
    "opp_sp_k_pct_10",
    "opp_sp_bb_pct",
    "opp_sp_xwoba",
    "opp_sp_hard_hit_pct",
    "opp_sp_whiff_pct",
    "opp_bp_era_10",
    "opp_bp_whip_10",
    "opp_bp_k9_10",
    "opp_bp_ip_last_3",
    "opp_bp_ip_last_7",
    "opp_team_k_pct_10",
    "opp_team_avg_10",
    "opp_team_obp_10",
    "opp_team_slg_10",
    "batter_vs_hand_hits_avg_10",
    "batter_vs_hand_tb_avg_10",
    "batter_vs_hand_hr_avg_10",
    "batter_vs_hand_iso_avg_10",
    "batter_vs_hand_k_rate_10",
    "batter_vs_hand_games_10",
    "batter_vs_rp_ba_30",
    "batter_vs_rp_slg_30",
    "batter_vs_rp_hr_rate_30",
    "batter_vs_rp_k_rate_30",
    "pinch_hit_risk",
]

HITTER_PA_V2_NUMERIC = [
    "lineup_confirmed_flag",
    "batting_order_missing_flag",
    "lineup_slot_pa_prior",
    "lineup_slot_low_pa_prior",
    "top_order_flag",
    "middle_order_flag",
    "bottom_order_flag",
    "projected_pa_slot_delta",
    "projected_pa_x_slot_prior",
    "team_runs_x_top_order",
    "team_runs_x_bottom_order",
    "platoon_sample_flag",
    "vs_hand_hit_per_pa",
    "vs_hand_tb_per_pa",
    "vs_hand_hr_per_pa",
    "vs_hand_iso_per_pa",
    "bullpen_quality_risk",
    "starter_quality_risk",
    "bench_or_removal_risk",
]

OPPORTUNITY_V3_NUMERIC = [
    "implied_run_diff",
    "abs_implied_run_diff",
    "favorite_flag",
    "underdog_flag",
    "close_game_flag",
    "blowout_risk",
    "high_total_flag",
    "low_total_flag",
    "home_favorite_ninth_penalty",
    "away_trailing_extra_pa_chance",
    "slot_prior_x_home_ninth_penalty",
    "slot_prior_x_blowout_risk",
    "projected_pa_x_implied_runs",
    "projected_pa_x_close_game",
    "projected_pitch_per_bf",
    "projected_bf_per_ip",
    "short_leash_projection_flag",
    "deep_leash_projection_flag",
    "pitcher_favorite_leash",
    "pitcher_blowout_hook_risk",
    "opponent_onbase_pressure",
]

PITCHER_LEASH_V2_NUMERIC = [
    "bullpen_fatigue_ip_3",
    "bullpen_fatigue_ip_7",
    "bullpen_fatigue_leash_support",
    "opponent_k_leash_support",
    "opponent_patience_hook_risk",
    "opponent_power_hook_risk",
    "pitcher_projection_leash_score",
    "pitcher_pitch_count_x_opponent_k",
    "pitcher_bf_x_close_game",
    "pitcher_bf_x_bullpen_fatigue",
    "pitcher_game_script_leash_support",
    "pitcher_leash_v2_score",
]

HITTER_NUMERIC = [*HITTER_NUMERIC, *HITTER_PA_V2_NUMERIC, *OPPORTUNITY_V3_NUMERIC]

PITCHER_NUMERIC = [
    "projected_ip",
    "projected_bf",
    "projected_pitch_count",
    "pitcher_starts",
    "is_home",
    "team_implied_runs",
    "opponent_implied_runs",
    "game_total_line",
    "opp_sp_k_pct_10",
    "opp_sp_bb_pct",
    "opp_sp_xwoba",
    "opp_sp_hard_hit_pct",
    "opp_sp_whiff_pct",
    "opp_bp_era_10",
    "opp_bp_whip_10",
    "opp_bp_k9_10",
    "opp_bp_ip_last_3",
    "opp_bp_ip_last_7",
    "opp_team_k_pct_10",
    "opp_team_avg_10",
    "opp_team_obp_10",
    "opp_team_slg_10",
    *OPPORTUNITY_V3_NUMERIC,
    *PITCHER_LEASH_V2_NUMERIC,
]

HITTER_CATEGORICAL = ["confirmed_lineup_source", "opp_sp_hand", "team_abbr", "opponent_abbr"]
PITCHER_CATEGORICAL = ["team_abbr", "opponent_abbr"]

PITCHER_HISTORY_NUMERIC = [
    "days_rest",
    "career_starts_before",
    "last_bf",
    "last_pitch",
    "last_ip",
    "recent_bf_mean_3",
    "recent_bf_mean_5",
    "recent_bf_std_5",
    "recent_pitch_mean_3",
    "recent_pitch_mean_5",
    "recent_pitch_std_5",
    "recent_ip_mean_3",
    "recent_ip_mean_5",
    "recent_ip_std_5",
    "recent_short_start_rate_5",
    "recent_workload_trend",
    "team_leash_bf_prior",
    "team_leash_pitch_prior",
    "team_leash_ip_prior",
]
PITCHER_JOINT_NUMERIC = list(dict.fromkeys([*PITCHER_NUMERIC, *PITCHER_HISTORY_NUMERIC]))
PITCHER_JOINT_TARGETS = {
    "bf": ("actual_bf", "projected_bf", 0.0, 40.0),
    "pitch_count": ("actual_pitch_count_proxy", "projected_pitch_count", 0.0, 130.0),
    "innings": ("actual_ip", "projected_ip", 0.0, 9.0),
}
PITCHER_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


_LINEUP_PA_PRIORS = {
    1: 4.65,
    2: 4.55,
    3: 4.45,
    4: 4.35,
    5: 4.20,
    6: 4.05,
    7: 3.90,
    8: 3.75,
    9: 3.60,
}
_LINEUP_LOW_PA_PRIORS = {
    1: 0.04,
    2: 0.05,
    3: 0.05,
    4: 0.06,
    5: 0.07,
    6: 0.09,
    7: 0.12,
    8: 0.15,
    9: 0.18,
}


SQL = """
SELECT
    game_date_et,
    game_slug,
    player_id,
    player_name,
    team_abbr,
    opponent_abbr,
    market,
    side,
    market_line::float AS market_line,
    pred_count::float AS pred_count,
    actual_value::float AS actual_value,
    won,
    push,
    confirmed_batting_order::float AS confirmed_batting_order,
    COALESCE(confirmed_lineup_source, 'unknown') AS confirmed_lineup_source,
    projected_pa::float AS projected_pa,
    pa_games::float AS pa_games,
    projected_ip::float AS projected_ip,
    projected_bf::float AS projected_bf,
    projected_pitch_count::float AS projected_pitch_count,
    pitcher_starts::float AS pitcher_starts,
    is_home::float AS is_home,
    team_implied_runs::float AS team_implied_runs,
    opponent_implied_runs::float AS opponent_implied_runs,
    game_total_line::float AS game_total_line,
    opp_sp_hand,
    opp_sp_hand_l::float AS opp_sp_hand_l,
    opp_sp_k_pct_10::float AS opp_sp_k_pct_10,
    opp_sp_bb_pct::float AS opp_sp_bb_pct,
    opp_sp_xwoba::float AS opp_sp_xwoba,
    opp_sp_hard_hit_pct::float AS opp_sp_hard_hit_pct,
    opp_sp_whiff_pct::float AS opp_sp_whiff_pct,
    opp_bp_era_10::float AS opp_bp_era_10,
    opp_bp_whip_10::float AS opp_bp_whip_10,
    opp_bp_k9_10::float AS opp_bp_k9_10,
    opp_bp_ip_last_3::float AS opp_bp_ip_last_3,
    opp_bp_ip_last_7::float AS opp_bp_ip_last_7,
    opp_team_k_pct_10::float AS opp_team_k_pct_10,
    opp_team_avg_10::float AS opp_team_avg_10,
    opp_team_obp_10::float AS opp_team_obp_10,
    opp_team_slg_10::float AS opp_team_slg_10,
    batter_vs_hand_hits_avg_10::float AS batter_vs_hand_hits_avg_10,
    batter_vs_hand_tb_avg_10::float AS batter_vs_hand_tb_avg_10,
    batter_vs_hand_hr_avg_10::float AS batter_vs_hand_hr_avg_10,
    batter_vs_hand_iso_avg_10::float AS batter_vs_hand_iso_avg_10,
    batter_vs_hand_k_rate_10::float AS batter_vs_hand_k_rate_10,
    batter_vs_hand_games_10::float AS batter_vs_hand_games_10,
    batter_vs_rp_ba_30::float AS batter_vs_rp_ba_30,
    batter_vs_rp_slg_30::float AS batter_vs_rp_slg_30,
    batter_vs_rp_hr_rate_30::float AS batter_vs_rp_hr_rate_30,
    batter_vs_rp_k_rate_30::float AS batter_vs_rp_k_rate_30,
    pinch_hit_risk::float AS pinch_hit_risk,
    actual_pa::float AS actual_pa,
    actual_bf::float AS actual_bf,
    actual_ip::float AS actual_ip,
    actual_pitch_count_proxy::float AS actual_pitch_count_proxy,
    low_pa_flag::float AS low_pa_flag
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND result_status = 'graded'
  AND market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
ORDER BY game_date_et, game_slug, player_id, market
"""


@dataclass(frozen=True)
class OpportunityConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_opportunity_models.json"
    runtime_file: str = "prop_opportunity_models.joblib"
    report_file: str = "mlb_prop_opportunity_models_latest.md"
    lookback_days: int = 365
    holdout_days: int = 28
    min_train_rows: int = 100
    min_holdout_rows: int = 30


def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1 FROM information_schema.tables
              WHERE table_schema = %s AND table_name = %s
            )
            """,
            (schema, table),
        )
        return bool(cur.fetchone()[0])


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _numeric_series(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def add_hitter_pa_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add opportunity features that describe lineup slot and playing-time risk."""
    if df.empty:
        return df
    out = df.copy()
    order = _numeric_series(out, "confirmed_batting_order")
    projected_pa = _numeric_series(out, "projected_pa")
    team_runs = _numeric_series(out, "team_implied_runs")
    pinch_risk = _numeric_series(out, "pinch_hit_risk", 0.0).fillna(0.0).clip(0.0, 1.0)

    rounded_slot = order.round()
    valid_slot = rounded_slot.between(1, 9)
    slot_int = rounded_slot.where(valid_slot).astype("Int64")
    slot_prior = slot_int.map(_LINEUP_PA_PRIORS).astype("float64").fillna(4.05)
    low_pa_prior = slot_int.map(_LINEUP_LOW_PA_PRIORS).astype("float64").fillna(0.14)

    out["lineup_confirmed_flag"] = valid_slot.astype(float)
    out["batting_order_missing_flag"] = (~valid_slot).astype(float)
    out["lineup_slot_pa_prior"] = slot_prior
    out["lineup_slot_low_pa_prior"] = low_pa_prior
    out["top_order_flag"] = slot_int.between(1, 4).fillna(False).astype(float)
    out["middle_order_flag"] = slot_int.between(5, 6).fillna(False).astype(float)
    out["bottom_order_flag"] = slot_int.between(7, 9).fillna(False).astype(float)
    out["projected_pa_slot_delta"] = projected_pa - slot_prior
    out["projected_pa_x_slot_prior"] = projected_pa * slot_prior
    out["team_runs_x_top_order"] = team_runs * out["top_order_flag"]
    out["team_runs_x_bottom_order"] = team_runs * out["bottom_order_flag"]

    vs_hand_games = _numeric_series(out, "batter_vs_hand_games_10", 0.0).fillna(0.0)
    pa_denom = projected_pa.where(projected_pa > 0.5, slot_prior).clip(lower=0.5)
    out["platoon_sample_flag"] = (vs_hand_games >= 5).astype(float)
    out["vs_hand_hit_per_pa"] = (_numeric_series(out, "batter_vs_hand_hits_avg_10") / pa_denom).clip(0.0, 1.0)
    out["vs_hand_tb_per_pa"] = (_numeric_series(out, "batter_vs_hand_tb_avg_10") / pa_denom).clip(0.0, 4.0)
    out["vs_hand_hr_per_pa"] = (_numeric_series(out, "batter_vs_hand_hr_avg_10") / pa_denom).clip(0.0, 1.0)
    out["vs_hand_iso_per_pa"] = (_numeric_series(out, "batter_vs_hand_iso_avg_10") / pa_denom).clip(0.0, 2.0)

    bp_era = _numeric_series(out, "opp_bp_era_10")
    bp_whip = _numeric_series(out, "opp_bp_whip_10")
    sp_xwoba = _numeric_series(out, "opp_sp_xwoba")
    sp_hard = _numeric_series(out, "opp_sp_hard_hit_pct")
    out["bullpen_quality_risk"] = (
        ((bp_era - 4.20) / 1.50).fillna(0.0)
        + ((bp_whip - 1.30) / 0.25).fillna(0.0)
    ).clip(-3.0, 3.0)
    out["starter_quality_risk"] = (
        ((sp_xwoba - 0.320) / 0.050).fillna(0.0)
        + ((sp_hard - 0.390) / 0.070).fillna(0.0)
    ).clip(-3.0, 3.0)
    out["bench_or_removal_risk"] = (
        0.55 * out["batting_order_missing_flag"]
        + 0.25 * out["bottom_order_flag"]
        + 0.20 * pinch_risk
    ).clip(0.0, 1.0)

    add_opportunity_v3_features(out, in_place=True)

    for col in [*HITTER_PA_V2_NUMERIC, *OPPORTUNITY_V3_NUMERIC]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    return out


def add_opportunity_v3_features(df: pd.DataFrame, *, in_place: bool = False) -> pd.DataFrame:
    """Add leakage-safe opportunity context used by PA/BF shadow models.

    These are deterministic lock-time transforms of existing market-training
    fields: game environment, projected usage, and lineup slot priors. They do
    not use actual PA/BF outcomes.
    """
    out = df if in_place else df.copy()
    if out.empty:
        return out

    team_runs = _numeric_series(out, "team_implied_runs")
    opp_runs = _numeric_series(out, "opponent_implied_runs")
    total = _numeric_series(out, "game_total_line")
    is_home = _numeric_series(out, "is_home", 0.0).fillna(0.0).clip(0.0, 1.0)
    projected_pa = _numeric_series(out, "projected_pa")
    projected_bf = _numeric_series(out, "projected_bf")
    projected_ip = _numeric_series(out, "projected_ip")
    projected_pitch_count = _numeric_series(out, "projected_pitch_count")
    slot_prior = _numeric_series(out, "lineup_slot_pa_prior", 4.05).fillna(4.05)

    run_diff = team_runs - opp_runs
    abs_diff = run_diff.abs()
    favorite = (run_diff > 0.15).astype(float)
    underdog = (run_diff < -0.15).astype(float)
    close_game = (abs_diff <= 0.75).astype(float)
    blowout_risk = ((abs_diff - 1.25) / 2.25).clip(lower=0.0, upper=1.0)
    high_total = (total >= 9.0).astype(float)
    low_total = (total <= 7.5).astype(float)

    # Home favorites can lose a bottom-ninth plate appearance. Away teams that
    # trail retain ninth-inning opportunity more often.
    home_fav_penalty = (is_home * favorite * (0.45 + 0.55 * blowout_risk)).clip(0.0, 1.0)
    away_trailing_extra = ((1.0 - is_home) * underdog * (0.35 + 0.65 * close_game)).clip(0.0, 1.0)

    out["implied_run_diff"] = run_diff
    out["abs_implied_run_diff"] = abs_diff
    out["favorite_flag"] = favorite
    out["underdog_flag"] = underdog
    out["close_game_flag"] = close_game
    out["blowout_risk"] = blowout_risk
    out["high_total_flag"] = high_total
    out["low_total_flag"] = low_total
    out["home_favorite_ninth_penalty"] = home_fav_penalty
    out["away_trailing_extra_pa_chance"] = away_trailing_extra
    out["slot_prior_x_home_ninth_penalty"] = slot_prior * home_fav_penalty
    out["slot_prior_x_blowout_risk"] = slot_prior * blowout_risk
    out["projected_pa_x_implied_runs"] = projected_pa * team_runs
    out["projected_pa_x_close_game"] = projected_pa * close_game

    bf_denom = projected_bf.where(projected_bf > 0.5, np.nan)
    ip_denom = projected_ip.where(projected_ip > 0.1, np.nan)
    out["projected_pitch_per_bf"] = (projected_pitch_count / bf_denom).clip(2.5, 6.5)
    out["projected_bf_per_ip"] = (projected_bf / ip_denom).clip(2.5, 7.5)
    out["short_leash_projection_flag"] = (
        (projected_pitch_count < 78.0) | (projected_bf < 20.0) | (projected_ip < 5.0)
    ).astype(float)
    out["deep_leash_projection_flag"] = (
        (projected_pitch_count >= 94.0) & (projected_bf >= 24.0) & (projected_ip >= 5.8)
    ).astype(float)
    out["pitcher_favorite_leash"] = (favorite * (0.65 + 0.35 * close_game)).clip(0.0, 1.0)
    out["pitcher_blowout_hook_risk"] = (blowout_risk * (0.5 + 0.5 * high_total)).clip(0.0, 1.0)
    out["opponent_onbase_pressure"] = (
        _numeric_series(out, "opp_team_obp_10").fillna(0.315)
        + 0.45 * _numeric_series(out, "opp_team_slg_10").fillna(0.405)
        - 0.45
    ).clip(-0.20, 0.35)

    add_pitcher_leash_v2_features(out, in_place=True)

    for col in OPPORTUNITY_V3_NUMERIC:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    return out


def add_pitcher_leash_v2_features(df: pd.DataFrame, *, in_place: bool = False) -> pd.DataFrame:
    """Add pitcher opportunity features for BF and pitch-count modeling."""
    out = df if in_place else df.copy()
    if out.empty:
        return out

    projected_bf = _numeric_series(out, "projected_bf")
    projected_ip = _numeric_series(out, "projected_ip")
    projected_pitch_count = _numeric_series(out, "projected_pitch_count")
    opp_k = _numeric_series(out, "opp_team_k_pct_10").fillna(0.225)
    opp_obp = _numeric_series(out, "opp_team_obp_10").fillna(0.315)
    opp_slg = _numeric_series(out, "opp_team_slg_10").fillna(0.405)
    opp_sp_bb = _numeric_series(out, "opp_sp_bb_pct").fillna(0.080)
    opp_sp_xwoba = _numeric_series(out, "opp_sp_xwoba").fillna(0.320)
    bp_ip_3 = _numeric_series(out, "opp_bp_ip_last_3").fillna(9.0)
    bp_ip_7 = _numeric_series(out, "opp_bp_ip_last_7").fillna(21.0)
    close_game = _numeric_series(out, "close_game_flag", 0.0).fillna(0.0).clip(0.0, 1.0)
    favorite = _numeric_series(out, "favorite_flag", 0.0).fillna(0.0).clip(0.0, 1.0)
    blowout = _numeric_series(out, "blowout_risk", 0.0).fillna(0.0).clip(0.0, 1.0)

    out["bullpen_fatigue_ip_3"] = bp_ip_3
    out["bullpen_fatigue_ip_7"] = bp_ip_7
    out["bullpen_fatigue_leash_support"] = (
        0.55 * ((bp_ip_3 - 8.5) / 5.0).clip(-1.0, 2.0)
        + 0.45 * ((bp_ip_7 - 20.0) / 10.0).clip(-1.0, 2.0)
    ).clip(-1.0, 2.5)
    out["opponent_k_leash_support"] = ((opp_k - 0.215) / 0.060).clip(-2.0, 2.5)
    out["opponent_patience_hook_risk"] = (
        0.60 * ((opp_obp - 0.320) / 0.045).clip(-2.0, 2.5)
        + 0.40 * ((opp_sp_bb - 0.080) / 0.035).clip(-2.0, 2.5)
    ).clip(-2.0, 2.5)
    out["opponent_power_hook_risk"] = (
        0.55 * ((opp_slg - 0.405) / 0.070).clip(-2.0, 2.5)
        + 0.45 * ((opp_sp_xwoba - 0.320) / 0.055).clip(-2.0, 2.5)
    ).clip(-2.0, 2.5)
    out["pitcher_projection_leash_score"] = (
        0.35 * ((projected_pitch_count - 85.0) / 15.0).clip(-2.0, 2.5)
        + 0.35 * ((projected_bf - 21.0) / 5.0).clip(-2.0, 2.5)
        + 0.30 * ((projected_ip - 5.2) / 1.2).clip(-2.0, 2.5)
    ).clip(-2.5, 3.0)
    out["pitcher_pitch_count_x_opponent_k"] = projected_pitch_count * opp_k
    out["pitcher_bf_x_close_game"] = projected_bf * close_game
    out["pitcher_bf_x_bullpen_fatigue"] = projected_bf * out["bullpen_fatigue_leash_support"]
    out["pitcher_game_script_leash_support"] = (
        0.55 * favorite
        + 0.35 * close_game
        - 0.45 * blowout
    ).clip(-1.0, 1.5)
    out["pitcher_leash_v2_score"] = (
        out["pitcher_projection_leash_score"]
        + 0.30 * out["bullpen_fatigue_leash_support"]
        + 0.22 * out["opponent_k_leash_support"]
        + 0.18 * out["pitcher_game_script_leash_support"]
        - 0.30 * out["opponent_patience_hook_risk"]
        - 0.25 * out["opponent_power_hook_risk"]
    ).clip(-3.5, 4.0)

    for col in PITCHER_LEASH_V2_NUMERIC:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    return out


def _pitcher_history_state(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    if df.empty:
        return state
    ordered = df.sort_values(["player_id", "game_date_et", "game_slug"])
    for player_id, group in ordered.groupby("player_id", dropna=False):
        last = group.iloc[-1]
        rec: dict[str, Any] = {
            "last_game_date": str(last.get("game_date_et")),
            "career_starts_before": int(len(group)),
        }
        for source, prefix in (
            ("actual_bf", "bf"),
            ("actual_pitch_count_proxy", "pitch"),
            ("actual_ip", "ip"),
        ):
            values = pd.to_numeric(group[source], errors="coerce").dropna().tail(5)
            rec[f"last_{prefix}"] = float(values.iloc[-1]) if not values.empty else None
            rec[f"recent_{prefix}_mean_3"] = float(values.tail(3).mean()) if not values.empty else None
            rec[f"recent_{prefix}_mean_5"] = float(values.mean()) if not values.empty else None
            rec[f"recent_{prefix}_std_5"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        short = (
            (pd.to_numeric(group["actual_bf"], errors="coerce") < 18.0)
            | (pd.to_numeric(group["actual_pitch_count_proxy"], errors="coerce") < 75.0)
        ).tail(5)
        rec["recent_short_start_rate_5"] = float(short.mean()) if len(short) else None
        pitch3 = rec.get("recent_pitch_mean_3")
        pitch5 = rec.get("recent_pitch_mean_5")
        rec["recent_workload_trend"] = (
            float(pitch3) - float(pitch5) if pitch3 is not None and pitch5 is not None else None
        )
        state[str(player_id)] = rec
    return state


def add_pitcher_history_features(
    df: pd.DataFrame,
    *,
    prior_state: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Add pregame-only workload history, always shifted before the target start."""
    if df.empty:
        return df.copy()
    out = df.copy()
    out["game_date_et"] = pd.to_datetime(out["game_date_et"]).dt.date
    if prior_state is not None:
        for idx, row in out.iterrows():
            rec = prior_state.get(str(row.get("player_id"))) or {}
            last_date = pd.to_datetime(rec.get("last_game_date"), errors="coerce")
            game_date = pd.to_datetime(row.get("game_date_et"), errors="coerce")
            out.at[idx, "days_rest"] = (
                float((game_date - last_date).days) if pd.notna(last_date) and pd.notna(game_date) else np.nan
            )
            for name in PITCHER_HISTORY_NUMERIC:
                if name == "days_rest":
                    continue
                out.at[idx, name] = rec.get(name)
        return out

    out = out.sort_values(["player_id", "game_date_et", "game_slug"]).copy()
    dates = pd.to_datetime(out["game_date_et"])
    out["days_rest"] = dates.groupby(out["player_id"]).diff().dt.days.astype(float)
    out["career_starts_before"] = out.groupby("player_id").cumcount().astype(float)
    mappings = (
        ("actual_bf", "bf"),
        ("actual_pitch_count_proxy", "pitch"),
        ("actual_ip", "ip"),
    )
    for source, prefix in mappings:
        numeric = pd.to_numeric(out[source], errors="coerce")
        out[f"last_{prefix}"] = numeric.groupby(out["player_id"]).shift(1)
        for window in (3, 5):
            out[f"recent_{prefix}_mean_{window}"] = numeric.groupby(out["player_id"]).transform(
                lambda values, w=window: values.shift(1).rolling(w, min_periods=1).mean()
            )
        out[f"recent_{prefix}_std_5"] = numeric.groupby(out["player_id"]).transform(
            lambda values: values.shift(1).rolling(5, min_periods=2).std(ddof=0)
        )
    short = (
        (pd.to_numeric(out["actual_bf"], errors="coerce") < 18.0)
        | (pd.to_numeric(out["actual_pitch_count_proxy"], errors="coerce") < 75.0)
    ).astype(float)
    out["recent_short_start_rate_5"] = short.groupby(out["player_id"]).transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean()
    )
    out["recent_workload_trend"] = out["recent_pitch_mean_3"] - out["recent_pitch_mean_5"]
    for source, name in (
        ("actual_bf", "team_leash_bf_prior"),
        ("actual_pitch_count_proxy", "team_leash_pitch_prior"),
        ("actual_ip", "team_leash_ip_prior"),
    ):
        numeric = pd.to_numeric(out[source], errors="coerce")
        out[name] = numeric.groupby(out["team_abbr"].fillna("unknown")).transform(
            lambda values: values.shift(1).expanding(min_periods=3).mean()
        )
    for name in PITCHER_HISTORY_NUMERIC:
        out[name] = pd.to_numeric(out.get(name), errors="coerce")
    return out.sort_index()


def _joint_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]), categorical),
    ], remainder="drop")


def _joint_regressor(
    numeric: list[str],
    categorical: list[str],
    *,
    quantile: float | None = None,
) -> Pipeline:
    params: dict[str, Any] = {
        "objective": "quantile" if quantile is not None else "regression_l1",
        "alpha": quantile,
        "n_estimators": 260,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 6,
        "min_child_samples": 18,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.20,
        "reg_lambda": 1.50,
        "random_state": 71 + int((quantile or 0.0) * 100),
        "n_jobs": -1,
        "verbosity": -1,
    }
    if quantile is None:
        params.pop("alpha")
    return Pipeline([
        ("features", _joint_preprocessor(numeric, categorical)),
        ("model", LGBMRegressor(**params)),
    ])


def _k_probability_brier(
    offers: pd.DataFrame,
    opportunity: pd.DataFrame,
) -> dict[str, Any]:
    if offers.empty or opportunity.empty:
        return {"rows": 0}
    keys = ["game_slug", "player_id"]
    cols = keys + ["joint_pred_bf", "joint_pred_pitch_count", "joint_pred_ip"]
    work = offers.merge(opportunity[cols], on=keys, how="inner")
    work = work.loc[~work["push"].fillna(False)].copy()
    work = work.dropna(subset=["market_line", "pred_count", "won", "side"])
    work = work.drop_duplicates([*keys, "side", "market_line"])
    if work.empty:
        return {"rows": 0}
    ratios = []
    for pred, base in (
        ("joint_pred_bf", "projected_bf"),
        ("joint_pred_pitch_count", "projected_pitch_count"),
        ("joint_pred_ip", "projected_ip"),
    ):
        ratio = pd.to_numeric(work[pred], errors="coerce") / pd.to_numeric(work[base], errors="coerce").replace(0.0, np.nan)
        ratios.append(ratio.clip(0.65, 1.35))
    factor = (0.45 * ratios[0] + 0.35 * ratios[1] + 0.20 * ratios[2]).fillna(1.0).clip(0.70, 1.30)
    base_mu = pd.to_numeric(work["pred_count"], errors="coerce").clip(0.05, 15.0)
    joint_mu = (base_mu * factor).clip(0.05, 15.0)
    line_floor = np.floor(pd.to_numeric(work["market_line"], errors="coerce")).astype(int)
    base_over = poisson.sf(line_floor, base_mu)
    joint_over = poisson.sf(line_floor, joint_mu)
    over_side = work["side"].astype(str).str.lower().eq("over").to_numpy()
    base_prob = np.where(over_side, base_over, 1.0 - base_over)
    joint_prob = np.where(over_side, joint_over, 1.0 - joint_over)
    target = work["won"].astype(int).to_numpy()
    return {
        "rows": int(len(work)),
        "baseline_brier": float(brier_score_loss(target, np.clip(base_prob, 1e-6, 1 - 1e-6))),
        "joint_brier": float(brier_score_loss(target, np.clip(joint_prob, 1e-6, 1 - 1e-6))),
        "brier_gain": float(
            brier_score_loss(target, np.clip(base_prob, 1e-6, 1 - 1e-6))
            - brier_score_loss(target, np.clip(joint_prob, 1e-6, 1 - 1e-6))
        ),
        "avg_opportunity_factor": float(factor.mean()),
    }


def _fit_pitcher_joint_opportunity(
    pitchers: pd.DataFrame,
    k_offers: pd.DataFrame,
    cfg: OpportunityConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    work = add_pitcher_history_features(pitchers)
    train, holdout, split = _joint_date_split(work, cfg) if not work.empty else (work, work, "none")
    record: dict[str, Any] = {
        "kind": "joint_quantile_opportunity",
        "status": "insufficient_rows",
        "rows": int(len(work)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "split_strategy": split,
        "numeric_features": PITCHER_JOINT_NUMERIC,
        "categorical_features": PITCHER_CATEGORICAL,
        "targets": {},
        "use_for_distribution": False,
    }
    if len(train) < cfg.min_train_rows or len(holdout) < cfg.min_holdout_rows:
        return record, None
    usable_numeric = [
        name for name in PITCHER_JOINT_NUMERIC
        if name in train.columns and pd.to_numeric(train[name], errors="coerce").notna().any()
    ]
    record["numeric_features"] = usable_numeric
    record["dropped_all_null_features"] = [name for name in PITCHER_JOINT_NUMERIC if name not in usable_numeric]
    feature_cols = usable_numeric + PITCHER_CATEGORICAL
    runtime_models: dict[str, Any] = {}
    holdout_predictions = holdout[["game_slug", "player_id", "projected_bf", "projected_pitch_count", "projected_ip"]].copy()
    residuals: dict[str, np.ndarray] = {}
    improved_targets = 0
    for label, (target, baseline, lo, hi) in PITCHER_JOINT_TARGETS.items():
        fit_train = train.dropna(subset=[target]).copy()
        fit_holdout = holdout.dropna(subset=[target]).copy()
        if len(fit_train) < cfg.min_train_rows or len(fit_holdout) < cfg.min_holdout_rows:
            record["targets"][label] = {"status": "insufficient_rows"}
            continue
        mean_model = _joint_regressor(usable_numeric, PITCHER_CATEGORICAL)
        train_baseline = pd.to_numeric(fit_train[baseline], errors="coerce")
        holdout_baseline = pd.to_numeric(fit_holdout[baseline], errors="coerce")
        residual_target = fit_train[target].astype(float) - train_baseline
        mean_model.fit(fit_train[feature_cols], residual_target)
        mean_pred = np.clip(holdout_baseline + mean_model.predict(fit_holdout[feature_cols]), lo, hi)
        quantile_models: dict[str, Pipeline] = {}
        quantile_predictions: dict[str, np.ndarray] = {}
        for quantile in PITCHER_QUANTILES:
            key = f"q{int(quantile * 100):02d}"
            model = _joint_regressor(usable_numeric, PITCHER_CATEGORICAL, quantile=quantile)
            model.fit(fit_train[feature_cols], residual_target)
            quantile_models[key] = model
            quantile_predictions[key] = np.clip(
                holdout_baseline + model.predict(fit_holdout[feature_cols]), lo, hi
            )
        q_matrix = np.column_stack([quantile_predictions[f"q{int(q * 100):02d}"] for q in PITCHER_QUANTILES])
        q_matrix = np.sort(q_matrix, axis=1)
        baseline_metrics = _baseline_metrics_reg(fit_holdout, target, baseline)
        model_metrics = _model_metrics_reg(fit_holdout, target, mean_pred)
        improved = bool(
            baseline_metrics.get("mae") is not None
            and model_metrics.get("mae") is not None
            and model_metrics["mae"] < baseline_metrics["mae"]
        )
        improved_targets += int(improved)
        y = fit_holdout[target].astype(float).to_numpy()
        coverage_10_90 = float(np.mean((y >= q_matrix[:, 0]) & (y <= q_matrix[:, -1])))
        coverage_25_75 = float(np.mean((y >= q_matrix[:, 1]) & (y <= q_matrix[:, 3])))
        record["targets"][label] = {
            "status": "trained",
            "model_type": "boosted_residual_from_existing_projection",
            "baseline_feature": baseline,
            "baseline": baseline_metrics,
            "mean_model": model_metrics,
            "mae_gain": (
                float(baseline_metrics["mae"] - model_metrics["mae"])
                if baseline_metrics.get("mae") is not None and model_metrics.get("mae") is not None else None
            ),
            "improved": improved,
            "quantile_coverage_10_90": coverage_10_90,
            "quantile_coverage_25_75": coverage_25_75,
        }
        runtime_models[label] = {
            "mean": mean_model,
            "quantiles": quantile_models,
            "baseline_feature": baseline,
        }
        full_pred = pd.Series(np.nan, index=holdout.index, dtype=float)
        full_pred.loc[fit_holdout.index] = mean_pred
        holdout_predictions[f"joint_pred_{'ip' if label == 'innings' else label}"] = full_pred
        residuals[label] = pd.Series(y - mean_pred, index=fit_holdout.index)

    common_residuals = pd.DataFrame(residuals).dropna()
    record["residual_correlation"] = (
        common_residuals.corr().round(6).to_dict() if len(common_residuals) >= 10 else {}
    )
    holdout_dates = set(holdout["game_date_et"])
    k_holdout = k_offers.loc[k_offers["game_date_et"].isin(holdout_dates)].copy()
    record["k_line_holdout"] = _k_probability_brier(k_holdout, holdout_predictions.dropna())
    k_gain = (record["k_line_holdout"] or {}).get("brier_gain")
    use_for_distribution = bool(improved_targets >= 2 and k_gain is not None and float(k_gain) > 0.0)
    record.update({
        "status": "trained",
        "improved_targets": improved_targets,
        "use_for_distribution": use_for_distribution,
        "activation_gate": "at_least_two_opportunity_targets_and_k_line_brier_must_improve",
    })
    runtime = {
        "models": runtime_models,
        "numeric_features": usable_numeric,
        "categorical_features": PITCHER_CATEGORICAL,
        "player_history_state": _pitcher_history_state(work),
        "use_for_distribution": use_for_distribution,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return record, runtime


def _load(cfg: OpportunityConfig) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, cfg.lookback_days))
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if not _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            return pd.DataFrame()
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    numeric = sorted(set(HITTER_NUMERIC + PITCHER_NUMERIC + [
        "market_line",
        "pred_count",
        "actual_value",
        "actual_pa",
        "actual_bf",
        "actual_ip",
        "actual_pitch_count_proxy",
        "low_pa_flag",
    ]))
    for col in numeric:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_hitter_pa_v2_features(df)
    for col in sorted(set(HITTER_CATEGORICAL + PITCHER_CATEGORICAL)):
        df[col] = df[col].fillna("unknown").astype(str)
    return df.replace([np.inf, -np.inf], np.nan)


def _split(df: pd.DataFrame, cfg: OpportunityConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    split = max(df["game_date_et"]) - timedelta(days=cfg.holdout_days)
    train = df.loc[df["game_date_et"] < split].copy()
    holdout = df.loc[df["game_date_et"] >= split].copy()
    if len(train) >= cfg.min_train_rows and len(holdout) >= cfg.min_holdout_rows:
        return train, holdout, f"last_{cfg.holdout_days}_days"
    dates = sorted(df["game_date_et"].unique())
    if len(dates) > 1:
        holdout_date = dates[-1]
        return df.loc[df["game_date_et"] < holdout_date].copy(), df.loc[df["game_date_et"] >= holdout_date].copy(), "last_available_date"
    return train, holdout, f"last_{cfg.holdout_days}_days"


def _joint_date_split(df: pd.DataFrame, cfg: OpportunityConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Expand the terminal date block until a usable, still-pure holdout exists."""
    train, holdout, split = _split(df, cfg)
    if len(train) >= cfg.min_train_rows and len(holdout) >= cfg.min_holdout_rows:
        return train, holdout, split
    dates = sorted(df["game_date_et"].dropna().unique())
    for date_count in range(2, len(dates)):
        holdout_dates = set(dates[-date_count:])
        candidate_holdout = df.loc[df["game_date_et"].isin(holdout_dates)].copy()
        candidate_train = df.loc[~df["game_date_et"].isin(holdout_dates)].copy()
        if len(candidate_holdout) >= cfg.min_holdout_rows and len(candidate_train) >= cfg.min_train_rows:
            return candidate_train, candidate_holdout, f"last_{date_count}_available_dates"
    return train, holdout, split


def _prepare(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    *,
    means: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
    cats: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, float], dict[str, float], dict[str, list[str]]]:
    means = dict(means or {})
    scales = dict(scales or {})
    cats = {key: list(value) for key, value in (cats or {}).items()}
    parts: list[np.ndarray] = []
    names: list[str] = []
    out_means: dict[str, float] = {}
    out_scales: dict[str, float] = {}
    for name in numeric:
        s = pd.to_numeric(df.get(name), errors="coerce")
        mean = float(means.get(name, s.mean() if not s.dropna().empty else 0.0))
        filled = s.fillna(mean)
        std = float(filled.std(ddof=0) or 0.0)
        scale = float(scales.get(name, std if std > 1e-9 else 1.0))
        if scale <= 1e-9:
            scale = 1.0
        parts.append(((filled - mean) / scale).to_numpy().reshape(-1, 1))
        names.append(name)
        out_means[name] = mean
        out_scales[name] = scale
    out_cats: dict[str, list[str]] = {}
    for name in categorical:
        values = cats.get(name)
        if values is None:
            values = sorted(str(v) for v in df.get(name, pd.Series([], dtype=str)).fillna("unknown").unique())
        out_cats[name] = values
        series = df.get(name, pd.Series(["unknown"] * len(df))).fillna("unknown").astype(str)
        for value in values:
            parts.append((series == value).astype(float).to_numpy().reshape(-1, 1))
            names.append(f"{name}={value}")
    return np.hstack(parts) if parts else np.zeros((len(df), 0)), names, out_means, out_scales, out_cats


def _score_linear(df: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    numeric = list(model.get("numeric_features") or [])
    categorical = list(model.get("categorical_features") or [])
    X, names, _, _, _ = _prepare(
        df,
        numeric,
        categorical,
        means=model.get("numeric_means") or {},
        scales=model.get("numeric_scales") or {},
        cats=model.get("categorical_values") or {},
    )
    coef_map = model.get("coef") or {}
    coef = np.array([float(coef_map.get(name, 0.0)) for name in names], dtype="float64")
    raw = float(model.get("intercept", 0.0)) + X.dot(coef)
    if model.get("kind") == "classifier":
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40)))
    return raw


def _baseline_metrics_reg(df: pd.DataFrame, target: str, baseline: str) -> dict[str, Any]:
    work = df.dropna(subset=[target, baseline])
    if work.empty:
        return {"rows": 0}
    y = work[target].astype(float)
    p = work[baseline].astype(float)
    rmse = math.sqrt(mean_squared_error(y, p))
    return {
        "rows": int(len(work)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(rmse),
        "bias": float((p - y).mean()),
        "r2": float(r2_score(y, p)) if len(work) > 1 else None,
    }


def _model_metrics_reg(df: pd.DataFrame, target: str, pred: np.ndarray) -> dict[str, Any]:
    work = df.dropna(subset=[target]).copy()
    if work.empty:
        return {"rows": 0}
    y = work[target].astype(float)
    p = pd.Series(pred, index=df.index).loc[work.index].astype(float)
    rmse = math.sqrt(mean_squared_error(y, p))
    return {
        "rows": int(len(work)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(rmse),
        "bias": float((p - y).mean()),
        "r2": float(r2_score(y, p)) if len(work) > 1 else None,
    }


def _model_metrics_cls(df: pd.DataFrame, target: str, prob: np.ndarray) -> dict[str, Any]:
    work = df.dropna(subset=[target]).copy()
    if work.empty:
        return {"rows": 0}
    y = work[target].astype(int)
    p = pd.Series(prob, index=df.index).loc[work.index].astype(float).clip(1e-6, 1 - 1e-6)
    return {
        "rows": int(len(work)),
        "actual_rate": float(y.mean()),
        "avg_prob": float(p.mean()),
        "calibration_error": float(y.mean() - p.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if y.nunique() == 2 else None,
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
    }


def _fit_regression(
    df: pd.DataFrame,
    target: str,
    baseline: str,
    numeric: list[str],
    categorical: list[str],
    cfg: OpportunityConfig,
) -> dict[str, Any]:
    work = df.dropna(subset=[target]).copy()
    train, holdout, split = _split(work, cfg) if not work.empty else (work, work, "none")
    payload: dict[str, Any] = {
        "kind": "regression",
        "target": target,
        "baseline_feature": baseline,
        "rows": int(len(work)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "split_strategy": split,
        "numeric_features": numeric,
        "categorical_features": categorical,
    }
    payload["baseline_holdout"] = _baseline_metrics_reg(holdout, target, baseline)
    if len(train) < cfg.min_train_rows or len(holdout) < cfg.min_holdout_rows:
        payload["status"] = "insufficient_rows"
        return payload
    X_train, names, means, scales, cats = _prepare(train, numeric, categorical)
    y_train = train[target].astype(float).to_numpy()
    model = Ridge(alpha=2.0)
    model.fit(X_train, y_train)
    X_hold, _, _, _, _ = _prepare(holdout, numeric, categorical, means=means, scales=scales, cats=cats)
    pred = model.predict(X_hold)
    if target == "actual_pa":
        pred = np.clip(pred, 0.0, 7.0)
    elif target in {"actual_bf", "actual_pitch_count_proxy"}:
        pred = np.clip(pred, 0.0, 130.0)
    elif target == "actual_ip":
        pred = np.clip(pred, 0.0, 9.0)
    holdout_metrics = _model_metrics_reg(holdout, target, pred)
    base_metrics = payload.get("baseline_holdout") or {}
    use_for_distribution = (
        holdout_metrics.get("mae") is not None
        and base_metrics.get("mae") is not None
        and holdout_metrics["mae"] < base_metrics["mae"]
        and (
            holdout_metrics.get("rmse") is None
            or base_metrics.get("rmse") is None
            or holdout_metrics["rmse"] <= base_metrics["rmse"]
        )
    )
    payload.update({
        "status": "trained",
        "use_for_distribution": bool(use_for_distribution),
        "intercept": float(model.intercept_),
        "coef": {name: float(value) for name, value in zip(names, model.coef_) if abs(float(value)) > 1e-12},
        "numeric_means": means,
        "numeric_scales": scales,
        "categorical_values": cats,
        "holdout": holdout_metrics,
    })
    return payload


def _fit_classifier(
    df: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    cfg: OpportunityConfig,
) -> dict[str, Any]:
    work = df.dropna(subset=[target]).copy()
    train, holdout, split = _split(work, cfg) if not work.empty else (work, work, "none")
    payload: dict[str, Any] = {
        "kind": "classifier",
        "target": target,
        "rows": int(len(work)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "split_strategy": split,
        "numeric_features": numeric,
        "categorical_features": categorical,
    }
    if len(train) < cfg.min_train_rows or len(holdout) < cfg.min_holdout_rows or train[target].nunique() < 2:
        payload["status"] = "insufficient_rows"
        return payload
    X_train, names, means, scales, cats = _prepare(train, numeric, categorical)
    y_train = train[target].astype(int).to_numpy()
    model = LogisticRegression(max_iter=3000, solver="lbfgs")
    model.fit(X_train, y_train)
    X_hold, _, _, _, _ = _prepare(holdout, numeric, categorical, means=means, scales=scales, cats=cats)
    prob = np.clip(model.predict_proba(X_hold)[:, 1], 1e-6, 1 - 1e-6)
    payload.update({
        "status": "trained",
        "intercept": float(model.intercept_[0]),
        "coef": {name: float(value) for name, value in zip(names, model.coef_[0]) if abs(float(value)) > 1e-12},
        "numeric_means": means,
        "numeric_scales": scales,
        "categorical_values": cats,
        "holdout": _model_metrics_cls(holdout, target, prob),
    })
    return payload


def _lineup_slot_impact(hitters: pd.DataFrame) -> list[dict[str, Any]]:
    work = hitters.dropna(subset=["confirmed_batting_order", "actual_pa"]).copy()
    if work.empty:
        return []
    rows = []
    for slot, group in work.groupby(work["confirmed_batting_order"].round().astype(int)):
        rows.append({
            "slot": int(slot),
            "rows": int(len(group)),
            "avg_actual_pa": float(group["actual_pa"].mean()),
            "low_pa_rate": float(group["low_pa_flag"].mean()) if group["low_pa_flag"].notna().any() else None,
        })
    rows.sort(key=lambda rec: rec["slot"])
    return rows


def _top_coefficients(model: dict[str, Any], limit: int = 12) -> list[tuple[str, float]]:
    coefs = [(name, float(value)) for name, value in (model.get("coef") or {}).items()]
    coefs.sort(key=lambda item: abs(item[1]), reverse=True)
    return coefs[:limit]


def _fmt_num(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"


def _fmt_pct(value: Any, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    v = float(value) * 100.0
    return f"{v:+.1f}%" if signed else f"{v:.1f}%"


def _write_text_with_lock_fallback(path: Path, text: str) -> Path:
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        fallback.write_text(text, encoding="utf-8")
        return fallback


def _write_report(payload: dict[str, Any], cfg: OpportunityConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / cfg.report_file
    lines = [
        "# MLB Prop Opportunity Models",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Rows: {payload.get('rows', 0)}",
        f"Date range: {payload.get('date_min')} to {payload.get('date_max')}",
        f"Status: {payload.get('status')}",
        "",
        "## Regression Holdouts",
        "",
        "| Model | Rows | Base MAE | Model MAE | Base RMSE | Model RMSE | Model Bias | R2 | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("hitter_pa", "pitcher_bf", "pitcher_ip", "pitcher_pitch_count_proxy"):
        rec = payload.get("models", {}).get(name) or {}
        base = rec.get("baseline_holdout") or {}
        hold = rec.get("holdout") or {}
        decision = "model_helped" if rec.get("use_for_distribution") else (
            "keep_baseline" if rec.get("status") == "trained" else rec.get("status", "unknown")
        )
        lines.append(
            f"| {name} | {hold.get('rows', rec.get('holdout_rows', 0))} | "
            f"{_fmt_num(base.get('mae'))} | {_fmt_num(hold.get('mae'))} | "
            f"{_fmt_num(base.get('rmse'))} | {_fmt_num(hold.get('rmse'))} | "
            f"{_fmt_num(hold.get('bias'), signed=True)} | {_fmt_num(hold.get('r2'))} | {decision} |"
        )

    joint = payload.get("models", {}).get("pitcher_joint_opportunity") or {}
    lines.extend([
        "",
        "## Pitcher Joint Opportunity Rebuild",
        "",
        f"Status: {joint.get('status', 'unknown')} | Live gate: {joint.get('use_for_distribution', False)} | "
        f"Improved targets: {joint.get('improved_targets', 0)}/3",
        "",
        "| Target | Base MAE | Joint MAE | MAE Gain | 10-90 Coverage | 25-75 Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for label, rec in (joint.get("targets") or {}).items():
        lines.append(
            f"| {label} | {_fmt_num((rec.get('baseline') or {}).get('mae'))} | "
            f"{_fmt_num((rec.get('mean_model') or {}).get('mae'))} | "
            f"{_fmt_num(rec.get('mae_gain'), signed=True)} | "
            f"{_fmt_pct(rec.get('quantile_coverage_10_90'))} | "
            f"{_fmt_pct(rec.get('quantile_coverage_25_75'))} |"
        )
    k_holdout = joint.get("k_line_holdout") or {}
    lines.extend([
        "",
        f"K-line holdout rows: {k_holdout.get('rows', 0)} | baseline Brier: "
        f"{_fmt_num(k_holdout.get('baseline_brier'))} | joint Brier: {_fmt_num(k_holdout.get('joint_brier'))} | "
        f"gain: {_fmt_num(k_holdout.get('brier_gain'), signed=True)}.",
        "",
        "The joint model remains shadow-only unless at least two workload targets and K-line Brier improve on the same date holdout.",
    ])

    low_pa = payload.get("models", {}).get("hitter_low_pa") or {}
    hold = low_pa.get("holdout") or {}
    lines.extend([
        "",
        "## Low-PA / Removal Risk",
        "",
        "| Rows | Actual | Avg Prob | Brier | Log Loss | AUC | Status |",
        "|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {hold.get('rows', low_pa.get('holdout_rows', 0))} | {_fmt_pct(hold.get('actual_rate'))} | "
            f"{_fmt_pct(hold.get('avg_prob'))} | {_fmt_num(hold.get('brier'))} | "
            f"{_fmt_num(hold.get('log_loss'))} | {_fmt_num(hold.get('auc'))} | {low_pa.get('status')} |"
        ),
        "",
        "## Lineup Slot Impact",
        "",
        "| Slot | Rows | Avg Actual PA | Low-PA Rate |",
        "|---:|---:|---:|---:|",
    ])
    for rec in payload.get("lineup_slot_impact", []):
        lines.append(
            f"| {rec['slot']} | {rec['rows']} | {_fmt_num(rec.get('avg_actual_pa'))} | {_fmt_pct(rec.get('low_pa_rate'))} |"
        )

    lines.extend(["", "## Largest Coefficients", ""])
    for name, rec in (payload.get("models") or {}).items():
        lines.extend([f"### {name}", "", "| Feature | Coef |", "|---|---:|"])
        for feature, coef in _top_coefficients(rec):
            lines.append(f"| {feature} | {_fmt_num(coef, signed=True)} |")
        lines.append("")
    return str(_write_text_with_lock_fallback(path, "\n".join(lines) + "\n"))


def train(cfg: OpportunityConfig) -> dict[str, Any]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    df = _load(cfg)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "features.mlb_prop_market_training_examples",
        "usage": "shadow_only",
        "rows": int(len(df)),
        "date_min": str(min(df["game_date_et"])) if not df.empty else None,
        "date_max": str(max(df["game_date_et"])) if not df.empty else None,
        "models": {},
        "lineup_slot_impact": [],
    }
    if df.empty:
        payload["status"] = "no_rows"
        payload["report_path"] = _write_report(payload, cfg)
        (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    hitters = (
        df.loc[df["market"].isin(["batter_hits", "batter_total_bases", "batter_home_runs"])]
        .drop_duplicates(["game_slug", "player_id"])
        .copy()
    )
    k_offers = df.loc[df["market"] == "pitcher_strikeouts"].copy()
    pitchers = (
        k_offers
        .drop_duplicates(["game_slug", "player_id"])
        .copy()
    )
    payload["hitter_player_games"] = int(len(hitters))
    payload["pitcher_player_games"] = int(len(pitchers))
    payload["lineup_slot_impact"] = _lineup_slot_impact(hitters)

    payload["models"]["hitter_pa"] = _fit_regression(
        hitters, "actual_pa", "projected_pa", HITTER_NUMERIC, HITTER_CATEGORICAL, cfg
    )
    payload["models"]["hitter_low_pa"] = _fit_classifier(
        hitters, "low_pa_flag", HITTER_NUMERIC, HITTER_CATEGORICAL, cfg
    )
    payload["models"]["pitcher_bf"] = _fit_regression(
        pitchers, "actual_bf", "projected_bf", PITCHER_NUMERIC, PITCHER_CATEGORICAL, cfg
    )
    payload["models"]["pitcher_ip"] = _fit_regression(
        pitchers, "actual_ip", "projected_ip", PITCHER_NUMERIC, PITCHER_CATEGORICAL, cfg
    )
    payload["models"]["pitcher_pitch_count_proxy"] = _fit_regression(
        pitchers, "actual_pitch_count_proxy", "projected_pitch_count", PITCHER_NUMERIC, PITCHER_CATEGORICAL, cfg
    )
    joint_record, joint_runtime = _fit_pitcher_joint_opportunity(pitchers, k_offers, cfg)
    payload["models"]["pitcher_joint_opportunity"] = joint_record
    joblib.dump(
        joint_runtime or {
            "models": {},
            "use_for_distribution": False,
            "status": joint_record.get("status"),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        cfg.model_dir / cfg.runtime_file,
    )
    payload["status"] = "ready"
    payload["report_path"] = _write_report(payload, cfg)
    (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB prop opportunity shadow models")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    args = parser.parse_args()
    payload = train(OpportunityConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
    ))
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows", 0),
        "hitter_player_games": payload.get("hitter_player_games", 0),
        "pitcher_player_games": payload.get("pitcher_player_games", 0),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
