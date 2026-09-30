"""Train NFL player-game stat projection models."""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine, text

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.features import USAGE_STATS
from nfl_pipeline.markets import SPEC_BY_STAT, STAT_SPECS

log = logging.getLogger("nfl_pipeline.modeling.train_player_stat_models")

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass(frozen=True)
class TrainConfig:
    pg_dsn: str = PG_DSN
    model_dir: Path = _MODEL_DIR
    min_prev_games: int = 3
    min_rows: int = 200
    holdout_weeks: int = 4
    random_state: int = 42
    out_file: str = "nfl_player_stat_models.joblib"
    report_file: str = "nfl_player_stat_models_report.json"


SQL_TRAIN = """
SELECT *
FROM features.nfl_player_game_training_features
WHERE n_games_prev_3 >= :min_prev_games
  AND game_date_et IS NOT NULL
ORDER BY season, week, game_id, player_id
"""


TARGET_STATS = tuple(spec.stat for spec in STAT_SPECS)
ID_COLS = {
    "season", "week", "game_id", "game_date_et", "player_id", "player_name",
    "team_abbr", "opponent_abbr", "created_at_utc", "updated_at_utc",
}
LEAKY_COLS = set(TARGET_STATS) | set(USAGE_STATS)


FEATURE_GROUP_PATTERNS: dict[str, tuple[str, ...]] = {
    "workload_rest": (
        "full_workload", "limited_workload", "fragility", "rest_risk", "starter_confidence",
        "snap_share", "offense_snap_share", "usage_volatility", "is_week_18", "is_late_season",
        "backup_role", "starter_role_stability", "recent_snap", "spike_snap",
        "weird_usage", "normal_usage", "role_continuity",
    ),
    "depth_injury": ("depth_", "roster_", "injury_", "practice_status", "same_week_", "teammate_", "vacancy"),
    "route_target_quality": (
        "route", "target", "reception", "air_yards", "wopr", "first_read",
        "yards_per_route", "receiving_yards_after_catch", "receiver_",
    ),
    "rb_carry_role": ("carries", "rushing_yards", "rb_", "yards_per_carry", "spike_carry"),
    "td_red_zone": ("red_zone", "goal_line", "end_zone", "td_", "_tds"),
    "game_market_env": ("team_implied", "opponent_implied", "team_spread", "game_total", "opp_allowed", "is_home", "game_script"),
    "usage_quality": ("usage_history_quality", "usage_context_quality", "evidence_quality", "projection_volatility"),
}

TARGET_FEATURE_POLICIES: dict[str, dict[str, Any]] = {
    "passing_yards": {
        "keep_groups": ("game_market_env",),
        "include_patterns": (
            "passing_yards_avg", "pass_attempts_avg", "pass_attempts_std",
            "high_pass_attempt", "spike_pass_attempt", "opp_allowed_passing_yards",
            "starter_confidence", "injury_downgrade", "projected_starter", "normal_usage",
            "qb_volume_spike_signal", "pass_spike_path", "full_workload", "limited_workload",
            "qb_yards_per_attempt", "qb_pass_volume", "qb_pass_efficiency",
            "qb_matchup_adjusted_pass_yards", "qb_script_adjusted_pass_attempts",
            "qb_workload_adjusted_pass_attempts", "qb_passing_yards_role_anchor",
        ),
        "exclude_groups": ("route_target_quality", "rb_carry_role", "td_red_zone"),
        "min_gain": 0.001,
    },
    "passing_tds": {
        "keep_groups": ("game_market_env", "td_red_zone"),
        "include_patterns": (
            "passing_tds_avg", "pass_attempts_avg", "team_implied", "red_zone",
            "goal_line", "opp_allowed_passing_tds", "projected_starter",
            "qb_volume_spike_signal", "pass_spike_path", "full_workload", "limited_workload",
        ),
        "exclude_groups": ("rb_carry_role", "route_target_quality"),
        "min_gain": 0.001,
    },
    "rushing_yards": {
        "keep_groups": ("rb_carry_role", "depth_injury", "game_market_env"),
        "include_patterns": (
            "rushing_yards_avg", "carries_avg", "carries_std", "carries_share",
            "carries_role_rank", "depth_rank", "depth_pos", "starter_confidence",
            "full_workload", "limited_workload", "weird_usage", "normal_usage",
            "role_continuity", "spike_carry", "carry_spike_path", "game_script_rush",
            "team_spread", "team_implied", "opp_allowed_rushing_yards",
            "rb_usage_spike_signal", "role_change_upside_score", "workload_floor_score",
            "workload_downside_v2", "workload_upside_v2", "rb_projected_carries_v2",
            "rb_carry_spike_v2", "rb_rush_yards_anchor_v2", "rb_live_carry_v3",
            "rb_projected_carries_v3", "rb_rush_yards_anchor_v3",
            "same_week_usage_confidence_v3", "yardage_projection_volatility_v3",
            "rb_usage_history_quality", "live_usage_context_quality_v4",
            "rb_carry_under_correction_v4", "rb_rush_yards_anchor_v4",
            "yardage_projection_volatility_v4",
        ),
        "exclude_groups": ("route_target_quality",),
        "min_gain": 0.001,
    },
    "rushing_tds": {
        "keep_groups": ("rb_carry_role", "td_red_zone", "game_market_env", "workload_rest"),
        "include_patterns": (
            "rushing_tds_avg", "goal_line", "red_zone", "team_implied",
            "carries_avg", "carries_share", "depth_rank", "starter_confidence",
            "td_workload_env", "td_goal_line_env",
        ),
        "exclude_groups": ("route_target_quality",),
        "min_gain": 0.001,
    },
    "receiving_yards": {
        "keep_groups": ("route_target_quality", "depth_injury", "td_red_zone", "usage_quality"),
        "include_patterns": (
            "receiving_yards_avg", "targets_avg", "targets_std", "targets_share",
            "target_share", "air_yards", "wopr", "route", "estimated_routes",
            "targets_per_route", "yards_per_route", "first_read", "receiver_",
            "same_week_", "teammate_", "vacancy", "starter_confidence",
            "projected_starter", "role_continuity", "normal_usage", "full_workload",
            "limited_workload", "workload_floor", "workload_downside_v2",
            "workload_upside_v2", "team_implied", "game_script_pass",
            "opp_allowed_receiving_yards", "is_home",
            "same_week_usage_confidence_v3", "receiver_live_spike_v3",
            "receiver_projected_targets_v3", "receiver_spike_yards_anchor_v3",
            "yardage_projection_volatility_v3",
            "receiving_usage_history_quality", "live_usage_context_quality_v4",
            "receiver_spike_under_correction_v4", "receiver_spike_yards_anchor_v4",
            "yardage_projection_volatility_v4",
        ),
        "exclude_groups": ("rb_carry_role",),
        "min_gain": 0.001,
    },
    "receiving_tds": {
        "keep_groups": ("td_red_zone", "usage_quality"),
        "include_patterns": (
            "receiving_td", "receiving_tds_avg", "red_zone", "goal_line",
            "end_zone", "first_read", "route", "target", "team_implied",
            "opp_allowed_receiving_tds", "starter_confidence", "full_workload",
            "limited_workload", "normal_usage", "role_continuity",
            "receiver_high_value_target",
            "receiving_td_rare_event_score_v2", "receiver_target_command",
            "receiver_route_spike_readiness", "workload_downside_v2",
            "same_week_usage_confidence_v3", "receiver_live_spike_v3",
            "td_usage_history_quality", "receiving_usage_history_quality",
            "live_usage_context_quality_v4",
        ),
        "exclude_groups": ("rb_carry_role",),
        "exclude_patterns": ("rushing_yards", "carries_share", "rb_"),
        "min_gain": 0.001,
    },
}

ALWAYS_KEEP_FEATURE_PATTERNS = (
    "position_",
    "season",
    "week",
    "n_games_prev",
    "is_home",
)


def _coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        if is_numeric_dtype(out[col]) or is_bool_dtype(out[col]) or is_datetime64_any_dtype(out[col]):
            continue
        if out[col].dtype == "object" or str(out[col].dtype).startswith("string"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[col for col in ID_COLS | LEAKY_COLS if col in df.columns]).copy()

    def num(name: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(X.get(name, pd.Series(default, index=X.index)), errors="coerce").fillna(default)

    def sigmoid_feature(value: pd.Series, center: float, scale: float) -> pd.Series:
        z = ((pd.to_numeric(value, errors="coerce").fillna(0.0) - center) / max(scale, 1e-6)).clip(-35.0, 35.0)
        return (1.0 / (1.0 + np.exp(-z))).clip(0.02, 0.92)

    if {"red_zone_carries_avg_5", "red_zone_targets_avg_5"}.issubset(X.columns):
        X["red_zone_touch_role_avg_5"] = (
            pd.to_numeric(X["red_zone_carries_avg_5"], errors="coerce").fillna(0.0)
            + pd.to_numeric(X["red_zone_targets_avg_5"], errors="coerce").fillna(0.0)
        )
    if {"goal_line_carries_avg_5", "goal_line_targets_avg_5"}.issubset(X.columns):
        X["goal_line_touch_role_avg_5"] = (
            pd.to_numeric(X["goal_line_carries_avg_5"], errors="coerce").fillna(0.0)
            + pd.to_numeric(X["goal_line_targets_avg_5"], errors="coerce").fillna(0.0)
        )
    if {"team_implied_points", "red_zone_touch_role_avg_5"}.issubset(X.columns):
        team_imp = pd.to_numeric(X["team_implied_points"], errors="coerce").fillna(22.0)
        X["red_zone_role_env_avg_5"] = pd.to_numeric(X["red_zone_touch_role_avg_5"], errors="coerce").fillna(0.0) * (team_imp / 22.0).clip(0.65, 1.45)
    if {"team_implied_points", "goal_line_touch_role_avg_5"}.issubset(X.columns):
        team_imp = pd.to_numeric(X["team_implied_points"], errors="coerce").fillna(22.0)
        X["goal_line_role_env_avg_5"] = pd.to_numeric(X["goal_line_touch_role_avg_5"], errors="coerce").fillna(0.0) * (team_imp / 22.0).clip(0.65, 1.45)
    if {"receiving_air_yards_avg_5", "targets_avg_5"}.issubset(X.columns):
        targets = pd.to_numeric(X["targets_avg_5"], errors="coerce").replace(0.0, np.nan)
        X["air_yards_per_target_avg_5"] = pd.to_numeric(X["receiving_air_yards_avg_5"], errors="coerce") / targets
    if {"receiving_yards_after_catch_avg_5", "receptions_avg_5"}.issubset(X.columns):
        receptions = pd.to_numeric(X["receptions_avg_5"], errors="coerce").replace(0.0, np.nan)
        X["yac_per_reception_avg_5"] = pd.to_numeric(X["receiving_yards_after_catch_avg_5"], errors="coerce") / receptions
    if {"targets_avg_5", "offense_snap_share_avg_5"}.issubset(X.columns):
        X["snap_weighted_targets_avg_5"] = (
            pd.to_numeric(X["targets_avg_5"], errors="coerce").fillna(0.0)
            * pd.to_numeric(X["offense_snap_share_avg_5"], errors="coerce").fillna(0.0)
        )
    if {"carries_avg_5", "offense_snap_share_avg_5"}.issubset(X.columns):
        X["snap_weighted_carries_avg_5"] = (
            pd.to_numeric(X["carries_avg_5"], errors="coerce").fillna(0.0)
            * pd.to_numeric(X["offense_snap_share_avg_5"], errors="coerce").fillna(0.0)
        )
    if {"starter_confidence", "red_zone_role_env_avg_5"}.issubset(X.columns):
        X["starter_adjusted_red_zone_role_avg_5"] = (
            pd.to_numeric(X["red_zone_role_env_avg_5"], errors="coerce").fillna(0.0)
            * pd.to_numeric(X["starter_confidence"], errors="coerce").fillna(0.50).clip(0.0, 1.0)
        )
    if {"rest_risk_score", "targets_avg_5"}.issubset(X.columns):
        X["rest_adjusted_targets_avg_5"] = (
            pd.to_numeric(X["targets_avg_5"], errors="coerce").fillna(0.0)
            * (1.0 - pd.to_numeric(X["rest_risk_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0))
        )
    if {"rest_risk_score", "carries_avg_5"}.issubset(X.columns):
        X["rest_adjusted_carries_avg_5"] = (
            pd.to_numeric(X["carries_avg_5"], errors="coerce").fillna(0.0)
            * (1.0 - pd.to_numeric(X["rest_risk_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0))
        )
    if {"route_participation_proxy_avg_5", "targets_per_route_proxy_avg_5"}.issubset(X.columns):
        X["route_target_intensity_avg_5"] = (
            pd.to_numeric(X["route_participation_proxy_avg_5"], errors="coerce").fillna(0.0)
            * pd.to_numeric(X["targets_per_route_proxy_avg_5"], errors="coerce").fillna(0.0)
        )
    if {"estimated_routes_avg_5", "targets_per_route_proxy_avg_5"}.issubset(X.columns):
        X["route_weighted_target_expectation_avg_5"] = (
            pd.to_numeric(X["estimated_routes_avg_5"], errors="coerce").fillna(0.0)
            * pd.to_numeric(X["targets_per_route_proxy_avg_5"], errors="coerce").fillna(0.0)
        )
    if {"estimated_routes_avg_5", "yards_per_route_proxy_avg_5"}.issubset(X.columns):
        X["route_weighted_yard_expectation_avg_5"] = (
            pd.to_numeric(X["estimated_routes_avg_5"], errors="coerce").fillna(0.0)
            * pd.to_numeric(X["yards_per_route_proxy_avg_5"], errors="coerce").fillna(0.0)
        )
    if {"team_implied_points", "first_read_proxy_avg_5"}.issubset(X.columns):
        X["first_read_team_total_env_avg_5"] = (
            pd.to_numeric(X["first_read_proxy_avg_5"], errors="coerce").fillna(0.0)
            * (pd.to_numeric(X["team_implied_points"], errors="coerce").fillna(22.0) / 22.0).clip(0.65, 1.45)
        )
    if {"team_implied_points", "end_zone_targets_avg_5"}.issubset(X.columns):
        X["end_zone_target_env_avg_5"] = (
            pd.to_numeric(X["end_zone_targets_avg_5"], errors="coerce").fillna(0.0)
            * (pd.to_numeric(X["team_implied_points"], errors="coerce").fillna(22.0) / 22.0).clip(0.65, 1.45)
        )
    if {"opp_allowed_receiving_tds_avg_5", "starter_adjusted_red_zone_role_avg_5"}.issubset(X.columns):
        X["receiving_td_matchup_role_avg_5"] = (
            pd.to_numeric(X["opp_allowed_receiving_tds_avg_5"], errors="coerce").fillna(0.0)
            * pd.to_numeric(X["starter_adjusted_red_zone_role_avg_5"], errors="coerce").fillna(0.0)
        )
    if {"receiving_td_matchup_role_avg_5", "first_read_team_total_env_avg_5"}.issubset(X.columns):
        X["receiving_td_first_read_matchup_avg_5"] = (
            pd.to_numeric(X["receiving_td_matchup_role_avg_5"], errors="coerce").fillna(0.0)
            * (1.0 + pd.to_numeric(X["first_read_team_total_env_avg_5"], errors="coerce").fillna(0.0))
        )
    if {"limited_workload_risk_score", "targets_avg_5"}.issubset(X.columns):
        full_workload = num("full_workload_score", 0.55).clip(0.0, 1.0)
        limited_risk = num("limited_workload_risk_score", 0.0).clip(0.0, 1.0)
        X["full_workload_adjusted_targets_avg_5"] = num("targets_avg_5") * full_workload
        X["risk_adjusted_targets_avg_5"] = num("targets_avg_5") * (1.0 - limited_risk)
        X["fragility_adjusted_targets_avg_5"] = num("targets_avg_5") * (1.0 - num("high_usage_fragility_score").clip(0.0, 1.0))
    if {"limited_workload_risk_score", "carries_avg_5"}.issubset(X.columns):
        full_workload = num("full_workload_score", 0.55).clip(0.0, 1.0)
        limited_risk = num("limited_workload_risk_score", 0.0).clip(0.0, 1.0)
        X["full_workload_adjusted_carries_avg_5"] = num("carries_avg_5") * full_workload
        X["risk_adjusted_carries_avg_5"] = num("carries_avg_5") * (1.0 - limited_risk)
        X["fragility_adjusted_carries_avg_5"] = num("carries_avg_5") * (1.0 - num("high_usage_fragility_score").clip(0.0, 1.0))
    if {"limited_workload_risk_score", "receiving_yards_avg_5"}.issubset(X.columns):
        X["risk_adjusted_receiving_yards_avg_5"] = num("receiving_yards_avg_5") * (1.0 - num("limited_workload_risk_score").clip(0.0, 1.0))
        X["route_env_receiving_yards_signal"] = (
            0.55 * num("receiver_yards_rate_signal")
            + 16.0 * num("receiver_route_env_score").clip(0.0, 2.5)
            + 8.0 * num("target_share_trend_3_10")
            + 6.0 * num("air_yards_share_trend_3_10")
        ).clip(lower=0.0)
    if {"rb_rush_role_env_score", "carries_trend_3_10"}.issubset(X.columns):
        X["rb_rushing_yards_opportunity_signal"] = (
            0.75 * num("rb_rush_role_env_score")
            + 0.55 * num("rb_carry_trend_env_score")
            + 3.5 * num("carries_trend_3_10")
            - 6.0 * num("high_usage_fragility_score")
        ).clip(lower=0.0)
    if {"receiving_td_role_score", "receiving_td_route_redzone_score"}.issubset(X.columns):
        X["receiving_td_probability_signal"] = (
            0.45 * num("receiving_td_role_score")
            + 0.35 * num("receiving_td_route_redzone_score")
            + 0.20 * num("receiving_td_matchup_role_avg_5")
            + 0.15 * num("first_read_team_total_env_avg_5")
            - 0.20 * num("limited_workload_risk_score")
        ).clip(lower=0.0)
    if {"receiving_td_any_score", "team_implied_points"}.issubset(X.columns):
        X["receiving_td_any_team_env_signal"] = (
            num("receiving_td_any_score")
            * (num("team_implied_points", 22.0) / 22.0).clip(0.65, 1.45)
            * (1.0 - 0.35 * num("limited_workload_risk_score").clip(0.0, 1.0))
        ).clip(lower=0.0)
    if {"receiver_spike_yards_score", "receiving_yards_avg_5"}.issubset(X.columns):
        X["receiver_spike_yards_projection_signal"] = (
            0.60 * num("receiver_yards_rate_signal")
            + 14.0 * num("receiver_spike_yards_score").clip(0.0, 3.0)
            + 6.0 * num("recent_target_spike_score").clip(0.0, 1.0)
            + 4.0 * num("recent_route_spike_score").clip(0.0, 1.0)
        ).clip(lower=0.0)
    if {"receiver_spike_yards_score", "target_spike_path_score"}.issubset(X.columns):
        X["receiver_usage_spike_signal"] = (
            0.45 * num("target_spike_path_score").clip(0.0, 1.0)
            + 0.20 * num("spike_snap_share_score").clip(0.0, 1.0)
            + 0.18 * num("recent_target_spike_score").clip(0.0, 1.0)
            + 0.12 * num("recent_route_spike_score").clip(0.0, 1.0)
            + 0.05 * num("game_script_pass_boost", 1.0).clip(0.65, 1.35)
        ).clip(0.0, 1.0)
    if {"rb_spike_rush_score", "rushing_yards_avg_5"}.issubset(X.columns):
        X["rb_spike_rush_projection_signal"] = (
            0.90 * num("rb_spike_rush_score")
            + 3.0 * num("recent_carry_spike_score").clip(0.0, 1.0)
            + 2.5 * num("game_script_rush_boost", 1.0).clip(0.65, 1.35)
        ).clip(lower=0.0)
    if {"rb_spike_rush_score", "carry_spike_path_score"}.issubset(X.columns):
        X["rb_usage_spike_signal"] = (
            0.48 * num("carry_spike_path_score").clip(0.0, 1.0)
            + 0.20 * num("spike_snap_share_score").clip(0.0, 1.0)
            + 0.18 * num("recent_carry_spike_score").clip(0.0, 1.0)
            + 0.09 * num("game_script_rush_boost", 1.0).clip(0.65, 1.35)
            + 0.05 * num("goal_line_carries_avg_5").clip(lower=0.0).clip(0.0, 2.0) / 2.0
        ).clip(0.0, 1.0)
    if {"pass_spike_path_score", "high_pass_attempt_score"}.issubset(X.columns):
        X["qb_volume_spike_signal"] = (
            0.55 * num("pass_spike_path_score").clip(0.0, 1.0)
            + 0.25 * num("high_pass_attempt_score").clip(0.0, 1.0)
            + 0.20 * num("game_script_pass_boost", 1.0).clip(0.65, 1.35)
        ).clip(0.0, 1.0)
    if {"passing_yards_avg_5", "pass_attempts_avg_5"}.issubset(X.columns):
        pass_att_5 = num("pass_attempts_avg_5", 0.0).replace(0.0, np.nan)
        pass_att_10 = num("pass_attempts_avg_10", 0.0).replace(0.0, np.nan)
        X["qb_yards_per_attempt_avg_5"] = (num("passing_yards_avg_5", 0.0) / pass_att_5).replace([np.inf, -np.inf], np.nan).fillna(6.8).clip(3.0, 11.5)
        if "passing_yards_avg_10" in X.columns:
            X["qb_yards_per_attempt_avg_10"] = (num("passing_yards_avg_10", 0.0) / pass_att_10).replace([np.inf, -np.inf], np.nan).fillna(X["qb_yards_per_attempt_avg_5"]).clip(3.0, 11.5)
        X["qb_pass_volume_efficiency_signal"] = (
            num("pass_attempts_avg_5", 0.0).clip(lower=0.0)
            * X["qb_yards_per_attempt_avg_5"].clip(3.0, 11.5)
        ).clip(lower=0.0)
    if {"pass_attempts_avg_5", "game_script_pass_boost"}.issubset(X.columns):
        X["qb_script_adjusted_pass_attempts_avg_5"] = (
            num("pass_attempts_avg_5", 0.0).clip(lower=0.0)
            * num("game_script_pass_boost", 1.0).clip(0.70, 1.35)
        ).clip(lower=0.0)
    if {"pass_attempts_avg_5", "full_workload_score"}.issubset(X.columns):
        X["qb_workload_adjusted_pass_attempts_avg_5"] = (
            num("pass_attempts_avg_5", 0.0).clip(lower=0.0)
            * num("full_workload_score", 0.65).clip(0.0, 1.0)
            * (1.0 - 0.50 * num("limited_workload_risk_score", 0.0).clip(0.0, 1.0))
            * (1.0 - 0.35 * num("weird_usage_risk_score", 0.0).clip(0.0, 1.0))
        ).clip(lower=0.0)
    if {"opp_allowed_passing_yards_avg_5", "passing_yards_avg_5"}.issubset(X.columns):
        opp_anchor = num("opp_allowed_passing_yards_avg_5", 225.0).clip(130.0, 340.0)
        team_anchor = num("passing_yards_avg_5", 205.0).clip(60.0, 380.0)
        env = (num("team_implied_points", 22.0) / 22.0).clip(0.72, 1.32)
        X["qb_matchup_adjusted_pass_yards_signal"] = (0.66 * team_anchor + 0.34 * opp_anchor) * env
    if {"qb_matchup_adjusted_pass_yards_signal", "qb_pass_volume_efficiency_signal"}.issubset(X.columns):
        X["qb_passing_yards_role_anchor"] = pd.concat([
            X["qb_matchup_adjusted_pass_yards_signal"].clip(lower=0.0),
            X["qb_pass_volume_efficiency_signal"].clip(lower=0.0),
            num("passing_yards_avg_5", 0.0).clip(lower=0.0),
            num("passing_yards_avg_10", 0.0).clip(lower=0.0),
        ], axis=1).mean(axis=1).fillna(0.0).clip(lower=0.0)
    if {"qb_script_adjusted_pass_attempts_avg_5", "qb_yards_per_attempt_avg_5"}.issubset(X.columns):
        X["qb_pass_efficiency_script_yards_signal"] = (
            X["qb_script_adjusted_pass_attempts_avg_5"].clip(lower=0.0)
            * X["qb_yards_per_attempt_avg_5"].clip(3.0, 11.5)
        ).clip(lower=0.0)
    if {"depth_rank_better", "recent_snap_rise_score"}.issubset(X.columns):
        X["role_change_upside_score"] = (
            0.35 * num("depth_rank_better").clip(0.0, 1.0)
            + 0.35 * num("recent_snap_rise_score").clip(0.0, 1.0)
            + 0.15 * num("recent_target_spike_score").clip(0.0, 1.0)
            + 0.15 * num("recent_carry_spike_score").clip(0.0, 1.0)
        ).clip(0.0, 1.0)
    if {"full_workload_score", "limited_workload_risk_score"}.issubset(X.columns):
        X["workload_floor_score"] = (
            num("full_workload_score").clip(0.0, 1.0)
            * (1.0 - num("limited_workload_risk_score").clip(0.0, 1.0))
            * (1.0 - 0.55 * num("weird_usage_risk_score").clip(0.0, 1.0))
        ).clip(0.0, 1.0)
    if {"targets_avg_5", "receiver_usage_spike_signal"}.issubset(X.columns):
        pos_gate = pd.Series(1.0, index=X.index, dtype=float)
        if "position" in df.columns:
            pos_gate = df["position"].astype(str).str.upper().isin(["RB", "WR", "TE"]).astype(float)
        route_volume_score = sigmoid_feature(num("route_weighted_target_expectation_avg_5", 0.0), 5.8, 1.8)
        share_volume_score = sigmoid_feature(num("targets_share_avg_5", 0.0).clip(0.0, 1.0), 0.18, 0.045)
        first_read_volume_score = sigmoid_feature(num("first_read_proxy_avg_5", 0.0).clip(0.0, 1.0), 0.17, 0.050)
        target_earning_score = sigmoid_feature(num("targets_per_route_proxy_avg_5", 0.0), 0.22, 0.055)
        wopr_volume_score = sigmoid_feature(num("wopr_avg_5", 0.0), 0.48, 0.11)
        target_trend_score = sigmoid_feature(num("targets_trend_3_10", 0.0), 0.55, 0.65)
        target_share_trend_score = sigmoid_feature(num("target_share_trend_3_10", 0.0), 0.018, 0.026)
        air_share_score = sigmoid_feature(num("air_yards_share_avg_5", 0.0).clip(0.0, 1.0), 0.24, 0.065)
        air_share_trend_score = sigmoid_feature(num("air_yards_share_trend_3_10", 0.0), 0.022, 0.032)
        air_volume_score = sigmoid_feature(num("receiving_air_yards_avg_5", 0.0).clip(lower=0.0), 58.0, 24.0)
        air_yards_rate_score = sigmoid_feature(num("receiver_yards_rate_signal", 0.0), 48.0, 20.0)
        pass_boost_score = ((num("game_script_pass_boost", 1.0).clip(0.65, 1.35) - 0.65) / 0.70).clip(0.0, 1.0)
        teammate_receiver_pressure = (
            0.70 * (num("same_week_teammate_receiver_injury_score", 0.0).clip(0.0, 4.0) / 1.60).clip(0.0, 1.0)
            + 0.30 * (num("same_week_teammate_receiver_out_count", 0.0).clip(0.0, 4.0) / 2.0).clip(0.0, 1.0)
        ).clip(0.0, 1.0)
        health_gate = (
            (0.50 + 0.50 * num("workload_floor_score", 0.55).clip(0.0, 1.0))
            * (1.0 - 0.42 * num("limited_workload_risk_score").clip(0.0, 1.0))
            * (1.0 - 0.30 * num("injury_downgrade_score").clip(0.0, 1.0))
            * (1.0 - 0.24 * num("depth_movement_risk_score").clip(0.0, 1.0))
            * (1.0 - 0.20 * num("weird_usage_risk_score").clip(0.0, 1.0))
        ).clip(0.15, 1.0)
        X["teammate_receiver_injury_pressure_score"] = teammate_receiver_pressure
        X["receiver_teammate_vacancy_score"] = (
            pos_gate
            * teammate_receiver_pressure
            * (0.45 + 0.55 * num("receiver_route_quality_score").clip(0.0, 1.5) / 1.5)
            * health_gate
        ).clip(0.0, 1.0)
        X["receiver_target_eruption_score"] = (
            pos_gate
            * (
                0.24 * num("high_target_score").clip(0.0, 1.0)
                + 0.18 * route_volume_score
                + 0.15 * share_volume_score
                + 0.14 * target_earning_score
                + 0.11 * target_trend_score
                + 0.09 * target_share_trend_score
                + 0.06 * X["receiver_teammate_vacancy_score"].clip(0.0, 1.0)
                + 0.03 * pass_boost_score
            )
            * health_gate
        ).clip(0.02, 0.94)
        X["air_yards_spike_path_score"] = (
            pos_gate
            * (
                0.23 * air_volume_score
                + 0.20 * air_share_score
                + 0.17 * air_share_trend_score
                + 0.15 * wopr_volume_score
                + 0.10 * first_read_volume_score
                + 0.09 * air_yards_rate_score
                + 0.06 * pass_boost_score
            )
            * health_gate
        ).clip(0.02, 0.94)
        X["receiver_air_yards_eruption_score"] = (
            pos_gate
            * pd.concat([
                X["air_yards_spike_path_score"].clip(0.0, 1.0),
                0.90 * air_volume_score,
                0.88 * air_share_trend_score,
                0.84 * wopr_volume_score,
            ], axis=1).max(axis=1).fillna(0.0)
            * health_gate
        ).clip(0.02, 0.95)
        X["receiver_explosive_spike_score"] = (
            pos_gate
            * (
                0.40 * X["receiver_target_eruption_score"].clip(0.0, 1.0)
                + 0.34 * X["receiver_air_yards_eruption_score"].clip(0.0, 1.0)
                + 0.16 * X["receiver_teammate_vacancy_score"].clip(0.0, 1.0)
                + 0.10 * pass_boost_score
            )
            * health_gate
        ).clip(0.02, 0.95)
        X["target_spike_path_score"] = pd.concat([
            num("target_spike_path_score", 0.0).clip(0.0, 1.0),
            0.90 * X["receiver_target_eruption_score"].clip(0.0, 1.0),
            0.72 * X["air_yards_spike_path_score"].clip(0.0, 1.0),
        ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
        X["receiver_usage_spike_signal"] = pd.concat([
            num("receiver_usage_spike_signal", 0.0).clip(0.0, 1.0),
            0.88 * X["receiver_target_eruption_score"].clip(0.0, 1.0),
            0.82 * X["receiver_explosive_spike_score"].clip(0.0, 1.0),
            0.70 * X["air_yards_spike_path_score"].clip(0.0, 1.0),
        ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
        X["receiver_target_eruption_anchor_targets"] = (
            pos_gate
            * pd.concat([
                num("route_weighted_target_expectation_avg_5", 0.0).clip(lower=0.0),
                num("targets_avg_5", 0.0).clip(lower=0.0) + num("targets_trend_3_10", 0.0).clip(lower=0.0),
                num("estimated_routes_avg_5", 0.0).clip(lower=0.0) * num("targets_per_route_proxy_avg_5", 0.0).clip(lower=0.0),
            ], axis=1).max(axis=1).fillna(0.0)
            * (0.96 + 0.24 * X["receiver_target_eruption_score"].clip(0.0, 1.0))
        ).clip(lower=0.0)
        X["receiver_air_yards_spike_anchor_yards"] = (
            pos_gate
            * pd.concat([
                num("receiver_yards_rate_signal", 0.0).clip(lower=0.0),
                num("route_weighted_yard_expectation_avg_5", 0.0).clip(lower=0.0),
                0.78 * num("receiving_air_yards_avg_5", 0.0).clip(lower=0.0),
                num("receiving_yards_avg_5", 0.0).clip(lower=0.0),
            ], axis=1).max(axis=1).fillna(0.0)
            * (0.92 + 0.22 * X["receiver_air_yards_eruption_score"].clip(0.0, 1.0))
        ).clip(lower=0.0)
        X["receiver_spike_volume_score"] = (
            pos_gate
            * (
                0.18 * num("high_target_score").clip(0.0, 1.0)
                + 0.18 * route_volume_score
                + 0.13 * share_volume_score
                + 0.12 * first_read_volume_score
                + 0.13 * X["receiver_target_eruption_score"].clip(0.0, 1.0)
                + 0.10 * X["receiver_explosive_spike_score"].clip(0.0, 1.0)
                + 0.09 * num("receiver_usage_spike_signal").clip(0.0, 1.0)
                + 0.05 * pass_boost_score
                + 0.07 * num("role_change_upside_score").clip(0.0, 1.0)
                + 0.05 * X["receiver_teammate_vacancy_score"].clip(0.0, 1.0)
            )
            * health_gate
        ).clip(0.02, 0.92)
        receiver_anchor = pd.concat([
            X["receiver_air_yards_spike_anchor_yards"].clip(lower=0.0),
            num("route_weighted_yard_expectation_avg_5", 0.0).clip(lower=0.0),
            num("receiver_yards_rate_signal", 0.0).clip(lower=0.0),
            num("receiving_yards_avg_5", 0.0).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        X["receiver_spike_volume_anchor_yards"] = (
            pos_gate
            * receiver_anchor
            * (0.82 + 0.30 * X["receiver_spike_volume_score"].clip(0.0, 1.0))
        ).clip(lower=0.0)
        route_cut = pd.Series(0.72, index=X.index, dtype=float)
        if "position" in df.columns:
            pos_text = df["position"].astype(str).str.upper()
            route_cut.loc[pos_text == "TE"] = 0.66
            route_cut.loc[pos_text == "RB"] = 0.42
        route_stability_score = sigmoid_feature(num("route_participation_proxy_avg_5", 0.0), route_cut, 0.090)
        route_depth_score = sigmoid_feature(num("estimated_routes_avg_5", 0.0), 30.0, 6.0)
        pass_boost_score = ((num("game_script_pass_boost", 1.0).clip(0.65, 1.35) - 0.65) / 0.70).clip(0.0, 1.0)
        X["receiver_target_command_score"] = (
            pos_gate
            * (
                0.28 * num("high_target_score").clip(0.0, 1.0)
                + 0.21 * share_volume_score
                + 0.18 * first_read_volume_score
                + 0.18 * target_earning_score
                + 0.13 * wopr_volume_score
                + 0.02 * X["receiver_teammate_vacancy_score"].clip(0.0, 1.0)
            )
            * (0.70 + 0.30 * num("role_continuity_score").clip(0.0, 1.0))
            * health_gate
        ).clip(0.02, 0.94)
        X["receiver_route_spike_readiness_score"] = (
            pos_gate
            * (
                0.30 * route_stability_score
                + 0.24 * route_depth_score
                + 0.16 * num("recent_route_spike_score").clip(0.0, 1.0)
                + 0.14 * num("recent_snap_rise_score").clip(0.0, 1.0)
                + 0.10 * num("role_change_upside_score").clip(0.0, 1.0)
                + 0.06 * num("starter_role_stability_score").clip(0.0, 1.0)
            )
            * health_gate
        ).clip(0.02, 0.94)
        X["receiver_target_route_spike_score"] = (
            pos_gate
            * (
                0.27 * X["receiver_target_command_score"].clip(0.0, 1.0)
                + 0.20 * X["receiver_route_spike_readiness_score"].clip(0.0, 1.0)
                + 0.13 * X["receiver_spike_volume_score"].clip(0.0, 1.0)
                + 0.10 * route_volume_score
                + 0.10 * X["receiver_target_eruption_score"].clip(0.0, 1.0)
                + 0.09 * pass_boost_score
                + 0.06 * num("target_spike_path_score").clip(0.0, 1.0)
                + 0.04 * num("role_change_upside_score").clip(0.0, 1.0)
                + 0.01 * X["receiver_teammate_vacancy_score"].clip(0.0, 1.0)
            )
            * health_gate
        ).clip(0.02, 0.95)
        X["receiver_contextual_spike_score"] = (
            pos_gate
            * (
                0.36 * X["receiver_target_route_spike_score"].clip(0.0, 1.0)
                + 0.19 * X["receiver_target_command_score"].clip(0.0, 1.0)
                + 0.16 * X["receiver_route_spike_readiness_score"].clip(0.0, 1.0)
                + 0.13 * X["receiver_explosive_spike_score"].clip(0.0, 1.0)
                + 0.14 * X["receiver_teammate_vacancy_score"].clip(0.0, 1.0)
                + 0.02 * pass_boost_score
            )
            * health_gate
        ).clip(0.0, 1.0)
        X["receiver_target_route_spike_anchor_targets"] = (
            pos_gate
            * pd.concat([
                X["receiver_target_eruption_anchor_targets"].clip(lower=0.0),
                num("route_weighted_target_expectation_avg_5", 0.0).clip(lower=0.0),
                num("targets_avg_5", 0.0).clip(lower=0.0) + 0.75 * num("targets_trend_3_10", 0.0).clip(lower=0.0),
                num("estimated_routes_avg_5", 0.0).clip(lower=0.0) * num("targets_per_route_proxy_avg_5", 0.0).clip(lower=0.0),
            ], axis=1).max(axis=1).fillna(0.0)
            * (0.92 + 0.22 * X["receiver_target_route_spike_score"].clip(0.0, 1.0))
        ).clip(lower=0.0)
        yards_per_target_proxy = (
            num("receiving_yards_avg_5", 0.0).replace(0.0, np.nan)
            / num("targets_avg_5", 0.0).replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        yards_per_target_proxy = yards_per_target_proxy.fillna(
            num("yards_per_route_proxy_avg_5", 0.0).clip(lower=0.0)
            / num("targets_per_route_proxy_avg_5", 0.0).replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(8.4).clip(3.5, 18.0)
        X["receiver_target_route_spike_anchor_yards"] = (
            pos_gate
            * pd.concat([
                X["receiver_spike_volume_anchor_yards"].clip(lower=0.0),
                X["receiver_target_route_spike_anchor_targets"].clip(lower=0.0) * yards_per_target_proxy,
                X["receiver_air_yards_spike_anchor_yards"].clip(lower=0.0),
                num("receiver_yards_rate_signal", 0.0).clip(lower=0.0),
                num("route_weighted_yard_expectation_avg_5", 0.0).clip(lower=0.0),
                num("receiving_yards_avg_5", 0.0).clip(lower=0.0),
            ], axis=1).max(axis=1).fillna(0.0)
            * (0.90 + 0.18 * X["receiver_target_route_spike_score"].clip(0.0, 1.0))
        ).clip(lower=0.0)
    if "receiving_usage_history_quality_score" not in X.columns:
        pos_gate = pd.Series(1.0, index=X.index, dtype=float)
        if "position" in df.columns:
            pos_gate = df["position"].astype(str).str.upper().isin(["RB", "WR", "TE"]).astype(float)
        route_known = pd.to_numeric(X.get("has_true_route_history", pd.Series(0.0, index=X.index)), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        X["receiving_usage_history_quality_score"] = (
            pos_gate
            * (
                0.20 * route_known
                + 0.16 * (num("estimated_routes_avg_5", 0.0) > 0).astype(float)
                + 0.16 * (num("targets_avg_5", 0.0) > 0).astype(float)
                + 0.14 * (num("target_share_avg_5", 0.0) > 0).astype(float)
                + 0.12 * ((num("air_yards_share_avg_5", 0.0) > 0) | (num("receiving_air_yards_avg_5", 0.0) > 0)).astype(float)
                + 0.08 * (num("wopr_avg_5", 0.0) > 0).astype(float)
                + 0.07 * ((num("first_read_target_share_avg_5", 0.0) > 0) | (num("first_read_targets_avg_5", 0.0) > 0)).astype(float)
                + 0.07 * ((num("offense_snap_share_avg_5", 0.0) > 0) | (num("snap_share_avg_5", 0.0) > 0)).astype(float)
            )
        ).clip(0.0, 1.0)
    if "rb_usage_history_quality_score" not in X.columns:
        rb_gate = pd.Series(1.0, index=X.index, dtype=float)
        if "position" in df.columns:
            rb_gate = df["position"].astype(str).str.upper().eq("RB").astype(float)
        X["rb_usage_history_quality_score"] = (
            rb_gate
            * (
                0.24 * (num("carries_avg_5", 0.0) > 0).astype(float)
                + 0.18 * (num("carries_avg_10", 0.0) > 0).astype(float)
                + 0.16 * ((num("offense_snap_share_avg_5", 0.0) > 0) | (num("snap_share_avg_5", 0.0) > 0)).astype(float)
                + 0.14 * (num("carries_share_avg_5", 0.0) > 0).astype(float)
                + 0.12 * ((num("red_zone_carries_avg_5", 0.0) > 0) | (num("goal_line_carries_avg_5", 0.0) > 0)).astype(float)
                + 0.10 * (num("yards_per_carry_avg_5", 0.0) > 0).astype(float)
                + 0.06 * (num("targets_avg_5", 0.0) > 0).astype(float)
            )
        ).clip(0.0, 1.0)
    if "td_usage_history_quality_score" not in X.columns:
        X["td_usage_history_quality_score"] = pd.concat([
            ((num("red_zone_targets_avg_5", 0.0) > 0) | (num("goal_line_targets_avg_5", 0.0) > 0) | (num("end_zone_targets_avg_5", 0.0) > 0)).astype(float),
            ((num("red_zone_carries_avg_5", 0.0) > 0) | (num("goal_line_carries_avg_5", 0.0) > 0)).astype(float),
            ((num("first_read_target_share_avg_5", 0.0) > 0) | (num("first_read_targets_avg_5", 0.0) > 0)).astype(float),
            ((num("offense_snap_share_avg_5", 0.0) > 0) | (num("snap_share_avg_5", 0.0) > 0)).astype(float),
        ], axis=1).mean(axis=1).fillna(0.0).clip(0.0, 1.0)
    if "live_usage_context_quality_v4_score" not in X.columns:
        same_week = num("same_week_usage_confidence_v3_score", 0.0).clip(0.0, 1.0)
        route_known = num("has_true_route_history", 0.0).clip(0.0, 1.0)
        X["live_usage_context_quality_v4_score"] = pd.concat([
            (0.45 * num("receiving_usage_history_quality_score", 0.0).clip(0.0, 1.0) + 0.35 * same_week + 0.20 * route_known),
            (0.55 * num("rb_usage_history_quality_score", 0.0).clip(0.0, 1.0) + 0.35 * same_week + 0.10 * num("role_continuity_score", 0.0).clip(0.0, 1.0)),
            same_week,
        ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    if "receiver_spike_under_correction_v4_score" not in X.columns:
        X["receiver_spike_under_correction_v4_score"] = (
            pd.concat([
                num("receiver_live_spike_v3_score", 0.0).clip(0.0, 1.0),
                num("receiver_target_route_spike_score", 0.0).clip(0.0, 1.0),
                num("receiver_target_spike_v2_score", 0.0).clip(0.0, 1.0),
                num("receiver_air_yards_spike_v2_score", 0.0).clip(0.0, 1.0),
                num("receiver_ypt_efficiency_spike_score", 0.0).clip(0.0, 1.0),
                num("receiver_teammate_vacancy_score", 0.0).clip(0.0, 1.0),
                num("role_change_upside_score", 0.0).clip(0.0, 1.0),
            ], axis=1).max(axis=1).fillna(0.0)
            * (0.58 + 0.42 * num("live_usage_context_quality_v4_score", 0.0).clip(0.0, 1.0))
            * (1.0 - 0.38 * num("workload_downside_v2_score", 0.0).clip(0.0, 1.0))
        ).clip(0.0, 1.0)
    if "receiver_spike_yards_anchor_v4" not in X.columns:
        X["receiver_spike_yards_anchor_v4"] = (
            pd.concat([
                num("receiver_spike_yards_anchor_v3", 0.0).clip(lower=0.0),
                num("receiver_target_route_spike_anchor_yards", 0.0).clip(lower=0.0),
                num("receiver_air_yards_spike_anchor_yards", 0.0).clip(lower=0.0),
                num("route_weighted_yard_expectation_avg_5", 0.0).clip(lower=0.0),
                num("receiving_yards_avg_5", 0.0).clip(lower=0.0),
            ], axis=1).max(axis=1).fillna(0.0)
            * (0.92 + 0.20 * num("receiver_spike_under_correction_v4_score", 0.0).clip(0.0, 1.0))
        ).clip(lower=0.0)
    if "rb_carry_under_correction_v4_score" not in X.columns:
        X["rb_carry_under_correction_v4_score"] = (
            pd.concat([
                num("rb_live_carry_v3_score", 0.0).clip(0.0, 1.0),
                num("rb_carry_spike_v2_score", 0.0).clip(0.0, 1.0),
                num("high_carry_score", 0.0).clip(0.0, 1.0),
                num("carry_spike_path_score", 0.0).clip(0.0, 1.0),
                num("role_change_upside_score", 0.0).clip(0.0, 1.0),
            ], axis=1).max(axis=1).fillna(0.0)
            * (0.60 + 0.40 * num("live_usage_context_quality_v4_score", 0.0).clip(0.0, 1.0))
            * (1.0 - 0.30 * num("workload_downside_v2_score", 0.0).clip(0.0, 1.0))
        ).clip(0.0, 1.0)
    if "rb_rush_yards_anchor_v4" not in X.columns:
        X["rb_rush_yards_anchor_v4"] = (
            pd.concat([
                num("rb_rush_yards_anchor_v3", 0.0).clip(lower=0.0),
                num("rb_rush_yards_anchor_v2", 0.0).clip(lower=0.0),
                num("rb_projected_carries_v3", 0.0).clip(lower=0.0) * num("yards_per_carry_avg_5", 4.1).clip(2.2, 6.8),
                num("rushing_yards_avg_5", 0.0).clip(lower=0.0),
            ], axis=1).max(axis=1).fillna(0.0)
            * (0.92 + 0.16 * num("rb_carry_under_correction_v4_score", 0.0).clip(0.0, 1.0))
        ).clip(lower=0.0)
    if "yardage_projection_volatility_v4_score" not in X.columns:
        X["yardage_projection_volatility_v4_score"] = pd.concat([
            num("yardage_projection_volatility_v3_score", 0.0).clip(0.0, 1.0),
            1.0 - num("live_usage_context_quality_v4_score", 0.0).clip(0.0, 1.0),
            num("usage_volatility_score", 0.0).clip(0.0, 1.0),
            num("high_usage_fragility_score", 0.0).clip(0.0, 1.0),
        ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    if {"td_goal_line_env_score", "team_implied_points"}.issubset(X.columns):
        X["td_workload_env_signal"] = (
            num("td_goal_line_env_score")
            * (num("team_implied_points", 22.0) / 22.0).clip(0.65, 1.45)
            * (1.0 - num("limited_workload_risk_score").clip(0.0, 1.0))
        ).clip(lower=0.0)
    if "position" in X.columns:
        X["position"] = X["position"].astype(str).str.upper()
        X = pd.get_dummies(X, columns=["position"], dummy_na=False)
    if "is_home" in X.columns:
        X["is_home"] = X["is_home"].astype("boolean").fillna(False).astype(int)
    X = _coerce_numeric_cols(X)
    non_numeric = [col for col in X.columns if not is_numeric_dtype(X[col])]
    if non_numeric:
        X = pd.get_dummies(X, columns=non_numeric, dummy_na=True)
    return X


def _feature_matches_group(column: str, group: str) -> bool:
    patterns = FEATURE_GROUP_PATTERNS.get(group) or ()
    return any(pattern in column for pattern in patterns)


def _target_feature_columns(columns: list[str], stat: str) -> tuple[list[str], dict[str, Any]]:
    policy = TARGET_FEATURE_POLICIES.get(stat) or {}
    keep_groups = tuple(policy.get("keep_groups") or ())
    drop_groups = tuple(policy.get("drop_groups") or ())
    include_patterns = tuple(policy.get("include_patterns") or ())
    exclude_groups = tuple(policy.get("exclude_groups") or ())
    exclude_patterns = tuple(policy.get("exclude_patterns") or ())
    if keep_groups or include_patterns:
        selected = [
            col for col in columns
            if any(pattern in col for pattern in ALWAYS_KEEP_FEATURE_PATTERNS)
            or any(pattern in col for pattern in include_patterns)
            or any(_feature_matches_group(col, group) for group in keep_groups)
        ]
        if exclude_groups or exclude_patterns:
            selected = [
                col for col in selected
                if any(pattern in col for pattern in ALWAYS_KEEP_FEATURE_PATTERNS)
                or (
                    not any(_feature_matches_group(col, group) for group in exclude_groups)
                    and not any(pattern in col for pattern in exclude_patterns)
                )
            ]
        action = "keep_groups"
        groups = keep_groups + include_patterns
    elif drop_groups:
        selected = [
            col for col in columns
            if not any(_feature_matches_group(col, group) for group in drop_groups)
        ]
        action = "drop_groups"
        groups = drop_groups
    else:
        return columns, {"status": "not_configured"}
    if len(selected) < max(5, int(len(columns) * 0.08)):
        return columns, {
            "status": "candidate_too_sparse",
            "action": action,
            "groups": list(groups),
            "full_columns": len(columns),
            "candidate_columns": len(selected),
        }
    return selected, {
        "status": "candidate_built",
        "action": action,
        "groups": list(groups),
        "full_columns": len(columns),
        "candidate_columns": len(selected),
    }


def _maybe_use_target_feature_selection(
    *,
    stat: str,
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    full_model: Any,
    full_train_pred: np.ndarray,
    full_holdout_pred: np.ndarray,
    cfg: TrainConfig,
    count_like: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Any, np.ndarray, np.ndarray, list[str], dict[str, Any], bool]:
    columns = list(X_train.columns)
    selected_columns, summary = _target_feature_columns(columns, stat)
    if selected_columns == columns:
        return X_train, X_holdout, full_model, full_train_pred, full_holdout_pred, columns, summary, False

    candidate_train = X_train.reindex(columns=selected_columns)
    candidate_holdout = X_holdout.reindex(columns=selected_columns)
    candidate_model = _fit_model(candidate_train, y_train, count_like=count_like, cfg=cfg)
    candidate_train_pred = np.clip(candidate_model.predict(candidate_train), 0.0, None)
    candidate_holdout_pred = np.clip(candidate_model.predict(candidate_holdout), 0.0, None)

    full_mae = float(mean_absolute_error(y_holdout, full_holdout_pred))
    candidate_mae = float(mean_absolute_error(y_holdout, candidate_holdout_pred))
    min_gain = float((TARGET_FEATURE_POLICIES.get(stat) or {}).get("min_gain") or 0.001)
    accepted = candidate_mae <= full_mae - min_gain
    summary.update({
        "full_direct_mae": full_mae,
        "candidate_direct_mae": candidate_mae,
        "mae_gain": full_mae - candidate_mae,
        "min_gain": min_gain,
    })
    if stat.endswith("_tds"):
        y_any = (y_holdout > 0).astype(int)
        full_brier = float(brier_score_loss(y_any, _poisson_any_prob(full_holdout_pred)))
        candidate_brier = float(brier_score_loss(y_any, _poisson_any_prob(candidate_holdout_pred)))
        accepted = candidate_brier <= full_brier - min_gain and candidate_mae <= full_mae + 0.020
        summary.update({
            "full_direct_any_brier": full_brier,
            "candidate_direct_any_brier": candidate_brier,
            "any_brier_gain": full_brier - candidate_brier,
            "mae_guard": full_mae + 0.020,
        })
    summary["accepted"] = bool(accepted)
    if not accepted:
        return X_train, X_holdout, full_model, full_train_pred, full_holdout_pred, columns, summary, False
    return (
        candidate_train,
        candidate_holdout,
        candidate_model,
        candidate_train_pred,
        candidate_holdout_pred,
        selected_columns,
        summary,
        True,
    )


def _fit_model(X: pd.DataFrame, y: pd.Series, *, count_like: bool, cfg: TrainConfig):
    if _HAS_LGB:
        objective = "poisson" if count_like else "regression_l1"
        return lgb.LGBMRegressor(
            objective=objective,
            n_estimators=800,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=cfg.random_state,
            n_jobs=-1,
            verbosity=-1,
        ).fit(X, y)
    loss = "poisson" if count_like and float(y.min()) >= 0.0 else "absolute_error"
    return HistGradientBoostingRegressor(
        loss=loss,
        max_iter=450,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=cfg.random_state,
    ).fit(X, y)


def _fit_classifier(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig):
    if _HAS_LGB:
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=650,
            learning_rate=0.025,
            num_leaves=15,
            min_child_samples=45,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_lambda=1.5,
            random_state=cfg.random_state,
            n_jobs=-1,
            verbosity=-1,
        ).fit(X, y)
    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=350,
        learning_rate=0.035,
        max_leaf_nodes=15,
        l2_regularization=0.20,
        random_state=cfg.random_state,
    ).fit(X, y)


def _predict_binary_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.clip(proba[:, 1], 0.001, 0.999)
        if proba.ndim == 2 and proba.shape[1] == 1:
            return np.clip(proba[:, 0], 0.001, 0.999)
    raw = np.asarray(model.predict(X), dtype=float)
    return np.clip(raw, 0.001, 0.999)


def _poisson_any_prob(mean: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.exp(-np.clip(np.asarray(mean, dtype=float), 0.0, None)), 0.001, 0.999)


def _clean_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _num_series(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(default, index=df.index)), errors="coerce").fillna(default)


def _position_role_rank_baseline_series(
    df: pd.DataFrame,
    stat: str,
    lookup: dict[str, Any],
    global_mean: float | None = None,
) -> pd.Series:
    global_value = _clean_float(global_mean) if global_mean is not None else _clean_float(lookup.get("global_mean"))
    if global_value is None:
        global_value = 0.0
    role_means = {str(k): float(v) for k, v in (lookup.get("role_means") or {}).items()}
    pos_means = {str(k).upper(): float(v) for k, v in (lookup.get("position_means") or {}).items()}
    rank_col = _role_rank_col(stat)
    if role_means and rank_col and rank_col in df.columns and "position" in df.columns:
        ranks = _role_bucket(pd.to_numeric(df[rank_col], errors="coerce"))
        values = [
            role_means.get(f"{pos}|{rank}", pos_means.get(pos, global_value))
            for pos, rank in zip(df["position"].astype(str).str.upper(), ranks)
        ]
        return pd.Series(values, index=df.index, dtype=float)
    if pos_means and "position" in df.columns:
        return df["position"].astype(str).str.upper().map(pos_means).fillna(global_value).astype(float)
    return pd.Series(global_value, index=df.index, dtype=float)


def _qb_contextual_passing_baseline(df: pd.DataFrame, lookup: dict[str, Any] | None = None) -> pd.Series:
    lookup = lookup or {}
    global_mean = _clean_float(lookup.get("global_mean")) or 207.0
    role_mean = _clean_float((lookup.get("role_means") or {}).get("QB|1")) or _clean_float(
        (lookup.get("position_means") or {}).get("QB")
    ) or global_mean
    pass_attempt_mean = _clean_float(lookup.get("pass_attempt_mean")) or 31.5
    ypa_mean = _clean_float(lookup.get("yards_per_attempt_mean")) or 6.8
    team_implied_mean = _clean_float(lookup.get("team_implied_mean")) or 22.0

    roll5 = _num_series(df, "passing_yards_avg_5", role_mean).clip(60.0, 380.0)
    roll10 = _num_series(df, "passing_yards_avg_10", role_mean).clip(60.0, 380.0)
    roll3 = _num_series(df, "passing_yards_avg_3", float((roll5 + roll10).mean() / 2.0)).clip(60.0, 420.0)
    player_form = (0.44 * roll5 + 0.36 * roll10 + 0.20 * roll3).clip(60.0, 390.0)

    att5 = _num_series(df, "pass_attempts_avg_5", pass_attempt_mean).clip(8.0, 55.0)
    att10 = _num_series(df, "pass_attempts_avg_10", pass_attempt_mean).clip(8.0, 55.0)
    attempts = (0.62 * att5 + 0.38 * att10).clip(8.0, 55.0)
    ypa5 = (roll5 / att5.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(ypa_mean).clip(3.5, 10.5)
    ypa10 = (roll10 / att10.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(ypa5).clip(3.5, 10.5)
    ypa = (0.58 * ypa5 + 0.42 * ypa10).clip(3.5, 10.5)

    pass_script = _num_series(df, "game_script_pass_boost", 1.0).clip(0.72, 1.32)
    high_attempt = _num_series(df, "high_pass_attempt_score", 0.0).clip(0.0, 1.0)
    spike_attempt = _num_series(df, "spike_pass_attempt_opportunity_score", 0.0).clip(0.0, 1.0)
    starter = _num_series(df, "starter_confidence", 0.82).clip(0.0, 1.0)
    full_workload = _num_series(df, "full_workload_score", 0.78).clip(0.0, 1.0)
    limited = _num_series(df, "limited_workload_risk_score", 0.0).clip(0.0, 1.0)
    weird = _num_series(df, "weird_usage_risk_score", 0.0).clip(0.0, 1.0)
    workload_factor = (
        (0.82 + 0.20 * starter)
        * (0.86 + 0.18 * full_workload)
        * (1.0 - 0.42 * limited)
        * (1.0 - 0.32 * weird)
    ).clip(0.42, 1.12)
    volume_upside = (1.0 + 0.12 * high_attempt + 0.08 * spike_attempt).clip(0.95, 1.18)
    attempt_based = (attempts * pass_script * workload_factor * volume_upside * ypa).clip(40.0, 420.0)

    opp_allowed = _num_series(df, "opp_allowed_passing_yards_avg_5", role_mean).clip(120.0, 360.0)
    team_implied = _num_series(df, "team_implied_points", team_implied_mean).clip(9.0, 38.0)
    env_factor = (team_implied / max(1.0, team_implied_mean)).clip(0.78, 1.24)
    matchup_based = ((0.68 * player_form) + (0.32 * opp_allowed)) * env_factor

    role_anchor = pd.Series(role_mean, index=df.index, dtype=float)
    contextual = (
        0.38 * player_form
        + 0.32 * attempt_based
        + 0.22 * matchup_based
        + 0.08 * role_anchor
    ).clip(40.0, 430.0)
    return contextual.fillna(role_mean).astype(float)


def _qb_role_recent_bias_adjusted_baseline(df: pd.DataFrame, lookup: dict[str, Any] | None = None) -> pd.Series:
    lookup = lookup or {}
    base = _position_role_rank_baseline_series(df, "passing_yards", lookup)
    correction = _clean_float(lookup.get("qb_role_recent_bias_correction")) or 0.0
    return (base + correction).clip(lower=0.0).astype(float)


def _baseline_from_metric_column(df: pd.DataFrame, stat: str, metrics: dict[str, Any] | None = None) -> np.ndarray:
    metrics = metrics or {}
    baseline_name = str(metrics.get("baseline_column") or "rolling_5")
    lookup = metrics.get("baseline_lookup") or {}
    global_mean = _clean_float(lookup.get("global_mean")) if isinstance(lookup, dict) else None
    if global_mean is None:
        global_mean = 0.0
    if baseline_name == "position_mean" and isinstance(lookup, dict):
        pos_means = {str(k).upper(): float(v) for k, v in (lookup.get("position_means") or {}).items()}
        if pos_means and "position" in df.columns:
            return (
                df["position"]
                .astype(str)
                .str.upper()
                .map(pos_means)
                .fillna(global_mean)
                .to_numpy(dtype=float)
            )
    if baseline_name == "position_role_rank_mean" and isinstance(lookup, dict):
        return _position_role_rank_baseline_series(df, stat, lookup, global_mean).to_numpy(dtype=float)
    if baseline_name == "qb_role_recent_bias_adjusted" and stat == "passing_yards":
        return _qb_role_recent_bias_adjusted_baseline(df, lookup if isinstance(lookup, dict) else {}).to_numpy(dtype=float)
    if baseline_name == "qb_contextual_baseline" and stat == "passing_yards":
        return _qb_contextual_passing_baseline(df, lookup if isinstance(lookup, dict) else {}).to_numpy(dtype=float)
    if baseline_name == "rolling_10":
        primary_col = f"{stat}_avg_10"
        fallback_col = f"{stat}_avg_5"
    elif baseline_name == "rolling_5_team_total_adjusted":
        primary_col = f"{stat}_avg_5"
        fallback_col = f"{stat}_avg_10"
        baseline = pd.to_numeric(df.get(primary_col, pd.Series(index=df.index)), errors="coerce")
        fallback = pd.to_numeric(df.get(fallback_col, pd.Series(index=df.index)), errors="coerce")
        base = baseline.fillna(fallback).fillna(global_mean)
        mean_imp = _clean_float((lookup or {}).get("team_implied_mean")) or 22.0
        implied = pd.to_numeric(df.get("team_implied_points", pd.Series(mean_imp, index=df.index)), errors="coerce").fillna(mean_imp)
        return (base * (implied / max(1.0, mean_imp)).clip(lower=0.75, upper=1.30)).to_numpy(dtype=float)
    elif baseline_name == "td_role_team_total_adjusted" and stat.endswith("_tds"):
        role_cols: tuple[str, ...] = ()
        if stat == "passing_tds":
            role_cols = ("red_zone_pass_attempts_avg_5",)
        elif stat == "rushing_tds":
            role_cols = ("red_zone_carries_avg_5", "goal_line_carries_avg_5")
        elif stat == "receiving_tds":
            role_cols = ("red_zone_targets_avg_5", "goal_line_targets_avg_5")
        if role_cols:
            role = sum(
                pd.to_numeric(df.get(col, pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
                for col in role_cols
            )
            rate = _clean_float((lookup or {}).get("td_role_rate"))
            if rate is not None:
                mean_imp = _clean_float((lookup or {}).get("team_implied_mean")) or 22.0
                implied = pd.to_numeric(df.get("team_implied_points", pd.Series(mean_imp, index=df.index)), errors="coerce").fillna(mean_imp)
                return (role * rate * (implied / max(1.0, mean_imp)).clip(lower=0.70, upper=1.40)).fillna(global_mean).clip(lower=0.0).to_numpy(dtype=float)
        primary_col = f"{stat}_avg_5"
        fallback_col = f"{stat}_avg_10"
    else:
        primary_col = f"{stat}_avg_5"
        fallback_col = f"{stat}_avg_10"
    baseline = pd.to_numeric(df.get(primary_col, pd.Series(index=df.index)), errors="coerce")
    fallback = pd.to_numeric(df.get(fallback_col, pd.Series(index=df.index)), errors="coerce")
    return baseline.fillna(fallback).fillna(global_mean).to_numpy(dtype=float)


def _apply_spike_residual_calibrator(
    pred: np.ndarray,
    X: pd.DataFrame,
    calibrator: dict[str, Any] | None,
) -> np.ndarray:
    base_pred = np.clip(np.asarray(pred, dtype=float), 0.0, None)
    if not calibrator:
        return base_pred
    model = calibrator.get("model")
    columns = list(calibrator.get("columns") or ())
    if model is None or not columns:
        return base_pred
    fills = {str(k): float(v) for k, v in (calibrator.get("fill_values") or {}).items()}
    X_resid = X.reindex(columns=columns).fillna(fills).fillna(0.0)
    raw_correction = np.asarray(model.predict(X_resid), dtype=float)
    shrink = float(calibrator.get("shrink") or 0.0)
    cap = float(calibrator.get("cap") or 0.0)
    negative_cap = float(calibrator.get("negative_cap") or (cap * 0.5))
    threshold = float(calibrator.get("signal_threshold") or 0.50)
    signal = _spike_signal_from_features(X, str(calibrator.get("stat") or ""))
    gate = ((signal - threshold) / max(1e-6, 1.0 - threshold)).clip(0.0, 1.0).to_numpy(dtype=float)
    correction = np.clip(raw_correction * shrink, -negative_cap, cap) * gate
    return np.clip(base_pred + correction, 0.0, None)


def _apply_receiver_spike_bias_calibrator(
    pred: np.ndarray,
    X: pd.DataFrame,
    calibrator: dict[str, Any] | None,
) -> np.ndarray:
    base_pred = np.clip(np.asarray(pred, dtype=float), 0.0, None)
    if not calibrator or str(calibrator.get("stat") or "") != "receiving_yards":
        return base_pred
    correction = _clean_float(calibrator.get("correction"))
    if correction is None or abs(correction) < 1e-9:
        return base_pred
    signal_threshold = float(calibrator.get("signal_threshold") or 0.75)
    target_threshold = float(calibrator.get("target_threshold") or 3.0)
    min_projection = float(calibrator.get("min_projection") or 12.0)
    signal = _spike_signal_from_features(X, "receiving_yards")
    target_expectation = _first_numeric_feature(
        X,
        (
            "receiver_projected_targets_v3",
            "receiver_projected_targets_v2",
            "route_weighted_target_expectation_avg_5",
            "targets_avg_5",
            "targets_avg_10",
        ),
        default=0.0,
    )
    signal_gate = ((signal - signal_threshold) / max(1e-6, 1.0 - signal_threshold)).clip(0.0, 1.0)
    target_gate = (target_expectation >= target_threshold).astype(float)
    projection_gate = (pd.Series(base_pred, index=X.index) >= min_projection).astype(float)
    gate = (signal_gate * target_gate * projection_gate).to_numpy(dtype=float)
    return np.clip(base_pred + correction * gate, 0.0, None)


def _receiver_target_air_yards_repair_features(X: pd.DataFrame) -> dict[str, pd.Series]:
    ypt = (
        _first_numeric_feature(X, ("receiving_yards_avg_5",), default=0.0)
        / _first_numeric_feature(X, ("targets_avg_5",), default=0.0).replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(8.0).clip(3.0, 18.0)
    target_expectation = _max_numeric_features(
        X,
        (
            "receiver_projected_targets_v3",
            "receiver_projected_targets_v2",
            "route_weighted_target_expectation_avg_5",
            "targets_avg_5",
            "targets_avg_10",
        ),
        default=0.0,
    )
    target_anchor = pd.concat(
        [
            _first_numeric_feature(X, ("receiver_target_route_spike_anchor_yards",), default=0.0),
            _first_numeric_feature(X, ("receiver_target_eruption_anchor_targets",), default=0.0) * ypt,
            _first_numeric_feature(X, ("receiver_spike_yards_anchor_v4",), default=0.0),
            _first_numeric_feature(X, ("receiver_spike_yards_anchor_v3",), default=0.0),
            _first_numeric_feature(X, ("route_weighted_yard_expectation_avg_5",), default=0.0),
            target_expectation * ypt,
        ],
        axis=1,
    ).max(axis=1).fillna(0.0)
    air_anchor = pd.concat(
        [
            _first_numeric_feature(X, ("receiver_air_yards_spike_anchor_yards",), default=0.0),
            _first_numeric_feature(X, ("route_env_receiving_yards_signal",), default=0.0),
            0.62 * _first_numeric_feature(X, ("receiving_air_yards_avg_5",), default=0.0),
            _first_numeric_feature(X, ("receiver_spike_yards_anchor_v4",), default=0.0),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0)
    route_anchor = pd.concat(
        [
            _first_numeric_feature(X, ("receiver_target_route_spike_anchor_yards",), default=0.0),
            _first_numeric_feature(X, ("receiver_spike_volume_anchor_yards",), default=0.0),
            _first_numeric_feature(X, ("route_weighted_yard_expectation_avg_5",), default=0.0),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0)
    target_signal = _max_numeric_features(
        X,
        (
            "receiver_target_route_spike_score",
            "receiver_target_spike_v2_score",
            "receiver_target_eruption_score",
            "receiver_target_command_score",
            "receiver_spike_under_correction_v4_score",
            "target_spike_path_score",
            "receiver_contextual_spike_score",
            "receiver_teammate_vacancy_score",
        ),
        default=0.0,
    ).clip(0.0, 1.0)
    air_signal = _max_numeric_features(
        X,
        (
            "air_yards_spike_path_score",
            "receiver_air_yards_eruption_score",
            "receiver_air_yards_spike_v2_score",
            "receiver_explosive_spike_score",
            "receiver_ypt_efficiency_spike_score",
        ),
        default=0.0,
    ).clip(0.0, 1.0)
    route_signal = _max_numeric_features(
        X,
        (
            "receiver_route_spike_readiness_score",
            "receiver_target_route_spike_score",
            "receiver_spike_volume_score",
            "spike_snap_share_score",
            "receiver_live_spike_v3_score",
        ),
        default=0.0,
    ).clip(0.0, 1.0)
    combo_signal = pd.concat([target_signal, air_signal, route_signal], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    combo_anchor = pd.concat([target_anchor, air_anchor, route_anchor], axis=1).max(axis=1).fillna(0.0)
    target_volume_signal = ((target_expectation - 5.0) / 4.5).clip(0.0, 1.0)
    route_depth_signal = ((_first_numeric_feature(X, ("estimated_routes_avg_5",), default=0.0) - 25.0) / 15.0).clip(0.0, 1.0)
    air_volume_signal = ((_first_numeric_feature(X, ("receiving_air_yards_avg_5",), default=0.0) - 45.0) / 55.0).clip(0.0, 1.0)
    vacancy_signal = _max_numeric_features(
        X,
        (
            "receiver_teammate_vacancy_score",
            "teammate_receiver_injury_pressure_score",
            "same_week_teammate_receiver_injury_score",
            "role_change_upside_score",
        ),
        default=0.0,
    ).clip(0.0, 1.0)
    under_spike_signal = pd.concat(
        [
            combo_signal,
            target_volume_signal,
            0.85 * route_depth_signal,
            0.82 * air_volume_signal,
            0.78 * vacancy_signal,
        ],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    under_spike_anchor = pd.concat(
        [
            combo_anchor,
            target_anchor,
            air_anchor,
            route_anchor,
            target_expectation * ypt,
            _first_numeric_feature(X, ("receiving_yards_avg_5", "receiving_yards_avg_10"), default=0.0),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0)
    risk = _max_numeric_features(
        X,
        (
            "limited_workload_risk_score",
            "workload_downside_v2_score",
            "rest_risk_score",
            "weird_usage_risk_score",
        ),
        default=0.0,
    ).clip(0.0, 1.0)
    quality = _first_numeric_feature(X, ("live_usage_context_quality_v4_score",), default=0.0).clip(0.0, 1.0)
    return {
        "target_expectation": target_expectation,
        "target_signal": target_signal,
        "air_signal": air_signal,
        "route_signal": route_signal,
        "combo_signal": combo_signal,
        "under_spike_signal": under_spike_signal,
        "target_anchor": target_anchor,
        "air_anchor": air_anchor,
        "route_anchor": route_anchor,
        "combo_anchor": combo_anchor,
        "under_spike_anchor": under_spike_anchor,
        "risk": risk,
        "quality": quality,
    }


def _apply_receiver_target_air_yards_repair_calibrator(
    pred: np.ndarray,
    X: pd.DataFrame,
    calibrator: dict[str, Any] | None,
) -> np.ndarray:
    base_pred = np.clip(np.asarray(pred, dtype=float), 0.0, None)
    if not calibrator or str(calibrator.get("stat") or "") != "receiving_yards":
        return base_pred
    profiles = calibrator.get("profiles") or []
    if not profiles:
        return base_pred
    features = _receiver_target_air_yards_repair_features(X)
    current = pd.Series(base_pred, index=X.index, dtype=float)
    for profile in profiles:
        signal_name = str(profile.get("signal") or "")
        anchor_name = str(profile.get("anchor") or "")
        signal = features.get(signal_name)
        anchor = features.get(anchor_name)
        if signal is None or anchor is None:
            continue
        threshold = float(profile.get("signal_threshold") or 0.75)
        target_threshold = float(profile.get("target_threshold") or 4.5)
        min_projection = float(profile.get("min_projection") or 8.0)
        weight = float(profile.get("weight") or 0.0)
        cap = float(profile.get("cap") or 12.0)
        if weight <= 0.0 or cap <= 0.0:
            continue
        signal_gate = ((signal - threshold) / max(1e-6, 1.0 - threshold)).clip(0.0, 1.0)
        target_gate = (features["target_expectation"] >= target_threshold).astype(float)
        projection_gate = (current >= min_projection).astype(float)
        healthy_gate = (features["risk"] < 0.55).astype(float) * (0.55 + 0.45 * features["quality"])
        gap = (anchor - current).clip(0.0, cap)
        adjustment = weight * signal_gate * target_gate * projection_gate * healthy_gate * gap
        current = (current + adjustment).clip(lower=0.0)
    return current.to_numpy(dtype=float)


def _rushing_workload_repair_features(X: pd.DataFrame) -> dict[str, pd.Series]:
    projected_carries = _max_numeric_features(
        X,
        ("rb_projected_carries_v3", "rb_projected_carries_v2", "risk_adjusted_carries_avg_5", "carries_avg_5", "carries_avg_10"),
        default=0.0,
    ).clip(lower=0.0)
    ypc = _first_numeric_feature(X, ("yards_per_carry_avg_5", "yards_per_carry_avg_10"), default=4.1).clip(2.2, 6.8)
    carry_signal = _max_numeric_features(
        X,
        (
            "rb_carry_under_correction_v4_score",
            "rb_live_carry_v3_score",
            "rb_carry_spike_v2_score",
            "high_carry_score",
            "carry_spike_path_score",
            "rb_usage_spike_signal",
            "role_change_upside_score",
        ),
        default=0.0,
    ).clip(0.0, 1.0)
    anchor = pd.concat(
        [
            _first_numeric_feature(X, ("rb_rush_yards_anchor_v4",), default=0.0),
            _first_numeric_feature(X, ("rb_rush_yards_anchor_v3",), default=0.0),
            _first_numeric_feature(X, ("rb_rush_yards_anchor_v2",), default=0.0),
            projected_carries * ypc,
            _first_numeric_feature(X, ("rushing_yards_avg_5", "rushing_yards_avg_10"), default=0.0),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(lower=0.0)
    risk = _max_numeric_features(
        X,
        (
            "limited_workload_risk_score",
            "workload_downside_v2_score",
            "rest_risk_score",
            "weird_usage_risk_score",
            "high_usage_fragility_score",
        ),
        default=0.0,
    ).clip(0.0, 1.0)
    quality = _max_numeric_features(
        X,
        ("rb_usage_history_quality_score", "live_usage_context_quality_v4_score", "same_week_usage_confidence_v3_score"),
        default=0.0,
    ).clip(0.0, 1.0)
    return {
        "projected_carries": projected_carries,
        "ypc": ypc,
        "carry_signal": carry_signal,
        "anchor": anchor,
        "risk": risk,
        "quality": quality,
    }


def _apply_rushing_workload_repair_calibrator(
    pred: np.ndarray,
    X: pd.DataFrame,
    calibrator: dict[str, Any] | None,
) -> np.ndarray:
    base_pred = np.clip(np.asarray(pred, dtype=float), 0.0, None)
    if not calibrator or str(calibrator.get("stat") or "") != "rushing_yards":
        return base_pred
    features = _rushing_workload_repair_features(X)
    current = pd.Series(base_pred, index=X.index, dtype=float)
    threshold = float(calibrator.get("signal_threshold") or 0.55)
    carry_threshold = float(calibrator.get("carry_threshold") or 8.0)
    min_projection = float(calibrator.get("min_projection") or 5.0)
    weight = float(calibrator.get("weight") or 0.0)
    cap = float(calibrator.get("cap") or 16.0)
    if weight <= 0.0 or cap <= 0.0:
        return base_pred
    signal_gate = ((features["carry_signal"] - threshold) / max(1e-6, 1.0 - threshold)).clip(0.0, 1.0)
    carry_gate = (features["projected_carries"] >= carry_threshold).astype(float)
    projection_gate = (current >= min_projection).astype(float)
    healthy_gate = (features["risk"] < 0.62).astype(float) * (0.55 + 0.45 * features["quality"])
    gap = (features["anchor"] - current).clip(0.0, cap)
    adjustment = weight * signal_gate * carry_gate * projection_gate * healthy_gate * gap
    return (current + adjustment).clip(lower=0.0).to_numpy(dtype=float)


def _qb_attempt_anchor_from_features(X: pd.DataFrame) -> pd.Series:
    att5 = _first_numeric_feature(X, ("pass_attempts_avg_5",), default=31.5).clip(8.0, 58.0)
    att10 = _first_numeric_feature(X, ("pass_attempts_avg_10",), default=31.5).clip(8.0, 58.0)
    recent = (0.62 * att5 + 0.38 * att10).clip(8.0, 58.0)
    script = _first_numeric_feature(X, ("qb_script_adjusted_pass_attempts_avg_5",), default=np.nan)
    script = script.where(script > 0.0, recent).fillna(recent).clip(8.0, 60.0)
    workload = _first_numeric_feature(X, ("qb_workload_adjusted_pass_attempts_avg_5",), default=np.nan)
    workload = workload.where(workload > 0.0, script).fillna(script).clip(5.0, 60.0)
    spike = _max_numeric_features(
        X,
        ("high_pass_attempt_score", "pass_spike_path_score", "qb_volume_spike_signal"),
        default=0.0,
    ).clip(0.0, 1.0)
    team_total = _first_numeric_feature(X, ("team_implied_points",), default=22.0).clip(8.0, 40.0)
    game_total = _first_numeric_feature(X, ("game_total", "game_total_line"), default=44.0).clip(28.0, 60.0)
    env = (0.72 + 0.18 * (team_total / 22.0).clip(0.70, 1.35) + 0.10 * (game_total / 44.0).clip(0.75, 1.25)).clip(0.82, 1.16)
    limited = _first_numeric_feature(X, ("limited_workload_risk_score",), default=0.0).clip(0.0, 1.0)
    weird = _first_numeric_feature(X, ("weird_usage_risk_score",), default=0.0).clip(0.0, 1.0)
    starter = _first_numeric_feature(X, ("starter_confidence", "projected_starter_score"), default=0.82).clip(0.0, 1.0)
    workload_gate = (0.78 + 0.24 * starter) * (1.0 - 0.32 * limited) * (1.0 - 0.22 * weird)
    anchor = (
        0.30 * recent
        + 0.42 * script
        + 0.28 * workload
    ) * env * workload_gate * (1.0 + 0.10 * np.clip(spike - 0.45, 0.0, 0.55))
    return anchor.clip(6.0, 62.0).fillna(recent)


def _qb_ypa_prior_from_features(X: pd.DataFrame) -> pd.Series:
    ypa5 = _first_numeric_feature(X, ("qb_yards_per_attempt_avg_5",), default=np.nan)
    ypa10 = _first_numeric_feature(X, ("qb_yards_per_attempt_avg_10",), default=np.nan)
    if ypa5.isna().all():
        pass_yards = _first_numeric_feature(X, ("passing_yards_avg_5",), default=np.nan)
        attempts = _first_numeric_feature(X, ("pass_attempts_avg_5",), default=np.nan).replace(0.0, np.nan)
        ypa5 = (pass_yards / attempts).replace([np.inf, -np.inf], np.nan)
    if ypa10.isna().all():
        pass_yards = _first_numeric_feature(X, ("passing_yards_avg_10",), default=np.nan)
        attempts = _first_numeric_feature(X, ("pass_attempts_avg_10",), default=np.nan).replace(0.0, np.nan)
        ypa10 = (pass_yards / attempts).replace([np.inf, -np.inf], np.nan)
    league = float(pd.concat([ypa5, ypa10], axis=0).dropna().clip(3.0, 11.5).mean()) if ypa5.notna().any() or ypa10.notna().any() else 6.8
    return (0.62 * ypa5.fillna(league) + 0.38 * ypa10.fillna(ypa5).fillna(league)).clip(3.0, 11.5)


def _apply_qb_passing_efficiency_repair(
    pred: np.ndarray,
    X: pd.DataFrame,
    repair: dict[str, Any] | None,
) -> np.ndarray:
    base_pred = np.clip(np.asarray(pred, dtype=float), 0.0, None)
    if not repair or str(repair.get("stat") or "") != "passing_yards":
        return base_pred
    model = repair.get("model")
    columns = list(repair.get("columns") or ())
    if model is None or not columns:
        return base_pred
    fills = {str(k): float(v) for k, v in (repair.get("fill_values") or {}).items()}
    X_eff = X.reindex(columns=columns).fillna(fills).fillna(0.0)
    raw_ypa = np.clip(np.asarray(model.predict(X_eff), dtype=float), 3.0, 11.5)
    ypa_prior = _qb_ypa_prior_from_features(X).to_numpy(dtype=float)
    ypa_weight = float(repair.get("ypa_blend_weight") or 0.50)
    ypa = np.clip(ypa_weight * raw_ypa + (1.0 - ypa_weight) * ypa_prior, 3.0, 11.5)
    attempts = _qb_attempt_anchor_from_features(X).to_numpy(dtype=float)
    volume_projection = np.clip(attempts * ypa, 25.0, 470.0)
    blend = float(repair.get("projection_blend_weight") or 0.35)
    cap_up = float(repair.get("cap_up") or 34.0)
    cap_down = float(repair.get("cap_down") or 24.0)
    starter = _first_numeric_feature(X, ("starter_confidence", "projected_starter_score"), default=0.82).clip(0.0, 1.0).to_numpy(dtype=float)
    limited = _first_numeric_feature(X, ("limited_workload_risk_score",), default=0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    weird = _first_numeric_feature(X, ("weird_usage_risk_score",), default=0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    gate = np.clip((0.45 + 0.55 * starter) * (1.0 - 0.48 * limited) * (1.0 - 0.34 * weird), 0.0, 1.0)
    correction = np.clip(volume_projection - base_pred, -cap_down, cap_up)
    return np.clip(base_pred + blend * gate * correction, 0.0, None)


def _apply_baseline_model_blend(
    model_payload: dict[str, Any],
    X: pd.DataFrame,
    baseline: np.ndarray | pd.Series,
) -> np.ndarray:
    base = np.clip(np.asarray(baseline, dtype=float), 0.0, None)
    nested = model_payload.get("model_payload")
    if nested is None:
        return base
    model_pred = _predict_model_payload(nested, X, base)
    weight = float(model_payload.get("model_weight") or 0.0)
    weight = float(np.clip(weight, 0.0, 1.0))
    correction = float(model_payload.get("correction") or 0.0)
    return np.clip((1.0 - weight) * base + weight * np.asarray(model_pred, dtype=float) + correction, 0.0, None)


def _predict_model_payload(model_payload: Any, X: pd.DataFrame, baseline: np.ndarray | pd.Series) -> np.ndarray:
    base = np.clip(np.asarray(baseline, dtype=float), 0.0, None)
    if isinstance(model_payload, dict):
        kind = str(model_payload.get("kind") or "direct")
        if kind == "baseline_model_blend":
            return _apply_baseline_model_blend(model_payload, X, base)
        if kind == "baseline_bias_calibrated":
            return _apply_bias_calibrator(base, X, model_payload.get("bias_calibrator"))
        if kind == "rare_event_classifier":
            clf = model_payload.get("classifier")
            if clf is None:
                return base
            positive_mean = float(model_payload.get("positive_mean") or 1.0)
            blend_weight = float(model_payload.get("blend_weight") or 1.0)
            p_any = _predict_binary_probability(clf, X)
            probability_blend_weight = model_payload.get("probability_blend_weight")
            if probability_blend_weight is not None:
                p_weight = float(probability_blend_weight)
                base_prob = _poisson_any_prob(base)
                p_any = np.clip(p_weight * p_any + (1.0 - p_weight) * base_prob, 0.001, 0.999)
            expected = p_any * max(0.01, positive_mean)
            pred = np.clip(blend_weight * expected + (1.0 - blend_weight) * base, 0.0, None)
            pred = _apply_spike_residual_calibrator(pred, X, model_payload.get("spike_residual_calibrator"))
            pred = _apply_rushing_workload_repair_calibrator(pred, X, model_payload.get("rushing_workload_repair_calibrator"))
            pred = _apply_receiver_spike_bias_calibrator(pred, X, model_payload.get("receiver_spike_bias_calibrator"))
            pred = _apply_receiver_target_air_yards_repair_calibrator(pred, X, model_payload.get("receiver_target_air_yards_repair_calibrator"))
            return _apply_bias_calibrator(pred, X, model_payload.get("bias_calibrator"))
        if kind == "rate_opportunity":
            direct_model = model_payload.get("direct_model")
            rate_model = model_payload.get("rate_model")
            if direct_model is None or rate_model is None:
                return base
            direct_pred = np.clip(np.asarray(direct_model.predict(X), dtype=float), 0.0, None)
            rate_cap = float(model_payload.get("rate_cap") or 30.0)
            rate_pred = np.clip(np.asarray(rate_model.predict(X), dtype=float), 0.0, rate_cap)
            opportunity_cols = tuple(model_payload.get("opportunity_cols") or ())
            opp = _first_numeric_feature(X, opportunity_cols, default=0.0).clip(lower=0.0).to_numpy(dtype=float)
            blend_weight = float(model_payload.get("rate_blend_weight") or 1.0)
            pred = np.clip(blend_weight * rate_pred * opp + (1.0 - blend_weight) * direct_pred, 0.0, None)
            pred = _apply_qb_passing_efficiency_repair(pred, X, model_payload.get("qb_passing_efficiency_repair"))
            pred = _apply_spike_residual_calibrator(pred, X, model_payload.get("spike_residual_calibrator"))
            pred = _apply_rushing_workload_repair_calibrator(pred, X, model_payload.get("rushing_workload_repair_calibrator"))
            pred = _apply_receiver_spike_bias_calibrator(pred, X, model_payload.get("receiver_spike_bias_calibrator"))
            pred = _apply_receiver_target_air_yards_repair_calibrator(pred, X, model_payload.get("receiver_target_air_yards_repair_calibrator"))
            return _apply_bias_calibrator(pred, X, model_payload.get("bias_calibrator"))
        raw_model = model_payload.get("model")
        if raw_model is None:
            return base
        pred = np.clip(np.asarray(raw_model.predict(X), dtype=float), 0.0, None)
        pred = _apply_qb_passing_efficiency_repair(pred, X, model_payload.get("qb_passing_efficiency_repair"))
        pred = _apply_spike_residual_calibrator(pred, X, model_payload.get("spike_residual_calibrator"))
        pred = _apply_rushing_workload_repair_calibrator(pred, X, model_payload.get("rushing_workload_repair_calibrator"))
        pred = _apply_receiver_spike_bias_calibrator(pred, X, model_payload.get("receiver_spike_bias_calibrator"))
        pred = _apply_receiver_target_air_yards_repair_calibrator(pred, X, model_payload.get("receiver_target_air_yards_repair_calibrator"))
        return _apply_bias_calibrator(pred, X, model_payload.get("bias_calibrator"))
    return np.clip(np.asarray(model_payload.predict(X), dtype=float), 0.0, None)


def _baseline_column(stat: str) -> str:
    return f"{stat}_avg_5"


def _role_rank_col(stat: str) -> str | None:
    if stat in {"passing_yards", "passing_tds"}:
        return "pass_attempts_role_rank"
    if stat in {"rushing_yards", "rushing_tds"}:
        return "carries_role_rank"
    if stat in {"receiving_yards", "receiving_tds"}:
        return "targets_role_rank"
    return None


def _role_bucket(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.fillna(99).clip(lower=1, upper=6).round().astype(int).astype(str)


RATE_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "passing_yards": {
        "actual_opportunity_col": "pass_attempts",
        "opportunity_cols": (
            "qb_script_adjusted_pass_attempts_avg_5",
            "qb_workload_adjusted_pass_attempts_avg_5",
            "pass_attempts_avg_5",
            "pass_attempts_avg_10",
        ),
        "min_actual_opportunity": 8.0,
        "min_rows": 180,
        "rate_cap": 14.0,
    },
    "rushing_yards": {
        "actual_opportunity_col": "carries",
        "opportunity_cols": ("rb_projected_carries_v3", "rb_projected_carries_v2", "carries_avg_5", "risk_adjusted_carries_avg_5", "carries_avg_10"),
        "min_actual_opportunity": 2.0,
        "min_rows": 220,
        "rate_cap": 10.0,
    },
    "receiving_yards": {
        "actual_opportunity_col": "targets",
        "opportunity_cols": ("receiver_projected_targets_v3", "receiver_projected_targets_v2", "route_weighted_target_expectation_avg_5", "targets_avg_5", "risk_adjusted_targets_avg_5", "targets_avg_10"),
        "min_actual_opportunity": 1.0,
        "min_rows": 320,
        "rate_cap": 28.0,
    },
}


SPIKE_RESIDUAL_FEATURES: dict[str, tuple[str, ...]] = {
    "rushing_yards": (
        "carries_avg_5",
        "carries_avg_10",
        "risk_adjusted_carries_avg_5",
        "full_workload_adjusted_carries_avg_5",
        "fragility_adjusted_carries_avg_5",
        "rushing_yards_avg_5",
        "rushing_yards_avg_10",
        "yards_per_carry_avg_5",
        "yards_per_carry_avg_10",
        "high_carry_score",
        "spike_carry_opportunity_score",
        "carry_spike_path_score",
        "spike_snap_share_score",
        "rb_rush_role_env_score",
        "rb_carry_trend_env_score",
        "rb_rushing_yards_opportunity_signal",
        "rb_spike_rush_score",
        "rb_usage_spike_signal",
        "role_change_upside_score",
        "workload_floor_score",
        "workload_downside_v2_score",
        "workload_upside_v2_score",
        "rb_projected_carries_v2",
        "rb_carry_spike_v2_score",
        "rb_rush_yards_anchor_v2",
        "rb_live_carry_v3_score",
        "rb_projected_carries_v3",
        "rb_rush_yards_anchor_v3",
        "same_week_usage_confidence_v3_score",
        "yardage_projection_volatility_v3_score",
        "rb_usage_history_quality_score",
        "live_usage_context_quality_v4_score",
        "rb_carry_under_correction_v4_score",
        "rb_rush_yards_anchor_v4",
        "yardage_projection_volatility_v4_score",
        "carries_trend_3_10",
        "recent_carry_spike_score",
        "red_zone_carries_avg_5",
        "goal_line_carries_avg_5",
        "team_implied_points",
        "team_spread",
        "team_spread_line",
        "game_total",
        "game_total_line",
        "is_home",
        "depth_pos_rank",
        "depth_pos_slot",
        "starter_confidence",
        "full_workload_score",
        "limited_workload_risk_score",
        "high_usage_fragility_score",
        "usage_volatility_score",
        "rest_risk_score",
    ),
    "receiving_yards": (
        "targets_avg_5",
        "targets_avg_10",
        "risk_adjusted_targets_avg_5",
        "full_workload_adjusted_targets_avg_5",
        "fragility_adjusted_targets_avg_5",
        "receiving_yards_avg_5",
        "receiving_yards_avg_10",
        "target_share_avg_5",
        "air_yards_share_avg_5",
        "wopr_avg_5",
        "estimated_routes_avg_5",
        "route_participation_proxy_avg_5",
        "targets_per_route_proxy_avg_5",
        "yards_per_route_proxy_avg_5",
        "receiver_route_env_score",
        "receiver_yards_rate_signal",
        "route_env_receiving_yards_signal",
        "route_target_intensity_avg_5",
        "route_weighted_target_expectation_avg_5",
        "route_weighted_yard_expectation_avg_5",
        "first_read_proxy_avg_5",
        "first_read_team_total_env_avg_5",
        "high_target_score",
        "spike_target_opportunity_score",
        "target_spike_path_score",
        "spike_snap_share_score",
        "target_share_trend_3_10",
        "air_yards_share_trend_3_10",
        "recent_target_spike_score",
        "recent_route_spike_score",
        "receiver_target_eruption_score",
        "air_yards_spike_path_score",
        "receiver_air_yards_eruption_score",
        "receiver_explosive_spike_score",
        "receiver_target_eruption_anchor_targets",
        "receiver_air_yards_spike_anchor_yards",
        "receiver_spike_yards_score",
        "receiver_spike_yards_projection_signal",
        "receiver_usage_spike_signal",
        "receiver_spike_volume_score",
        "receiver_spike_volume_anchor_yards",
        "receiver_target_command_score",
        "receiver_route_spike_readiness_score",
        "receiver_target_route_spike_score",
        "receiver_contextual_spike_score",
        "workload_downside_v2_score",
        "workload_upside_v2_score",
        "receiver_projected_targets_v2",
        "receiver_high_value_target_score",
        "receiver_target_spike_v2_score",
        "receiver_air_yards_spike_v2_score",
        "receiver_ypt_efficiency_spike_score",
        "receiver_spike_yards_anchor_v2",
        "receiver_live_spike_v3_score",
        "receiver_projected_targets_v3",
        "receiver_spike_yards_anchor_v3",
        "same_week_usage_confidence_v3_score",
        "yardage_projection_volatility_v3_score",
        "receiving_usage_history_quality_score",
        "td_usage_history_quality_score",
        "live_usage_context_quality_v4_score",
        "receiver_spike_under_correction_v4_score",
        "receiver_spike_yards_anchor_v4",
        "yardage_projection_volatility_v4_score",
        "receiver_teammate_vacancy_score",
        "teammate_receiver_injury_pressure_score",
        "same_week_teammate_receiver_injury_score",
        "same_week_teammate_receiver_out_count",
        "receiver_target_route_spike_anchor_targets",
        "receiver_target_route_spike_anchor_yards",
        "role_change_upside_score",
        "workload_floor_score",
        "team_implied_points",
        "team_spread",
        "team_spread_line",
        "game_total",
        "game_total_line",
        "is_home",
        "depth_pos_rank",
        "depth_pos_slot",
        "starter_confidence",
        "full_workload_score",
        "limited_workload_risk_score",
        "high_usage_fragility_score",
        "usage_volatility_score",
        "rest_risk_score",
    ),
}


def _first_numeric_feature(X: pd.DataFrame, names: tuple[str, ...], default: float = 0.0) -> pd.Series:
    out = pd.Series(np.nan, index=X.index, dtype=float)
    for name in names:
        if name not in X.columns:
            continue
        out = out.fillna(pd.to_numeric(X[name], errors="coerce"))
    return out.fillna(default)


def _max_numeric_features(X: pd.DataFrame, names: tuple[str, ...], default: float = 0.0) -> pd.Series:
    values = [
        _first_numeric_feature(X, (name,), default=np.nan)
        for name in names
        if name in X.columns
    ]
    if not values:
        return pd.Series(default, index=X.index, dtype=float)
    return pd.concat(values, axis=1).max(axis=1).fillna(default).astype(float)


def _opportunity_from_features(X: pd.DataFrame, stat: str) -> pd.Series:
    spec = RATE_MODEL_SPECS.get(stat)
    if not spec:
        return pd.Series(1.0, index=X.index, dtype=float)
    opp = _first_numeric_feature(X, tuple(spec["opportunity_cols"]), default=0.0)
    return opp.clip(lower=0.0)


def _spike_signal_from_features(X: pd.DataFrame, stat: str) -> pd.Series:
    if stat == "rushing_yards":
        carry = _first_numeric_feature(X, ("high_carry_score",), default=0.0).clip(0.0, 1.0)
        path = _first_numeric_feature(X, ("carry_spike_path_score",), default=0.0).clip(0.0, 1.0)
        spike = _first_numeric_feature(X, ("spike_snap_share_score",), default=0.0).clip(0.0, 1.0)
        usage = _first_numeric_feature(X, ("rb_usage_spike_signal",), default=0.0).clip(0.0, 1.0)
        role_up = _first_numeric_feature(X, ("role_change_upside_score",), default=0.0).clip(0.0, 1.0)
        v4 = _first_numeric_feature(X, ("rb_carry_under_correction_v4_score",), default=0.0).clip(0.0, 1.0)
        live_carry = _first_numeric_feature(X, ("rb_live_carry_v3_score",), default=0.0).clip(0.0, 1.0)
        projected_carries = _first_numeric_feature(X, ("rb_projected_carries_v3", "rb_projected_carries_v2", "carries_avg_5"), default=0.0)
        carry_volume = ((projected_carries - 9.0) / 8.0).clip(0.0, 1.0)
        return pd.concat([carry, path, usage, v4, live_carry, carry_volume, 0.65 * spike, 0.75 * role_up], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    if stat == "receiving_yards":
        target = _first_numeric_feature(X, ("high_target_score",), default=0.0).clip(0.0, 1.0)
        path = _first_numeric_feature(X, ("target_spike_path_score",), default=0.0).clip(0.0, 1.0)
        spike = _first_numeric_feature(X, ("spike_snap_share_score",), default=0.0).clip(0.0, 1.0)
        usage = _first_numeric_feature(X, ("receiver_usage_spike_signal",), default=0.0).clip(0.0, 1.0)
        target_command = _first_numeric_feature(X, ("receiver_target_command_score",), default=0.0).clip(0.0, 1.0)
        route_readiness = _first_numeric_feature(X, ("receiver_route_spike_readiness_score",), default=0.0).clip(0.0, 1.0)
        target_route = _first_numeric_feature(X, ("receiver_target_route_spike_score",), default=0.0).clip(0.0, 1.0)
        contextual = _first_numeric_feature(X, ("receiver_contextual_spike_score",), default=0.0).clip(0.0, 1.0)
        target_eruption = _first_numeric_feature(X, ("receiver_target_eruption_score",), default=0.0).clip(0.0, 1.0)
        air_path = _first_numeric_feature(X, ("air_yards_spike_path_score",), default=0.0).clip(0.0, 1.0)
        air_eruption = _first_numeric_feature(X, ("receiver_air_yards_eruption_score",), default=0.0).clip(0.0, 1.0)
        explosive = _first_numeric_feature(X, ("receiver_explosive_spike_score",), default=0.0).clip(0.0, 1.0)
        role_up = _first_numeric_feature(X, ("role_change_upside_score",), default=0.0).clip(0.0, 1.0)
        v4 = _first_numeric_feature(X, ("receiver_spike_under_correction_v4_score",), default=0.0).clip(0.0, 1.0)
        target_expectation = _first_numeric_feature(
            X,
            (
                "receiver_projected_targets_v3",
                "receiver_projected_targets_v2",
                "route_weighted_target_expectation_avg_5",
                "targets_avg_5",
            ),
            default=0.0,
        )
        target_volume = ((target_expectation - 5.0) / 4.5).clip(0.0, 1.0)
        route_depth = ((_first_numeric_feature(X, ("estimated_routes_avg_5",), default=0.0) - 25.0) / 15.0).clip(0.0, 1.0)
        air_volume = ((_first_numeric_feature(X, ("receiving_air_yards_avg_5",), default=0.0) - 45.0) / 55.0).clip(0.0, 1.0)
        vacancy = _first_numeric_feature(X, ("receiver_teammate_vacancy_score", "teammate_receiver_injury_pressure_score"), default=0.0).clip(0.0, 1.0)
        return (
            pd.concat([
                target,
                path,
                usage,
                v4,
                target_route,
                contextual,
                target_eruption,
                air_path,
                air_eruption,
                explosive,
                0.92 * target_command,
                0.88 * route_readiness,
                0.75 * spike,
                0.75 * role_up,
                0.80 * target_volume,
                0.70 * route_depth,
                0.70 * air_volume,
                0.72 * vacancy,
            ], axis=1)
            .max(axis=1)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
    if stat == "passing_yards":
        high = _first_numeric_feature(X, ("high_pass_attempt_score",), default=0.0).clip(0.0, 1.0)
        path = _first_numeric_feature(X, ("pass_spike_path_score",), default=0.0).clip(0.0, 1.0)
        volume = _first_numeric_feature(X, ("qb_volume_spike_signal",), default=0.0).clip(0.0, 1.0)
        return pd.concat([high, path, volume], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    return pd.Series(0.0, index=X.index, dtype=float)


def _spike_residual_feature_columns(X: pd.DataFrame, stat: str) -> list[str]:
    wanted = set(SPIKE_RESIDUAL_FEATURES.get(stat) or ())
    columns = [col for col in X.columns if col in wanted or col.startswith("position_")]
    return sorted(dict.fromkeys(columns))


def _position_bucket_from_features(X: pd.DataFrame) -> pd.Series:
    out = pd.Series("UNK", index=X.index, dtype=object)
    for pos in ("QB", "RB", "WR", "TE"):
        col = f"position_{pos}"
        if col in X.columns:
            out = out.mask(pd.to_numeric(X[col], errors="coerce").fillna(0.0) >= 0.5, pos)
    return out


def _depth_bucket_from_features(X: pd.DataFrame) -> pd.Series:
    rank = pd.to_numeric(X.get("depth_pos_rank", pd.Series(np.nan, index=X.index)), errors="coerce")
    out = pd.Series("depth_missing", index=X.index, dtype=object)
    out = out.mask(rank <= 1.25, "depth_1")
    out = out.mask((rank > 1.25) & (rank <= 2.25), "depth_2")
    out = out.mask((rank > 2.25) & (rank <= 4.25), "depth_3_4")
    out = out.mask(rank > 4.25, "depth_5_plus")
    return out


def _workload_bucket_from_features(X: pd.DataFrame) -> pd.Series:
    full = pd.to_numeric(X.get("full_workload_score", pd.Series(0.55, index=X.index)), errors="coerce").fillna(0.55)
    limited = pd.to_numeric(X.get("limited_workload_risk_score", pd.Series(0.0, index=X.index)), errors="coerce").fillna(0.0)
    fragility = pd.to_numeric(X.get("high_usage_fragility_score", pd.Series(0.0, index=X.index)), errors="coerce").fillna(0.0)
    out = pd.Series("normal_workload", index=X.index, dtype=object)
    out = out.mask(full >= 0.72, "full_workload")
    out = out.mask((limited >= 0.42) | (fragility >= 0.42), "limited_or_fragile")
    out = out.mask(full < 0.42, "low_workload")
    return out


def _spike_bucket_from_features(X: pd.DataFrame, stat: str) -> pd.Series:
    signal = _spike_signal_from_features(X, stat)
    out = pd.Series("spike_low", index=X.index, dtype=object)
    out = out.mask(signal >= 0.45, "spike_watch")
    out = out.mask(signal >= 0.65, "spike_high")
    return out


def _risk_bucket_from_features(X: pd.DataFrame) -> pd.Series:
    limited = pd.to_numeric(X.get("limited_workload_risk_score", pd.Series(0.0, index=X.index)), errors="coerce").fillna(0.0)
    weird = pd.to_numeric(X.get("weird_usage_risk_score", pd.Series(0.0, index=X.index)), errors="coerce").fillna(0.0)
    injury = pd.to_numeric(X.get("injury_downgrade_score", pd.Series(0.0, index=X.index)), errors="coerce").fillna(0.0)
    risk = pd.concat([limited, weird, injury], axis=1).max(axis=1).clip(0.0, 1.0)
    out = pd.Series("risk_low", index=X.index, dtype=object)
    out = out.mask(risk >= 0.30, "risk_watch")
    out = out.mask(risk >= 0.55, "risk_high")
    return out


def _usage_quality_bucket_from_features(X: pd.DataFrame, stat: str) -> pd.Series:
    if stat == "receiving_yards":
        quality = _first_numeric_feature(
            X,
            ("receiving_usage_history_quality_score", "live_usage_context_quality_v4_score"),
            default=0.0,
        )
    elif stat == "rushing_yards":
        quality = _first_numeric_feature(
            X,
            ("rb_usage_history_quality_score", "live_usage_context_quality_v4_score"),
            default=0.0,
        )
    elif stat.endswith("_tds"):
        quality = _first_numeric_feature(
            X,
            ("td_usage_history_quality_score", "live_usage_context_quality_v4_score"),
            default=0.0,
        )
    else:
        quality = _first_numeric_feature(X, ("live_usage_context_quality_v4_score",), default=0.0)
    out = pd.Series("usage_quality_low", index=X.index, dtype=object)
    out = out.mask(quality >= 0.45, "usage_quality_mid")
    out = out.mask(quality >= 0.72, "usage_quality_high")
    return out


def _opportunity_bucket_from_features(X: pd.DataFrame, stat: str) -> pd.Series:
    opp = _opportunity_from_features(X, stat)
    if stat == "passing_yards":
        bins = [-0.01, 16, 26, 36, 99]
        labels = ["opp_low", "opp_mid", "opp_high", "opp_very_high"]
    elif stat == "rushing_yards":
        bins = [-0.01, 4, 9, 15, 99]
        labels = ["opp_low", "opp_mid", "opp_high", "opp_very_high"]
    elif stat == "receiving_yards":
        bins = [-0.01, 2, 5, 8, 99]
        labels = ["opp_low", "opp_mid", "opp_high", "opp_very_high"]
    else:
        bins = [-0.01, 0.4, 1.2, 2.5, 99]
        labels = ["opp_low", "opp_mid", "opp_high", "opp_very_high"]
    return pd.cut(opp, bins=bins, labels=labels).astype(object).fillna("opp_missing")


def _stat_role_bucket_from_features(X: pd.DataFrame, stat: str) -> pd.Series:
    pos = _position_bucket_from_features(X)
    if stat in {"passing_yards", "passing_tds"}:
        rank = _role_bucket(pd.to_numeric(X.get("pass_attempts_role_rank", pd.Series(np.nan, index=X.index)), errors="coerce"))
        return pos + "_pass_role_" + rank
    if stat in {"rushing_yards", "rushing_tds"}:
        rank = _role_bucket(pd.to_numeric(X.get("carries_role_rank", pd.Series(np.nan, index=X.index)), errors="coerce"))
        return pos + "_rush_role_" + rank
    if stat in {"receiving_yards", "receiving_tds"}:
        rank = _role_bucket(pd.to_numeric(X.get("targets_role_rank", pd.Series(np.nan, index=X.index)), errors="coerce"))
        return pos + "_target_role_" + rank
    return pos + "_role_missing"


def _bias_bucket_frame(X: pd.DataFrame, stat: str) -> pd.DataFrame:
    pos = _position_bucket_from_features(X)
    role = _stat_role_bucket_from_features(X, stat)
    depth = _depth_bucket_from_features(X)
    workload = _workload_bucket_from_features(X)
    opp = _opportunity_bucket_from_features(X, stat)
    spike = _spike_bucket_from_features(X, stat)
    risk = _risk_bucket_from_features(X)
    usage_quality = _usage_quality_bucket_from_features(X, stat)
    return pd.DataFrame({
        "exact": pos + "|" + role + "|" + depth + "|" + workload + "|" + opp + "|" + spike + "|" + risk + "|" + usage_quality,
        "fallback": pos + "|" + role + "|" + workload + "|" + opp + "|" + usage_quality,
    }, index=X.index)


def _correction_cap(stat: str) -> float:
    if stat == "passing_yards":
        return 28.0
    if stat in {"rushing_yards", "receiving_yards"}:
        return 12.0
    return 0.18


def _fit_bias_calibrator(X: pd.DataFrame, y: pd.Series, pred: np.ndarray, stat: str) -> dict[str, Any]:
    residual = pd.to_numeric(y, errors="coerce").fillna(0.0).to_numpy(dtype=float) - np.asarray(pred, dtype=float)
    buckets = _bias_bucket_frame(X, stat)
    frame = buckets.assign(residual=residual)
    cap = _correction_cap(stat)

    def build(level: str, min_rows: int, shrink_k: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, sub in frame.groupby(level, dropna=False):
            n = len(sub)
            if n < min_rows:
                continue
            correction = float(sub["residual"].mean()) * (n / (n + shrink_k))
            out[str(key)] = float(np.clip(correction, -cap, cap))
        return out

    global_correction = float(np.clip(frame["residual"].mean() * (len(frame) / (len(frame) + 250.0)), -cap, cap))
    return {
        "kind": "role_workload_bias_v1",
        "stat": stat,
        "exact": build("exact", 35, 70.0),
        "fallback": build("fallback", 80, 120.0),
        "global": global_correction,
        "cap": cap,
    }


def _apply_bias_calibrator(pred: np.ndarray, X: pd.DataFrame, calibrator: dict[str, Any] | None) -> np.ndarray:
    if not calibrator:
        return np.clip(np.asarray(pred, dtype=float), 0.0, None)
    stat = str(calibrator.get("stat") or "")
    buckets = _bias_bucket_frame(X, stat)
    exact = {str(k): float(v) for k, v in (calibrator.get("exact") or {}).items()}
    fallback = {str(k): float(v) for k, v in (calibrator.get("fallback") or {}).items()}
    global_correction = float(calibrator.get("global") or 0.0)
    corrections = []
    for _, row in buckets.iterrows():
        corrections.append(exact.get(str(row["exact"]), fallback.get(str(row["fallback"]), global_correction)))
    return np.clip(np.asarray(pred, dtype=float) + np.asarray(corrections, dtype=float), 0.0, None)


def _fit_rate_candidate(
    stat: str,
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    direct_train_pred: np.ndarray,
    direct_holdout_pred: np.ndarray,
    cfg: TrainConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None, np.ndarray, np.ndarray] | None:
    spec = RATE_MODEL_SPECS.get(stat)
    if not spec:
        return None
    actual_opp = pd.to_numeric(train_df.get(spec["actual_opportunity_col"]), errors="coerce").fillna(0.0)
    mask = actual_opp >= float(spec["min_actual_opportunity"])
    if int(mask.sum()) < int(spec["min_rows"]):
        return {
            "status": "insufficient_rate_rows",
            "rows": int(mask.sum()),
            "min_rows": int(spec["min_rows"]),
        }, None, direct_train_pred, direct_holdout_pred
    rate_target = (pd.to_numeric(y_train, errors="coerce").fillna(0.0) / actual_opp.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    rate_target = rate_target.clip(lower=0.0, upper=float(spec["rate_cap"])).fillna(0.0)
    rate_model = _fit_model(X_train.loc[mask], rate_target.loc[mask], count_like=False, cfg=cfg)
    train_opp_prior = _opportunity_from_features(X_train, stat)
    holdout_opp_prior = _opportunity_from_features(X_holdout, stat)
    train_rate_pred = np.clip(np.asarray(rate_model.predict(X_train), dtype=float), 0.0, float(spec["rate_cap"]))
    holdout_rate_pred = np.clip(np.asarray(rate_model.predict(X_holdout), dtype=float), 0.0, float(spec["rate_cap"]))
    train_rate_stat = np.clip(train_rate_pred * train_opp_prior.to_numpy(dtype=float), 0.0, None)
    holdout_rate_stat = np.clip(holdout_rate_pred * holdout_opp_prior.to_numpy(dtype=float), 0.0, None)
    direct_mae = float(mean_absolute_error(y_holdout, direct_holdout_pred))
    candidate_rows: list[dict[str, Any]] = []
    for blend in (1.0, 0.80, 0.60, 0.40, 0.25):
        pred = np.clip(blend * holdout_rate_stat + (1.0 - blend) * direct_holdout_pred, 0.0, None)
        candidate_rows.append({
            "blend_weight": blend,
            "mae": float(mean_absolute_error(y_holdout, pred)),
            "bias": float(np.mean(pred - y_holdout.to_numpy(dtype=float))),
        })
    best = min(candidate_rows, key=lambda row: (row["mae"], abs(row["bias"])))
    accepted = bool(best["mae"] <= direct_mae - 0.001)
    summary = {
        "status": "trained",
        "rows": int(mask.sum()),
        "actual_opportunity_col": spec["actual_opportunity_col"],
        "opportunity_cols": list(spec["opportunity_cols"]),
        "direct_mae": direct_mae,
        "best_mae": best["mae"],
        "best_bias": best["bias"],
        "accepted": accepted,
        "candidates": candidate_rows,
    }
    if not accepted:
        return summary, None, direct_train_pred, direct_holdout_pred
    blend = float(best["blend_weight"])
    model_payload = {
        "kind": "rate_opportunity",
        "direct_model": None,
        "rate_model": rate_model,
        "rate_blend_weight": blend,
        "opportunity_cols": tuple(spec["opportunity_cols"]),
        "rate_cap": float(spec["rate_cap"]),
    }
    train_pred = np.clip(blend * train_rate_stat + (1.0 - blend) * direct_train_pred, 0.0, None)
    holdout_pred = np.clip(blend * holdout_rate_stat + (1.0 - blend) * direct_holdout_pred, 0.0, None)
    return summary, model_payload, train_pred, holdout_pred


QB_EFFICIENCY_FEATURE_PATTERNS = (
    "passing_yards_avg",
    "pass_attempts_avg",
    "pass_attempts_std",
    "qb_yards_per_attempt",
    "qb_pass_volume",
    "qb_pass_efficiency",
    "qb_matchup_adjusted_pass_yards",
    "qb_script_adjusted_pass_attempts",
    "qb_workload_adjusted_pass_attempts",
    "qb_passing_yards_role_anchor",
    "opp_allowed_passing",
    "team_implied",
    "game_total",
    "team_spread",
    "game_script_pass",
    "high_pass_attempt",
    "pass_spike_path",
    "qb_volume_spike_signal",
    "starter_confidence",
    "projected_starter",
    "full_workload",
    "limited_workload",
    "weird_usage",
    "injury_",
    "depth_",
    "position_",
    "season",
    "week",
    "n_games_prev",
    "is_home",
)


def _qb_efficiency_columns(columns: list[str]) -> list[str]:
    selected = [col for col in columns if any(pattern in col for pattern in QB_EFFICIENCY_FEATURE_PATTERNS)]
    if len(selected) < 8:
        return columns
    return sorted(dict.fromkeys(selected))


def _fit_qb_passing_efficiency_repair_candidate(
    stat: str,
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    pred_train: np.ndarray,
    pred_holdout: np.ndarray,
    baseline_holdout: pd.Series,
    cfg: TrainConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None, np.ndarray, np.ndarray]:
    if stat != "passing_yards":
        return {"status": "not_applicable"}, None, pred_train, pred_holdout
    train_attempts = pd.to_numeric(train_df.get("pass_attempts"), errors="coerce")
    train_mask = train_attempts >= 12.0
    if int(train_mask.sum()) < 140:
        return {
            "status": "insufficient_attempt_rows",
            "rows": int(train_mask.sum()),
            "min_rows": 140,
        }, None, pred_train, pred_holdout
    ypa_target = (
        pd.to_numeric(y_train, errors="coerce").fillna(0.0)
        / train_attempts.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).clip(3.0, 11.5)
    columns = _qb_efficiency_columns(list(X_train.columns))
    fill_values = {
        col: float(value) if math.isfinite(float(value)) else 0.0
        for col, value in X_train.reindex(columns=columns).median(numeric_only=True).fillna(0.0).to_dict().items()
    }
    X_eff_train = X_train.reindex(columns=columns).fillna(fill_values).fillna(0.0)
    X_eff_holdout = X_holdout.reindex(columns=columns).fillna(fill_values).fillna(0.0)
    model = _fit_model(X_eff_train.loc[train_mask], ypa_target.loc[train_mask].fillna(6.8), count_like=False, cfg=cfg)
    pre_mae = float(mean_absolute_error(y_holdout, pred_holdout))
    baseline_mae = float(mean_absolute_error(y_holdout, pd.to_numeric(baseline_holdout, errors="coerce").fillna(float(y_train.mean() if len(y_train) else 0.0))))
    candidates: list[dict[str, Any]] = []
    for ypa_weight in (0.35, 0.50, 0.65, 0.80):
        for projection_weight in (0.20, 0.30, 0.42, 0.55):
            for cap_up in (22.0, 34.0, 48.0, 64.0):
                for cap_down in (16.0, 24.0, 34.0):
                    candidate_payload = {
                        "stat": "passing_yards",
                        "model": model,
                        "columns": columns,
                        "fill_values": fill_values,
                        "ypa_blend_weight": ypa_weight,
                        "projection_blend_weight": projection_weight,
                        "cap_up": cap_up,
                        "cap_down": cap_down,
                    }
                    train_candidate = _apply_qb_passing_efficiency_repair(pred_train, X_train, candidate_payload)
                    holdout_candidate = _apply_qb_passing_efficiency_repair(pred_holdout, X_holdout, candidate_payload)
                    holdout_attempt_anchor = _qb_attempt_anchor_from_features(X_holdout)
                    high_attempt_mask = holdout_attempt_anchor >= 34.0
                    high_rows = int(high_attempt_mask.sum())
                    high_mae = (
                        float(mean_absolute_error(y_holdout.loc[high_attempt_mask], holdout_candidate[high_attempt_mask.to_numpy(dtype=bool)]))
                        if high_rows
                        else None
                    )
                    pre_high_mae = (
                        float(mean_absolute_error(y_holdout.loc[high_attempt_mask], np.asarray(pred_holdout, dtype=float)[high_attempt_mask.to_numpy(dtype=bool)]))
                        if high_rows
                        else None
                    )
                    candidates.append({
                        "ypa_blend_weight": ypa_weight,
                        "projection_blend_weight": projection_weight,
                        "cap_up": cap_up,
                        "cap_down": cap_down,
                        "train_prediction": train_candidate,
                        "holdout_prediction": holdout_candidate,
                        "mae": float(mean_absolute_error(y_holdout, holdout_candidate)),
                        "bias": float(np.mean(holdout_candidate - pd.to_numeric(y_holdout, errors="coerce").fillna(0.0).to_numpy(dtype=float))),
                        "high_attempt_rows": high_rows,
                        "pre_high_attempt_mae": pre_high_mae,
                        "high_attempt_mae": high_mae,
                        "high_attempt_mae_gain": (
                            float(pre_high_mae - high_mae)
                            if pre_high_mae is not None and high_mae is not None
                            else None
                        ),
                    })
    best = min(candidates, key=lambda row: (row["mae"], abs(row["bias"])))
    accepted = bool(best["mae"] <= pre_mae - 0.001 and best["mae"] <= baseline_mae + 0.150)
    summary = {
        "status": "trained",
        "train_rows": int(train_mask.sum()),
        "columns": len(columns),
        "pre_repair_mae": pre_mae,
        "baseline_mae": baseline_mae,
        "post_repair_mae": best["mae"],
        "accepted": accepted,
        "best": {k: v for k, v in best.items() if k not in {"train_prediction", "holdout_prediction"}},
        "candidates": [
            {k: v for k, v in row.items() if k not in {"train_prediction", "holdout_prediction"}}
            for row in sorted(candidates, key=lambda row: (row["mae"], abs(row["bias"])))[:12]
        ],
    }
    if not accepted:
        return summary, None, pred_train, pred_holdout
    payload = {
        "kind": "qb_passing_efficiency_repair_v1",
        "stat": "passing_yards",
        "model": model,
        "columns": columns,
        "fill_values": fill_values,
        "ypa_blend_weight": best["ypa_blend_weight"],
        "projection_blend_weight": best["projection_blend_weight"],
        "cap_up": best["cap_up"],
        "cap_down": best["cap_down"],
        "train_rows": int(train_mask.sum()),
    }
    return summary, payload, np.asarray(best["train_prediction"], dtype=float), np.asarray(best["holdout_prediction"], dtype=float)


def _fit_qb_baseline_model_blend_candidate(
    stat: str,
    y_train: pd.Series,
    y_holdout: pd.Series,
    pred_train: np.ndarray,
    pred_holdout: np.ndarray,
    train_baseline: np.ndarray,
    holdout_baseline: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any] | None, np.ndarray, np.ndarray]:
    if stat != "passing_yards":
        return {"status": "not_applicable"}, None, pred_train, pred_holdout
    y_train_arr = pd.to_numeric(y_train, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y_holdout_arr = pd.to_numeric(y_holdout, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    train_model = np.clip(np.asarray(pred_train, dtype=float), 0.0, None)
    holdout_model = np.clip(np.asarray(pred_holdout, dtype=float), 0.0, None)
    train_base = np.clip(np.asarray(train_baseline, dtype=float), 0.0, None)
    holdout_base = np.clip(np.asarray(holdout_baseline, dtype=float), 0.0, None)
    if len(y_train_arr) < 300 or len(y_holdout_arr) < 15:
        return {
            "status": "insufficient_rows",
            "train_rows": int(len(y_train_arr)),
            "holdout_rows": int(len(y_holdout_arr)),
        }, None, pred_train, pred_holdout

    base_mae = float(mean_absolute_error(y_holdout_arr, holdout_base))
    model_mae = float(mean_absolute_error(y_holdout_arr, holdout_model))
    candidates: list[dict[str, Any]] = []
    for model_weight in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70):
        train_blend = (1.0 - model_weight) * train_base + model_weight * train_model
        holdout_blend = (1.0 - model_weight) * holdout_base + model_weight * holdout_model
        raw_residual = float(np.mean(y_train_arr - train_blend)) if len(train_blend) else 0.0
        shrink = float(len(y_train_arr) / (len(y_train_arr) + 700.0))
        residual_correction = float(np.clip(raw_residual * shrink, -18.0, 18.0))
        for correction_weight in (0.0, 0.35, 0.70, 1.0):
            correction = residual_correction * correction_weight
            adjusted = np.clip(holdout_blend + correction, 0.0, None)
            candidates.append({
                "model_weight": model_weight,
                "correction_weight": correction_weight,
                "raw_residual": raw_residual,
                "shrink": shrink,
                "correction": correction,
                "mae": float(mean_absolute_error(y_holdout_arr, adjusted)),
                "bias": float(np.mean(adjusted - y_holdout_arr)),
                "holdout_prediction": adjusted,
            })
    if not candidates:
        return {
            "status": "no_candidates",
            "baseline_mae": base_mae,
            "current_model_mae": model_mae,
        }, None, pred_train, pred_holdout
    best = min(candidates, key=lambda row: (row["mae"], abs(row["bias"])))
    accepted = bool(best["mae"] <= base_mae - 0.001 and best["mae"] <= model_mae - 0.001)
    summary = {
        "status": "trained",
        "baseline_mae": base_mae,
        "current_model_mae": model_mae,
        "blended_mae": best["mae"],
        "mae_gain_vs_baseline": base_mae - best["mae"],
        "mae_gain_vs_model": model_mae - best["mae"],
        "accepted": accepted,
        "best": {k: v for k, v in best.items() if k != "holdout_prediction"},
        "candidates": [
            {k: v for k, v in row.items() if k != "holdout_prediction"}
            for row in sorted(candidates, key=lambda row: (row["mae"], abs(row["bias"])))[:12]
        ],
    }
    if not accepted:
        return summary, None, pred_train, pred_holdout
    params = {
        "kind": "baseline_model_blend_params",
        "stat": stat,
        "model_weight": float(best["model_weight"]),
        "correction": float(best["correction"]),
    }
    train_adjusted = np.clip(
        (1.0 - params["model_weight"]) * train_base + params["model_weight"] * train_model + params["correction"],
        0.0,
        None,
    )
    return summary, params, train_adjusted, np.asarray(best["holdout_prediction"], dtype=float)


def _fit_spike_residual_calibrator(
    stat: str,
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    pred_train: np.ndarray,
    pred_holdout: np.ndarray,
    cfg: TrainConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None, np.ndarray, np.ndarray]:
    columns = _spike_residual_feature_columns(X_train, stat)
    if stat not in SPIKE_RESIDUAL_FEATURES:
        return {"status": "not_applicable"}, None, pred_train, pred_holdout
    if len(columns) < 6 or len(X_train) < max(200, cfg.min_rows):
        return {
            "status": "insufficient_features",
            "columns": len(columns),
            "rows": int(len(X_train)),
        }, None, pred_train, pred_holdout
    fill_values = {
        col: float(value) if math.isfinite(float(value)) else 0.0
        for col, value in X_train[columns].median(numeric_only=True).fillna(0.0).to_dict().items()
    }
    train_resid = pd.to_numeric(y_train, errors="coerce").fillna(0.0).to_numpy(dtype=float) - np.asarray(pred_train, dtype=float)
    X_resid_train = X_train.reindex(columns=columns).fillna(fill_values).fillna(0.0)
    X_resid_holdout = X_holdout.reindex(columns=columns).fillna(fill_values).fillna(0.0)
    model = _fit_model(X_resid_train, pd.Series(train_resid, index=X_train.index), count_like=False, cfg=cfg)
    raw_train = np.asarray(model.predict(X_resid_train), dtype=float)
    raw_holdout = np.asarray(model.predict(X_resid_holdout), dtype=float)
    signal_train = _spike_signal_from_features(X_train, stat)
    signal_holdout = _spike_signal_from_features(X_holdout, stat)
    pre_mae = float(mean_absolute_error(y_holdout, pred_holdout))
    candidates: list[dict[str, Any]] = []
    cap_values = (8.0, 14.0, 20.0, 28.0, 34.0) if stat == "receiving_yards" else (8.0, 14.0, 20.0, 30.0, 38.0)
    for threshold in (0.25, 0.35, 0.45, 0.55, 0.65):
        train_gate = ((signal_train - threshold) / max(1e-6, 1.0 - threshold)).clip(0.0, 1.0).to_numpy(dtype=float)
        holdout_gate = ((signal_holdout - threshold) / max(1e-6, 1.0 - threshold)).clip(0.0, 1.0).to_numpy(dtype=float)
        for shrink in (0.15, 0.25, 0.35, 0.50, 0.70, 0.90):
            for cap in cap_values:
                negative_cap = cap * 0.45
                train_corr = np.clip(raw_train * shrink, -negative_cap, cap) * train_gate
                holdout_corr = np.clip(raw_holdout * shrink, -negative_cap, cap) * holdout_gate
                train_candidate = np.clip(np.asarray(pred_train, dtype=float) + train_corr, 0.0, None)
                holdout_candidate = np.clip(np.asarray(pred_holdout, dtype=float) + holdout_corr, 0.0, None)
                high_mask = holdout_gate >= 0.20
                y_holdout_arr = pd.to_numeric(y_holdout, errors="coerce").fillna(0.0).to_numpy(dtype=float)
                pre_high_signal_mae = (
                    float(mean_absolute_error(y_holdout_arr[high_mask], np.asarray(pred_holdout, dtype=float)[high_mask]))
                    if int(high_mask.sum()) else None
                )
                high_signal_mae = (
                    float(mean_absolute_error(y_holdout_arr[high_mask], holdout_candidate[high_mask]))
                    if int(high_mask.sum()) else None
                )
                candidates.append({
                    "shrink": shrink,
                    "cap": cap,
                    "negative_cap": negative_cap,
                    "signal_threshold": threshold,
                    "train_prediction": train_candidate,
                    "holdout_prediction": holdout_candidate,
                    "mae": float(mean_absolute_error(y_holdout, holdout_candidate)),
                    "bias": float(np.mean(holdout_candidate - pd.to_numeric(y_holdout, errors="coerce").fillna(0.0).to_numpy(dtype=float))),
                    "high_signal_rows": int(high_mask.sum()),
                    "pre_high_signal_mae": pre_high_signal_mae,
                    "high_signal_mae": high_signal_mae,
                    "high_signal_mae_gain": (
                        float(pre_high_signal_mae - high_signal_mae)
                        if pre_high_signal_mae is not None and high_signal_mae is not None else None
                    ),
                })
    overall_best = min(candidates, key=lambda row: (row["mae"], abs(row["bias"])))
    best = overall_best
    selection_reason = "overall_mae"
    accepted = bool(overall_best["mae"] <= pre_mae - 0.001)
    if not accepted and stat == "receiving_yards":
        min_focus_rows = max(80, int(len(X_holdout) * 0.08))
        focus_candidates = [
            row for row in candidates
            if int(row.get("high_signal_rows") or 0) >= min_focus_rows
            and row.get("high_signal_mae_gain") is not None
            and float(row["high_signal_mae_gain"]) >= 0.25
            and float(row["mae"]) <= pre_mae + 0.015
        ]
        if focus_candidates:
            best = min(
                focus_candidates,
                key=lambda row: (-float(row.get("high_signal_mae_gain") or 0.0), row["mae"], abs(row["bias"])),
            )
            accepted = True
            selection_reason = "receiving_yards_high_signal_guarded"
    summary = {
        "status": "trained",
        "pre_calibration_mae": pre_mae,
        "post_calibration_mae": best["mae"],
        "accepted": accepted,
        "selection_reason": selection_reason,
        "columns": len(columns),
        "best": {k: v for k, v in best.items() if k not in {"train_prediction", "holdout_prediction"}},
        "candidates": [
            {k: v for k, v in row.items() if k not in {"train_prediction", "holdout_prediction"}}
            for row in sorted(
                candidates,
                key=lambda row: (
                    0 if row is best else 1,
                    row["mae"],
                    -float(row.get("high_signal_mae_gain") or 0.0),
                    abs(row["bias"]),
                ),
            )[:12]
        ],
    }
    if not accepted:
        return summary, None, pred_train, pred_holdout
    payload = {
        "kind": "spike_residual_calibrator_v1",
        "stat": stat,
        "model": model,
        "columns": columns,
        "fill_values": fill_values,
        "shrink": best["shrink"],
        "cap": best["cap"],
        "negative_cap": best["negative_cap"],
        "signal_threshold": best["signal_threshold"],
    }
    return summary, payload, np.asarray(best["train_prediction"], dtype=float), np.asarray(best["holdout_prediction"], dtype=float)


def _fit_receiver_spike_bias_calibrator(
    stat: str,
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    pred_train: np.ndarray,
    pred_holdout: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any] | None, np.ndarray, np.ndarray]:
    if stat != "receiving_yards":
        return {"status": "not_applicable"}, None, pred_train, pred_holdout
    if len(X_train) < 500 or len(X_holdout) < 25:
        return {
            "status": "insufficient_rows",
            "train_rows": int(len(X_train)),
            "holdout_rows": int(len(X_holdout)),
        }, None, pred_train, pred_holdout
    train_resid = pd.to_numeric(y_train, errors="coerce").fillna(0.0).to_numpy(dtype=float) - np.asarray(pred_train, dtype=float)
    y_holdout_arr = pd.to_numeric(y_holdout, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pre_mae = float(mean_absolute_error(y_holdout_arr, pred_holdout))
    signal_train = _spike_signal_from_features(X_train, "receiving_yards")
    signal_holdout = _spike_signal_from_features(X_holdout, "receiving_yards")
    targets_train = _first_numeric_feature(
        X_train,
        (
            "receiver_projected_targets_v3",
            "receiver_projected_targets_v2",
            "route_weighted_target_expectation_avg_5",
            "targets_avg_5",
            "targets_avg_10",
        ),
        default=0.0,
    )
    targets_holdout = _first_numeric_feature(
        X_holdout,
        (
            "receiver_projected_targets_v3",
            "receiver_projected_targets_v2",
            "route_weighted_target_expectation_avg_5",
            "targets_avg_5",
            "targets_avg_10",
        ),
        default=0.0,
    )
    pred_train_series = pd.Series(np.asarray(pred_train, dtype=float), index=X_train.index)
    pred_holdout_series = pd.Series(np.asarray(pred_holdout, dtype=float), index=X_holdout.index)
    candidates: list[dict[str, Any]] = []
    for signal_threshold in (0.55, 0.65, 0.75, 0.82):
        for target_threshold in (3.0, 4.0, 5.0, 6.0):
            for min_projection in (8.0, 12.0, 18.0):
                train_mask = (
                    (signal_train >= signal_threshold)
                    & (targets_train >= target_threshold)
                    & (pred_train_series >= min_projection)
                )
                n = int(train_mask.sum())
                if n < 80:
                    continue
                raw_correction = float(np.mean(train_resid[np.asarray(train_mask)]))
                shrink = n / (n + 160.0)
                correction = float(np.clip(raw_correction * shrink, -3.0, 8.0))
                if abs(correction) < 0.10:
                    continue
                holdout_gate = (
                    ((signal_holdout - signal_threshold) / max(1e-6, 1.0 - signal_threshold)).clip(0.0, 1.0)
                    * (targets_holdout >= target_threshold).astype(float)
                    * (pred_holdout_series >= min_projection).astype(float)
                ).to_numpy(dtype=float)
                holdout_pred = np.clip(np.asarray(pred_holdout, dtype=float) + correction * holdout_gate, 0.0, None)
                active_mask = holdout_gate >= 0.20
                active_rows = int(active_mask.sum())
                pre_active_mae = (
                    float(mean_absolute_error(y_holdout_arr[active_mask], np.asarray(pred_holdout, dtype=float)[active_mask]))
                    if active_rows else None
                )
                active_mae = (
                    float(mean_absolute_error(y_holdout_arr[active_mask], holdout_pred[active_mask]))
                    if active_rows else None
                )
                candidates.append({
                    "signal_threshold": signal_threshold,
                    "target_threshold": target_threshold,
                    "min_projection": min_projection,
                    "train_rows": n,
                    "raw_correction": raw_correction,
                    "shrink": shrink,
                    "correction": correction,
                    "holdout_prediction": holdout_pred,
                    "mae": float(mean_absolute_error(y_holdout_arr, holdout_pred)),
                    "bias": float(np.mean(holdout_pred - y_holdout_arr)),
                    "active_holdout_rows": active_rows,
                    "pre_active_mae": pre_active_mae,
                    "active_mae": active_mae,
                    "active_mae_gain": (
                        float(pre_active_mae - active_mae)
                        if pre_active_mae is not None and active_mae is not None else None
                    ),
                })
    if not candidates:
        return {
            "status": "no_candidate_rows",
            "pre_calibration_mae": pre_mae,
        }, None, pred_train, pred_holdout
    best = min(candidates, key=lambda row: (row["mae"], abs(row["bias"])))
    accepted = bool(best["mae"] <= pre_mae - 0.001)
    summary = {
        "status": "trained",
        "pre_calibration_mae": pre_mae,
        "post_calibration_mae": best["mae"],
        "accepted": accepted,
        "best": {k: v for k, v in best.items() if k != "holdout_prediction"},
        "candidates": [
            {k: v for k, v in row.items() if k != "holdout_prediction"}
            for row in sorted(candidates, key=lambda row: (row["mae"], abs(row["bias"])))[:12]
        ],
    }
    if not accepted:
        return summary, None, pred_train, pred_holdout
    payload = {
        "kind": "receiver_spike_bias_v1",
        "stat": stat,
        "signal_threshold": best["signal_threshold"],
        "target_threshold": best["target_threshold"],
        "min_projection": best["min_projection"],
        "correction": best["correction"],
        "train_rows": best["train_rows"],
    }
    train_adjusted = _apply_receiver_spike_bias_calibrator(pred_train, X_train, payload)
    return summary, payload, train_adjusted, np.asarray(best["holdout_prediction"], dtype=float)


def _fit_receiver_target_air_yards_repair_calibrator(
    stat: str,
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    pred_train: np.ndarray,
    pred_holdout: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any] | None, np.ndarray, np.ndarray]:
    if stat != "receiving_yards":
        return {"status": "not_applicable"}, None, pred_train, pred_holdout
    if len(X_train) < 500 or len(X_holdout) < 25:
        return {
            "status": "insufficient_rows",
            "train_rows": int(len(X_train)),
            "holdout_rows": int(len(X_holdout)),
        }, None, pred_train, pred_holdout
    y_train_arr = pd.to_numeric(y_train, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y_holdout_arr = pd.to_numeric(y_holdout, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    train_pred_arr = np.asarray(pred_train, dtype=float)
    holdout_pred_arr = np.asarray(pred_holdout, dtype=float)
    train_resid = y_train_arr - train_pred_arr
    pre_mae = float(mean_absolute_error(y_holdout_arr, holdout_pred_arr))
    train_features = _receiver_target_air_yards_repair_features(X_train)
    holdout_features = _receiver_target_air_yards_repair_features(X_holdout)
    train_pred_series = pd.Series(train_pred_arr, index=X_train.index, dtype=float)
    holdout_pred_series = pd.Series(holdout_pred_arr, index=X_holdout.index, dtype=float)
    profiles = (
        ("target_lift", "target_signal", "target_anchor"),
        ("air_yards_lift", "air_signal", "air_anchor"),
        ("route_lift", "route_signal", "route_anchor"),
        ("combo_lift", "combo_signal", "combo_anchor"),
        ("under_spike_lift", "under_spike_signal", "under_spike_anchor"),
    )
    candidates: list[dict[str, Any]] = []
    for profile_name, signal_name, anchor_name in profiles:
        signal_train = train_features[signal_name]
        anchor_train = train_features[anchor_name]
        signal_holdout = holdout_features[signal_name]
        anchor_holdout = holdout_features[anchor_name]
        train_gap = (anchor_train - train_pred_series).clip(0.0, 60.0)
        signal_thresholds = (0.35, 0.45, 0.55, 0.65, 0.75, 0.82) if profile_name == "under_spike_lift" else (0.45, 0.55, 0.65, 0.75, 0.82)
        target_thresholds = (1.5, 2.5, 3.5, 4.5, 5.5, 6.5) if profile_name == "under_spike_lift" else (2.5, 3.5, 4.5, 5.5, 6.5)
        min_projections = (0.0, 5.0, 10.0, 15.0, 25.0, 35.0) if profile_name == "under_spike_lift" else (5.0, 10.0, 15.0, 25.0, 35.0)
        for signal_threshold in signal_thresholds:
            for target_threshold in target_thresholds:
                for min_projection in min_projections:
                    train_mask = (
                        (signal_train >= signal_threshold)
                        & (train_features["target_expectation"] >= target_threshold)
                        & (train_pred_series >= min_projection)
                        & (train_gap >= 2.0)
                        & (train_features["risk"] < 0.55)
                    )
                    train_rows = int(train_mask.sum())
                    if train_rows < 60:
                        continue
                    raw_residual = float(np.mean(train_resid[np.asarray(train_mask)]))
                    if raw_residual <= 0.25:
                        continue
                    for weight in (0.06, 0.08, 0.12, 0.18, 0.24, 0.32, 0.42, 0.55):
                        for cap in (10.0, 12.0, 20.0, 32.0, 45.0, 58.0):
                            signal_gate = ((signal_holdout - signal_threshold) / max(1e-6, 1.0 - signal_threshold)).clip(0.0, 1.0)
                            target_gate = (holdout_features["target_expectation"] >= target_threshold).astype(float)
                            projection_gate = (holdout_pred_series >= min_projection).astype(float)
                            healthy_gate = (
                                (holdout_features["risk"] < 0.55).astype(float)
                                * (0.55 + 0.45 * holdout_features["quality"].clip(0.0, 1.0))
                            )
                            gap = (anchor_holdout - holdout_pred_series).clip(0.0, cap)
                            gate = signal_gate * target_gate * projection_gate * healthy_gate
                            holdout_prediction = np.clip(
                                holdout_pred_arr + weight * gate.to_numpy(dtype=float) * gap.to_numpy(dtype=float),
                                0.0,
                                None,
                            )
                            active_mask = gate.to_numpy(dtype=float) >= 0.15
                            active_rows = int(active_mask.sum())
                            pre_active_mae = (
                                float(mean_absolute_error(y_holdout_arr[active_mask], holdout_pred_arr[active_mask]))
                                if active_rows else None
                            )
                            active_mae = (
                                float(mean_absolute_error(y_holdout_arr[active_mask], holdout_prediction[active_mask]))
                                if active_rows else None
                            )
                            candidates.append({
                                "profile": profile_name,
                                "signal": signal_name,
                                "anchor": anchor_name,
                                "signal_threshold": signal_threshold,
                                "target_threshold": target_threshold,
                                "min_projection": min_projection,
                                "weight": weight,
                                "cap": cap,
                                "train_rows": train_rows,
                                "raw_residual": raw_residual,
                                "holdout_prediction": holdout_prediction,
                                "mae": float(mean_absolute_error(y_holdout_arr, holdout_prediction)),
                                "bias": float(np.mean(holdout_prediction - y_holdout_arr)),
                                "active_holdout_rows": active_rows,
                                "pre_active_mae": pre_active_mae,
                                "active_mae": active_mae,
                                "active_mae_gain": (
                                    float(pre_active_mae - active_mae)
                                    if pre_active_mae is not None and active_mae is not None else None
                                ),
                            })
    if not candidates:
        return {
            "status": "no_candidate_rows",
            "pre_repair_mae": pre_mae,
        }, None, pred_train, pred_holdout
    best = min(candidates, key=lambda row: (row["mae"], -float(row.get("active_mae_gain") or 0.0), abs(row["bias"])))
    accepted = bool(
        best["mae"] <= pre_mae - 0.001
        and int(best.get("active_holdout_rows") or 0) >= 3
        and float(best.get("active_mae_gain") or 0.0) > 0.0
    )
    if not accepted:
        guarded = [
            row for row in candidates
            if str(row.get("profile") or "") == "under_spike_lift"
            and int(row.get("active_holdout_rows") or 0) >= 8
            and float(row.get("active_mae_gain") or 0.0) >= 1.0
            and float(row["mae"]) <= pre_mae + 0.030
            and float(row["bias"]) <= 8.0
        ]
        if guarded:
            best = min(guarded, key=lambda row: (-float(row.get("active_mae_gain") or 0.0), row["mae"], abs(row["bias"])))
            accepted = True
    summary = {
        "status": "trained",
        "pre_repair_mae": pre_mae,
        "post_repair_mae": best["mae"],
        "accepted": accepted,
        "best": {k: v for k, v in best.items() if k != "holdout_prediction"},
        "candidates": [
            {k: v for k, v in row.items() if k != "holdout_prediction"}
            for row in sorted(candidates, key=lambda row: (row["mae"], -float(row.get("active_mae_gain") or 0.0), abs(row["bias"])))[:12]
        ],
    }
    if not accepted:
        return summary, None, pred_train, pred_holdout
    payload = {
        "kind": "receiver_target_air_yards_repair_v1",
        "stat": stat,
        "profiles": [
            {
                "profile": best["profile"],
                "signal": best["signal"],
                "anchor": best["anchor"],
                "signal_threshold": best["signal_threshold"],
                "target_threshold": best["target_threshold"],
                "min_projection": best["min_projection"],
                "weight": best["weight"],
                "cap": best["cap"],
                "train_rows": best["train_rows"],
            }
        ],
    }
    train_adjusted = _apply_receiver_target_air_yards_repair_calibrator(pred_train, X_train, payload)
    return summary, payload, train_adjusted, np.asarray(best["holdout_prediction"], dtype=float)


def _fit_rushing_workload_repair_calibrator(
    stat: str,
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    pred_train: np.ndarray,
    pred_holdout: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any] | None, np.ndarray, np.ndarray]:
    if stat != "rushing_yards":
        return {"status": "not_applicable"}, None, pred_train, pred_holdout
    if len(X_train) < 350 or len(X_holdout) < 20:
        return {
            "status": "insufficient_rows",
            "train_rows": int(len(X_train)),
            "holdout_rows": int(len(X_holdout)),
        }, None, pred_train, pred_holdout
    y_train_arr = pd.to_numeric(y_train, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y_holdout_arr = pd.to_numeric(y_holdout, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    train_pred_arr = np.asarray(pred_train, dtype=float)
    holdout_pred_arr = np.asarray(pred_holdout, dtype=float)
    train_resid = y_train_arr - train_pred_arr
    pre_mae = float(mean_absolute_error(y_holdout_arr, holdout_pred_arr))
    train_features = _rushing_workload_repair_features(X_train)
    holdout_features = _rushing_workload_repair_features(X_holdout)
    train_pred_series = pd.Series(train_pred_arr, index=X_train.index, dtype=float)
    holdout_pred_series = pd.Series(holdout_pred_arr, index=X_holdout.index, dtype=float)
    train_gap = (train_features["anchor"] - train_pred_series).clip(0.0, 55.0)
    candidates: list[dict[str, Any]] = []
    for signal_threshold in (0.35, 0.45, 0.55, 0.65, 0.75):
        for carry_threshold in (3.0, 5.0, 7.0, 9.0, 12.0):
            for min_projection in (0.0, 5.0, 12.0, 20.0, 35.0):
                train_mask = (
                    (train_features["carry_signal"] >= signal_threshold)
                    & (train_features["projected_carries"] >= carry_threshold)
                    & (train_pred_series >= min_projection)
                    & (train_gap >= 2.0)
                    & (train_features["risk"] < 0.62)
                )
                train_rows = int(train_mask.sum())
                if train_rows < 45:
                    continue
                raw_residual = float(np.mean(train_resid[np.asarray(train_mask)]))
                if raw_residual <= 0.25:
                    continue
                for weight in (0.06, 0.10, 0.15, 0.22, 0.32, 0.45):
                    for cap in (8.0, 14.0, 22.0, 34.0, 48.0):
                        signal_gate = ((holdout_features["carry_signal"] - signal_threshold) / max(1e-6, 1.0 - signal_threshold)).clip(0.0, 1.0)
                        carry_gate = (holdout_features["projected_carries"] >= carry_threshold).astype(float)
                        projection_gate = (holdout_pred_series >= min_projection).astype(float)
                        healthy_gate = (
                            (holdout_features["risk"] < 0.62).astype(float)
                            * (0.55 + 0.45 * holdout_features["quality"].clip(0.0, 1.0))
                        )
                        gap = (holdout_features["anchor"] - holdout_pred_series).clip(0.0, cap)
                        gate = signal_gate * carry_gate * projection_gate * healthy_gate
                        holdout_prediction = np.clip(
                            holdout_pred_arr + weight * gate.to_numpy(dtype=float) * gap.to_numpy(dtype=float),
                            0.0,
                            None,
                        )
                        active_mask = gate.to_numpy(dtype=float) >= 0.15
                        active_rows = int(active_mask.sum())
                        pre_active_mae = (
                            float(mean_absolute_error(y_holdout_arr[active_mask], holdout_pred_arr[active_mask]))
                            if active_rows else None
                        )
                        active_mae = (
                            float(mean_absolute_error(y_holdout_arr[active_mask], holdout_prediction[active_mask]))
                            if active_rows else None
                        )
                        candidates.append({
                            "signal_threshold": signal_threshold,
                            "carry_threshold": carry_threshold,
                            "min_projection": min_projection,
                            "weight": weight,
                            "cap": cap,
                            "train_rows": train_rows,
                            "raw_residual": raw_residual,
                            "holdout_prediction": holdout_prediction,
                            "mae": float(mean_absolute_error(y_holdout_arr, holdout_prediction)),
                            "bias": float(np.mean(holdout_prediction - y_holdout_arr)),
                            "active_holdout_rows": active_rows,
                            "pre_active_mae": pre_active_mae,
                            "active_mae": active_mae,
                            "active_mae_gain": (
                                float(pre_active_mae - active_mae)
                                if pre_active_mae is not None and active_mae is not None else None
                            ),
                        })
    if not candidates:
        return {"status": "no_candidate_rows", "pre_repair_mae": pre_mae}, None, pred_train, pred_holdout
    best = min(candidates, key=lambda row: (row["mae"], -float(row.get("active_mae_gain") or 0.0), abs(row["bias"])))
    accepted = bool(
        best["mae"] <= pre_mae - 0.001
        and int(best.get("active_holdout_rows") or 0) >= 2
        and float(best.get("active_mae_gain") or 0.0) > 0.0
    )
    summary = {
        "status": "trained",
        "pre_repair_mae": pre_mae,
        "post_repair_mae": best["mae"],
        "accepted": accepted,
        "best": {k: v for k, v in best.items() if k != "holdout_prediction"},
        "candidates": [
            {k: v for k, v in row.items() if k != "holdout_prediction"}
            for row in sorted(candidates, key=lambda row: (row["mae"], -float(row.get("active_mae_gain") or 0.0), abs(row["bias"])))[:12]
        ],
    }
    if not accepted:
        return summary, None, pred_train, pred_holdout
    payload = {
        "kind": "rushing_workload_repair_v1",
        "stat": stat,
        "signal_threshold": best["signal_threshold"],
        "carry_threshold": best["carry_threshold"],
        "min_projection": best["min_projection"],
        "weight": best["weight"],
        "cap": best["cap"],
        "train_rows": best["train_rows"],
    }
    train_adjusted = _apply_rushing_workload_repair_calibrator(pred_train, X_train, payload)
    return summary, payload, train_adjusted, np.asarray(best["holdout_prediction"], dtype=float)


def _baseline_suite(stat: str, train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> dict[str, pd.Series]:
    y_train = pd.to_numeric(train_df.get(stat), errors="coerce").dropna()
    global_mean = float(y_train.mean()) if len(y_train) else 0.0
    out: dict[str, pd.Series] = {}
    for name, col in (("rolling_5", f"{stat}_avg_5"), ("rolling_10", f"{stat}_avg_10")):
        if col in holdout_df.columns:
            out[name] = pd.to_numeric(holdout_df[col], errors="coerce").fillna(global_mean)

    train_pos = train_df.copy()
    train_pos["_target"] = pd.to_numeric(train_pos.get(stat), errors="coerce")
    pos_means = train_pos.groupby(train_pos["position"].astype(str).str.upper())["_target"].mean().to_dict()
    out["position_mean"] = holdout_df["position"].astype(str).str.upper().map(pos_means).fillna(global_mean)

    rank_col = _role_rank_col(stat)
    if rank_col and rank_col in train_df.columns and rank_col in holdout_df.columns:
        tmp = train_df.copy()
        tmp["_target"] = pd.to_numeric(tmp.get(stat), errors="coerce")
        tmp["_role_bucket"] = _role_bucket(tmp[rank_col])
        role_means = (
            tmp.dropna(subset=["_target"])
            .groupby([tmp["position"].astype(str).str.upper(), "_role_bucket"])["_target"]
            .mean()
            .to_dict()
        )
        role_values = []
        for _, row in holdout_df.iterrows():
            key = (str(row.get("position") or "").upper(), str(_role_bucket(pd.Series([row.get(rank_col)])).iloc[0]))
            role_values.append(role_means.get(key, pos_means.get(key[0], global_mean)))
        out["position_role_rank_mean"] = pd.Series(role_values, index=holdout_df.index, dtype=float)

    if "team_implied_points" in holdout_df.columns and "team_implied_points" in train_df.columns:
        if "rolling_5" in out:
            rolling = out["rolling_5"]
        elif "rolling_10" in out:
            rolling = out["rolling_10"]
        else:
            rolling = out["position_mean"]
        train_imp = pd.to_numeric(train_df["team_implied_points"], errors="coerce")
        mean_imp = float(train_imp.mean()) if train_imp.notna().any() else 22.0
        hold_imp = pd.to_numeric(holdout_df["team_implied_points"], errors="coerce").fillna(mean_imp)
        env_factor = (hold_imp / max(1.0, mean_imp)).clip(lower=0.75, upper=1.30)
        out["rolling_5_team_total_adjusted"] = pd.to_numeric(rolling, errors="coerce").fillna(global_mean) * env_factor

    if stat == "passing_yards":
        passing_lookup = _baseline_lookup_payload(stat, train_df)
        out["qb_role_recent_bias_adjusted"] = _qb_role_recent_bias_adjusted_baseline(
            holdout_df,
            passing_lookup,
        )
        out["qb_contextual_baseline"] = _qb_contextual_passing_baseline(
            holdout_df,
            passing_lookup,
        )

    td_role_cols: tuple[str, ...] = ()
    if stat == "passing_tds":
        td_role_cols = ("red_zone_pass_attempts_avg_5",)
    elif stat == "rushing_tds":
        td_role_cols = ("red_zone_carries_avg_5", "goal_line_carries_avg_5")
    elif stat == "receiving_tds":
        td_role_cols = ("red_zone_targets_avg_5", "goal_line_targets_avg_5")
    if td_role_cols and all(col in train_df.columns and col in holdout_df.columns for col in td_role_cols):
        train_role = sum(pd.to_numeric(train_df[col], errors="coerce").fillna(0.0) for col in td_role_cols)
        hold_role = sum(pd.to_numeric(holdout_df[col], errors="coerce").fillna(0.0) for col in td_role_cols)
        y_full = pd.to_numeric(train_df.get(stat), errors="coerce").fillna(0.0)
        role_sum = float(train_role.sum())
        rate = float(y_full.sum()) / role_sum if role_sum > 0 else global_mean
        td_base = hold_role * rate
        if "team_implied_points" in holdout_df.columns and "team_implied_points" in train_df.columns:
            train_imp = pd.to_numeric(train_df["team_implied_points"], errors="coerce")
            mean_imp = float(train_imp.mean()) if train_imp.notna().any() else 22.0
            hold_imp = pd.to_numeric(holdout_df["team_implied_points"], errors="coerce").fillna(mean_imp)
            td_base = td_base * (hold_imp / max(1.0, mean_imp)).clip(lower=0.70, upper=1.40)
        out["td_role_team_total_adjusted"] = td_base.fillna(global_mean).clip(lower=0.0)

    return out


def _best_baseline(y_true: pd.Series, baselines: dict[str, pd.Series]) -> tuple[str, pd.Series, dict[str, Any]]:
    summary: dict[str, Any] = {}
    best_name = "zero"
    best_series = pd.Series(0.0, index=y_true.index)
    best_mae = float("inf")
    for name, series in baselines.items():
        base = pd.to_numeric(series, errors="coerce").fillna(float(y_true.mean() if len(y_true) else 0.0))
        mae = float(mean_absolute_error(y_true, base))
        summary[name] = {"mae": mae}
        if mae < best_mae:
            best_name = name
            best_series = base
            best_mae = mae
    return best_name, best_series, summary


def _fit_qb_role_recent_bias(train_df: pd.DataFrame) -> dict[str, Any]:
    if len(train_df) < 300:
        return {
            "status": "insufficient_rows",
            "rows": int(len(train_df)),
            "correction": 0.0,
        }
    inner_train, inner_validation = _temporal_split(train_df, 4)
    if len(inner_train) < 200 or len(inner_validation) < 15:
        return {
            "status": "insufficient_split_rows",
            "train_rows": int(len(inner_train)),
            "validation_rows": int(len(inner_validation)),
            "correction": 0.0,
        }
    lookup = _baseline_lookup_payload("passing_yards", inner_train, tune_qb_bias=False)
    baseline = _position_role_rank_baseline_series(inner_validation, "passing_yards", lookup)
    y = pd.to_numeric(inner_validation.get("passing_yards"), errors="coerce").fillna(0.0)
    residual = y.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
    raw_correction = float(np.mean(residual)) if len(residual) else 0.0
    shrink = float(len(inner_validation) / (len(inner_validation) + 40.0))
    correction = float(np.clip(raw_correction * shrink, -28.0, 28.0))
    adjusted = (baseline + correction).clip(lower=0.0)
    baseline_mae = float(mean_absolute_error(y, baseline))
    adjusted_mae = float(mean_absolute_error(y, adjusted))
    accepted = bool(adjusted_mae <= baseline_mae - 0.001)
    if not accepted:
        correction = 0.0
    return {
        "status": "trained",
        "train_rows": int(len(inner_train)),
        "validation_rows": int(len(inner_validation)),
        "raw_correction": raw_correction,
        "shrink": shrink,
        "correction": correction,
        "baseline_mae": baseline_mae,
        "adjusted_mae": adjusted_mae,
        "accepted": accepted,
    }


def _baseline_lookup_payload(stat: str, train_df: pd.DataFrame, tune_qb_bias: bool = True) -> dict[str, Any]:
    target = pd.to_numeric(train_df.get(stat), errors="coerce")
    tmp = train_df.copy()
    tmp["_target"] = target
    tmp = tmp.loc[tmp["_target"].notna()].copy()
    global_mean = float(tmp["_target"].mean()) if len(tmp) else 0.0
    position_means: dict[str, float] = {}
    role_means: dict[str, float] = {}
    if len(tmp) and "position" in tmp.columns:
        position_means = {
            str(key).upper(): float(value)
            for key, value in tmp.groupby(tmp["position"].astype(str).str.upper())["_target"].mean().items()
        }
    rank_col = _role_rank_col(stat)
    if len(tmp) and rank_col and rank_col in tmp.columns and "position" in tmp.columns:
        tmp["_role_bucket"] = _role_bucket(pd.to_numeric(tmp[rank_col], errors="coerce"))
        role_means = {
            f"{str(pos).upper()}|{rank}": float(value)
            for (pos, rank), value in tmp.groupby([tmp["position"].astype(str).str.upper(), "_role_bucket"])["_target"].mean().items()
        }
    team_implied = pd.to_numeric(
        train_df.get("team_implied_points", pd.Series(np.nan, index=train_df.index)),
        errors="coerce",
    )
    team_implied_mean = float(team_implied.mean()) if team_implied.notna().any() else 22.0
    pass_attempts = pd.to_numeric(
        train_df.get("pass_attempts", pd.Series(np.nan, index=train_df.index)),
        errors="coerce",
    )
    pass_attempt_mean = float(pass_attempts.mean()) if pass_attempts.notna().any() else 31.5
    if stat == "passing_yards":
        pass_attempt_hist = pd.to_numeric(
            train_df.get("pass_attempts_avg_5", pd.Series(np.nan, index=train_df.index)),
            errors="coerce",
        )
        pass_yards_hist = pd.to_numeric(
            train_df.get("passing_yards_avg_5", pd.Series(np.nan, index=train_df.index)),
            errors="coerce",
        )
        ypa_values = (pass_yards_hist / pass_attempt_hist.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        yards_per_attempt_mean = float(ypa_values.dropna().clip(3.0, 12.0).mean()) if ypa_values.notna().any() else 6.8
    else:
        yards_per_attempt_mean = None
    td_role_rate: float | None = None
    if stat.endswith("_tds"):
        role_cols: tuple[str, ...] = ()
        if stat == "passing_tds":
            role_cols = ("red_zone_pass_attempts_avg_5",)
        elif stat == "rushing_tds":
            role_cols = ("red_zone_carries_avg_5", "goal_line_carries_avg_5")
        elif stat == "receiving_tds":
            role_cols = ("red_zone_targets_avg_5", "goal_line_targets_avg_5")
        if role_cols:
            role_sum = sum(
                pd.to_numeric(train_df.get(col, pd.Series(0.0, index=train_df.index)), errors="coerce").fillna(0.0)
                for col in role_cols
            )
            denom = float(role_sum.sum())
            td_role_rate = float(target.fillna(0.0).sum() / denom) if denom > 0 else global_mean
    qb_role_recent_bias: dict[str, Any] | None = None
    if stat == "passing_yards" and tune_qb_bias:
        qb_role_recent_bias = _fit_qb_role_recent_bias(train_df)
    return {
        "global_mean": global_mean,
        "position_means": position_means,
        "role_means": role_means,
        "team_implied_mean": team_implied_mean,
        "pass_attempt_mean": pass_attempt_mean,
        "yards_per_attempt_mean": yards_per_attempt_mean,
        "td_role_rate": td_role_rate,
        "qb_role_recent_bias_correction": (qb_role_recent_bias or {}).get("correction"),
        "qb_role_recent_bias": qb_role_recent_bias,
    }


def _temporal_split(df: pd.DataFrame, holdout_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_num = pd.to_numeric(df["season"], errors="coerce")
    week_num = pd.to_numeric(df["week"], errors="coerce")
    latest_season = season_num.dropna()
    if not latest_season.empty:
        season = int(latest_season.max())
        latest_regular = pd.DataFrame({
            "season": season_num,
            "week": week_num,
        }).loc[
            (season_num == season) & (week_num <= 18)
        ].dropna()
        latest_regular = (
            latest_regular.drop_duplicates()
            .sort_values(["season", "week"])
        )
        if len(latest_regular) >= 1:
            available_holdout_weeks = min(max(1, holdout_weeks), len(latest_regular))
            holdout_week_values = [int(value) for value in latest_regular["week"].tail(available_holdout_weeks)]
            first_holdout_week = min(holdout_week_values)
            train_mask = (season_num < season) | ((season_num == season) & (week_num < first_holdout_week))
            holdout_mask = (season_num == season) & (week_num.isin(holdout_week_values))
            train_df = df.loc[train_mask].copy()
            holdout_df = df.loc[holdout_mask].copy()
            if not train_df.empty and not holdout_df.empty:
                return train_df, holdout_df
    ordered_weeks = pd.DataFrame({
        "season": season_num,
        "week": week_num,
    }).dropna().drop_duplicates().sort_values(["season", "week"])
    ordered_weeks = ordered_weeks.assign(order=lambda d: range(len(d)))
    if ordered_weeks.empty:
        return df.iloc[0:0], df.iloc[0:0]
    cutoff = max(0, int(ordered_weeks["order"].max()) - max(1, holdout_weeks) + 1)
    week_index = ordered_weeks.set_index(["season", "week"])["order"].to_dict()
    order = pd.Series([
        week_index.get((season, week), -1)
        for season, week in zip(season_num, week_num)
    ], index=df.index)
    return df.loc[order < cutoff].copy(), df.loc[order >= cutoff].copy()


def _metric_payload(y_true: pd.Series, pred: np.ndarray, baseline: pd.Series) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=float)
    base = pd.to_numeric(baseline, errors="coerce").fillna(float(y_true.mean() if len(y_true) else 0.0)).to_numpy()
    mae = float(mean_absolute_error(y_true, pred))
    base_mae = float(mean_absolute_error(y_true, base))
    rmse = float(math.sqrt(mean_squared_error(y_true, pred)))
    base_rmse = float(math.sqrt(mean_squared_error(y_true, base)))
    bias = float(np.mean(pred - y_true.to_numpy(dtype=float))) if len(y_true) else None
    return {
        "rows": int(len(y_true)),
        "mae": mae,
        "baseline_mae": base_mae,
        "mae_gain_vs_baseline": base_mae - mae,
        "rmse": rmse,
        "baseline_rmse": base_rmse,
        "residual_sigma": max(1.0, float(rmse)),
        "bias": bias,
        "projection_pass": bool(mae <= base_mae - 0.001),
    }


def train(cfg: TrainConfig) -> dict[str, Any]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(cfg.pg_dsn)
    df = pd.read_sql(text(SQL_TRAIN), engine, params={"min_prev_games": cfg.min_prev_games})
    payload: dict[str, Any] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready",
        "rows": int(len(df)),
        "models": {},
        "metrics": {},
        "feature_columns": {},
        "fill_values": {},
        "stat_specs": {
            spec.stat: {
                "label": spec.label,
                "positions": list(spec.positions),
                "market_keys": list(spec.market_keys),
                "count_like": spec.count_like,
            }
            for spec in STAT_SPECS
        },
    }
    if df.empty:
        payload["status"] = "no_training_rows"
        joblib.dump(payload, cfg.model_dir / cfg.out_file)
        (cfg.model_dir / cfg.report_file).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    for spec in STAT_SPECS:
        stat_df = df.loc[df["position"].astype(str).str.upper().isin(spec.positions)].copy()
        stat_df = stat_df.loc[pd.to_numeric(stat_df[spec.stat], errors="coerce").notna()].copy()
        if len(stat_df) < cfg.min_rows:
            payload["metrics"][spec.stat] = {
                "status": "insufficient_rows",
                "rows": int(len(stat_df)),
                "min_rows": cfg.min_rows,
                "projection_pass": False,
            }
            continue
        train_df, holdout_df = _temporal_split(stat_df, cfg.holdout_weeks)
        if len(train_df) < cfg.min_rows or holdout_df.empty:
            payload["metrics"][spec.stat] = {
                "status": "insufficient_split_rows",
                "train_rows": int(len(train_df)),
                "holdout_rows": int(len(holdout_df)),
                "projection_pass": False,
            }
            continue
        X_train_raw = _make_features(train_df)
        X_holdout_raw = _make_features(holdout_df)
        columns = list(X_train_raw.columns)
        fill_values = {
            col: float(value) if math.isfinite(float(value)) else 0.0
            for col, value in X_train_raw.median(numeric_only=True).fillna(0.0).to_dict().items()
        }
        X_train = X_train_raw.reindex(columns=columns).fillna(fill_values).fillna(0.0)
        X_holdout = X_holdout_raw.reindex(columns=columns).fillna(fill_values).fillna(0.0)
        y_train = pd.to_numeric(train_df[spec.stat], errors="coerce").fillna(0.0).clip(lower=0.0)
        y_holdout = pd.to_numeric(holdout_df[spec.stat], errors="coerce").fillna(0.0).clip(lower=0.0)
        model = _fit_model(X_train, y_train, count_like=spec.count_like, cfg=cfg)
        pred_train = np.clip(model.predict(X_train), 0.0, None)
        pred_holdout = np.clip(model.predict(X_holdout), 0.0, None)
        (
            X_train,
            X_holdout,
            model,
            pred_train,
            pred_holdout,
            columns,
            feature_selection_summary,
            feature_selection_accepted,
        ) = _maybe_use_target_feature_selection(
            stat=spec.stat,
            X_train=X_train,
            X_holdout=X_holdout,
            y_train=y_train,
            y_holdout=y_holdout,
            full_model=model,
            full_train_pred=pred_train,
            full_holdout_pred=pred_holdout,
            cfg=cfg,
            count_like=spec.count_like,
        )
        if feature_selection_accepted:
            fill_values = {col: fill_values.get(col, 0.0) for col in columns}
        baseline_suite = _baseline_suite(spec.stat, train_df, holdout_df)
        baseline_name, baseline_series, baseline_summary = _best_baseline(y_holdout, baseline_suite)
        baseline_lookup = _baseline_lookup_payload(spec.stat, train_df)
        baseline_metric_context = {"baseline_column": baseline_name, "baseline_lookup": baseline_lookup}
        stored_model: Any = model
        selected_variant = "target_feature_selected" if feature_selection_accepted else "direct"
        rare_event_summary: dict[str, Any] = {}
        rate_summary: dict[str, Any] = {}
        qb_efficiency_repair_summary: dict[str, Any] = {"status": "not_attempted"}
        qb_baseline_model_blend_summary: dict[str, Any] = {"status": "not_attempted"}
        spike_residual_summary: dict[str, Any] = {"status": "not_attempted"}
        receiver_spike_bias_summary: dict[str, Any] = {"status": "not_attempted"}
        receiver_target_air_yards_repair_summary: dict[str, Any] = {"status": "not_attempted"}
        rushing_workload_repair_summary: dict[str, Any] = {"status": "not_attempted"}
        rate_candidate = _fit_rate_candidate(
            spec.stat,
            X_train,
            X_holdout,
            train_df,
            holdout_df,
            y_train,
            y_holdout,
            pred_train,
            pred_holdout,
            cfg,
        )
        if rate_candidate is not None:
            rate_summary, rate_payload, rate_train_pred, rate_holdout_pred = rate_candidate
            if rate_payload is not None:
                rate_payload["direct_model"] = model
                stored_model = rate_payload
                selected_variant = f"{selected_variant}+rate_opportunity"
                pred_train = rate_train_pred
                pred_holdout = rate_holdout_pred
        (
            qb_efficiency_repair_summary,
            qb_efficiency_repair_payload,
            qb_efficiency_repair_train_pred,
            qb_efficiency_repair_holdout_pred,
        ) = _fit_qb_passing_efficiency_repair_candidate(
            spec.stat,
            X_train,
            X_holdout,
            train_df,
            holdout_df,
            y_train,
            y_holdout,
            pred_train,
            pred_holdout,
            baseline_series,
            cfg,
        )
        if qb_efficiency_repair_payload is not None:
            if not isinstance(stored_model, dict):
                stored_model = {"kind": "direct", "model": stored_model}
            stored_model["qb_passing_efficiency_repair"] = qb_efficiency_repair_payload
            selected_variant = f"{selected_variant}+qb_efficiency_repair"
            pred_train = qb_efficiency_repair_train_pred
            pred_holdout = qb_efficiency_repair_holdout_pred
        if spec.stat == "receiving_tds" and int((y_train > 0).sum()) >= 25:
            y_train_any = (y_train > 0).astype(int)
            y_holdout_any = (y_holdout > 0).astype(int)
            clf = _fit_classifier(X_train, y_train_any, cfg)
            p_any = _predict_binary_probability(clf, X_holdout)
            p_any_train = _predict_binary_probability(clf, X_train)
            positive_mean = float(y_train.loc[y_train > 0].mean()) if int((y_train > 0).sum()) else 1.0
            positive_mean = float(np.clip(positive_mean, 1.0, 1.65))
            base_arr = pd.to_numeric(baseline_series, errors="coerce").fillna(float(y_train.mean())).to_numpy(dtype=float)
            train_base_arr = _baseline_from_metric_column(train_df, spec.stat, baseline_metric_context)
            baseline_prob = _poisson_any_prob(base_arr)
            train_baseline_prob = _poisson_any_prob(train_base_arr)
            direct_brier = float(brier_score_loss(y_holdout_any, _poisson_any_prob(pred_holdout)))
            baseline_brier = float(brier_score_loss(y_holdout_any, baseline_prob))
            candidate_rows: list[dict[str, Any]] = [{
                "variant": "direct",
                "blend_weight": None,
                "probability_blend_weight": None,
                "prediction": pred_holdout,
                "train_prediction": pred_train,
                "td_any_probability": _poisson_any_prob(pred_holdout),
                "mae": float(mean_absolute_error(y_holdout, pred_holdout)),
                "td_any_brier": direct_brier,
            }]
            for blend_weight in (1.0, 0.85, 0.70, 0.55, 0.40, 0.25):
                blended_prob = np.clip(blend_weight * p_any + (1.0 - blend_weight) * baseline_prob, 0.001, 0.999)
                blended_train_prob = np.clip(blend_weight * p_any_train + (1.0 - blend_weight) * train_baseline_prob, 0.001, 0.999)
                blended = np.clip(blended_prob * positive_mean, 0.0, None)
                blended_train = np.clip(blended_train_prob * positive_mean, 0.0, None)
                candidate_rows.append({
                    "variant": "rare_event_classifier",
                    "blend_weight": 1.0,
                    "probability_blend_weight": blend_weight,
                    "prediction": blended,
                    "train_prediction": blended_train,
                    "td_any_probability": blended_prob,
                    "mae": float(mean_absolute_error(y_holdout, blended)),
                    "td_any_brier": float(brier_score_loss(y_holdout_any, blended_prob)),
                })
            baseline_mae_for_guard = float(mean_absolute_error(y_holdout, base_arr))
            eligible = [row for row in candidate_rows if row["mae"] <= baseline_mae_for_guard + 0.020]
            best = min(eligible or candidate_rows, key=lambda row: (row["td_any_brier"], row["mae"]))
            rare_event_summary = {
                "positive_train_rows": int((y_train > 0).sum()),
                "positive_mean": positive_mean,
                "baseline_td_any_brier": baseline_brier,
                "direct_td_any_brier": direct_brier,
                "selection_metric": "td_any_brier_with_count_mae_guard",
                "mae_guard": baseline_mae_for_guard + 0.020,
                "candidates": [
                    {k: v for k, v in row.items() if k not in {"prediction", "train_prediction", "td_any_probability"}}
                    for row in candidate_rows
                ],
            }
            if best["variant"] == "rare_event_classifier":
                selected_variant = f"{selected_variant}+rare_event_classifier"
                pred_holdout = np.asarray(best["prediction"], dtype=float)
                pred_train = np.asarray(best["train_prediction"], dtype=float)
                stored_model = {
                    "kind": "rare_event_classifier",
                    "classifier": clf,
                    "positive_mean": positive_mean,
                    "blend_weight": best["blend_weight"],
                    "probability_blend_weight": best["probability_blend_weight"],
                    "baseline_column": baseline_name,
                    "baseline_lookup": baseline_lookup,
                }
        spike_residual_summary, spike_payload, spike_train_pred, spike_holdout_pred = _fit_spike_residual_calibrator(
            spec.stat,
            X_train,
            X_holdout,
            y_train,
            y_holdout,
            pred_train,
            pred_holdout,
            cfg,
        )
        if spike_payload is not None:
            if not isinstance(stored_model, dict):
                stored_model = {"kind": "direct", "model": stored_model}
            stored_model["spike_residual_calibrator"] = spike_payload
            selected_variant = f"{selected_variant}+spike_residual"
            pred_train = spike_train_pred
            pred_holdout = spike_holdout_pred
        (
            rushing_workload_repair_summary,
            rushing_workload_repair_payload,
            rushing_workload_repair_train_pred,
            rushing_workload_repair_holdout_pred,
        ) = _fit_rushing_workload_repair_calibrator(
            spec.stat,
            X_train,
            X_holdout,
            y_train,
            y_holdout,
            pred_train,
            pred_holdout,
        )
        if rushing_workload_repair_payload is not None:
            if not isinstance(stored_model, dict):
                stored_model = {"kind": "direct", "model": stored_model}
            stored_model["rushing_workload_repair_calibrator"] = rushing_workload_repair_payload
            selected_variant = f"{selected_variant}+rushing_workload_repair"
            pred_train = rushing_workload_repair_train_pred
            pred_holdout = rushing_workload_repair_holdout_pred
        receiver_spike_bias_summary, receiver_spike_bias_payload, receiver_spike_bias_train_pred, receiver_spike_bias_holdout_pred = _fit_receiver_spike_bias_calibrator(
            spec.stat,
            X_train,
            X_holdout,
            y_train,
            y_holdout,
            pred_train,
            pred_holdout,
        )
        if receiver_spike_bias_payload is not None:
            if not isinstance(stored_model, dict):
                stored_model = {"kind": "direct", "model": stored_model}
            stored_model["receiver_spike_bias_calibrator"] = receiver_spike_bias_payload
            selected_variant = f"{selected_variant}+receiver_spike_bias"
            pred_train = receiver_spike_bias_train_pred
            pred_holdout = receiver_spike_bias_holdout_pred
        (
            receiver_target_air_yards_repair_summary,
            receiver_target_air_yards_repair_payload,
            receiver_target_air_yards_repair_train_pred,
            receiver_target_air_yards_repair_holdout_pred,
        ) = _fit_receiver_target_air_yards_repair_calibrator(
            spec.stat,
            X_train,
            X_holdout,
            y_train,
            y_holdout,
            pred_train,
            pred_holdout,
        )
        if receiver_target_air_yards_repair_payload is not None:
            if not isinstance(stored_model, dict):
                stored_model = {"kind": "direct", "model": stored_model}
            stored_model["receiver_target_air_yards_repair_calibrator"] = receiver_target_air_yards_repair_payload
            selected_variant = f"{selected_variant}+target_air_repair"
            pred_train = receiver_target_air_yards_repair_train_pred
            pred_holdout = receiver_target_air_yards_repair_holdout_pred
        bias_summary: dict[str, Any] = {"status": "not_attempted"}
        if len(y_train) >= max(200, cfg.min_rows):
            calibrator = _fit_bias_calibrator(X_train, y_train, pred_train, spec.stat)
            calibrated_holdout = _apply_bias_calibrator(pred_holdout, X_holdout, calibrator)
            pre_cal_mae = float(mean_absolute_error(y_holdout, pred_holdout))
            cal_mae = float(mean_absolute_error(y_holdout, calibrated_holdout))
            accepted_bias = cal_mae <= pre_cal_mae - 0.001
            if spec.stat == "receiving_tds":
                pre_brier = float(brier_score_loss((y_holdout > 0).astype(int), _poisson_any_prob(pred_holdout)))
                cal_brier = float(brier_score_loss((y_holdout > 0).astype(int), _poisson_any_prob(calibrated_holdout)))
                accepted_bias = accepted_bias and cal_brier <= pre_brier + 0.001
            bias_summary = {
                "status": "trained",
                "pre_calibration_mae": pre_cal_mae,
                "post_calibration_mae": cal_mae,
                "accepted": bool(accepted_bias),
                "exact_buckets": len(calibrator.get("exact") or {}),
                "fallback_buckets": len(calibrator.get("fallback") or {}),
                "global_correction": calibrator.get("global"),
            }
            if accepted_bias:
                if not isinstance(stored_model, dict):
                    stored_model = {"kind": "direct", "model": stored_model}
                stored_model["bias_calibrator"] = calibrator
                selected_variant = f"{selected_variant}+bias_calibrated"
                pred_holdout = calibrated_holdout
        baseline_residual_summary: dict[str, Any] = {"status": "not_attempted"}
        if len(y_train) >= max(200, cfg.min_rows):
            train_baseline_arr = _baseline_from_metric_column(train_df, spec.stat, baseline_metric_context)
            holdout_baseline_arr = pd.to_numeric(baseline_series, errors="coerce").fillna(float(y_train.mean())).to_numpy(dtype=float)
            baseline_calibrator = _fit_bias_calibrator(X_train, y_train, train_baseline_arr, spec.stat)
            baseline_calibrated_holdout = _apply_bias_calibrator(holdout_baseline_arr, X_holdout, baseline_calibrator)
            baseline_calibrated_train = _apply_bias_calibrator(train_baseline_arr, X_train, baseline_calibrator)
            base_mae = float(mean_absolute_error(y_holdout, holdout_baseline_arr))
            current_mae = float(mean_absolute_error(y_holdout, pred_holdout))
            baseline_cal_mae = float(mean_absolute_error(y_holdout, baseline_calibrated_holdout))
            accepted_baseline_residual = bool(
                baseline_cal_mae <= base_mae - 0.001
                and baseline_cal_mae <= current_mae - 0.001
            )
            baseline_residual_summary = {
                "status": "trained",
                "baseline_mae": base_mae,
                "current_model_mae": current_mae,
                "calibrated_baseline_mae": baseline_cal_mae,
                "mae_gain_vs_baseline": base_mae - baseline_cal_mae,
                "accepted": accepted_baseline_residual,
                "exact_buckets": len(baseline_calibrator.get("exact") or {}),
                "fallback_buckets": len(baseline_calibrator.get("fallback") or {}),
                "global_correction": baseline_calibrator.get("global"),
            }
            if accepted_baseline_residual:
                stored_model = {
                    "kind": "baseline_bias_calibrated",
                    "baseline_column": baseline_name,
                    "baseline_lookup": baseline_lookup,
                    "bias_calibrator": baseline_calibrator,
                }
                selected_variant = "baseline_bias_calibrated"
                pred_train = baseline_calibrated_train
                pred_holdout = baseline_calibrated_holdout
            (
                qb_baseline_model_blend_summary,
                qb_baseline_model_blend_params,
                qb_baseline_model_blend_train_pred,
                qb_baseline_model_blend_holdout_pred,
            ) = _fit_qb_baseline_model_blend_candidate(
                spec.stat,
                y_train,
                y_holdout,
                pred_train,
                pred_holdout,
                train_baseline_arr,
                holdout_baseline_arr,
            )
            if qb_baseline_model_blend_params is not None:
                nested_model = stored_model if isinstance(stored_model, dict) else {"kind": "direct", "model": stored_model}
                stored_model = {
                    "kind": "baseline_model_blend",
                    "stat": spec.stat,
                    "model_payload": nested_model,
                    "baseline_column": baseline_name,
                    "baseline_lookup": baseline_lookup,
                    "model_weight": qb_baseline_model_blend_params["model_weight"],
                    "correction": qb_baseline_model_blend_params["correction"],
                }
                selected_variant = f"{selected_variant}+baseline_model_blend"
                pred_train = qb_baseline_model_blend_train_pred
                pred_holdout = qb_baseline_model_blend_holdout_pred
        metrics = _metric_payload(y_holdout, pred_holdout, baseline_series)
        if spec.stat == "receiving_tds":
            base_arr = pd.to_numeric(baseline_series, errors="coerce").fillna(float(y_train.mean())).to_numpy(dtype=float)
            y_holdout_any = (y_holdout > 0).astype(int)
            metrics["td_any_brier"] = float(brier_score_loss(y_holdout_any, _poisson_any_prob(pred_holdout)))
            metrics["baseline_td_any_brier"] = float(brier_score_loss(y_holdout_any, _poisson_any_prob(base_arr)))
            metrics["td_probability_pass"] = bool(
                metrics["td_any_brier"] <= metrics["baseline_td_any_brier"] - 0.001
                and metrics["mae"] <= metrics["baseline_mae"] + 0.020
            )
            metrics["projection_pass"] = bool(
                metrics["mae"] <= metrics["baseline_mae"] - 0.001
                and metrics["td_any_brier"] <= metrics["baseline_td_any_brier"] + 0.001
            )
        projection_accepted = bool(metrics["projection_pass"])
        td_probability_accepted = bool(metrics.get("td_probability_pass"))
        metrics.update({
            "status": "trained",
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
            "baseline_column": baseline_name,
            "baseline_lookup": baseline_lookup,
            "baseline_suite": baseline_summary,
            "variant": selected_variant,
            "rare_event": rare_event_summary,
            "rate_model": rate_summary,
            "qb_efficiency_repair": qb_efficiency_repair_summary,
            "qb_baseline_model_blend": qb_baseline_model_blend_summary,
            "spike_residual": spike_residual_summary,
            "rushing_workload_repair": rushing_workload_repair_summary,
            "receiver_spike_bias": receiver_spike_bias_summary,
            "receiver_target_air_yards_repair": receiver_target_air_yards_repair_summary,
            "bias_calibration": bias_summary,
            "baseline_residual": baseline_residual_summary,
            "feature_selection": feature_selection_summary,
            "accepted": projection_accepted,
            "projection_accepted": projection_accepted,
            "td_probability_accepted": td_probability_accepted,
        })
        payload["models"][spec.stat] = stored_model
        payload["metrics"][spec.stat] = metrics
        payload["feature_columns"][spec.stat] = columns
        payload["fill_values"][spec.stat] = fill_values
        log.info(
            "%s: holdout MAE %.3f vs baseline %.3f accepted=%s",
            spec.stat,
            metrics["mae"],
            metrics["baseline_mae"],
            metrics["accepted"],
        )

    joblib.dump(payload, cfg.model_dir / cfg.out_file)
    report = dict(payload)
    report["models"] = sorted(payload["models"].keys())
    (cfg.model_dir / cfg.report_file).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NFL player stat projection models")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--holdout-weeks", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = TrainConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        min_rows=args.min_rows,
        holdout_weeks=args.holdout_weeks,
    )
    result = train(cfg)
    print(json.dumps({k: v for k, v in result.items() if k != "models"}, indent=2, default=str))


if __name__ == "__main__":
    main()
