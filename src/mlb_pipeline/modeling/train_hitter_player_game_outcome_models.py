"""Train hitter player-game opportunity and outcome models.

These models validate the projection layer before betting-market logic gets a
vote.  They learn from one row per hitter/game:

* PA opportunity
* per-PA singles, doubles, triples, HR, and walk rates
* HR any-game rare-event probability

The artifact is diagnostic until holdout metrics prove it beats simple lineup
slot priors and existing prop projections.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psycopg2
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .build_hitter_player_game_training_table import (
    HitterPlayerGameTrainingConfig,
    refresh_hitter_player_game_training,
)
from mlb_pipeline.atomic_io import atomic_write_via
from .model_release import HITTER_CHALLENGER_ARTIFACT, ensure_hitter_production_artifact, hitter_model_release_id

from mlb_pipeline.db import PG_DSN as _PG_DSN
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

NUMERIC_FEATURES = [
    "lineup_slot",
    "confirmed_starter_num",
    "is_home",
    "team_implied_runs",
    "opponent_implied_runs",
    "game_total_line",
    "park_run_factor",
    "park_hr_factor",
    "park_babip_factor",
    "temperature_f",
    "wind_speed_mph",
    "wind_sin",
    "wind_cos",
    "precip_prob_pct",
    "is_dome",
    "is_day_game",
    "weather_pregame_flag",
    "own_lineup_xwoba_avg",
    "own_lineup_xslg_avg",
    "own_lineup_barrel_avg",
    "own_lineup_hard_hit_avg",
    "own_lineup_k_pct_cv",
    "own_lineup_pct_lhb",
    "lineup_confirmed_flag",
    "confirmed_team_lineup_slots",
    "team_lineup_confirmed_flag",
    "lineup_boxscore_proxy_flag",
    "lineup_slot_x_team_implied_runs",
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
    "batter_sc_barrel_rate",
    "batter_sc_hard_hit_pct",
    "batter_sc_avg_exit_velo",
    "batter_sc_avg_launch_angle",
    "batter_sc_sweet_spot_pct",
    "batter_sc_fb_pct",
    "batter_sc_gb_pct",
    "batter_sc_ld_pct",
    "batter_sc_xba",
    "batter_sc_xslg",
    "batter_sc_xwoba",
    "batter_sc_xiso",
    "batter_sc_brl_pa",
    "batter_sprint_speed",
    "batter_disc_oz_swing_pct",
    "batter_disc_iz_contact_pct",
    "batter_disc_oz_contact_pct",
    "batter_disc_whiff_pct",
    "batter_disc_out_zone_pct",
    "batter_disc_k_pct",
    "batter_disc_bb_pct",
    "opp_sp_sc_barrel_rate",
    "opp_sp_sc_hard_hit_pct",
    "opp_sp_sc_avg_exit_velo",
    "opp_sp_sc_avg_launch_angle",
    "opp_sp_sc_xba",
    "opp_sp_sc_xslg",
    "opp_sp_sc_xwoba",
    "opp_sp_sc_xiso",
    "opp_sp_fb_pct",
    "opp_sp_fb_hard_hit_pct",
    "opp_sp_fb_xwoba",
    "opp_sp_fb_run_value_per_100",
    "opp_sp_fb_whiff_pct",
    "opp_sp_fb_k_pct",
    "opp_sp_si_pct",
    "opp_sp_si_hard_hit_pct",
    "opp_sp_si_xwoba",
    "opp_sp_si_whiff_pct",
    "opp_sp_si_k_pct",
    "opp_sp_sl_pct",
    "opp_sp_sl_hard_hit_pct",
    "opp_sp_sl_xwoba",
    "opp_sp_sl_run_value_per_100",
    "opp_sp_sl_whiff_pct",
    "opp_sp_sl_k_pct",
    "opp_sp_ch_pct",
    "opp_sp_ch_hard_hit_pct",
    "opp_sp_ch_xwoba",
    "opp_sp_ch_run_value_per_100",
    "opp_sp_ch_whiff_pct",
    "opp_sp_ch_k_pct",
    "opp_sp_fastball_family_pct",
    "opp_sp_pitch_diversity",
    "projected_pa",
    "pa_games",
    "lineup_slot_pa_prior",
    "lineup_slot_low_pa_prior",
    "top_order_flag",
    "middle_order_flag",
    "bottom_order_flag",
    "projected_pa_slot_delta",
    "projected_pa_x_slot_prior",
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
    "catcher_flag",
    "catcher_low_pa_risk",
    "platoon_same_hand_flag",
    "platoon_advantage_flag",
    "batter_hand_known_flag",
    "opp_sp_hand_known_flag",
    "player_prior_pa",
    "player_prior_hit_rate",
    "player_prior_xbh_given_hit",
    "player_prior_hr_given_xbh",
    "player_prior_triple_given_non_hr_xbh",
    "player_prior_walk_given_non_hit",
]

CATEGORICAL_FEATURES = [
    "team_abbr",
    "opponent_abbr",
    "lineup_source",
    "starter_status_source",
    "primary_position",
    "batter_hand",
    "opp_sp_hand",
]

RATE_TARGETS = {
    "single_rate": "actual_singles",
    "double_rate": "actual_doubles",
    "triple_rate": "actual_triples",
    "hr_rate": "actual_home_runs",
    "walk_rate": "actual_walks",
}

EVENT_CLASSES = ["out", "walk", "single", "double", "triple", "hr"]
TB_STATE_NAMES = ["tb_0", "tb_1", "tb_2_3", "tb_4_plus_hr", "tb_4_plus_non_hr"]
_TB_STATE_HIERARCHICAL_HEADS = (
    "tb_positive",
    "tb_2_plus_given_positive",
    "tb_4_plus_given_2_plus",
    "tb_hr_given_4_plus",
)

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


@dataclass(frozen=True)
class HitterOutcomeModelConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    lookback_days: int = 365
    holdout_days: int = 28
    min_train_rows: int = 1000
    min_holdout_rows: int = 200
    min_prop_holdout_rows: int = 100
    rebuild_player_game_if_empty: bool = True
    fit_independent_boosted_candidate: bool = False
    report_file: str | None = None


SQL = """
SELECT *
FROM features.mlb_hitter_player_game_training
WHERE game_date_et >= %(cutoff)s
  AND actual_pa IS NOT NULL
ORDER BY game_date_et, game_slug, player_id
"""


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


def _row_count(conn) -> int:
    if not _table_exists(conn, "features", "mlb_hitter_player_game_training"):
        return 0
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM features.mlb_hitter_player_game_training")
        return int(cur.fetchone()[0] or 0)


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


def _load_head_feature_policy(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "hitter_event_feature_ablation.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    policy = payload.get("pruning_policy") or {}
    return policy if policy.get("status") == "ready" else {}


def _load(cfg: HitterOutcomeModelConfig) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if _row_count(conn) == 0:
            if not cfg.rebuild_player_game_if_empty:
                return pd.DataFrame()
            refresh_hitter_player_game_training(
                HitterPlayerGameTrainingConfig(
                    pg_dsn=cfg.pg_dsn,
                    lookback_days=cfg.lookback_days,
                )
            )
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    numeric = set(NUMERIC_FEATURES) | {
        "actual_pa",
        "actual_hits",
        "actual_singles",
        "actual_doubles",
        "actual_triples",
        "actual_home_runs",
        "actual_total_bases",
        "actual_walks",
        "model_pred_hits",
        "model_pred_total_bases",
        "model_pred_home_runs",
        "prop_example_rows",
    }
    lineup_slot = pd.to_numeric(df.get("lineup_slot"), errors="coerce")
    stored_starter = df["confirmed_starter"].fillna(False).astype(bool).astype(float)
    df["confirmed_starter_num"] = np.where(lineup_slot.between(1, 9), 1.0, stored_starter)
    for col in numeric:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col not in df:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _coalesce_column(df: pd.DataFrame, name: str, aliases: tuple[str, ...], default: Any = np.nan) -> pd.Series:
    if name in df:
        out = df[name].copy()
    else:
        out = pd.Series([default] * len(df), index=df.index)
    for alias in aliases:
        if alias in df:
            out = out.where(out.notna(), df[alias])
    return out


def _numeric_column(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    values = df[name] if name in df else pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(values, errors="coerce")


def prepare_hitter_outcome_features(
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    """Return a model-ready hitter outcome feature frame.

    Player-game training rows use names like ``lineup_slot`` while offer-level
    replay rows use names like ``confirmed_batting_order``.  This adapter keeps
    the direct event model usable in both places without duplicating feature
    mapping logic.
    """
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    out = df.copy()
    out["lineup_slot"] = _coalesce_column(out, "lineup_slot", ("confirmed_batting_order",))
    out["lineup_source"] = _coalesce_column(out, "lineup_source", ("confirmed_lineup_source",), "unknown")
    alias_map: dict[str, tuple[str, ...]] = {
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
        "batter_sc_brl_pa": ("sc_brl_pa",),
        "batter_sprint_speed": ("sprint_speed",),
        "batter_disc_oz_swing_pct": ("sc_b_oz_swing_pct",),
        "batter_disc_iz_contact_pct": ("sc_b_iz_contact_pct",),
        "batter_disc_oz_contact_pct": ("sc_b_oz_contact_pct",),
        "batter_disc_whiff_pct": ("sc_b_disc_whiff_pct",),
        "batter_disc_out_zone_pct": ("sc_b_out_zone_pct",),
        "batter_disc_k_pct": ("sc_b_k_pct",),
        "batter_disc_bb_pct": ("sc_b_bb_pct",),
    }
    for canonical, aliases in alias_map.items():
        out[canonical] = _coalesce_column(out, canonical, aliases)
    slot = _numeric_column(out, "lineup_slot")
    stored_starter = (
        out["confirmed_starter"].fillna(False).astype(bool).astype(float)
        if "confirmed_starter" in out
        else pd.Series(0.0, index=out.index)
    )
    out["confirmed_starter_num"] = np.where(slot.between(1, 9), 1.0, stored_starter)
    if "lineup_confirmed_flag" not in out:
        source = out.get("lineup_source", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
        out["lineup_confirmed_flag"] = source.str.contains("lineup|raw_lineups", case=False, regex=True).astype(float)
    if "lineup_boxscore_proxy_flag" not in out:
        source = out.get("lineup_source", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
        out["lineup_boxscore_proxy_flag"] = source.str.contains("boxscore", case=False, regex=False).astype(float)
    if "lineup_slot_x_team_implied_runs" not in out:
        team_runs = _numeric_column(out, "team_implied_runs")
        out["lineup_slot_x_team_implied_runs"] = slot * team_runs
    if "starter_status_source" not in out:
        src = out.get("lineup_source")
        out["starter_status_source"] = np.where(
            _numeric_column(out, "lineup_slot").between(1, 9),
            src.fillna("confirmed_or_projected_lineup") if isinstance(src, pd.Series) else "confirmed_or_projected_lineup",
            "unknown",
        )
    if "weather_pregame_flag" not in out:
        out["weather_pregame_flag"] = _numeric_column(out, "temperature_f").notna().astype(float)

    slot_i = slot.round().where(slot.between(1, 9)).astype("Int64")
    slot_prior = slot_i.map(_LINEUP_PA_PRIORS).astype("float64").fillna(4.05)
    low_pa_prior = slot_i.map(_LINEUP_LOW_PA_PRIORS).astype("float64").fillna(0.14)
    projected_pa = _numeric_column(out, "projected_pa")
    team_runs = _numeric_column(out, "team_implied_runs")
    opp_runs = _numeric_column(out, "opponent_implied_runs")
    total = _numeric_column(out, "game_total_line")
    is_home = _numeric_column(out, "is_home", 0.0).fillna(0.0).clip(0.0, 1.0)
    run_diff = team_runs - opp_runs
    abs_diff = run_diff.abs()
    favorite = (run_diff > 0.15).astype(float)
    underdog = (run_diff < -0.15).astype(float)
    close_game = (abs_diff <= 0.75).astype(float)
    blowout_risk = ((abs_diff - 1.25) / 2.25).clip(lower=0.0, upper=1.0)
    top_order = slot_i.between(1, 4).fillna(False).astype(float)
    middle_order = slot_i.between(5, 6).fillna(False).astype(float)
    bottom_order = slot_i.between(7, 9).fillna(False).astype(float)
    home_fav_penalty = (is_home * favorite * (0.45 + 0.55 * blowout_risk)).clip(0.0, 1.0)
    away_trailing_extra = ((1.0 - is_home) * underdog * (0.35 + 0.65 * close_game)).clip(0.0, 1.0)

    primary_pos = out.get("primary_position", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str).str.upper()
    catcher = primary_pos.str.contains(r"\bC\b|CATCHER", regex=True).astype(float)
    lineup_source = out.get("lineup_source", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str).str.lower()
    boxscore_proxy = lineup_source.str.contains("boxscore", regex=False).astype(float)

    batter_hand = out.get("batter_hand", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str).str.upper()
    opp_hand = out.get("opp_sp_hand", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str).str.upper()
    batter_known = batter_hand.isin(["L", "R", "S"]).astype(float)
    opp_known = opp_hand.isin(["L", "R"]).astype(float)
    same_hand = (
        ((batter_hand == "L") & (opp_hand == "L"))
        | ((batter_hand == "R") & (opp_hand == "R"))
    ).astype(float)
    platoon_adv = (
        ((batter_hand == "L") & (opp_hand == "R"))
        | ((batter_hand == "R") & (opp_hand == "L"))
        | (batter_hand == "S")
    ).astype(float)

    out["lineup_slot_pa_prior"] = slot_prior
    out["lineup_slot_low_pa_prior"] = low_pa_prior
    out["top_order_flag"] = top_order
    out["middle_order_flag"] = middle_order
    out["bottom_order_flag"] = bottom_order
    out["projected_pa_slot_delta"] = projected_pa - slot_prior
    out["projected_pa_x_slot_prior"] = projected_pa * slot_prior
    out["implied_run_diff"] = run_diff
    out["abs_implied_run_diff"] = abs_diff
    out["favorite_flag"] = favorite
    out["underdog_flag"] = underdog
    out["close_game_flag"] = close_game
    out["blowout_risk"] = blowout_risk
    out["high_total_flag"] = (total >= 9.0).astype(float)
    out["low_total_flag"] = (total <= 7.5).astype(float)
    out["home_favorite_ninth_penalty"] = home_fav_penalty
    out["away_trailing_extra_pa_chance"] = away_trailing_extra
    out["slot_prior_x_home_ninth_penalty"] = slot_prior * home_fav_penalty
    out["slot_prior_x_blowout_risk"] = slot_prior * blowout_risk
    out["projected_pa_x_implied_runs"] = projected_pa * team_runs
    out["projected_pa_x_close_game"] = projected_pa * close_game
    out["catcher_flag"] = catcher
    out["catcher_low_pa_risk"] = (catcher * (0.45 + 0.35 * boxscore_proxy + 0.20 * bottom_order)).clip(0.0, 1.0)
    out["platoon_same_hand_flag"] = same_hand
    out["platoon_advantage_flag"] = platoon_adv
    out["batter_hand_known_flag"] = batter_known
    out["opp_sp_hand_known_flag"] = opp_known

    data: dict[str, Any] = {}
    for col in numeric_features:
        data[col] = pd.to_numeric(out[col], errors="coerce") if col in out else pd.Series(np.nan, index=out.index)
    for col in categorical_features:
        data[col] = out[col].fillna("unknown").astype(str) if col in out else pd.Series("unknown", index=out.index)
    return pd.DataFrame(data, index=out.index)[numeric_features + categorical_features]


def _split(df: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = max(df["game_date_et"])
    split_date = max_date - timedelta(days=max(1, holdout_days - 1))
    train = df[df["game_date_et"] < split_date].copy()
    holdout = df[df["game_date_et"] >= split_date].copy()
    return train, holdout


def _one_hot_encoder(*, dense: bool = False) -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore", "min_frequency": 10}
    try:
        return OneHotEncoder(**kwargs, sparse_output=not dense)
    except TypeError:  # pragma: no cover - older sklearn
        return OneHotEncoder(**kwargs, sparse=not dense)


def _preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    *,
    dense: bool = False,
) -> ColumnTransformer:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                _one_hot_encoder(dense=dense),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0 if dense else 0.3,
    )


def _regression_pipeline(
    alpha: float = 8.0,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    return Pipeline([
        ("features", _preprocessor(numeric_features, categorical_features)),
        ("model", Ridge(alpha=alpha)),
    ])


def _boosted_rate_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    return Pipeline([
        ("features", _preprocessor(numeric_features, categorical_features)),
        ("model", LGBMRegressor(
            objective="cross_entropy",
            n_estimators=220,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=6,
            min_child_samples=90,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.15,
            reg_lambda=1.25,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )),
    ])


def _boosted_binary_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    return Pipeline([
        ("features", _preprocessor(numeric_features, categorical_features)),
        ("model", LGBMClassifier(
            objective="binary",
            n_estimators=260,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=6,
            min_child_samples=90,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.20,
            reg_lambda=1.50,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )),
    ])


def _boosted_count_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    return Pipeline([
        ("features", _preprocessor(numeric_features, categorical_features)),
        ("model", LGBMRegressor(
            objective="regression_l1",
            n_estimators=220,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=6,
            min_child_samples=90,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.15,
            reg_lambda=1.25,
            random_state=43,
            n_jobs=-1,
            verbosity=-1,
        )),
    ])


def _classifier_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    return Pipeline([
        ("features", _preprocessor(numeric_features, categorical_features)),
        ("model", LogisticRegression(
            max_iter=600,
            C=0.35,
            solver="saga",
            tol=1e-3,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _calibrated_hgb_classifier() -> Any:
    base = HistGradientBoostingClassifier(
        max_iter=90,
        learning_rate=0.055,
        max_leaf_nodes=17,
        min_samples_leaf=35,
        l2_regularization=0.08,
        early_stopping=True,
        random_state=42,
    )
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=2, ensemble=False)
    except TypeError:  # pragma: no cover - older sklearn
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=2)


def _boosted_classifier_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    return Pipeline([
        ("features", _preprocessor(numeric_features, categorical_features, dense=True)),
        ("model", _calibrated_hgb_classifier()),
    ])


def _clip(values: Any, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    return np.clip(arr, lower, upper)


def _rmse(y_true: Any, y_pred: Any) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _count_metrics(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(pred) & pd.notna(y_true).to_numpy()
    if not mask.any():
        return {"rows": 0}
    true = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)[mask]
    pred = pred[mask]
    return {
        "rows": int(mask.sum()),
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": _rmse(true, pred),
        "bias": float(np.mean(true - pred)),
    }


def _slot_prior(train: pd.DataFrame, holdout: pd.DataFrame, target: str) -> np.ndarray:
    overall = float(train[target].mean())
    by_slot = train.groupby(train["lineup_slot"].round())[target].mean().to_dict()
    vals = []
    for value in holdout["lineup_slot"]:
        slot = round(float(value)) if pd.notna(value) else None
        vals.append(float(by_slot.get(slot, overall)))
    return np.asarray(vals, dtype=float)


def _rate_prior(train: pd.DataFrame, holdout: pd.DataFrame, count_col: str, pa_pred: np.ndarray) -> np.ndarray:
    work = train[train["actual_pa"] > 0].copy()
    work["_rate"] = (work[count_col] / work["actual_pa"]).clip(lower=0.0, upper=1.0)
    overall = float(work["_rate"].mean())
    by_slot = work.groupby(work["lineup_slot"].round())["_rate"].mean().to_dict()
    rates = []
    for value in holdout["lineup_slot"]:
        slot = round(float(value)) if pd.notna(value) else None
        rates.append(float(by_slot.get(slot, overall)))
    return np.asarray(rates, dtype=float) * pa_pred


def _event_count_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pa = pd.to_numeric(df.get("actual_pa"), errors="coerce").fillna(0.0).clip(lower=0.0)
    walks = pd.to_numeric(df.get("actual_walks"), errors="coerce").fillna(0.0).clip(lower=0.0)
    singles = pd.to_numeric(df.get("actual_singles"), errors="coerce").fillna(0.0).clip(lower=0.0)
    doubles = pd.to_numeric(df.get("actual_doubles"), errors="coerce").fillna(0.0).clip(lower=0.0)
    triples = pd.to_numeric(df.get("actual_triples"), errors="coerce").fillna(0.0).clip(lower=0.0)
    hr = pd.to_numeric(df.get("actual_home_runs"), errors="coerce").fillna(0.0).clip(lower=0.0)
    non_out = walks + singles + doubles + triples + hr
    scale = pd.Series(1.0, index=df.index, dtype="float64")
    over = (non_out > pa) & (non_out > 0)
    scale.loc[over] = (pa.loc[over] / non_out.loc[over]).clip(lower=0.0, upper=1.0)
    walks *= scale
    singles *= scale
    doubles *= scale
    triples *= scale
    hr *= scale
    out = (pa - (walks + singles + doubles + triples + hr)).clip(lower=0.0)
    return pd.DataFrame(
        {
            "out": out,
            "walk": walks,
            "single": singles,
            "double": doubles,
            "triple": triples,
            "hr": hr,
        },
        index=df.index,
    )


def _player_key(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value).strip()


def _prior_rate(numerator: pd.Series, denominator: pd.Series, mean: float, strength: float) -> pd.Series:
    return ((numerator + mean * strength) / (denominator + strength)).clip(lower=1e-5, upper=1.0 - 1e-5)


def add_leakage_safe_player_priors(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Add empirical-Bayes player rates using only dates before each game."""
    out = df.copy()
    if out.empty or "player_id" not in out or "game_date_et" not in out:
        for name, default in _PLAYER_PRIOR_DEFAULTS.items():
            out[name] = default
        return out, {}

    counts = _event_count_matrix(out)
    daily = pd.DataFrame({
        "player_key": out["player_id"].map(_player_key),
        "game_date_et": pd.to_datetime(out["game_date_et"]).dt.date,
        "pa": pd.to_numeric(out["actual_pa"], errors="coerce").fillna(0.0).clip(lower=0.0),
        "hits": counts[["single", "double", "triple", "hr"]].sum(axis=1),
        "xbh": counts[["double", "triple", "hr"]].sum(axis=1),
        "non_hr_xbh": counts[["double", "triple"]].sum(axis=1),
        "hr": counts["hr"],
        "triple": counts["triple"],
        "walk": counts["walk"],
    })
    daily = daily.groupby(["player_key", "game_date_et"], as_index=False).sum(numeric_only=True)
    daily = daily.sort_values(["player_key", "game_date_et"]).reset_index(drop=True)
    cumulative_cols = ["pa", "hits", "xbh", "non_hr_xbh", "hr", "triple", "walk"]
    for col in cumulative_cols:
        daily[f"prior_{col}"] = daily.groupby("player_key", sort=False)[col].cumsum() - daily[col]

    strengths = _PLAYER_PRIOR_STRENGTHS
    defaults = _PLAYER_PRIOR_DEFAULTS
    daily["player_prior_pa"] = daily["prior_pa"]
    daily["player_prior_hit_rate"] = _prior_rate(
        daily["prior_hits"], daily["prior_pa"], defaults["player_prior_hit_rate"], strengths["hit"]
    )
    daily["player_prior_xbh_given_hit"] = _prior_rate(
        daily["prior_xbh"], daily["prior_hits"], defaults["player_prior_xbh_given_hit"], strengths["xbh"]
    )
    daily["player_prior_hr_given_xbh"] = _prior_rate(
        daily["prior_hr"], daily["prior_xbh"], defaults["player_prior_hr_given_xbh"], strengths["hr"]
    )
    daily["player_prior_triple_given_non_hr_xbh"] = _prior_rate(
        daily["prior_triple"],
        daily["prior_non_hr_xbh"],
        defaults["player_prior_triple_given_non_hr_xbh"],
        strengths["triple"],
    )
    prior_non_hit = (daily["prior_pa"] - daily["prior_hits"]).clip(lower=0.0)
    daily["player_prior_walk_given_non_hit"] = _prior_rate(
        daily["prior_walk"], prior_non_hit, defaults["player_prior_walk_given_non_hit"], strengths["walk"]
    )

    prior_cols = list(_PLAYER_PRIOR_DEFAULTS)
    join = daily[["player_key", "game_date_et", *prior_cols]]
    out["_player_key"] = out["player_id"].map(_player_key)
    out["_game_date_key"] = pd.to_datetime(out["game_date_et"]).dt.date
    out = out.merge(
        join,
        how="left",
        left_on=["_player_key", "_game_date_key"],
        right_on=["player_key", "game_date_et"],
        suffixes=("", "_prior"),
    )
    out = out.drop(columns=["_player_key", "_game_date_key", "player_key", "game_date_et_prior"], errors="ignore")
    for name, default in defaults.items():
        out[name] = pd.to_numeric(out.get(name), errors="coerce").fillna(default)

    totals = daily.groupby("player_key", as_index=False)[cumulative_cols].sum(numeric_only=True)
    state: dict[str, dict[str, float]] = {}
    for _, row in totals.iterrows():
        key = str(row["player_key"])
        pa = float(row["pa"])
        hits = float(row["hits"])
        xbh = float(row["xbh"])
        non_hr_xbh = float(row["non_hr_xbh"])
        non_hit = max(0.0, pa - hits)
        state[key] = {
            "player_prior_pa": pa,
            "player_prior_hit_rate": float((hits + defaults["player_prior_hit_rate"] * strengths["hit"]) / (pa + strengths["hit"])),
            "player_prior_xbh_given_hit": float((xbh + defaults["player_prior_xbh_given_hit"] * strengths["xbh"]) / (hits + strengths["xbh"])),
            "player_prior_hr_given_xbh": float((float(row["hr"]) + defaults["player_prior_hr_given_xbh"] * strengths["hr"]) / (xbh + strengths["hr"])),
            "player_prior_triple_given_non_hr_xbh": float((float(row["triple"]) + defaults["player_prior_triple_given_non_hr_xbh"] * strengths["triple"]) / (non_hr_xbh + strengths["triple"])),
            "player_prior_walk_given_non_hit": float((float(row["walk"]) + defaults["player_prior_walk_given_non_hit"] * strengths["walk"]) / (non_hit + strengths["walk"])),
        }
    return out, state


def apply_player_prior_state(df: pd.DataFrame, state: dict[str, dict[str, float]] | None) -> pd.DataFrame:
    out = df.copy()
    state = state or {}
    player_ids = out.get("player_id", pd.Series(index=out.index, dtype=object))
    keys = player_ids.map(_player_key)
    for name, default in _PLAYER_PRIOR_DEFAULTS.items():
        out[name] = [float((state.get(key) or {}).get(name, default)) for key in keys]
    return out


def _build_event_training_examples(
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    features = prepare_hitter_outcome_features(df, numeric_features, categorical_features)
    counts = _event_count_matrix(df)
    parts: list[pd.DataFrame] = []
    labels: list[str] = []
    weights: list[float] = []
    for cls in EVENT_CLASSES:
        w = pd.to_numeric(counts[cls], errors="coerce").fillna(0.0)
        mask = w > 0
        if not mask.any():
            continue
        parts.append(features.loc[mask].copy())
        labels.extend([cls] * int(mask.sum()))
        weights.extend(w.loc[mask].to_numpy(dtype=float).tolist())
    if not parts:
        return pd.DataFrame(columns=features.columns), np.asarray([], dtype=object), np.asarray([], dtype=float)
    return pd.concat(parts, ignore_index=True), np.asarray(labels, dtype=object), np.asarray(weights, dtype=float)


def _predict_event_probabilities(
    model: Pipeline,
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    features = prepare_hitter_outcome_features(df, numeric_features, categorical_features)
    raw = model.predict_proba(features[numeric_features + categorical_features])
    probs = pd.DataFrame(0.0, index=df.index, columns=[f"p_{cls}" for cls in EVENT_CLASSES])
    for i, cls in enumerate(model.classes_):
        if cls in EVENT_CLASSES:
            probs[f"p_{cls}"] = raw[:, i]
    row_sum = probs.sum(axis=1).replace(0.0, np.nan)
    probs = probs.div(row_sum, axis=0).fillna(0.0)
    return probs


def _fit_boosted_event_binary_models(
    X_event: pd.DataFrame,
    y_event: np.ndarray,
    w_event: np.ndarray,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict[str, Pipeline]:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    models: dict[str, Pipeline] = {}
    if len(y_event) == 0:
        return models
    feature_cols = numeric_features + categorical_features
    for cls in EVENT_CLASSES:
        y_bin = (y_event == cls).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        model = _classifier_pipeline(numeric_features, categorical_features)
        model.fit(X_event[feature_cols], y_bin, model__sample_weight=w_event)
        models[cls] = model
    return models


def _predict_boosted_event_probabilities(
    models: dict[str, Pipeline],
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    features = prepare_hitter_outcome_features(df, numeric_features, categorical_features)
    X = features[numeric_features + categorical_features]
    probs = pd.DataFrame(0.0, index=df.index, columns=[f"p_{cls}" for cls in EVENT_CLASSES])
    for cls, model in models.items():
        if cls not in EVENT_CLASSES:
            continue
        try:
            probs[f"p_{cls}"] = model.predict_proba(X)[:, 1]
        except Exception:
            continue
    row_sum = probs.sum(axis=1).replace(0.0, np.nan)
    fallback = 1.0 / float(len(EVENT_CLASSES))
    probs = probs.div(row_sum, axis=0).fillna(fallback)
    return probs


_HIERARCHICAL_EVENT_HEADS = {
    "hit": (("single", "double", "triple", "hr"), ("out", "walk")),
    "walk_given_non_hit": (("walk",), ("out",)),
    "xbh_given_hit": (("double", "triple", "hr"), ("single",)),
    "hr_given_xbh": (("hr",), ("double", "triple")),
    "triple_given_non_hr_xbh": (("triple",), ("double",)),
}

_PLAYER_PRIOR_DEFAULTS = {
    "player_prior_pa": 0.0,
    "player_prior_hit_rate": 0.225,
    "player_prior_xbh_given_hit": 0.35,
    "player_prior_hr_given_xbh": 0.42,
    "player_prior_triple_given_non_hr_xbh": 0.08,
    "player_prior_walk_given_non_hit": 0.11,
}

_PLAYER_PRIOR_STRENGTHS = {
    "hit": 80.0,
    "xbh": 35.0,
    "hr": 20.0,
    "triple": 20.0,
    "walk": 100.0,
}


def _weighted_binary_examples(
    df: pd.DataFrame,
    positive_classes: tuple[str, ...],
    negative_classes: tuple[str, ...],
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    features = prepare_hitter_outcome_features(df, numeric_features, categorical_features)
    counts = _event_count_matrix(df)
    positive = counts[list(positive_classes)].sum(axis=1)
    negative = counts[list(negative_classes)].sum(axis=1)
    parts: list[pd.DataFrame] = []
    labels: list[int] = []
    weights: list[float] = []
    for label, event_weight in ((1, positive), (0, negative)):
        mask = event_weight > 0
        if not mask.any():
            continue
        parts.append(features.loc[mask].copy())
        labels.extend([label] * int(mask.sum()))
        weights.extend(event_weight.loc[mask].to_numpy(dtype=float).tolist())
    if not parts:
        return pd.DataFrame(columns=features.columns), np.asarray([], dtype=int), np.asarray([], dtype=float)
    return pd.concat(parts, ignore_index=True), np.asarray(labels, dtype=int), np.asarray(weights, dtype=float)


def _fit_hierarchical_event_models(
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    head_numeric_features: dict[str, list[str]] | None = None,
) -> dict[str, Pipeline]:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    counts = _event_count_matrix(df)
    models: dict[str, Pipeline] = {}
    for head, (positive, negative) in _HIERARCHICAL_EVENT_HEADS.items():
        head_numeric = list((head_numeric_features or {}).get(head) or numeric_features)
        features = prepare_hitter_outcome_features(df, head_numeric, categorical_features)
        feature_cols = head_numeric + categorical_features
        positive_count = counts[list(positive)].sum(axis=1)
        negative_count = counts[list(negative)].sum(axis=1)
        denominator = positive_count + negative_count
        mask = denominator > 0
        if (
            int(mask.sum()) < 100
            or float(positive_count.loc[mask].sum()) < 25.0
            or float(negative_count.loc[mask].sum()) < 25.0
        ):
            continue
        target = (positive_count.loc[mask] / denominator.loc[mask]).clip(lower=0.0, upper=1.0)
        model = _boosted_rate_pipeline(
            numeric_features=head_numeric,
            categorical_features=categorical_features,
        )
        model.fit(
            features.loc[mask, feature_cols],
            target,
            model__sample_weight=denominator.loc[mask].to_numpy(dtype=float),
        )
        models[head] = model
    return models


def _predict_hierarchical_event_probabilities(
    models: dict[str, Pipeline],
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    player_prior_state: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    source = apply_player_prior_state(df, player_prior_state) if player_prior_state is not None else df
    features = prepare_hitter_outcome_features(source, numeric_features, categorical_features)
    X = features[numeric_features + categorical_features]

    def predict(head: str, fallback: float) -> np.ndarray:
        model = models.get(head)
        if model is None:
            return np.full(len(df), fallback, dtype=float)
        if hasattr(model, "predict_proba"):
            values = model.predict_proba(X)[:, 1]
        else:
            values = model.predict(X)
        return np.clip(values, 1e-6, 1.0 - 1e-6)

    p_hit = predict("hit", 0.225)
    p_walk_non_hit = predict("walk_given_non_hit", 0.11)
    p_xbh_hit = predict("xbh_given_hit", 0.35)
    p_hr_xbh = predict("hr_given_xbh", 0.42)
    p_triple_non_hr_xbh = predict("triple_given_non_hr_xbh", 0.08)

    p_non_hit = 1.0 - p_hit
    p_walk = p_non_hit * p_walk_non_hit
    p_out = p_non_hit - p_walk
    p_single = p_hit * (1.0 - p_xbh_hit)
    p_xbh = p_hit * p_xbh_hit
    p_hr = p_xbh * p_hr_xbh
    p_non_hr_xbh = p_xbh - p_hr
    p_triple = p_non_hr_xbh * p_triple_non_hr_xbh
    p_double = p_non_hr_xbh - p_triple
    probs = pd.DataFrame(
        {
            "p_out": p_out,
            "p_walk": p_walk,
            "p_single": p_single,
            "p_double": p_double,
            "p_triple": p_triple,
            "p_hr": p_hr,
        },
        index=df.index,
    )
    row_sum = probs.sum(axis=1).replace(0.0, np.nan)
    return probs.div(row_sum, axis=0).fillna(1.0 / float(len(EVENT_CLASSES)))


def _pa_uncertainty_key(row: pd.Series) -> str:
    slot = pd.to_numeric(pd.Series([row.get("lineup_slot")]), errors="coerce").iloc[0]
    if pd.isna(slot):
        slot_group = "unknown"
    elif float(slot) <= 3:
        slot_group = "top"
    elif float(slot) <= 6:
        slot_group = "middle"
    else:
        slot_group = "bottom"
    home = "home" if bool(row.get("is_home")) else "away"
    return f"{slot_group}|{home}"


def _fit_pa_uncertainty(train: pd.DataFrame, holdout_days: int) -> dict[str, Any]:
    dates = sorted(set(train["game_date_et"]))
    if len(dates) < 20:
        return {"status": "insufficient_dates"}
    cutoff = max(dates) - timedelta(days=max(14, min(holdout_days, 28)))
    fit = train.loc[train["game_date_et"] < cutoff].copy()
    calibration = train.loc[train["game_date_et"] >= cutoff].copy()
    if len(fit) < 1000 or len(calibration) < 300:
        return {"status": "insufficient_rows"}
    model = _boosted_count_pipeline()
    normal_fit = fit.loc[pd.to_numeric(fit["actual_pa"], errors="coerce") >= 3].copy()
    normal_calibration = calibration.loc[pd.to_numeric(calibration["actual_pa"], errors="coerce") >= 3].copy()
    if len(normal_fit) < 1000 or len(normal_calibration) < 200:
        return {"status": "insufficient_normal_pa_rows"}
    model.fit(
        prepare_hitter_outcome_features(normal_fit)[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        normal_fit["actual_pa"],
    )
    pred = np.clip(
        model.predict(prepare_hitter_outcome_features(normal_calibration)[NUMERIC_FEATURES + CATEGORICAL_FEATURES]),
        3.0,
        7.0,
    )
    residual = pd.to_numeric(normal_calibration["actual_pa"], errors="coerce").to_numpy(dtype=float) - pred
    work = normal_calibration.copy()
    work["_residual"] = residual
    work["_pa_group"] = work.apply(_pa_uncertainty_key, axis=1)

    low_calibration = calibration.loc[pd.to_numeric(calibration["actual_pa"], errors="coerce") <= 2].copy()

    def low_state_probs(frame: pd.DataFrame) -> dict[str, float]:
        values = pd.to_numeric(frame.get("actual_pa"), errors="coerce").dropna().round().clip(0, 2)
        counts = values.value_counts().to_dict()
        total = float(len(values))
        prior = {0: 0.05, 1: 0.20, 2: 0.75}
        strength = 20.0
        return {
            str(n): float((float(counts.get(n, 0.0)) + prior[n] * strength) / (total + strength))
            for n in range(3)
        }

    def summarize(values: pd.Series) -> dict[str, float]:
        arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.75
        return {
            "rows": int(len(arr)),
            "bias": float(np.mean(arr)) if len(arr) else 0.0,
            "sigma": max(0.45, min(1.75, sigma)),
        }

    global_rec = summarize(work["_residual"])
    global_rec["low_pa_state_probs"] = low_state_probs(low_calibration)
    groups = {
        str(key): summarize(group["_residual"])
        for key, group in work.groupby("_pa_group")
        if len(group) >= 100
    }
    calibration_groups = calibration.assign(_pa_group=calibration.apply(_pa_uncertainty_key, axis=1))
    for key, rec in groups.items():
        low_group = calibration_groups.loc[
            (calibration_groups["_pa_group"] == key)
            & (pd.to_numeric(calibration_groups["actual_pa"], errors="coerce") <= 2)
        ]
        rec["low_pa_state_probs"] = low_state_probs(low_group)
    return {
        "status": "trained",
        "method": "two_part_low_pa_plus_conditional_normal_pa",
        "calibration_start": str(cutoff),
        "global": global_rec,
        "groups": groups,
    }


def projected_pa_pmf(mean_pa: float, row: pd.Series, uncertainty: dict[str, Any] | None) -> dict[int, float]:
    uncertainty = uncertainty or {}
    rec = (uncertainty.get("groups") or {}).get(_pa_uncertainty_key(row)) or uncertainty.get("global") or {}
    low_prob = next(
        (
            float(row.get(name))
            for name in ("pa_low_probability", "p_event_low_pa", "opp_model_low_pa")
            if pd.notna(row.get(name))
        ),
        None,
    )
    normal_mean = next(
        (
            float(row.get(name))
            for name in ("pa_normal_mean", "p_event_normal_pa", "opp_model_normal_pa")
            if pd.notna(row.get(name))
        ),
        None,
    )
    mean = max(0.0, min(7.0, float(normal_mean if normal_mean is not None else mean_pa) + float(rec.get("bias") or 0.0)))
    sigma = max(0.35, float(rec.get("sigma") or 0.80))

    def cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf((value - mean) / (sigma * math.sqrt(2.0))))

    normal_weights: dict[int, float] = {}
    normal_start = 3 if low_prob is not None else 0
    for n in range(normal_start, 8):
        lower = -math.inf if n == normal_start else n - 0.5
        upper = math.inf if n == 7 else n + 0.5
        lo = 0.0 if not math.isfinite(lower) else cdf(lower)
        hi = 1.0 if not math.isfinite(upper) else cdf(upper)
        normal_weights[n] = max(0.0, hi - lo)
    normal_total = sum(normal_weights.values()) or 1.0
    normal_weights = {n: value / normal_total for n, value in normal_weights.items()}
    if low_prob is None:
        return normal_weights
    low_prob = max(1e-5, min(1.0 - 1e-5, low_prob))
    low_states = rec.get("low_pa_state_probs") or {"0": 0.05, "1": 0.20, "2": 0.75}
    low_total = sum(max(0.0, float(low_states.get(str(n), 0.0))) for n in range(3)) or 1.0
    weights = {
        n: low_prob * max(0.0, float(low_states.get(str(n), 0.0))) / low_total
        for n in range(3)
    }
    weights.update({n: (1.0 - low_prob) * value for n, value in normal_weights.items()})
    total = sum(weights.values()) or 1.0
    return {n: value / total for n, value in weights.items()}


def convolve_hitter_outcomes(event_probs: dict[str, float], pa_pmf: dict[int, float]) -> dict[str, Any]:
    p_out = max(0.0, float(event_probs.get("p_out", 0.0)))
    p_walk = max(0.0, float(event_probs.get("p_walk", 0.0)))
    p_single = max(0.0, float(event_probs.get("p_single", 0.0)))
    p_double = max(0.0, float(event_probs.get("p_double", 0.0)))
    p_triple = max(0.0, float(event_probs.get("p_triple", 0.0)))
    p_hr = max(0.0, float(event_probs.get("p_hr", 0.0)))
    total = p_out + p_walk + p_single + p_double + p_triple + p_hr
    if total <= 0:
        return {}
    outcomes = [
        (0, False, p_out / total),
        (0, False, p_walk / total),
        (1, False, p_single / total),
        (2, False, p_double / total),
        (3, False, p_triple / total),
        (4, True, p_hr / total),
    ]
    p_hit = (p_single + p_double + p_triple + p_hr) / total
    p_hr_norm = p_hr / total
    tb_joint: dict[tuple[int, bool], float] = {}
    hits_pmf: dict[int, float] = {}
    hr_pmf: dict[int, float] = {}
    for n, pa_weight in pa_pmf.items():
        if pa_weight <= 0:
            continue
        dp: dict[tuple[int, bool], float] = {(0, False): 1.0}
        for _ in range(int(n)):
            nxt: dict[tuple[int, bool], float] = {}
            for (tb, had_hr), base in dp.items():
                for bases, is_hr, probability in outcomes:
                    key = (tb + bases, had_hr or is_hr)
                    nxt[key] = nxt.get(key, 0.0) + base * probability
            dp = nxt
        for key, probability in dp.items():
            tb_joint[key] = tb_joint.get(key, 0.0) + pa_weight * probability
        for count in range(int(n) + 1):
            hits_pmf[count] = hits_pmf.get(count, 0.0) + pa_weight * math.comb(int(n), count) * (p_hit ** count) * ((1.0 - p_hit) ** (int(n) - count))
            hr_pmf[count] = hr_pmf.get(count, 0.0) + pa_weight * math.comb(int(n), count) * (p_hr_norm ** count) * ((1.0 - p_hr_norm) ** (int(n) - count))

    tb_pmf: dict[int, float] = {}
    states = {"tb_0": 0.0, "tb_1": 0.0, "tb_2_3": 0.0, "tb_4_plus_hr": 0.0, "tb_4_plus_non_hr": 0.0}
    for (tb, had_hr), probability in tb_joint.items():
        tb_pmf[tb] = tb_pmf.get(tb, 0.0) + probability
        if tb == 0:
            states["tb_0"] += probability
        elif tb == 1:
            states["tb_1"] += probability
        elif tb <= 3:
            states["tb_2_3"] += probability
        elif had_hr:
            states["tb_4_plus_hr"] += probability
        else:
            states["tb_4_plus_non_hr"] += probability
    return {"tb_pmf": tb_pmf, "tb_joint": tb_joint, "tb_states": states, "hits_pmf": hits_pmf, "hr_pmf": hr_pmf}


def _tb_state_metrics(df: pd.DataFrame, probs: pd.DataFrame, pa_pred: np.ndarray, pa_uncertainty: dict[str, Any]) -> dict[str, Any]:
    state_names = ["tb_0", "tb_1", "tb_2_3", "tb_4_plus_hr", "tb_4_plus_non_hr"]
    predicted: list[list[float]] = []
    actual: list[int] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        event = {f"p_{cls}": float(probs.iloc[pos][f"p_{cls}"]) for cls in EVENT_CLASSES}
        curve = convolve_hitter_outcomes(event, projected_pa_pmf(float(pa_pred[pos]), row, pa_uncertainty))
        states = curve.get("tb_states") or {}
        if not states:
            continue
        tb = float(row.get("actual_total_bases") or 0.0)
        hr = float(row.get("actual_home_runs") or 0.0)
        label = "tb_0" if tb <= 0 else "tb_1" if tb <= 1 else "tb_2_3" if tb < 4 else "tb_4_plus_hr" if hr > 0 else "tb_4_plus_non_hr"
        predicted.append([max(1e-9, float(states.get(name, 0.0))) for name in state_names])
        actual.append(state_names.index(label))
    if not predicted:
        return {"rows": 0}
    p = np.asarray(predicted, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    y = np.asarray(actual, dtype=int)
    one_hot = np.eye(len(state_names))[y]
    return {
        "rows": int(len(y)),
        "multiclass_brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.0)))),
        "states": state_names,
        "predicted_rates": {name: float(p[:, i].mean()) for i, name in enumerate(state_names)},
        "actual_rates": {name: float((y == i).mean()) for i, name in enumerate(state_names)},
    }


def _tb_state_labels(df: pd.DataFrame) -> pd.Series:
    tb = pd.to_numeric(df.get("actual_total_bases"), errors="coerce").fillna(0.0)
    hr = pd.to_numeric(df.get("actual_home_runs"), errors="coerce").fillna(0.0)
    labels = np.where(
        tb <= 0,
        "tb_0",
        np.where(
            tb <= 1,
            "tb_1",
            np.where(tb < 4, "tb_2_3", np.where(hr > 0, "tb_4_plus_hr", "tb_4_plus_non_hr")),
        ),
    )
    return pd.Series(labels, index=df.index, dtype="object")


def _fit_tb_state_models(
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict[str, Pipeline]:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    features = prepare_hitter_outcome_features(df, numeric_features, categorical_features)
    labels = _tb_state_labels(df)
    models: dict[str, Pipeline] = {}
    for state in TB_STATE_NAMES:
        target = (labels == state).astype(int)
        positives = int(target.sum())
        negatives = int(len(target) - positives)
        if positives < 25 or negatives < 25:
            continue
        model = _boosted_binary_pipeline(numeric_features, categorical_features)
        positive_weight = min(4.0, math.sqrt(negatives / max(positives, 1)))
        sample_weight = np.where(target.to_numpy() == 1, positive_weight, 1.0)
        model.fit(
            features[numeric_features + categorical_features],
            target,
            model__sample_weight=sample_weight,
        )
        models[state] = model
    return models


def _predict_tb_state_models(
    models: dict[str, Pipeline],
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    features = prepare_hitter_outcome_features(df, numeric_features, categorical_features)
    X = features[numeric_features + categorical_features]
    probs = pd.DataFrame(0.0, index=df.index, columns=TB_STATE_NAMES)
    for state, model in models.items():
        if state not in probs:
            continue
        probs[state] = np.clip(model.predict_proba(X)[:, 1], 1e-8, 1.0 - 1e-8)
    row_sum = probs.sum(axis=1).replace(0.0, np.nan)
    return probs.div(row_sum, axis=0).fillna(1.0 / float(len(TB_STATE_NAMES)))


def _fit_hierarchical_tb_state_models(
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict[str, Pipeline]:
    """Fit a probability tree that preserves the rare HR-driven 4+ tail."""
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    labels = _tb_state_labels(df)
    features = prepare_hitter_outcome_features(df, numeric_features, categorical_features)
    feature_cols = numeric_features + categorical_features
    specs = {
        "tb_positive": (
            labels.isin(["tb_1", "tb_2_3", "tb_4_plus_hr", "tb_4_plus_non_hr"]),
            pd.Series(True, index=df.index),
        ),
        "tb_2_plus_given_positive": (
            labels.isin(["tb_2_3", "tb_4_plus_hr", "tb_4_plus_non_hr"]),
            labels.isin(["tb_1", "tb_2_3", "tb_4_plus_hr", "tb_4_plus_non_hr"]),
        ),
        "tb_4_plus_given_2_plus": (
            labels.isin(["tb_4_plus_hr", "tb_4_plus_non_hr"]),
            labels.isin(["tb_2_3", "tb_4_plus_hr", "tb_4_plus_non_hr"]),
        ),
        "tb_hr_given_4_plus": (
            labels.eq("tb_4_plus_hr"),
            labels.isin(["tb_4_plus_hr", "tb_4_plus_non_hr"]),
        ),
    }
    models: dict[str, Pipeline] = {}
    for head, (target, eligible) in specs.items():
        y = target.loc[eligible].astype(int)
        if len(y) < 100 or int(y.sum()) < 25 or int((1 - y).sum()) < 25:
            continue
        model = _boosted_binary_pipeline(numeric_features, categorical_features)
        positives = int(y.sum())
        negatives = int((1 - y).sum())
        positive_weight = min(4.0, math.sqrt(negatives / max(positives, 1)))
        sample_weight = np.where(y.to_numpy() == 1, positive_weight, 1.0)
        model.fit(
            features.loc[eligible, feature_cols],
            y,
            model__sample_weight=sample_weight,
        )
        models[head] = model
    return models


def _predict_hierarchical_tb_state_models(
    models: dict[str, Pipeline],
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    features = prepare_hitter_outcome_features(df, numeric_features, categorical_features)
    X = features[numeric_features + categorical_features]

    def predict(head: str, fallback: float) -> np.ndarray:
        model = models.get(head)
        if model is None:
            return np.full(len(df), fallback, dtype=float)
        return np.clip(model.predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)

    p_positive = predict("tb_positive", 0.56)
    p_2_plus_positive = predict("tb_2_plus_given_positive", 0.43)
    p_4_plus_2_plus = predict("tb_4_plus_given_2_plus", 0.43)
    p_hr_4_plus = predict("tb_hr_given_4_plus", 0.82)
    p_0 = 1.0 - p_positive
    p_1 = p_positive * (1.0 - p_2_plus_positive)
    p_2_3 = p_positive * p_2_plus_positive * (1.0 - p_4_plus_2_plus)
    p_4_plus = p_positive * p_2_plus_positive * p_4_plus_2_plus
    probs = pd.DataFrame({
        "tb_0": p_0,
        "tb_1": p_1,
        "tb_2_3": p_2_3,
        "tb_4_plus_hr": p_4_plus * p_hr_4_plus,
        "tb_4_plus_non_hr": p_4_plus * (1.0 - p_hr_4_plus),
    }, index=df.index)
    return probs.div(probs.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(1.0 / len(TB_STATE_NAMES))


def _fit_tb_hr_tail_logit_offset(labels: pd.Series, probs: pd.DataFrame) -> float:
    """Fit a shrunk intercept-only calibration for the rare HR-driven tail."""
    target_rate = float((labels == "tb_4_plus_hr").mean())
    raw = np.clip(probs["tb_4_plus_hr"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(raw / (1.0 - raw))
    lo, hi = -2.0, 2.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        mean = float(np.mean(1.0 / (1.0 + np.exp(-np.clip(logits + mid, -30.0, 30.0)))))
        if mean < target_rate:
            lo = mid
        else:
            hi = mid
    offset = (lo + hi) / 2.0
    shrink = float(len(labels) / (len(labels) + 1000.0))
    return float(np.clip(offset * shrink, -1.0, 1.0))


def _apply_tb_hr_tail_logit_offset(probs: pd.DataFrame, offset: float) -> pd.DataFrame:
    if abs(float(offset)) <= 1e-12 or probs.empty:
        return probs.copy()
    out = probs.copy()
    raw = np.clip(out["tb_4_plus_hr"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(raw / (1.0 - raw))
    repaired = 1.0 / (1.0 + np.exp(-np.clip(logits + float(offset), -30.0, 30.0)))
    other_cols = [name for name in TB_STATE_NAMES if name != "tb_4_plus_hr"]
    other_total = out[other_cols].sum(axis=1).to_numpy(dtype=float)
    scale = np.divide(1.0 - repaired, other_total, out=np.ones(len(out)), where=other_total > 1e-12)
    out.loc[:, other_cols] = out[other_cols].mul(scale, axis=0)
    out["tb_4_plus_hr"] = repaired
    return out.div(out.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(1.0 / len(TB_STATE_NAMES))


def _convolution_tb_state_frame(
    df: pd.DataFrame,
    event_probs: pd.DataFrame,
    pa_pred: np.ndarray,
    pa_uncertainty: dict[str, Any],
    low_pa_prob: np.ndarray | None = None,
    normal_pa_pred: np.ndarray | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        row_for_curve = row.copy()
        if low_pa_prob is not None:
            row_for_curve["pa_low_probability"] = float(low_pa_prob[pos])
        if normal_pa_pred is not None:
            row_for_curve["pa_normal_mean"] = float(normal_pa_pred[pos])
        event = {f"p_{cls}": float(event_probs.iloc[pos][f"p_{cls}"]) for cls in EVENT_CLASSES}
        curve = convolve_hitter_outcomes(
            event,
            projected_pa_pmf(float(pa_pred[pos]), row_for_curve, pa_uncertainty),
        )
        states = curve.get("tb_states") or {}
        rows.append({state: float(states.get(state, 0.0)) for state in TB_STATE_NAMES})
    return pd.DataFrame(rows, index=df.index, columns=TB_STATE_NAMES)


def _tb_state_multiclass_metrics(labels: pd.Series, probs: pd.DataFrame) -> dict[str, Any]:
    y = labels.map({name: i for i, name in enumerate(TB_STATE_NAMES)}).astype(int).to_numpy()
    p = probs[TB_STATE_NAMES].to_numpy(dtype=float)
    p = np.clip(p, 1e-9, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    one_hot = np.eye(len(TB_STATE_NAMES))[y]
    hr_tail_target = (labels == "tb_4_plus_hr").astype(int).to_numpy()
    hr_tail_prob = p[:, TB_STATE_NAMES.index("tb_4_plus_hr")]
    return {
        "rows": int(len(y)),
        "multiclass_brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "predicted_rates": {name: float(p[:, i].mean()) for i, name in enumerate(TB_STATE_NAMES)},
        "actual_rates": {name: float((y == i).mean()) for i, name in enumerate(TB_STATE_NAMES)},
        "hr_driven_4_plus": {
            "actual_rate": float(hr_tail_target.mean()),
            "avg_probability": float(hr_tail_prob.mean()),
            "calibration_error": float(hr_tail_prob.mean() - hr_tail_target.mean()),
            "brier": float(np.mean((hr_tail_target - hr_tail_prob) ** 2)),
        },
    }


def _select_tb_state_blend(
    df: pd.DataFrame,
    base_probs: pd.DataFrame,
    direct_candidates: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    dates = sorted(pd.to_datetime(df["game_date_et"]).dt.date.unique())
    if len(dates) < 4 or len(df) < 200:
        return {"enabled": False, "reason": "insufficient_walk_forward_rows", "alpha": 0.0}
    validation_start = dates[max(1, len(dates) // 2)]
    tune_mask = pd.to_datetime(df["game_date_et"]).dt.date < validation_start
    validation_mask = ~tune_mask
    if int(tune_mask.sum()) < 80 or int(validation_mask.sum()) < 80:
        return {"enabled": False, "reason": "insufficient_walk_forward_rows", "alpha": 0.0}
    labels = _tb_state_labels(df)
    candidates: list[dict[str, Any]] = []
    for candidate_name, direct_probs in direct_candidates.items():
        for alpha in (0.25, 0.5, 0.75, 1.0):
            blended = (1.0 - alpha) * base_probs.loc[tune_mask] + alpha * direct_probs.loc[tune_mask]
            metrics = _tb_state_multiclass_metrics(labels.loc[tune_mask], blended)
            candidates.append({
                "candidate": candidate_name,
                "alpha": alpha,
                "tune_brier": float(metrics["multiclass_brier"]),
                "tune_hr_tail_brier": float((metrics.get("hr_driven_4_plus") or {}).get("brier")),
                "tune_joint_loss": float(
                    metrics["multiclass_brier"] + (metrics.get("hr_driven_4_plus") or {}).get("brier")
                ),
            })
    base_tune = _tb_state_multiclass_metrics(labels.loc[tune_mask], base_probs.loc[tune_mask])
    candidates.append({
        "candidate": "convolution",
        "alpha": 0.0,
        "tune_brier": float(base_tune["multiclass_brier"]),
        "tune_hr_tail_brier": float((base_tune.get("hr_driven_4_plus") or {}).get("brier")),
        "tune_joint_loss": float(
            base_tune["multiclass_brier"] + (base_tune.get("hr_driven_4_plus") or {}).get("brier")
        ),
    })
    best = min(candidates, key=lambda rec: rec["tune_joint_loss"])
    alpha = float(best["alpha"])
    candidate_name = str(best["candidate"])
    base_metrics = _tb_state_multiclass_metrics(labels.loc[validation_mask], base_probs.loc[validation_mask])
    selected_direct = direct_candidates.get(candidate_name, base_probs)
    blend_metrics = _tb_state_multiclass_metrics(
        labels.loc[validation_mask],
        (1.0 - alpha) * base_probs.loc[validation_mask] + alpha * selected_direct.loc[validation_mask],
    )
    gain = float(base_metrics["multiclass_brier"] - blend_metrics["multiclass_brier"])
    base_hr_brier = float((base_metrics.get("hr_driven_4_plus") or {}).get("brier"))
    blend_hr_brier = float((blend_metrics.get("hr_driven_4_plus") or {}).get("brier"))
    hr_tail_gain = base_hr_brier - blend_hr_brier
    aggregate_win = gain >= 0.0005 and hr_tail_gain >= -0.0001
    tail_repair = gain >= -0.0001 and hr_tail_gain >= 0.0001
    enabled = bool(alpha > 0.0 and (aggregate_win or tail_repair))
    return {
        "enabled": enabled,
        "reason": (
            "validation_aggregate_and_tail_confirmed" if enabled and aggregate_win
            else "validation_hr_tail_repair_without_material_aggregate_regression" if enabled
            else "no_joint_validation_gain"
        ),
        "selected_candidate": candidate_name if enabled else "convolution",
        "alpha": alpha if enabled else 0.0,
        "tune_end": str(dates[max(0, len(dates) // 2 - 1)]),
        "validation_start": str(validation_start),
        "candidates": candidates,
        "base_validation": base_metrics,
        "blended_validation": blend_metrics,
        "validation_brier_gain": gain,
        "validation_hr_tail_brier_gain": hr_tail_gain,
    }


def _weighted_multiclass_brier(counts: pd.DataFrame, probs: pd.DataFrame) -> float | None:
    total = float(counts[EVENT_CLASSES].sum(axis=1).sum())
    if total <= 0:
        return None
    p_mat = np.column_stack([
        pd.to_numeric(probs[f"p_{cls}"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for cls in EVENT_CLASSES
    ])
    p_sq = np.sum(p_mat * p_mat, axis=1)
    loss = 0.0
    for i, cls in enumerate(EVENT_CLASSES):
        weights = pd.to_numeric(counts[cls], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        loss += float(np.sum(weights * (p_sq - 2.0 * p_mat[:, i] + 1.0)))
    return loss / total


def _event_log_loss_from_counts(counts: pd.DataFrame, probs: pd.DataFrame) -> float | None:
    total = float(counts[EVENT_CLASSES].sum(axis=1).sum())
    if total <= 0:
        return None
    loss = 0.0
    for cls in EVENT_CLASSES:
        p = np.clip(pd.to_numeric(probs[f"p_{cls}"], errors="coerce").fillna(1e-6).to_numpy(dtype=float), 1e-6, 1.0)
        w = pd.to_numeric(counts[cls], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        loss += float(np.sum(w * -np.log(p)))
    return loss / total


def _event_rate_table(df: pd.DataFrame, probs: pd.DataFrame) -> dict[str, Any]:
    counts = _event_count_matrix(df)
    actual_pa = float(counts[EVENT_CLASSES].sum(axis=1).sum())
    pred_pa = float(len(df)) if len(df) else 0.0
    rows: dict[str, Any] = {"rows": int(len(df)), "actual_pa": actual_pa}
    for cls in EVENT_CLASSES:
        actual = float(counts[cls].sum())
        pred_rate = float(pd.to_numeric(probs[f"p_{cls}"], errors="coerce").fillna(0.0).mean()) if len(probs) else 0.0
        rows[cls] = {
            "actual_per_pa": actual / actual_pa if actual_pa > 0 else None,
            "pred_mean_prob": pred_rate,
            "bias_per_pa": (actual / actual_pa - pred_rate) if actual_pa > 0 else None,
        }
    rows["mean_row_weight"] = actual_pa / pred_pa if pred_pa > 0 else None
    return rows


def _event_projection_metrics(
    event_holdout: pd.DataFrame,
    event_probs: pd.DataFrame,
    event_pa_pred: np.ndarray,
) -> dict[str, Any]:
    event_hit_count = event_pa_pred * (
        event_probs["p_single"].to_numpy(dtype=float)
        + event_probs["p_double"].to_numpy(dtype=float)
        + event_probs["p_triple"].to_numpy(dtype=float)
        + event_probs["p_hr"].to_numpy(dtype=float)
    )
    event_tb_count = event_pa_pred * (
        event_probs["p_single"].to_numpy(dtype=float)
        + 2.0 * event_probs["p_double"].to_numpy(dtype=float)
        + 3.0 * event_probs["p_triple"].to_numpy(dtype=float)
        + 4.0 * event_probs["p_hr"].to_numpy(dtype=float)
    )
    event_hr_count = event_pa_pred * event_probs["p_hr"].to_numpy(dtype=float)
    event_counts = _event_count_matrix(event_holdout)
    return {
        "weighted_event_log_loss": _event_log_loss_from_counts(event_counts, event_probs),
        "weighted_event_brier": _weighted_multiclass_brier(event_counts, event_probs),
        "event_rates": _event_rate_table(event_holdout, event_probs),
        "hits": _count_metrics(event_holdout["actual_hits"], event_hit_count),
        "total_bases": _count_metrics(event_holdout["actual_total_bases"], event_tb_count),
        "home_runs": _count_metrics(event_holdout["actual_home_runs"], event_hr_count),
    }


def _select_count_repair(
    holdout: pd.DataFrame,
    direct_prediction: np.ndarray,
    *,
    actual_column: str,
    base_column: str,
    repair_name: str,
    minimum_gain: float = 0.002,
    max_bias_offset: float = 0.75,
    require_any_brier_gain: bool = False,
) -> dict[str, Any]:
    """Gate a direct count blend with expanding, date-purged folds."""
    work = pd.DataFrame({
        "game_date_et": pd.to_datetime(holdout["game_date_et"]).dt.date,
        "actual": pd.to_numeric(holdout[actual_column], errors="coerce"),
        "base": pd.to_numeric(holdout[base_column], errors="coerce"),
        "direct": np.asarray(direct_prediction, dtype=float),
    }, index=holdout.index).dropna(subset=["actual", "base", "direct"])
    dates = sorted(work["game_date_et"].unique())
    if len(work) < 200 or len(dates) < 4:
        return {
            "enabled": False,
            "reason": "insufficient_temporal_rows",
            "rows": int(len(work)),
            "dates": int(len(dates)),
            "alpha": 0.0,
        }
    def fit_calibration(prior: pd.DataFrame) -> tuple[float, float, list[dict[str, float]]]:
        offset = float(np.clip(
            (prior["actual"] - prior["direct"]).mean(),
            -max_bias_offset,
            max_bias_offset,
        ))
        adjusted = np.clip(prior["direct"] + offset, 0.0, 8.0)
        candidates = []
        for candidate_alpha in np.linspace(0.0, 1.0, 11):
            prediction = (1.0 - candidate_alpha) * prior["base"] + candidate_alpha * adjusted
            candidates.append({
                "alpha": float(candidate_alpha),
                "mae": float(mean_absolute_error(prior["actual"], prediction)),
                "rmse": _rmse(prior["actual"], prediction),
            })
        selected = min(candidates, key=lambda row: (row["mae"], row["rmse"], row["alpha"]))
        return float(selected["alpha"]), offset, candidates

    min_train_dates = max(4, min(7, len(dates) // 3))
    test_window_dates = max(2, min(7, len(dates) // 4))
    fold_frames: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    repair_alpha = 0.5
    for start in range(min_train_dates, len(dates), test_window_dates):
        test_dates = dates[start:start + test_window_dates]
        if not test_dates:
            continue
        prior = work[work["game_date_et"] < test_dates[0]]
        test = work[work["game_date_et"].isin(test_dates)].copy()
        if len(prior) < 100 or len(test) < 30:
            continue
        _, offset, _ = fit_calibration(prior)
        alpha = repair_alpha
        test["direct_adjusted"] = np.clip(test["direct"] + offset, 0.0, 8.0)
        test["blended"] = (1.0 - alpha) * test["base"] + alpha * test["direct_adjusted"]
        base_metrics = _count_metrics(test["actual"], test["base"])
        direct_metrics = _count_metrics(test["actual"], test["direct_adjusted"])
        blend_metrics = _count_metrics(test["actual"], test["blended"])
        base_any = 1.0 - np.exp(-np.clip(test["base"].to_numpy(dtype=float), 0.0, 8.0))
        blend_any = 1.0 - np.exp(-np.clip(test["blended"].to_numpy(dtype=float), 0.0, 8.0))
        actual_any = test["actual"].gt(0).astype(int).to_numpy()
        base_any_brier = float(brier_score_loss(actual_any, base_any))
        blend_any_brier = float(brier_score_loss(actual_any, blend_any))
        folds.append({
            "test_start": str(test_dates[0]),
            "test_end": str(test_dates[-1]),
            "train_rows": int(len(prior)),
            "test_rows": int(len(test)),
            "alpha": alpha,
            "bias_offset": offset,
            "base": base_metrics,
            "direct": direct_metrics,
            "blended": blend_metrics,
            "mae_gain": float(base_metrics["mae"] - blend_metrics["mae"]),
            "base_any_brier": base_any_brier,
            "blended_any_brier": blend_any_brier,
            "any_brier_gain": base_any_brier - blend_any_brier,
        })
        fold_frames.append(test)
    if not fold_frames:
        return {
            "enabled": False,
            "reason": "insufficient_walk_forward_folds",
            "rows": int(len(work)),
            "dates": int(len(dates)),
            "alpha": 0.0,
            "folds": [],
        }
    oof = pd.concat(fold_frames, ignore_index=True)
    base_validation = _count_metrics(oof["actual"], oof["base"])
    direct_validation = _count_metrics(oof["actual"], oof["direct_adjusted"])
    blended_validation = _count_metrics(oof["actual"], oof["blended"])
    mae_gain = float(base_validation["mae"] - blended_validation["mae"])
    positive_folds = sum(float(fold["mae_gain"]) > minimum_gain for fold in folds)
    base_any_oof = 1.0 - np.exp(-np.clip(oof["base"].to_numpy(dtype=float), 0.0, 8.0))
    blend_any_oof = 1.0 - np.exp(-np.clip(oof["blended"].to_numpy(dtype=float), 0.0, 8.0))
    actual_any_oof = oof["actual"].gt(0).astype(int).to_numpy()
    base_any_brier = float(brier_score_loss(actual_any_oof, base_any_oof))
    blended_any_brier = float(brier_score_loss(actual_any_oof, blend_any_oof))
    any_brier_gain = base_any_brier - blended_any_brier
    _, production_bias_offset, production_candidates = fit_calibration(work)
    production_alpha = repair_alpha
    enabled = bool(
        production_alpha > 0.0
        and mae_gain > minimum_gain
        and blended_validation["rmse"] <= base_validation["rmse"] + 0.005
        and abs(blended_validation["bias"]) <= abs(base_validation["bias"]) + 0.02
        and positive_folds >= max(1, math.ceil(len(folds) * 0.5))
        and (not require_any_brier_gain or any_brier_gain > 0.0002)
    )
    return {
        "method": f"expanding_walk_forward_gated_direct_player_game_{repair_name}_blend",
        "enabled": enabled,
        "reason": "repeated_oof_gain" if enabled else "no_repeated_joint_oof_gain",
        "rows": int(len(work)),
        "dates": int(len(dates)),
        "walk_forward_folds": int(len(folds)),
        "positive_folds": int(positive_folds),
        "folds": folds,
        "validation_rows": int(len(oof)),
        "production_bias_offset": production_bias_offset if enabled else 0.0,
        "production_candidate_metrics": production_candidates,
        "predeclared_repair_alpha": repair_alpha,
        "alpha": production_alpha if enabled else 0.0,
        "candidate_alpha": production_alpha,
        "base_validation": base_validation,
        "direct_validation": direct_validation,
        "blended_validation": blended_validation,
        "validation_mae_gain": mae_gain,
        "base_any_brier": base_any_brier,
        "blended_any_brier": blended_any_brier,
        "any_brier_gain": any_brier_gain,
    }


def _select_tb_count_repair(
    holdout: pd.DataFrame,
    direct_prediction: np.ndarray,
    *,
    minimum_gain: float = 0.002,
) -> dict[str, Any]:
    return _select_count_repair(
        holdout,
        direct_prediction,
        actual_column="actual_total_bases",
        base_column="model_pred_total_bases",
        repair_name="tb",
        minimum_gain=minimum_gain,
        max_bias_offset=0.75,
    )


def _fit_event_class_log_offsets(df: pd.DataFrame, probs: pd.DataFrame) -> dict[str, float]:
    """Fit shrunk multinomial intercept offsets from a past calibration block."""
    counts = _event_count_matrix(df)
    total_events = float(counts.sum(axis=1).sum())
    if total_events <= 0 or len(probs) != len(df):
        return {cls: 0.0 for cls in EVENT_CLASSES}
    weights = pd.to_numeric(df.get("actual_pa"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if float(weights.sum()) <= 0:
        weights = np.ones(len(df), dtype=float)
    shrink = total_events / (total_events + 2000.0)
    offsets: dict[str, float] = {}
    for cls in EVENT_CLASSES:
        actual_rate = float(counts[cls].sum()) / total_events
        predicted_rate = float(np.average(probs[f"p_{cls}"].to_numpy(dtype=float), weights=weights))
        ratio = max(1e-6, actual_rate) / max(1e-6, predicted_rate)
        offsets[cls] = float(np.clip(math.log(ratio) * shrink, -0.45, 0.45))
    return offsets


def _apply_event_class_log_offsets(probs: pd.DataFrame, offsets: dict[str, float] | None) -> pd.DataFrame:
    if not offsets:
        return probs.copy()
    cols = [f"p_{cls}" for cls in EVENT_CLASSES]
    values = np.clip(probs[cols].to_numpy(dtype=float), 1e-9, 1.0)
    logits = np.log(values) + np.asarray([float(offsets.get(cls, 0.0)) for cls in EVENT_CLASSES])
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    return pd.DataFrame(calibrated, index=probs.index, columns=cols)


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return math.log(value / (1.0 - value))


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def _fit_conditional_xbh_logit_offsets(df: pd.DataFrame, probs: pd.DataFrame) -> dict[str, float]:
    """Calibrate the conditional XBH hierarchy on an earlier date block."""
    counts = _event_count_matrix(df)
    hit_count = counts[["single", "double", "triple", "hr"]].sum(axis=1)
    xbh_count = counts[["double", "triple", "hr"]].sum(axis=1)
    non_hr_xbh_count = counts[["double", "triple"]].sum(axis=1)
    specs = {
        "xbh_given_hit": (xbh_count, hit_count, 500.0),
        "hr_given_xbh": (counts["hr"], xbh_count, 250.0),
        "triple_given_non_hr_xbh": (counts["triple"], non_hr_xbh_count, 120.0),
    }
    p_hit = probs[["p_single", "p_double", "p_triple", "p_hr"]].sum(axis=1).clip(1e-6)
    p_xbh = probs[["p_double", "p_triple", "p_hr"]].sum(axis=1).clip(1e-6)
    p_non_hr_xbh = probs[["p_double", "p_triple"]].sum(axis=1).clip(1e-6)
    predicted = {
        "xbh_given_hit": (p_xbh / p_hit).clip(1e-6, 1.0 - 1e-6),
        "hr_given_xbh": (probs["p_hr"] / p_xbh).clip(1e-6, 1.0 - 1e-6),
        "triple_given_non_hr_xbh": (probs["p_triple"] / p_non_hr_xbh).clip(1e-6, 1.0 - 1e-6),
    }
    offsets: dict[str, float] = {}
    for name, (positive, denominator, prior_strength) in specs.items():
        total = float(denominator.sum())
        if total <= 0:
            offsets[name] = 0.0
            continue
        actual_rate = float(positive.sum()) / total
        predicted_rate = float(np.average(predicted[name].to_numpy(dtype=float), weights=denominator))
        shrink = total / (total + prior_strength)
        offsets[name] = float(np.clip((_logit(actual_rate) - _logit(predicted_rate)) * shrink, -1.25, 1.25))
    return offsets


def _apply_conditional_xbh_logit_offsets(
    probs: pd.DataFrame,
    offsets: dict[str, float] | None,
) -> pd.DataFrame:
    if not offsets:
        return probs.copy()
    out = probs.copy()
    p_hit = out[["p_single", "p_double", "p_triple", "p_hr"]].sum(axis=1).to_numpy(dtype=float)
    p_non_hit = np.clip(1.0 - p_hit, 0.0, 1.0)
    p_walk_share = (
        out["p_walk"].to_numpy(dtype=float) / np.clip(p_non_hit, 1e-8, None)
    ).clip(0.0, 1.0)
    p_xbh_raw = out[["p_double", "p_triple", "p_hr"]].sum(axis=1).to_numpy(dtype=float)
    p_xbh_given_hit = p_xbh_raw / np.clip(p_hit, 1e-8, None)
    p_hr_given_xbh = out["p_hr"].to_numpy(dtype=float) / np.clip(p_xbh_raw, 1e-8, None)
    p_non_hr_xbh_raw = out[["p_double", "p_triple"]].sum(axis=1).to_numpy(dtype=float)
    p_triple_given_non_hr = out["p_triple"].to_numpy(dtype=float) / np.clip(p_non_hr_xbh_raw, 1e-8, None)
    p_xbh_given_hit = _sigmoid_array(
        np.log(np.clip(p_xbh_given_hit, 1e-6, 1 - 1e-6) / np.clip(1.0 - p_xbh_given_hit, 1e-6, 1.0))
        + float(offsets.get("xbh_given_hit", 0.0))
    )
    p_hr_given_xbh = _sigmoid_array(
        np.log(np.clip(p_hr_given_xbh, 1e-6, 1 - 1e-6) / np.clip(1.0 - p_hr_given_xbh, 1e-6, 1.0))
        + float(offsets.get("hr_given_xbh", 0.0))
    )
    p_triple_given_non_hr = _sigmoid_array(
        np.log(np.clip(p_triple_given_non_hr, 1e-6, 1 - 1e-6) / np.clip(1.0 - p_triple_given_non_hr, 1e-6, 1.0))
        + float(offsets.get("triple_given_non_hr_xbh", 0.0))
    )
    p_xbh = p_hit * p_xbh_given_hit
    p_hr = p_xbh * p_hr_given_xbh
    p_non_hr_xbh = p_xbh - p_hr
    p_triple = p_non_hr_xbh * p_triple_given_non_hr
    p_double = p_non_hr_xbh - p_triple
    out["p_single"] = p_hit - p_xbh
    out["p_double"] = p_double
    out["p_triple"] = p_triple
    out["p_hr"] = p_hr
    out["p_walk"] = p_non_hit * p_walk_share
    out["p_out"] = p_non_hit - out["p_walk"].to_numpy(dtype=float)
    return out[[f"p_{cls}" for cls in EVENT_CLASSES]].clip(1e-9, 1.0)


def _mae_from_metric(metrics: dict[str, Any], key: str) -> float | None:
    try:
        value = ((metrics.get(key) or {}).get("mae"))
        return None if value is None else float(value)
    except Exception:
        return None


def _event_model_composite(metrics: dict[str, Any]) -> float | None:
    brier = metrics.get("weighted_event_brier")
    hits = _mae_from_metric(metrics, "hits")
    tb = _mae_from_metric(metrics, "total_bases")
    hr = _mae_from_metric(metrics, "home_runs")
    pieces = [float(v) for v in (brier, hits, tb, hr) if v is not None and math.isfinite(float(v))]
    if not pieces:
        return None
    # Brier drives probability quality; count MAE keeps the event curve honest
    # for hits/TB/HR pricing.
    return (
        (float(brier) * 4.0 if brier is not None else 0.0)
        + (float(hits) * 0.20 if hits is not None else 0.0)
        + (float(tb) * 0.10 if tb is not None else 0.0)
        + (float(hr) * 0.35 if hr is not None else 0.0)
    )


def _safe_auc(y_true: Any, prob: Any) -> float | None:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def _safe_log_loss(y_true: Any, prob: Any) -> float | None:
    try:
        return float(log_loss(y_true, np.clip(prob, 1e-5, 1 - 1e-5), labels=[0, 1]))
    except Exception:
        return None


def _prop_projection_metrics(holdout: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    mapping = {
        "hits": ("actual_hits", "model_pred_hits"),
        "total_bases": ("actual_total_bases", "model_pred_total_bases"),
        "home_runs": ("actual_home_runs", "model_pred_home_runs"),
    }
    prop_rows = holdout[holdout["prop_example_rows"].fillna(0) > 0].copy()
    for name, (target, pred_col) in mapping.items():
        if prop_rows.empty or pred_col not in prop_rows:
            out[name] = {"rows": 0}
            continue
        mask = prop_rows[pred_col].notna() & prop_rows[target].notna()
        if not mask.any():
            out[name] = {"rows": 0}
            continue
        out[name] = _count_metrics(prop_rows.loc[mask, target], prop_rows.loc[mask, pred_col])
    return out


def train_hitter_player_game_outcomes(cfg: HitterOutcomeModelConfig) -> dict[str, Any]:
    df = _load(cfg)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_release_id": hitter_model_release_id(),
        "source": "features.mlb_hitter_player_game_training",
        "rows": int(len(df)),
        "status": "ok",
    }
    if df.empty:
        payload["status"] = "no_rows"
        _write_outputs(payload, cfg, models=None)
        return payload

    df, player_prior_state = add_leakage_safe_player_priors(df)
    payload["player_prior_summary"] = {
        "method": "date_shifted_empirical_bayes",
        "players": int(len(player_prior_state)),
        "features": list(_PLAYER_PRIOR_DEFAULTS),
    }
    train, holdout = _split(df, cfg.holdout_days)
    payload["train_rows"] = int(len(train))
    payload["holdout_rows"] = int(len(holdout))
    payload["holdout_start"] = str(min(holdout["game_date_et"])) if len(holdout) else None
    payload["holdout_end"] = str(max(holdout["game_date_et"])) if len(holdout) else None
    if len(train) < cfg.min_train_rows or len(holdout) < cfg.min_holdout_rows:
        payload["status"] = "insufficient_rows"
        _write_outputs(payload, cfg, models=None)
        return payload

    models: dict[str, Any] = {"player_prior_state": player_prior_state}
    metrics: dict[str, Any] = {}
    feature_policy = _load_head_feature_policy(cfg.model_dir)
    head_numeric_features = {
        str(head): [name for name in features if name in NUMERIC_FEATURES]
        for head, features in (feature_policy.get("numeric_features_by_head") or {}).items()
    }
    pa_numeric_features = list(head_numeric_features.get("pa") or NUMERIC_FEATURES)
    head_categorical_features = [
        name for name in (feature_policy.get("categorical_features") or CATEGORICAL_FEATURES)
        if name in CATEGORICAL_FEATURES
    ]
    payload["head_feature_policy"] = {
        "status": "applied" if feature_policy else "default_full_features",
        "generated_at_utc": feature_policy.get("generated_at_utc") if feature_policy else None,
        "heads": sorted(head_numeric_features),
        "categorical_features": head_categorical_features,
    }
    train_features = prepare_hitter_outcome_features(train)
    holdout_features = prepare_hitter_outcome_features(holdout)

    pa_model = _regression_pipeline(
        alpha=10.0,
        numeric_features=pa_numeric_features,
        categorical_features=head_categorical_features,
    )
    pa_model.fit(train_features[pa_numeric_features + head_categorical_features], train["actual_pa"])
    pa_ridge_pred = np.clip(pa_model.predict(holdout_features[pa_numeric_features + head_categorical_features]), 0.0, 6.5)
    pa_boosted_model = _boosted_count_pipeline(pa_numeric_features, head_categorical_features)
    pa_boosted_model.fit(
        train_features[pa_numeric_features + head_categorical_features],
        train["actual_pa"],
    )
    pa_boosted_pred = np.clip(
        pa_boosted_model.predict(holdout_features[pa_numeric_features + head_categorical_features]),
        0.0,
        7.0,
    )
    pa_boosted_use = bool(
        mean_absolute_error(holdout["actual_pa"], pa_boosted_pred)
        < mean_absolute_error(holdout["actual_pa"], pa_ridge_pred) - 0.002
    )
    pa_single_pred = pa_boosted_pred if pa_boosted_use else pa_ridge_pred
    selected_pa_model = pa_boosted_model if pa_boosted_use else pa_model
    pa_uncertainty = _fit_pa_uncertainty(train, cfg.holdout_days)
    models["pa_uncertainty"] = pa_uncertainty
    metrics["pa_uncertainty"] = pa_uncertainty

    low_target_train = (pd.to_numeric(train["actual_pa"], errors="coerce") <= 2).astype(int)
    low_target_holdout = (pd.to_numeric(holdout["actual_pa"], errors="coerce") <= 2).astype(int)
    pa_low_model = _boosted_binary_pipeline(pa_numeric_features, head_categorical_features)
    pa_low_model.fit(train_features[pa_numeric_features + head_categorical_features], low_target_train)
    pa_low_prob = np.clip(
        pa_low_model.predict_proba(holdout_features[pa_numeric_features + head_categorical_features])[:, 1],
        1e-5,
        1.0 - 1e-5,
    )
    normal_train_mask = pd.to_numeric(train["actual_pa"], errors="coerce") >= 3
    normal_holdout_mask = pd.to_numeric(holdout["actual_pa"], errors="coerce") >= 3
    pa_normal_model = _boosted_count_pipeline(pa_numeric_features, head_categorical_features)
    pa_normal_model.fit(
        train_features.loc[normal_train_mask, pa_numeric_features + head_categorical_features],
        train.loc[normal_train_mask, "actual_pa"],
    )
    pa_normal_pred = np.clip(
        pa_normal_model.predict(holdout_features[pa_numeric_features + head_categorical_features]),
        3.0,
        7.0,
    )
    low_states = ((pa_uncertainty.get("global") or {}).get("low_pa_state_probs") or {"0": 0.05, "1": 0.20, "2": 0.75})
    low_total = sum(float(low_states.get(str(n), 0.0)) for n in range(3)) or 1.0
    low_pa_mean = sum(n * float(low_states.get(str(n), 0.0)) for n in range(3)) / low_total
    pa_two_part_pred = np.clip(pa_low_prob * low_pa_mean + (1.0 - pa_low_prob) * pa_normal_pred, 0.0, 7.0)
    single_mae = float(mean_absolute_error(holdout["actual_pa"], pa_single_pred))
    two_part_mae = float(mean_absolute_error(holdout["actual_pa"], pa_two_part_pred))
    low_base_prob = float(low_target_train.mean())
    low_model_brier = float(brier_score_loss(low_target_holdout, pa_low_prob))
    low_base_brier = float(brier_score_loss(low_target_holdout, np.full(len(holdout), low_base_prob)))
    lineup_source = holdout.get("lineup_source", pd.Series("unknown", index=holdout.index)).fillna("unknown").astype(str).str.lower()
    immutable_low_pa_mask = ~lineup_source.str.contains("boxscore|postgame|actual", regex=True)
    lineup_proxy_mask = lineup_source.str.contains("boxscore", regex=False) & pd.to_numeric(
        holdout.get("lineup_slot"), errors="coerce"
    ).between(1, 9)
    distribution_validation_mask = immutable_low_pa_mask | lineup_proxy_mask
    safe_low_pa_rows = int(immutable_low_pa_mask.sum())
    distribution_rows = int(distribution_validation_mask.sum())
    if distribution_rows >= 200 and low_target_holdout.loc[distribution_validation_mask].nunique() >= 2:
        distribution_low_brier = float(brier_score_loss(
            low_target_holdout.loc[distribution_validation_mask],
            pa_low_prob[distribution_validation_mask.to_numpy()],
        ))
        distribution_low_base_brier = float(brier_score_loss(
            low_target_holdout.loc[distribution_validation_mask],
            np.full(distribution_rows, low_base_prob),
        ))
    else:
        distribution_low_brier = None
        distribution_low_base_brier = None
    pa_distribution_use = bool(
        distribution_low_brier is not None
        and distribution_low_base_brier is not None
        and distribution_low_brier < distribution_low_base_brier
    )
    pa_two_part_use = bool(
        two_part_mae < single_mae
        and pa_distribution_use
    )
    pa_pred = pa_two_part_pred if pa_two_part_use else pa_single_pred
    holdout["pa_low_probability"] = pa_low_prob
    holdout["pa_normal_mean"] = pa_normal_pred
    slot_pa_pred = _slot_prior(train, holdout, "actual_pa")
    projected_pa_mask = holdout["projected_pa"].notna()
    metrics["pa_model"] = {
        "model": _count_metrics(holdout["actual_pa"], pa_pred),
        "single_mean_model": _count_metrics(holdout["actual_pa"], pa_single_pred),
        "ridge_model": _count_metrics(holdout["actual_pa"], pa_ridge_pred),
        "boosted_model": _count_metrics(holdout["actual_pa"], pa_boosted_pred),
        "single_mean_selected": "boosted_lgbm" if pa_boosted_use else "ridge",
        "two_part_model": _count_metrics(holdout["actual_pa"], pa_two_part_pred),
        "two_part": {
            "enabled": pa_two_part_use,
            "low_pa_rows": int(low_target_holdout.sum()),
            "low_pa_rate": float(low_target_holdout.mean()),
            "low_pa_model_brier": low_model_brier,
            "low_pa_baseline_brier": low_base_brier,
            "distribution_enabled": pa_distribution_use,
            "immutable_lock_low_pa_rows": safe_low_pa_rows,
            "distribution_validation_rows": distribution_rows,
            "distribution_validation_brier": distribution_low_brier,
            "distribution_validation_baseline_brier": distribution_low_base_brier,
            "activation_reason": (
                "two_part_mean_and_distribution_gain" if pa_two_part_use
                else "distribution_only_gain" if pa_distribution_use
                else "insufficient_temporal_lineup_rows" if distribution_rows < 200
                else "no_temporal_distribution_brier_gain"
            ),
            "normal_pa_rows": int(normal_holdout_mask.sum()),
            "normal_pa_mae": float(mean_absolute_error(
                holdout.loc[normal_holdout_mask, "actual_pa"],
                pa_normal_pred[normal_holdout_mask.to_numpy()],
            )) if normal_holdout_mask.any() else None,
            "low_pa_conditional_mean": low_pa_mean,
        },
        "slot_prior": _count_metrics(holdout["actual_pa"], slot_pa_pred),
        "existing_projected_pa": (
            _count_metrics(holdout.loc[projected_pa_mask, "actual_pa"], holdout.loc[projected_pa_mask, "projected_pa"])
            if projected_pa_mask.any()
            else {"rows": 0}
        ),
    }
    models["pa_model"] = selected_pa_model
    models["pa_ridge_model"] = pa_model
    models["pa_boosted_model"] = pa_boosted_model
    models["pa_low_model"] = pa_low_model
    models["pa_normal_model"] = pa_normal_model
    models["pa_two_part_use"] = pa_two_part_use
    models["pa_distribution_use"] = pa_distribution_use

    rate_train = train[train["actual_pa"] > 0].copy()
    rate_holdout = holdout[holdout["actual_pa"] > 0].copy()
    holdout_rate_index = rate_holdout.index
    pa_pred_rates = pd.Series(pa_pred, index=holdout.index).loc[holdout_rate_index].to_numpy(dtype=float)
    rate_train_features = prepare_hitter_outcome_features(rate_train)
    rate_holdout_features = prepare_hitter_outcome_features(rate_holdout)
    rate_predictions: dict[str, np.ndarray] = {}
    rate_metrics: dict[str, Any] = {}
    for rate_name, count_col in RATE_TARGETS.items():
        rate_train[rate_name] = (rate_train[count_col] / rate_train["actual_pa"]).clip(lower=0.0, upper=1.0)
        model = _regression_pipeline(alpha=12.0)
        model.fit(rate_train_features[NUMERIC_FEATURES + CATEGORICAL_FEATURES], rate_train[rate_name])
        pred_rate = _clip(model.predict(rate_holdout_features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]), 0.0, 0.9)
        rate_predictions[rate_name] = pred_rate
        pred_count = pred_rate * pa_pred_rates
        prior_count = _rate_prior(train, rate_holdout, count_col, pa_pred_rates)
        rate_metrics[rate_name] = {
            "rate_rows": int(len(rate_holdout)),
            "count_model": _count_metrics(rate_holdout[count_col], pred_count),
            "slot_rate_prior": _count_metrics(rate_holdout[count_col], prior_count),
        }
        models[rate_name] = model
    metrics["rate_models"] = rate_metrics

    single = rate_predictions["single_rate"]
    double = rate_predictions["double_rate"]
    triple = rate_predictions["triple_rate"]
    hr = rate_predictions["hr_rate"]
    hit_count = pa_pred_rates * (single + double + triple + hr)
    tb_count = pa_pred_rates * (single + 2.0 * double + 3.0 * triple + 4.0 * hr)
    hr_count = pa_pred_rates * hr
    metrics["structured_counts"] = {
        "rows": int(len(rate_holdout)),
        "hits": _count_metrics(rate_holdout["actual_hits"], hit_count),
        "total_bases": _count_metrics(rate_holdout["actual_total_bases"], tb_count),
        "home_runs": _count_metrics(rate_holdout["actual_home_runs"], hr_count),
        "slot_prior_hits": _count_metrics(
            rate_holdout["actual_hits"],
            _rate_prior(train, rate_holdout, "actual_hits", pa_pred_rates),
        ),
        "slot_prior_total_bases": _count_metrics(
            rate_holdout["actual_total_bases"],
            _rate_prior(train, rate_holdout, "actual_total_bases", pa_pred_rates),
        ),
        "slot_prior_home_runs": _count_metrics(
            rate_holdout["actual_home_runs"],
            _rate_prior(train, rate_holdout, "actual_home_runs", pa_pred_rates),
        ),
    }

    # Test direct player-game count challengers against the exact legacy
    # forecasts used live. They activate independently after date-purged OOF
    # gains; a failed head remains diagnostic and receives zero blend weight.
    direct_specs = (
        ("hits", "actual_hits", "model_pred_hits", _boosted_count_pipeline, 0.002, 0.40, False, 4.0),
        ("tb", "actual_total_bases", "model_pred_total_bases", _boosted_count_pipeline, 0.002, 0.75, False, 8.0),
        ("hr", "actual_home_runs", "model_pred_home_runs", _boosted_poisson_count_pipeline, 0.0002, 0.15, True, 3.0),
    )
    for prefix, actual_col, base_col, factory, minimum_gain, max_offset, require_brier, upper in direct_specs:
        numeric_features = list(head_numeric_features.get(f"{prefix}_count") or NUMERIC_FEATURES)
        count_model = factory(numeric_features, head_categorical_features)
        count_model.fit(
            train_features[numeric_features + head_categorical_features],
            train[actual_col],
        )
        direct_prediction = np.clip(
            count_model.predict(holdout_features[numeric_features + head_categorical_features]),
            0.0,
            upper,
        )
        policy = _select_count_repair(
            holdout,
            direct_prediction,
            actual_column=actual_col,
            base_column=base_col,
            repair_name=prefix,
            minimum_gain=minimum_gain,
            max_bias_offset=max_offset,
            require_any_brier_gain=require_brier,
        )
        metrics[f"direct_{prefix}_count_repair"] = {
            **policy,
            "all_holdout": _count_metrics(holdout[actual_col], direct_prediction),
            "numeric_features": numeric_features,
            "categorical_features": head_categorical_features,
        }
        models[f"{prefix}_count_model"] = count_model
        models[f"{prefix}_count_numeric_features"] = numeric_features
        models[f"{prefix}_count_categorical_features"] = head_categorical_features
        models[f"{prefix}_count_blend_alpha"] = float(policy.get("alpha") or 0.0)
        models[f"{prefix}_count_bias_offset"] = float(policy.get("production_bias_offset") or 0.0)

    event_train = train[train["actual_pa"] > 0].copy()
    event_holdout = holdout[holdout["actual_pa"] > 0].copy()
    train_event_rows = int(_event_count_matrix(event_train).sum(axis=1).sum())
    event_metrics: dict[str, Any] = {
        "train_player_games": int(len(event_train)),
        "train_event_rows": train_event_rows,
        "holdout_rows": int(len(event_holdout)),
    }
    if len(event_train) > 0 and len(event_holdout) > 0:
        event_pa_pred = pd.Series(pa_pred, index=holdout.index).loc[event_holdout.index].to_numpy(dtype=float)
        event_low_pa_prob = pd.Series(pa_low_prob, index=holdout.index).loc[event_holdout.index].to_numpy(dtype=float)
        event_normal_pa_pred = pd.Series(pa_normal_pred, index=holdout.index).loc[event_holdout.index].to_numpy(dtype=float)
        active_event_probs: pd.DataFrame | None = None
        try:
            hierarchical_models = _fit_hierarchical_event_models(
                event_train,
                categorical_features=head_categorical_features,
                head_numeric_features={
                    head: features for head, features in head_numeric_features.items()
                    if head in _HIERARCHICAL_EVENT_HEADS
                },
            )
        except Exception as exc:
            event_metrics["hierarchical_conditional_error"] = str(exc)
            hierarchical_models = {}
        if len(hierarchical_models) == len(_HIERARCHICAL_EVENT_HEADS):
            hierarchical_probs = _predict_hierarchical_event_probabilities(
                hierarchical_models,
                event_holdout,
                categorical_features=head_categorical_features,
            )
            hierarchical_metrics = _event_projection_metrics(
                event_holdout,
                hierarchical_probs,
                event_pa_pred,
            )
            event_metrics["hierarchical_conditional"] = {
                "heads": sorted(hierarchical_models.keys()),
                **hierarchical_metrics,
            }
            hierarchical_score = _event_model_composite(hierarchical_metrics)
            event_metrics["model_selection"] = {
                "hierarchical_composite": hierarchical_score,
                "hierarchical_brier": hierarchical_metrics.get("weighted_event_brier"),
                "hierarchical_log_loss": hierarchical_metrics.get("weighted_event_log_loss"),
                "selected": "hierarchical_conditional_lgbm",
            }
            models["event_hierarchical_models"] = hierarchical_models
            event_metrics.update({
                "active_event_model": "hierarchical_conditional_lgbm",
                "classes": list(EVENT_CLASSES),
                **hierarchical_metrics,
            })
            active_event_probs = hierarchical_probs
            event_metrics["tb_state_distribution"] = _tb_state_metrics(
                event_holdout,
                hierarchical_probs,
                event_pa_pred,
                pa_uncertainty,
            )
        else:
            event_metrics["status"] = "insufficient_hierarchical_heads"

        if cfg.fit_independent_boosted_candidate:
            X_event, y_event, w_event = _build_event_training_examples(event_train)
            try:
                boosted_models = _fit_boosted_event_binary_models(X_event, y_event, w_event)
            except Exception as exc:
                event_metrics["boosted_binary_error"] = str(exc)
                boosted_models = {}
            if len(boosted_models) >= 4:
                boosted_probs = _predict_boosted_event_probabilities(boosted_models, event_holdout)
                boosted_metrics = _event_projection_metrics(event_holdout, boosted_probs, event_pa_pred)
                event_metrics["boosted_binary_benchmark"] = {
                    "classes": sorted(boosted_models.keys()),
                    **boosted_metrics,
                }
                event_metrics["model_selection"]["boosted_composite"] = _event_model_composite(boosted_metrics)
                models["event_binary_models"] = boosted_models

        if active_event_probs is not None:
            event_dates = pd.to_datetime(event_holdout["game_date_et"]).dt.date
            unique_event_dates = sorted(event_dates.unique())
            if len(unique_event_dates) >= 4:
                calibration_start = unique_event_dates[max(1, len(unique_event_dates) // 2)]
                event_calibration_mask = event_dates < calibration_start
                event_validation_mask = ~event_calibration_mask
                calibration_offsets = _fit_event_class_log_offsets(
                    event_holdout.loc[event_calibration_mask],
                    active_event_probs.loc[event_calibration_mask],
                )
                calibrated_event_probs = _apply_event_class_log_offsets(active_event_probs, calibration_offsets)
                validation_positions = event_validation_mask.to_numpy()
                base_event_validation = _event_projection_metrics(
                    event_holdout.loc[event_validation_mask],
                    active_event_probs.loc[event_validation_mask],
                    event_pa_pred[validation_positions],
                )
                calibrated_event_validation = _event_projection_metrics(
                    event_holdout.loc[event_validation_mask],
                    calibrated_event_probs.loc[event_validation_mask],
                    event_pa_pred[validation_positions],
                )
                base_tb_mae = _mae_from_metric(base_event_validation, "total_bases")
                calibrated_tb_mae = _mae_from_metric(calibrated_event_validation, "total_bases")
                base_hr_mae = _mae_from_metric(base_event_validation, "home_runs")
                calibrated_hr_mae = _mae_from_metric(calibrated_event_validation, "home_runs")
                base_event_brier = base_event_validation.get("weighted_event_brier")
                calibrated_event_brier = calibrated_event_validation.get("weighted_event_brier")
                event_calibration_enabled = bool(
                    base_tb_mae is not None
                    and calibrated_tb_mae is not None
                    and calibrated_tb_mae < base_tb_mae
                    and base_hr_mae is not None
                    and calibrated_hr_mae is not None
                    and calibrated_hr_mae <= base_hr_mae
                    and base_event_brier is not None
                    and calibrated_event_brier is not None
                    and calibrated_event_brier <= base_event_brier
                )
                event_metrics["event_class_calibration"] = {
                    "method": "temporal_multinomial_intercept_offsets",
                    "calibration_end": str(unique_event_dates[max(0, len(unique_event_dates) // 2 - 1)]),
                    "validation_start": str(calibration_start),
                    "calibration_rows": int(event_calibration_mask.sum()),
                    "validation_rows": int(event_validation_mask.sum()),
                    "candidate_offsets": calibration_offsets,
                    "base_validation": base_event_validation,
                    "calibrated_validation": calibrated_event_validation,
                    "enabled": event_calibration_enabled,
                    "reason": "tb_hr_and_event_brier_improved" if event_calibration_enabled else "no_joint_validation_gain",
                }
                if event_calibration_enabled:
                    production_offsets = _fit_event_class_log_offsets(event_holdout, active_event_probs)
                    active_event_probs = calibrated_event_probs
                    models["event_class_log_offsets"] = production_offsets
                    event_metrics["event_class_calibration"]["production_offsets"] = production_offsets
                    event_metrics["active_event_model"] = f"{event_metrics.get('active_event_model')}_event_rate_calibrated"

                xbh_offsets = _fit_conditional_xbh_logit_offsets(
                    event_holdout.loc[event_calibration_mask],
                    active_event_probs.loc[event_calibration_mask],
                )
                xbh_candidate = _apply_conditional_xbh_logit_offsets(active_event_probs, xbh_offsets)
                xbh_base_validation = _event_projection_metrics(
                    event_holdout.loc[event_validation_mask],
                    active_event_probs.loc[event_validation_mask],
                    event_pa_pred[validation_positions],
                )
                xbh_candidate_validation = _event_projection_metrics(
                    event_holdout.loc[event_validation_mask],
                    xbh_candidate.loc[event_validation_mask],
                    event_pa_pred[validation_positions],
                )
                base_xbh_tb_mae = _mae_from_metric(xbh_base_validation, "total_bases")
                candidate_xbh_tb_mae = _mae_from_metric(xbh_candidate_validation, "total_bases")
                base_xbh_brier = xbh_base_validation.get("weighted_event_brier")
                candidate_xbh_brier = xbh_candidate_validation.get("weighted_event_brier")
                base_xbh_hr_mae = _mae_from_metric(xbh_base_validation, "home_runs")
                candidate_xbh_hr_mae = _mae_from_metric(xbh_candidate_validation, "home_runs")
                xbh_enabled = bool(
                    base_xbh_tb_mae is not None
                    and candidate_xbh_tb_mae is not None
                    and candidate_xbh_tb_mae < base_xbh_tb_mae
                    and base_xbh_brier is not None
                    and candidate_xbh_brier is not None
                    and candidate_xbh_brier <= base_xbh_brier
                    and base_xbh_hr_mae is not None
                    and candidate_xbh_hr_mae is not None
                    and candidate_xbh_hr_mae <= base_xbh_hr_mae + 0.002
                )
                event_metrics["conditional_xbh_calibration"] = {
                    "method": "temporal_empirical_bayes_conditional_logit_offsets",
                    "calibration_rows": int(event_calibration_mask.sum()),
                    "validation_rows": int(event_validation_mask.sum()),
                    "candidate_offsets": xbh_offsets,
                    "base_validation": xbh_base_validation,
                    "candidate_validation": xbh_candidate_validation,
                    "enabled": xbh_enabled,
                    "reason": "tb_mae_and_event_brier_improved" if xbh_enabled else "no_joint_validation_gain",
                }
                if xbh_enabled:
                    production_xbh_offsets = _fit_conditional_xbh_logit_offsets(event_holdout, active_event_probs)
                    active_event_probs = xbh_candidate
                    models["conditional_xbh_logit_offsets"] = production_xbh_offsets
                    event_metrics["conditional_xbh_calibration"]["production_offsets"] = production_xbh_offsets
                    event_metrics["active_event_model"] = f"{event_metrics.get('active_event_model')}_conditional_xbh_calibrated"
            try:
                tb_state_models = _fit_tb_state_models(
                    event_train,
                    pa_numeric_features,
                    head_categorical_features,
                )
                hierarchical_tb_models = _fit_hierarchical_tb_state_models(
                    event_train,
                    pa_numeric_features,
                    head_categorical_features,
                )
                direct_candidates: dict[str, pd.DataFrame] = {}
                model_candidates: dict[str, dict[str, Pipeline]] = {}
                candidate_tail_offsets: dict[str, float] = {}
                production_tail_offsets: dict[str, float] = {}
                holdout_state_labels = _tb_state_labels(event_holdout)
                holdout_dates = sorted(pd.to_datetime(event_holdout["game_date_et"]).dt.date.unique())
                tail_calibration_start = holdout_dates[max(1, len(holdout_dates) // 2)]
                tail_calibration_mask = (
                    pd.to_datetime(event_holdout["game_date_et"]).dt.date < tail_calibration_start
                )
                if len(tb_state_models) == len(TB_STATE_NAMES):
                    one_vs_rest_raw = _predict_tb_state_models(
                        tb_state_models,
                        event_holdout,
                        pa_numeric_features,
                        head_categorical_features,
                    )
                    one_vs_rest_offset = _fit_tb_hr_tail_logit_offset(
                        holdout_state_labels.loc[tail_calibration_mask],
                        one_vs_rest_raw.loc[tail_calibration_mask],
                    )
                    direct_candidates["one_vs_rest"] = _apply_tb_hr_tail_logit_offset(
                        one_vs_rest_raw, one_vs_rest_offset
                    )
                    candidate_tail_offsets["one_vs_rest"] = one_vs_rest_offset
                    production_tail_offsets["one_vs_rest"] = _fit_tb_hr_tail_logit_offset(
                        holdout_state_labels, one_vs_rest_raw
                    )
                    model_candidates["one_vs_rest"] = tb_state_models
                if len(hierarchical_tb_models) == len(_TB_STATE_HIERARCHICAL_HEADS):
                    hierarchical_raw = _predict_hierarchical_tb_state_models(
                        hierarchical_tb_models,
                        event_holdout,
                        pa_numeric_features,
                        head_categorical_features,
                    )
                    hierarchical_offset = _fit_tb_hr_tail_logit_offset(
                        holdout_state_labels.loc[tail_calibration_mask],
                        hierarchical_raw.loc[tail_calibration_mask],
                    )
                    direct_candidates["hierarchical_hr_tail"] = _apply_tb_hr_tail_logit_offset(
                        hierarchical_raw, hierarchical_offset
                    )
                    candidate_tail_offsets["hierarchical_hr_tail"] = hierarchical_offset
                    production_tail_offsets["hierarchical_hr_tail"] = _fit_tb_hr_tail_logit_offset(
                        holdout_state_labels, hierarchical_raw
                    )
                    model_candidates["hierarchical_hr_tail"] = hierarchical_tb_models
                if direct_candidates:
                    base_state_probs = _convolution_tb_state_frame(
                        event_holdout,
                        active_event_probs,
                        event_pa_pred,
                        pa_uncertainty,
                        event_low_pa_prob if pa_distribution_use else None,
                        event_normal_pa_pred if pa_distribution_use else None,
                    )
                    blend_policy = _select_tb_state_blend(event_holdout, base_state_probs, direct_candidates)
                    candidate_metrics = {
                        name: _tb_state_multiclass_metrics(_tb_state_labels(event_holdout), probabilities)
                        for name, probabilities in direct_candidates.items()
                    }
                    selected_candidate = str(blend_policy.get("selected_candidate") or "convolution")
                    event_metrics["tb_state_residual"] = {
                        "method": "gated_direct_state_candidates_with_hr_tail_hierarchy",
                        "states": list(TB_STATE_NAMES),
                        "base_holdout": _tb_state_multiclass_metrics(_tb_state_labels(event_holdout), base_state_probs),
                        "candidate_holdout": candidate_metrics,
                        "candidate_hr_tail_logit_offsets": candidate_tail_offsets,
                        "production_hr_tail_logit_offsets": production_tail_offsets,
                        "tail_calibration_end": str(holdout_dates[max(0, len(holdout_dates) // 2 - 1)]),
                        **blend_policy,
                    }
                    if selected_candidate in model_candidates:
                        models["tb_state_models"] = model_candidates[selected_candidate]
                        models["tb_state_model_kind"] = selected_candidate
                        models["tb_state_hr_tail_logit_offset"] = float(
                            production_tail_offsets.get(selected_candidate, 0.0)
                        )
                    models["tb_state_blend_alpha"] = float(blend_policy.get("alpha") or 0.0)
                else:
                    event_metrics["tb_state_residual"] = {
                        "enabled": False,
                        "reason": "insufficient_tb_state_heads",
                        "trained_one_vs_rest_heads": sorted(tb_state_models),
                        "trained_hierarchical_heads": sorted(hierarchical_tb_models),
                    }
            except Exception as exc:
                event_metrics["tb_state_residual"] = {"enabled": False, "reason": "training_error", "error": str(exc)}
    else:
        event_metrics["status"] = "insufficient_event_classes"
    metrics["direct_event_model"] = event_metrics

    hr_train = train[train["actual_pa"] > 0].copy()
    hr_holdout = holdout[holdout["actual_pa"] > 0].copy()
    hr_train["hr_any"] = (hr_train["actual_home_runs"] > 0).astype(int)
    hr_holdout["hr_any"] = (hr_holdout["actual_home_runs"] > 0).astype(int)
    hr_any_model = _boosted_binary_pipeline()
    hr_any_model.fit(
        prepare_hitter_outcome_features(hr_train)[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        hr_train["hr_any"],
    )
    hr_prob = hr_any_model.predict_proba(
        prepare_hitter_outcome_features(hr_holdout)[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    )[:, 1]
    hr_prior = float(hr_train["hr_any"].mean())
    prior_prob = np.full(len(hr_holdout), hr_prior, dtype=float)
    metrics["hr_any_model"] = {
        "rows": int(len(hr_holdout)),
        "model_brier": float(brier_score_loss(hr_holdout["hr_any"], hr_prob)),
        "prior_brier": float(brier_score_loss(hr_holdout["hr_any"], prior_prob)),
        "model_log_loss": _safe_log_loss(hr_holdout["hr_any"], hr_prob),
        "prior_log_loss": _safe_log_loss(hr_holdout["hr_any"], prior_prob),
        "auc": _safe_auc(hr_holdout["hr_any"], hr_prob),
        "base_rate": hr_prior,
    }
    models["hr_any_model"] = hr_any_model

    metrics["existing_prop_projection_holdout"] = _prop_projection_metrics(holdout)
    payload["metrics"] = metrics
    payload["feature_coverage"] = _feature_coverage(df)
    payload["recommendation"] = _recommend(metrics, len(holdout), cfg)
    _write_outputs(payload, cfg, models=models)
    return payload


def _feature_coverage(df: pd.DataFrame) -> dict[str, float]:
    coverage: dict[str, float] = {}
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if col not in df:
            coverage[col] = 0.0
        else:
            coverage[col] = float(df[col].notna().mean())
    return coverage


def _gain(model_metric: dict[str, Any], prior_metric: dict[str, Any], key: str = "mae") -> float | None:
    try:
        model = float(model_metric[key])
        prior = float(prior_metric[key])
    except Exception:
        return None
    return prior - model


def _recommend(metrics: dict[str, Any], holdout_rows: int, cfg: HitterOutcomeModelConfig) -> dict[str, Any]:
    pa_gain = _gain(metrics.get("pa_model", {}).get("model", {}), metrics.get("pa_model", {}).get("slot_prior", {}))
    hits_gain = _gain(
        metrics.get("structured_counts", {}).get("hits", {}),
        metrics.get("structured_counts", {}).get("slot_prior_hits", {}),
    )
    tb_gain = _gain(
        metrics.get("structured_counts", {}).get("total_bases", {}),
        metrics.get("structured_counts", {}).get("slot_prior_total_bases", {}),
    )
    hr_gain = _gain(
        metrics.get("structured_counts", {}).get("home_runs", {}),
        metrics.get("structured_counts", {}).get("slot_prior_home_runs", {}),
    )
    hr_any = metrics.get("hr_any_model", {})
    direct = metrics.get("direct_event_model", {})
    hits_count_repair = metrics.get("direct_hits_count_repair", {})
    tb_count_repair = metrics.get("direct_tb_count_repair", {})
    hr_count_repair = metrics.get("direct_hr_count_repair", {})
    event_tb_gain = _gain(
        direct.get("total_bases", {}),
        metrics.get("structured_counts", {}).get("total_bases", {}),
    )
    event_hits_gain = _gain(
        direct.get("hits", {}),
        metrics.get("structured_counts", {}).get("hits", {}),
    )
    event_hr_gain = _gain(
        direct.get("home_runs", {}),
        metrics.get("structured_counts", {}).get("home_runs", {}),
    )
    event_hits_gain_vs_prior = _gain(
        direct.get("hits", {}),
        metrics.get("structured_counts", {}).get("slot_prior_hits", {}),
    )
    event_tb_gain_vs_prior = _gain(
        direct.get("total_bases", {}),
        metrics.get("structured_counts", {}).get("slot_prior_total_bases", {}),
    )
    try:
        hr_any_gain = float(hr_any.get("prior_brier")) - float(hr_any.get("model_brier"))
    except Exception:
        hr_any_gain = None
    passes_basic_gate = bool(
        holdout_rows >= cfg.min_holdout_rows
        and pa_gain is not None and pa_gain > 0.01
        and event_hits_gain_vs_prior is not None and event_hits_gain_vs_prior > 0.005
        and event_tb_gain_vs_prior is not None and event_tb_gain_vs_prior > 0.005
        and event_tb_gain is not None and event_tb_gain > 0.0
        and hr_any_gain is not None and hr_any_gain > 0.0005
    )
    return {
        "production_status": "usable_for_distribution" if passes_basic_gate else "diagnostic_only",
        "reason": (
            "Holdout gates passed; event curves may feed distribution pricing."
            if passes_basic_gate
            else "Require repeated holdout gains before replacing prop projections."
        ),
        "holdout_rows": holdout_rows,
        "pa_mae_gain_vs_slot_prior": pa_gain,
        "hits_mae_gain_vs_slot_rate_prior": hits_gain,
        "tb_mae_gain_vs_slot_rate_prior": tb_gain,
        "hr_mae_gain_vs_slot_rate_prior": hr_gain,
        "direct_event_hits_mae_gain_vs_independent_rates": event_hits_gain,
        "direct_event_tb_mae_gain_vs_independent_rates": event_tb_gain,
        "direct_event_hr_mae_gain_vs_independent_rates": event_hr_gain,
        "direct_event_hits_mae_gain_vs_slot_prior": event_hits_gain_vs_prior,
        "direct_event_tb_mae_gain_vs_slot_prior": event_tb_gain_vs_prior,
        "hr_any_brier_gain_vs_prior": hr_any_gain,
        "direct_tb_count_repair_enabled": bool(tb_count_repair.get("enabled")),
        "direct_tb_count_repair_mae_gain": tb_count_repair.get("validation_mae_gain"),
        "direct_tb_count_repair_alpha": tb_count_repair.get("alpha"),
        "direct_hits_count_repair_enabled": bool(hits_count_repair.get("enabled")),
        "direct_hits_count_repair_mae_gain": hits_count_repair.get("validation_mae_gain"),
        "direct_hits_count_repair_alpha": hits_count_repair.get("alpha"),
        "direct_hr_count_repair_enabled": bool(hr_count_repair.get("enabled")),
        "direct_hr_count_repair_mae_gain": hr_count_repair.get("validation_mae_gain"),
        "direct_hr_count_repair_brier_gain": hr_count_repair.get("any_brier_gain"),
        "direct_hr_count_repair_alpha": hr_count_repair.get("alpha"),
        "passes_basic_gate": passes_basic_gate,
    }


def _write_outputs(payload: dict[str, Any], cfg: HitterOutcomeModelConfig, models: dict[str, Any] | None) -> None:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    json_path = cfg.model_dir / "hitter_player_game_outcome_models.json"
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    if models:
        artifact = {
            "models": models,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "event_classes": EVENT_CLASSES,
            "active_event_model": ((payload.get("metrics") or {}).get("direct_event_model") or {}).get("active_event_model"),
            "trained_at_utc": payload.get("generated_at_utc"),
            "model_release_id": payload.get("model_release_id"),
            "artifact_role": "nightly_challenger",
            "recommendation": payload.get("recommendation"),
            "metrics": payload.get("metrics") or {},
            "player_prior_state": models.get("player_prior_state") or {},
            "pa_uncertainty": models.get("pa_uncertainty") or {},
        }
        atomic_write_via(
            cfg.model_dir / HITTER_CHALLENGER_ARTIFACT,
            lambda temp: joblib.dump(artifact, temp),
            attempts=6,
            retry_all_errors=True,
        )
        # Keep the legacy filename as the latest challenger for diagnostics and
        # compatibility. Prediction and distribution pricing load the separately
        # pinned production artifact.
        atomic_write_via(
            cfg.model_dir / "hitter_player_game_outcome_models.joblib",
            lambda temp: joblib.dump(artifact, temp),
            attempts=6,
            retry_all_errors=True,
        )
        ensure_hitter_production_artifact(cfg.model_dir)
    _write_report(payload, cfg.report_file)


def _write_report(payload: dict[str, Any], report_file: str | None) -> None:
    path = _REPORT_DIR / (report_file or "mlb_hitter_player_game_outcome_models_latest.md")
    path.parent.mkdir(parents=True, exist_ok=True)

    def num(value: Any, digits: int = 3) -> str:
        try:
            if value is None:
                return "-"
            return f"{float(value):.{digits}f}"
        except Exception:
            return "-"

    metrics = payload.get("metrics", {})
    rec = payload.get("recommendation", {})
    pa = metrics.get("pa_model", {})
    structured = metrics.get("structured_counts", {})
    direct = metrics.get("direct_event_model", {})
    hits_count_repair = metrics.get("direct_hits_count_repair", {})
    tb_count_repair = metrics.get("direct_tb_count_repair", {})
    hr_count_repair = metrics.get("direct_hr_count_repair", {})
    hr_any = metrics.get("hr_any_model", {})
    prop = metrics.get("existing_prop_projection_holdout", {})

    lines = [
        "# MLB Hitter Player-Game Outcome Models",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        f"Model release: {payload.get('model_release_id', '-')}",
        f"Rows: {payload.get('rows', 0)} | Train: {payload.get('train_rows', 0)} | Holdout: {payload.get('holdout_rows', 0)}",
        f"Holdout: {payload.get('holdout_start')} to {payload.get('holdout_end')}",
        f"Status: {payload.get('status')}",
        "",
        "## Recommendation",
        "",
        f"- Production status: {rec.get('production_status', 'unknown')}",
        f"- Passes basic gate: {rec.get('passes_basic_gate', False)}",
        f"- PA MAE gain vs slot prior: {num(rec.get('pa_mae_gain_vs_slot_prior'))}",
        f"- Hits MAE gain vs slot-rate prior: {num(rec.get('hits_mae_gain_vs_slot_rate_prior'))}",
        f"- TB MAE gain vs slot-rate prior: {num(rec.get('tb_mae_gain_vs_slot_rate_prior'))}",
        f"- HR MAE gain vs slot-rate prior: {num(rec.get('hr_mae_gain_vs_slot_rate_prior'))}",
        f"- Direct event hits MAE gain vs slot prior: {num(rec.get('direct_event_hits_mae_gain_vs_slot_prior'))}",
        f"- Direct event TB MAE gain vs slot prior: {num(rec.get('direct_event_tb_mae_gain_vs_slot_prior'))}",
        f"- Direct event TB MAE gain vs independent rates: {num(rec.get('direct_event_tb_mae_gain_vs_independent_rates'))}",
        f"- Direct hits count repair enabled / MAE gain: {bool(hits_count_repair.get('enabled'))} / {num(hits_count_repair.get('validation_mae_gain'))}",
        f"- Direct TB count repair enabled: {bool(tb_count_repair.get('enabled'))}",
        f"- Direct TB count repair alpha: {num(tb_count_repair.get('alpha'))}",
        f"- Direct TB count validation MAE gain: {num(tb_count_repair.get('validation_mae_gain'))}",
        f"- Direct HR count repair enabled / MAE gain: {bool(hr_count_repair.get('enabled'))} / {num(hr_count_repair.get('validation_mae_gain'))}",
        f"- Direct HR any-event Brier gain: {num(hr_count_repair.get('any_brier_gain'), 5)}",
        f"- HR-any Brier gain vs prior: {num(rec.get('hr_any_brier_gain_vs_prior'), 5)}",
        "",
        "## Feature Coverage",
        "",
        "| Feature | Coverage |",
        "|---|---:|",
    ]
    coverage = payload.get("feature_coverage", {})
    for col in [
        "park_run_factor",
        "park_hr_factor",
        "park_babip_factor",
        "own_lineup_xwoba_avg",
        "own_lineup_barrel_avg",
        "lineup_confirmed_flag",
        "confirmed_team_lineup_slots",
        "team_lineup_confirmed_flag",
        "lineup_slot_pa_prior",
        "home_favorite_ninth_penalty",
        "blowout_risk",
        "catcher_low_pa_risk",
        "platoon_advantage_flag",
        "batter_sc_barrel_rate",
        "batter_sc_xwoba",
        "batter_sc_xslg",
        "batter_sprint_speed",
        "batter_disc_whiff_pct",
        "opp_sp_sc_barrel_rate",
        "opp_sp_sc_xwoba",
        "opp_sp_fb_pct",
        "opp_sp_fb_xwoba",
        "opp_sp_sl_pct",
        "opp_sp_ch_pct",
        "opp_sp_fastball_family_pct",
        "opp_sp_pitch_diversity",
    ]:
        val = coverage.get(col)
        lines.append(f"| {col} | {num(None if val is None else float(val) * 100.0, 1)}% |")
    lines.extend([
        "",
        "## Opportunity",
        "",
        "| Model | Rows | MAE | RMSE | Bias |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, key in [
        ("Selected PA model", "model"),
        ("Single-mean PA", "single_mean_model"),
        ("Two-part PA", "two_part_model"),
        ("Slot prior", "slot_prior"),
        ("Existing projected PA", "existing_projected_pa"),
    ]:
        row = pa.get(key, {})
        lines.append(
            f"| {label} | {row.get('rows', 0)} | {num(row.get('mae'))} | "
            f"{num(row.get('rmse'))} | {num(row.get('bias'))} |"
        )
    two_part = pa.get("two_part", {})
    lines.extend([
        "",
        f"- Two-part PA enabled: {two_part.get('enabled', False)}",
        f"- Low-PA distribution enabled: {two_part.get('distribution_enabled', False)}",
        f"- Low-PA Brier: {num(two_part.get('low_pa_model_brier'), 5)} vs baseline {num(two_part.get('low_pa_baseline_brier'), 5)}",
        f"- Immutable lock-context low-PA rows: {two_part.get('immutable_lock_low_pa_rows', 0)}",
        f"- Temporal lineup/proxy validation rows: {two_part.get('distribution_validation_rows', 0)}",
        f"- Temporal distribution Brier: {num(two_part.get('distribution_validation_brier'), 5)} vs baseline {num(two_part.get('distribution_validation_baseline_brier'), 5)}",
        f"- Activation reason: {two_part.get('activation_reason', '-')}",
        f"- Conditional normal-play PA MAE: {num(two_part.get('normal_pa_mae'))}",
    ])
    lines.extend([
        "",
        "## Structured Counts",
        "",
        "| Target | Model Rows | Model MAE | Prior MAE | Model Bias |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, model_key, prior_key in [
        ("Hits", "hits", "slot_prior_hits"),
        ("Total bases", "total_bases", "slot_prior_total_bases"),
        ("Home runs", "home_runs", "slot_prior_home_runs"),
    ]:
        m = structured.get(model_key, {})
        p = structured.get(prior_key, {})
        lines.append(
            f"| {label} | {m.get('rows', 0)} | {num(m.get('mae'))} | "
            f"{num(p.get('mae'))} | {num(m.get('bias'))} |"
        )
    lines.extend([
        "",
        "## Direct Per-PA Event Model",
        "",
        f"- Active event curve: {direct.get('active_event_model', '-')}",
        f"- Train event rows: {direct.get('train_event_rows', 0)}",
        f"- Holdout player-games: {direct.get('holdout_rows', 0)}",
        f"- Weighted event Brier: {num(direct.get('weighted_event_brier'), 5)}",
        f"- Weighted event log loss: {num(direct.get('weighted_event_log_loss'), 5)}",
        f"- Classes: {', '.join(map(str, direct.get('classes', []))) if direct.get('classes') else '-'}",
        f"- TB-state residual enabled: {(direct.get('tb_state_residual') or {}).get('enabled', False)}",
        f"- TB-state blend alpha: {num((direct.get('tb_state_residual') or {}).get('alpha'))}",
        f"- TB-state validation Brier gain: {num((direct.get('tb_state_residual') or {}).get('validation_brier_gain'), 5)}",
        "",
        "| Target | Rows | Direct Event MAE | Independent Rate MAE | Direct Bias |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, key in [("Hits", "hits"), ("Total bases", "total_bases"), ("Home runs", "home_runs")]:
        d = direct.get(key, {})
        s = structured.get(key, {})
        lines.append(
            f"| {label} | {d.get('rows', 0)} | {num(d.get('mae'))} | "
            f"{num(s.get('mae'))} | {num(d.get('bias'))} |"
        )
    boosted = direct.get("boosted_binary") or {}
    linear = direct.get("linear_multinomial") or {}
    hierarchical = direct.get("hierarchical_conditional") or {}
    if boosted or linear or hierarchical:
        lines.extend([
            "",
            "## Event Model Candidates",
            "",
            f"- Selected: {(direct.get('model_selection') or {}).get('selected', direct.get('active_event_model', '-'))}",
            "",
            "| Candidate | Brier | Log Loss | Composite | Hits MAE | TB MAE | HR MAE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        selection = direct.get("model_selection") or {}
        for label, row in [
            ("linear_multinomial", linear),
            ("boosted_binary_calibrated", boosted),
            ("hierarchical_conditional_lgbm", hierarchical),
        ]:
            if not row:
                continue
            composite_key = {
                "linear_multinomial": "linear_composite",
                "boosted_binary_calibrated": "boosted_composite",
                "hierarchical_conditional_lgbm": "hierarchical_composite",
            }[label]
            composite = selection.get(composite_key)
            lines.append(
                f"| {label} | {num(row.get('weighted_event_brier'), 5)} | "
                f"{num(row.get('weighted_event_log_loss'), 5)} | {num(composite, 5)} | "
                f"{num((row.get('hits') or {}).get('mae'))} | "
                f"{num((row.get('total_bases') or {}).get('mae'))} | "
                f"{num((row.get('home_runs') or {}).get('mae'))} |"
            )
    event_rates = (direct.get("event_rates") or {})
    if event_rates:
        lines.extend([
            "",
            "| Event | Actual / PA | Predicted Prob | Bias / PA |",
            "|---|---:|---:|---:|",
        ])
        for cls in EVENT_CLASSES:
            rec_rate = event_rates.get(cls, {})
            lines.append(
                f"| {cls} | {num(rec_rate.get('actual_per_pa'), 4)} | "
                f"{num(rec_rate.get('pred_mean_prob'), 4)} | {num(rec_rate.get('bias_per_pa'), 4)} |"
            )
    xbh_calibration = direct.get("conditional_xbh_calibration") or {}
    xbh_base = xbh_calibration.get("base_validation") or {}
    xbh_candidate = xbh_calibration.get("candidate_validation") or {}
    lines.extend([
        "",
        "## Conditional XBH Calibration",
        "",
        f"- Enabled: {xbh_calibration.get('enabled', False)}",
        f"- Method: {xbh_calibration.get('method', '-')}",
        f"- Validation rows: {xbh_calibration.get('validation_rows', 0)}",
        f"- TB MAE before / after: {num((xbh_base.get('total_bases') or {}).get('mae'))} / "
        f"{num((xbh_candidate.get('total_bases') or {}).get('mae'))}",
        f"- Event Brier before / after: {num(xbh_base.get('weighted_event_brier'), 5)} / "
        f"{num(xbh_candidate.get('weighted_event_brier'), 5)}",
        f"- Offsets: {json.dumps(xbh_calibration.get('production_offsets') or xbh_calibration.get('candidate_offsets') or {}, sort_keys=True)}",
    ])
    lines.extend([
        "",
        "## HR Rare Event",
        "",
        f"- Rows: {hr_any.get('rows', 0)}",
        f"- Model Brier: {num(hr_any.get('model_brier'), 5)}",
        f"- Prior Brier: {num(hr_any.get('prior_brier'), 5)}",
        f"- AUC: {num(hr_any.get('auc'), 3)}",
        "",
        "## Existing Prop Projection Holdout",
        "",
        "| Target | Rows | MAE | RMSE | Bias |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, key in [("Hits", "hits"), ("Total bases", "total_bases"), ("Home runs", "home_runs")]:
        row = prop.get(key, {})
        lines.append(
            f"| {label} | {row.get('rows', 0)} | {num(row.get('mae'))} | "
            f"{num(row.get('rmse'))} | {num(row.get('bias'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _boosted_poisson_count_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    return Pipeline([
        ("features", _preprocessor(numeric_features, categorical_features)),
        ("model", LGBMRegressor(
            objective="poisson",
            n_estimators=240,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=6,
            min_child_samples=100,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.20,
            reg_lambda=1.50,
            random_state=44,
            n_jobs=-1,
            verbosity=-1,
        )),
    ])


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    try:
        import decimal

        if isinstance(value, decimal.Decimal):
            return float(value)
    except Exception:
        pass
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hitter player-game outcome models.")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--min-holdout-rows", type=int, default=200)
    parser.add_argument("--fit-independent-boosted", action="store_true")
    parser.add_argument("--report-file")
    args = parser.parse_args()

    cfg = HitterOutcomeModelConfig(
        pg_dsn=args.pg_dsn,
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
        min_train_rows=args.min_train_rows,
        min_holdout_rows=args.min_holdout_rows,
        fit_independent_boosted_candidate=args.fit_independent_boosted,
        report_file=args.report_file,
    )
    print(json.dumps(train_hitter_player_game_outcomes(cfg), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
