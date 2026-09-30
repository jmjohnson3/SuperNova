"""Walk-forward TB tail repair challenger.

The report focuses on the defect called out by the TB decomposition report:
4+ TB tail misses and single/double/triple/HR mix errors.  It trains a
diagnostic-only direct TB-state model on one row per hitter-game, then prices
true-paired TB offers from those states.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psycopg2
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text, atomic_write_via
from mlb_pipeline.db import PG_DSN

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_PRODUCTION_MODEL_FILE = "prop_tb_tail_state_model.joblib"

TB_STATES = ("zero", "one", "two_three", "four_plus_hr", "four_plus_non_hr")
_FOLD_TEST_DAYS = 5
_MIN_TRAIN_ROWS = 900
_CALIBRATION_MIN_ROWS = 50
_CALIBRATION_SHRINK_ROWS = 200.0
_BLEND_ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
_DK_TB15_MIN_TRUE_PAIR_ROWS = 500
_MAX_DK_TB15_CALIBRATION_ERROR = 0.05

NUMERIC_FEATURES = [
    "lineup_slot",
    "confirmed_starter_num",
    "is_home",
    "projected_pa",
    "pa_games",
    "model_pred_hits",
    "model_pred_total_bases",
    "model_pred_home_runs",
    "pred_hit_rate",
    "pred_tb_rate",
    "pred_hr_rate",
    "pred_extra_bases_per_hit",
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
    "is_dome",
    "is_day_game",
    "own_lineup_xwoba_avg",
    "own_lineup_xslg_avg",
    "own_lineup_barrel_avg",
    "own_lineup_hard_hit_avg",
    "lineup_confirmed_flag",
    "confirmed_team_lineup_slots",
    "team_lineup_confirmed_flag",
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
    "batter_vs_hand_hits_avg_10",
    "batter_vs_hand_tb_avg_10",
    "batter_vs_hand_hr_avg_10",
    "batter_vs_hand_iso_avg_10",
    "batter_vs_hand_k_rate_10",
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
    "batter_disc_whiff_pct",
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
    "opp_sp_fastball_family_pct",
    "opp_sp_pitch_diversity",
]
CATEGORICAL_FEATURES = [
    "team_abbr",
    "opponent_abbr",
    "lineup_source",
    "primary_position",
    "batter_hand",
    "opp_sp_hand",
]

SQL_PLAYER_GAMES = """
SELECT game_date_et, game_slug, player_id, player_name,
       team_abbr, opponent_abbr, lineup_slot, lineup_source,
       CASE WHEN confirmed_starter IS TRUE THEN 1.0 ELSE 0.0 END AS confirmed_starter_num,
       primary_position, batter_hand, opp_sp_hand,
       is_home::float, projected_pa::float, actual_pa::float,
       model_pred_hits::float, model_pred_total_bases::float, model_pred_home_runs::float,
       actual_hits::float, actual_singles::float, actual_doubles::float,
       actual_triples::float, actual_home_runs::float, actual_total_bases::float,
       pa_games::float,
       team_implied_runs::float, opponent_implied_runs::float, game_total_line::float,
       park_run_factor::float, park_hr_factor::float, park_babip_factor::float,
       temperature_f::float, wind_speed_mph::float, wind_sin::float, wind_cos::float,
       is_dome::float, is_day_game::float,
       own_lineup_xwoba_avg::float, own_lineup_xslg_avg::float,
       own_lineup_barrel_avg::float, own_lineup_hard_hit_avg::float,
       lineup_confirmed_flag::float, confirmed_team_lineup_slots::float,
       team_lineup_confirmed_flag::float,
       opp_sp_k_pct_10::float, opp_sp_bb_pct::float, opp_sp_xwoba::float,
       opp_sp_hard_hit_pct::float, opp_sp_whiff_pct::float,
       opp_bp_era_10::float, opp_bp_whip_10::float, opp_bp_k9_10::float,
       opp_bp_ip_last_3::float, opp_bp_ip_last_7::float,
       batter_vs_hand_hits_avg_10::float, batter_vs_hand_tb_avg_10::float,
       batter_vs_hand_hr_avg_10::float, batter_vs_hand_iso_avg_10::float,
       batter_vs_hand_k_rate_10::float,
       batter_vs_rp_ba_30::float, batter_vs_rp_slg_30::float,
       batter_vs_rp_hr_rate_30::float, batter_vs_rp_k_rate_30::float,
       batter_sc_barrel_rate::float, batter_sc_hard_hit_pct::float,
       batter_sc_avg_exit_velo::float, batter_sc_avg_launch_angle::float,
       batter_sc_sweet_spot_pct::float, batter_sc_fb_pct::float,
       batter_sc_gb_pct::float, batter_sc_ld_pct::float,
       batter_sc_xba::float, batter_sc_xslg::float, batter_sc_xwoba::float,
       batter_sc_xiso::float, batter_sc_brl_pa::float,
       batter_disc_whiff_pct::float, batter_disc_k_pct::float,
       batter_disc_bb_pct::float,
       opp_sp_sc_barrel_rate::float, opp_sp_sc_hard_hit_pct::float,
       opp_sp_sc_avg_exit_velo::float, opp_sp_sc_avg_launch_angle::float,
       opp_sp_sc_xba::float, opp_sp_sc_xslg::float,
       opp_sp_sc_xwoba::float, opp_sp_sc_xiso::float,
       opp_sp_fastball_family_pct::float, opp_sp_pitch_diversity::float
FROM features.mlb_hitter_player_game_training
WHERE game_date_et >= %(cutoff)s
  AND actual_total_bases IS NOT NULL
  AND model_pred_total_bases IS NOT NULL
  AND projected_pa IS NOT NULL
ORDER BY game_date_et, game_slug, player_id
"""

SQL_TB_OFFERS = """
SELECT game_date_et, game_slug, player_id,
       side, market_line::float AS market_line, line_bucket, line_surface,
       bookmaker_key, pair_quality,
       won, COALESCE(push, false) AS push,
       COALESCE(true_pair_flag::float, CASE WHEN pair_quality IN ('same_book','cross_book') THEN 1.0 ELSE 0.0 END) AS true_pair_flag,
       COALESCE(synthetic_pair_flag::float, CASE WHEN pair_quality = 'synthetic' THEN 1.0 ELSE 0.0 END) AS synthetic_pair_flag
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market = 'batter_total_bases'
  AND market_line IS NOT NULL
  AND won IS NOT NULL
  AND COALESCE(push, false) IS FALSE
ORDER BY game_date_et, game_slug, player_id, market_line, side, bookmaker_key
"""


def _table_exists(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (table_name,))
        return bool(cur.fetchone()[0])


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _model_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("cat", _one_hot_encoder(), categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=140,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=5,
        min_child_samples=90,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.25,
        reg_lambda=1.50,
        random_state=44,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline([("features", pre), ("model", model)])


def _safe_logit(prob: float) -> float:
    p = max(1e-5, min(1.0 - 1e-5, float(prob)))
    return math.log(p / (1.0 - p))


def _inv_logit(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(value)))))


def _poisson_pmf(lam: float, k: int) -> float:
    lam = max(1e-6, min(20.0, float(lam)))
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _baseline_state_probs(row: pd.Series) -> dict[str, float]:
    lam = max(1e-6, min(12.0, float(row.get("model_pred_total_bases") or 0.0)))
    p0 = _poisson_pmf(lam, 0)
    p1 = _poisson_pmf(lam, 1)
    p2 = _poisson_pmf(lam, 2)
    p3 = _poisson_pmf(lam, 3)
    tail = max(0.0, 1.0 - p0 - p1 - p2 - p3)
    pred_hr = max(0.0, float(row.get("model_pred_home_runs") or 0.0))
    hr_tail = min(tail, (1.0 - math.exp(-pred_hr)) * 0.95)
    non_hr_tail = max(0.0, tail - hr_tail)
    probs = {
        "zero": p0,
        "one": p1,
        "two_three": p2 + p3,
        "four_plus_hr": hr_tail,
        "four_plus_non_hr": non_hr_tail,
    }
    total = sum(probs.values()) or 1.0
    return {key: value / total for key, value in probs.items()}


def _label(row: pd.Series) -> str:
    tb = float(row.get("actual_total_bases") or 0.0)
    hr = float(row.get("actual_home_runs") or 0.0)
    if tb <= 0:
        return "zero"
    if abs(tb - 1.0) <= 1e-9:
        return "one"
    if tb < 4.0:
        return "two_three"
    return "four_plus_hr" if hr > 0 else "four_plus_non_hr"


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_date_et"] = pd.to_datetime(out["game_date_et"]).dt.date
    for col in NUMERIC_FEATURES:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col not in out:
            out[col] = "unknown"
        out[col] = out[col].fillna("unknown").astype(str)
    projected_pa = pd.to_numeric(out["projected_pa"], errors="coerce").clip(lower=0.25)
    pred_hits = pd.to_numeric(out["model_pred_hits"], errors="coerce").clip(lower=0.0)
    pred_tb = pd.to_numeric(out["model_pred_total_bases"], errors="coerce").clip(lower=0.0)
    pred_hr = pd.to_numeric(out["model_pred_home_runs"], errors="coerce").clip(lower=0.0)
    out["pred_hit_rate"] = pred_hits / projected_pa
    out["pred_tb_rate"] = pred_tb / projected_pa
    out["pred_hr_rate"] = pred_hr / projected_pa
    out["pred_extra_bases_per_hit"] = (pred_tb - pred_hits).clip(lower=0.0) / pred_hits.clip(lower=0.15)
    out["tb_state"] = out.apply(_label, axis=1)
    return out


def _available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [col for col in NUMERIC_FEATURES if col in df and df[col].notna().any()]
    categorical = [col for col in CATEGORICAL_FEATURES if col in df]
    return numeric, categorical


def _align_probabilities(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    raw = model.predict_proba(X)
    classes = [str(cls) for cls in model.named_steps["model"].classes_]
    out = pd.DataFrame(0.0, index=X.index, columns=list(TB_STATES))
    for idx, cls in enumerate(classes):
        if cls in out.columns:
            out[cls] = raw[:, idx]
    total = out.sum(axis=1).replace(0, np.nan)
    out = out.div(total, axis=0).fillna(1.0 / len(TB_STATES))
    return out


def _walk_forward_state_probs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dates = sorted(df["game_date_et"].unique())
    numeric, categorical = _available_features(df)
    direct = pd.DataFrame(np.nan, index=df.index, columns=list(TB_STATES), dtype=float)
    expected_values = pd.DataFrame(np.nan, index=df.index, columns=list(TB_STATES), dtype=float)
    folds: list[dict[str, Any]] = []
    for start in range(0, len(dates), _FOLD_TEST_DAYS):
        test_dates = dates[start:start + _FOLD_TEST_DAYS]
        if not test_dates:
            continue
        train_mask = df["game_date_et"] < test_dates[0]
        test_mask = df["game_date_et"].isin(test_dates)
        if int(train_mask.sum()) < _MIN_TRAIN_ROWS or int(test_mask.sum()) == 0:
            continue
        train = df.loc[train_mask]
        test = df.loc[test_mask]
        if train["tb_state"].nunique() < 4:
            continue
        model = _model_pipeline(numeric, categorical)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
            model.fit(train[numeric + categorical], train["tb_state"])
            probs = _align_probabilities(model, test[numeric + categorical])
        state_cols = list(TB_STATES)
        direct.loc[test.index, state_cols] = probs[state_cols]
        state_means = train.groupby("tb_state")["actual_total_bases"].mean().to_dict()
        for state in TB_STATES:
            expected_values.loc[test.index, state] = float(state_means.get(state, {
                "zero": 0.0,
                "one": 1.0,
                "two_three": 2.4,
                "four_plus_hr": 4.6,
                "four_plus_non_hr": 4.4,
            }[state]))
        folds.append({
            "holdout_start": str(test_dates[0]),
            "holdout_end": str(test_dates[-1]),
            "train_rows": int(train_mask.sum()),
            "holdout_rows": int(test_mask.sum()),
        })
    meta = {
        "enabled": bool(folds),
        "folds": folds,
        "numeric_features": numeric,
        "categorical_features": categorical,
    }
    return direct, expected_values, meta


def _baseline_probs_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = [_baseline_state_probs(row) for _, row in df.iterrows()]
    return pd.DataFrame(rows, index=df.index, columns=list(TB_STATES)).fillna(1.0 / len(TB_STATES))


def _state_brier(labels: pd.Series, probs: pd.DataFrame) -> dict[str, Any]:
    work = probs.dropna()
    labels = labels.loc[work.index]
    if work.empty:
        return {"rows": 0, "brier": None}
    target = pd.DataFrame(0.0, index=work.index, columns=list(TB_STATES))
    for state in TB_STATES:
        target[state] = labels.eq(state).astype(float)
    state_cols = list(TB_STATES)
    loss = np.square(target[state_cols] - work[state_cols]).sum(axis=1)
    return {"rows": int(len(loss)), "brier": float(loss.mean())}


def _expected_tb(probs: pd.DataFrame, expected_values: pd.DataFrame) -> pd.Series:
    state_cols = list(TB_STATES)
    return (probs[state_cols] * expected_values[state_cols]).sum(axis=1)


def _blend_probs(base: pd.DataFrame, direct: pd.DataFrame, alpha: float) -> pd.DataFrame:
    state_cols = list(TB_STATES)
    out = (1.0 - float(alpha)) * base[state_cols] + float(alpha) * direct[state_cols]
    total = out.sum(axis=1).replace(0.0, np.nan)
    return out.div(total, axis=0).fillna(1.0 / len(state_cols))


def _count_metrics(actual: pd.Series, pred: pd.Series) -> dict[str, Any]:
    work = pd.DataFrame({"actual": actual, "pred": pred}).dropna()
    if work.empty:
        return {"rows": 0}
    error = work["pred"] - work["actual"]
    return {
        "rows": int(len(work)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
    }


def _line_bucket(line: Any) -> str:
    try:
        value = float(line)
    except (TypeError, ValueError):
        return "unknown"
    if value <= 1.5:
        return "TB 1.5"
    if value <= 2.5:
        return "TB 2.5"
    return "TB 3.5+"


def _p_over_from_state_probs(probs: pd.DataFrame, line: pd.Series) -> pd.Series:
    line_bucket = line.apply(_line_bucket)
    out = pd.Series(np.nan, index=probs.index, dtype=float)
    tail = probs["four_plus_hr"] + probs["four_plus_non_hr"]
    out.loc[line_bucket.eq("TB 1.5")] = (
        probs.loc[line_bucket.eq("TB 1.5"), "two_three"]
        + tail.loc[line_bucket.eq("TB 1.5")]
    )
    out.loc[line_bucket.eq("TB 2.5")] = (
        0.42 * probs.loc[line_bucket.eq("TB 2.5"), "two_three"]
        + tail.loc[line_bucket.eq("TB 2.5")]
    )
    out.loc[line_bucket.eq("TB 3.5+")] = tail.loc[line_bucket.eq("TB 3.5+")]
    return out.clip(1e-5, 1.0 - 1e-5)


def _apply_line_calibration(joined: pd.DataFrame) -> pd.DataFrame:
    out = joined.copy()
    out["direct_p_over_calibrated"] = out["direct_p_over"]
    rows = []
    for day in sorted(out["game_date_et"].dropna().unique()):
        current_mask = out["game_date_et"].eq(day)
        for line_key in sorted(out.loc[current_mask, "line_key"].dropna().unique()):
            current = current_mask & out["line_key"].eq(line_key)
            prior = out["game_date_et"].lt(day) & out["line_key"].eq(line_key)
            hist = out.loc[prior].dropna(subset=["direct_p_over", "over_won"])
            if len(hist) < _CALIBRATION_MIN_ROWS:
                continue
            pred_rate = float(hist["direct_p_over"].mean())
            actual_rate = float(hist["over_won"].mean())
            shrink = len(hist) / (len(hist) + _CALIBRATION_SHRINK_ROWS)
            offset = shrink * (_safe_logit(actual_rate) - _safe_logit(pred_rate))
            out.loc[current, "direct_p_over_calibrated"] = out.loc[current, "direct_p_over"].map(
                lambda p: _inv_logit(_safe_logit(float(p)) + offset)
            )
            rows.append({
                "date": str(day),
                "line_key": line_key,
                "prior_rows": int(len(hist)),
                "prior_pred_rate": pred_rate,
                "prior_actual_rate": actual_rate,
                "offset": float(offset),
            })
    out.attrs["calibration_offsets"] = rows
    return out


def _brier(target: pd.Series, prob: pd.Series) -> float | None:
    work = pd.DataFrame({"target": target, "prob": prob}).dropna()
    if work.empty:
        return None
    return float(np.square(work["target"] - work["prob"].clip(1e-5, 1.0 - 1e-5)).mean())


def _line_metrics(joined: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for key, group in joined.groupby(["line_key", "bookmaker_key", "side"], dropna=False):
        if len(group) < 10:
            continue
        side = str(key[2] or "").lower()
        base_side = group["base_p_over"] if side == "over" else 1.0 - group["base_p_over"]
        direct_side = group["direct_p_over"] if side == "over" else 1.0 - group["direct_p_over"]
        calibrated_side = group["direct_p_over_calibrated"] if side == "over" else 1.0 - group["direct_p_over_calibrated"]
        target = group["won"].astype(bool).astype(float)
        rows.append({
            "line_key": key[0],
            "bookmaker_key": key[1],
            "side": side,
            "rows": int(len(group)),
            "base_brier": _brier(target, base_side),
            "direct_brier": _brier(target, direct_side),
            "calibrated_brier": _brier(target, calibrated_side),
            "base_calibration_error": abs(float(base_side.mean()) - float(target.mean())),
            "direct_calibration_error": abs(float(direct_side.mean()) - float(target.mean())),
            "calibrated_calibration_error": abs(float(calibrated_side.mean()) - float(target.mean())),
            "win_rate": float(target.mean()),
            "base_mean_prob": float(base_side.mean()),
            "direct_mean_prob": float(direct_side.mean()),
            "calibrated_mean_prob": float(calibrated_side.mean()),
        })
    rows.sort(key=lambda row: (row["line_key"], row["bookmaker_key"], row["side"]))
    return rows


def _dk_tb15_over_gate(line_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rec = next(
        (
            row for row in line_rows
            if str(row.get("line_key") or "") == "TB 1.5"
            and str(row.get("bookmaker_key") or "").lower() == "draftkings"
            and str(row.get("side") or "").lower() == "over"
        ),
        None,
    )
    if not rec:
        return {
            "passed": False,
            "reason": "dk_tb15_over_true_pair_metric_missing",
            "required_rows": _DK_TB15_MIN_TRUE_PAIR_ROWS,
            "max_calibration_error": _MAX_DK_TB15_CALIBRATION_ERROR,
        }
    rows = int(rec.get("rows") or 0)
    base_brier = rec.get("base_brier")
    direct_brier = rec.get("direct_brier")
    calibrated_brier = rec.get("calibrated_brier")
    base_cal = rec.get("base_calibration_error")
    calibrated_cal = rec.get("calibrated_calibration_error")
    blockers: list[str] = []
    if rows < _DK_TB15_MIN_TRUE_PAIR_ROWS:
        blockers.append("not_enough_dk_tb15_true_pair_rows")
    if base_brier is None or calibrated_brier is None or float(calibrated_brier) >= float(base_brier):
        blockers.append("calibrated_brier_not_better_than_baseline")
    if base_brier is None or direct_brier is None or float(direct_brier) >= float(base_brier):
        blockers.append("direct_brier_not_better_than_baseline")
    if calibrated_cal is None or float(calibrated_cal) > _MAX_DK_TB15_CALIBRATION_ERROR:
        blockers.append("calibration_error_above_gate")
    if base_cal is not None and calibrated_cal is not None and float(calibrated_cal) > float(base_cal):
        blockers.append("calibration_worse_than_baseline")
    return {
        **rec,
        "passed": not blockers,
        "reason": "passed" if not blockers else ";".join(blockers),
        "required_rows": _DK_TB15_MIN_TRUE_PAIR_ROWS,
        "max_calibration_error": _MAX_DK_TB15_CALIBRATION_ERROR,
        "brier_gain": (
            float(base_brier) - float(calibrated_brier)
            if base_brier is not None and calibrated_brier is not None
            else None
        ),
        "calibration_error_gain": (
            float(base_cal) - float(calibrated_cal)
            if base_cal is not None and calibrated_cal is not None
            else None
        ),
    }


def _overall_side_brier(joined: pd.DataFrame, probability_col: str) -> float | None:
    if joined.empty:
        return None
    return _brier(joined["won"].astype(bool).astype(float), np.where(
        joined["side"].astype(str).str.lower().eq("over"),
        joined[probability_col],
        1.0 - joined[probability_col],
    ))


def _join_offers(df: pd.DataFrame, offers: pd.DataFrame, base_probs: pd.DataFrame, direct_probs: pd.DataFrame) -> pd.DataFrame:
    if offers.empty:
        return pd.DataFrame()
    clean_offers = offers.copy()
    clean_offers["game_date_et"] = pd.to_datetime(clean_offers["game_date_et"]).dt.date
    clean_offers["true_pair_flag"] = pd.to_numeric(clean_offers["true_pair_flag"], errors="coerce").fillna(0.0)
    clean_offers["synthetic_pair_flag"] = pd.to_numeric(clean_offers["synthetic_pair_flag"], errors="coerce").fillna(0.0)
    clean_offers = clean_offers.loc[
        clean_offers["true_pair_flag"].ge(0.5)
        & clean_offers["synthetic_pair_flag"].lt(0.5)
    ].copy()
    if clean_offers.empty:
        return pd.DataFrame()
    keys = ["game_date_et", "game_slug", "player_id"]
    state_frame = df[keys].copy()
    state_frame["base_p_over_tmp"] = 0.0
    state_frame["direct_p_over_tmp"] = 0.0
    joined = clean_offers.merge(state_frame[keys], on=keys, how="inner")
    if joined.empty:
        return joined
    joined = joined.merge(
        df[keys].reset_index().rename(columns={"index": "_pg_index"}),
        on=keys,
        how="left",
    )
    base_selected = base_probs.loc[joined["_pg_index"].to_numpy()]
    direct_selected = direct_probs.loc[joined["_pg_index"].to_numpy()]
    base_selected.index = joined.index
    direct_selected.index = joined.index
    joined["line_key"] = joined["market_line"].apply(_line_bucket)
    joined["base_p_over"] = _p_over_from_state_probs(base_selected, joined["market_line"])
    joined["direct_p_over"] = _p_over_from_state_probs(direct_selected, joined["market_line"])
    joined["over_won"] = np.where(
        joined["side"].astype(str).str.lower().eq("over"),
        joined["won"].astype(bool),
        ~joined["won"].astype(bool),
    ).astype(float)
    return _apply_line_calibration(joined)


def _blend_gate(
    *,
    df: pd.DataFrame,
    offers: pd.DataFrame,
    labels: pd.Series,
    base_probs: pd.DataFrame,
    direct_probs: pd.DataFrame,
    direct_expected_tb: pd.Series,
    base_count: dict[str, Any],
    base_state_brier: dict[str, Any],
    base_line_brier: float | None,
) -> dict[str, Any]:
    baseline_count = pd.to_numeric(df["model_pred_total_bases"], errors="coerce")
    records: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for alpha in _BLEND_ALPHAS:
        blended_probs = _blend_probs(base_probs, direct_probs, alpha)
        blended_expected = (baseline_count + alpha * (direct_expected_tb - baseline_count)).clip(lower=0.0, upper=12.0)
        state_brier = _state_brier(labels, blended_probs)
        count = _count_metrics(df["actual_total_bases"], blended_expected)
        joined = _join_offers(df, offers, base_probs, blended_probs)
        line_brier = _overall_side_brier(joined, "direct_p_over_calibrated")
        state_gain = (
            float(base_state_brier["brier"] - state_brier["brier"])
            if base_state_brier.get("brier") is not None and state_brier.get("brier") is not None
            else None
        )
        count_gain = (
            float(base_count["mae"] - count["mae"])
            if base_count.get("mae") is not None and count.get("mae") is not None
            else None
        )
        line_gain = (
            float(base_line_brier - line_brier)
            if base_line_brier is not None and line_brier is not None
            else None
        )
        rec = {
            "alpha": float(alpha),
            "state_brier": state_brier.get("brier"),
            "state_brier_gain": state_gain,
            "tb_mae": count.get("mae"),
            "tb_mae_gain": count_gain,
            "line_brier_rows": int(len(joined)),
            "line_brier": line_brier,
            "line_brier_gain": line_gain,
            "accepted": bool(
                alpha > 0
                and state_gain is not None and state_gain > 0.0
                and count_gain is not None and count_gain > 0.0
                and line_gain is not None and line_gain > 0.0
            ),
        }
        records.append(rec)
    accepted = [rec for rec in records if rec.get("accepted")]
    if accepted:
        selected = sorted(
            accepted,
            key=lambda rec: (
                float(rec.get("line_brier_gain") or 0.0),
                float(rec.get("tb_mae_gain") or 0.0),
                float(rec.get("state_brier_gain") or 0.0),
            ),
            reverse=True,
        )[0]
        reason = "blended_state_count_and_true_pair_line_brier_improved"
    else:
        selected = sorted(
            records,
            key=lambda rec: (
                int(float(rec.get("alpha") or 0.0) > 0.0),
                float(rec.get("line_brier_gain") or -999.0),
                float(rec.get("tb_mae_gain") or -999.0),
                float(rec.get("state_brier_gain") or -999.0),
            ),
            reverse=True,
        )[0] if records else {}
        reason = "diagnostic_only_until_blended_state_count_and_line_brier_all_improve"
    return {
        "alphas": list(_BLEND_ALPHAS),
        "records": records,
        "selected": selected,
        "accepted": bool(selected and selected.get("accepted")),
        "reason": reason,
    }


def _fit_production_artifact(
    df: pd.DataFrame,
    *,
    selected_blend: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Fit the reusable scorer after the walk-forward proof gate passes."""
    if not payload.get("accepted"):
        return None
    numeric, categorical = _available_features(df)
    if len(df) < _MIN_TRAIN_ROWS or df["tb_state"].nunique() < 4:
        return None
    model = _model_pipeline(numeric, categorical)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
        model.fit(df[numeric + categorical], df["tb_state"])
    state_expected_values = {
        state: float(value)
        for state, value in df.groupby("tb_state")["actual_total_bases"].mean().to_dict().items()
    }
    for state, fallback in {
        "zero": 0.0,
        "one": 1.0,
        "two_three": 2.4,
        "four_plus_hr": 4.6,
        "four_plus_non_hr": 4.4,
    }.items():
        state_expected_values.setdefault(state, fallback)
    return {
        "model": model,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "states": list(TB_STATES),
        "state_expected_values": state_expected_values,
        "selected_blend_alpha": float(selected_blend.get("alpha") or 0.0),
        "accepted": bool(payload.get("accepted")),
        "reason": payload.get("reason"),
        "trained_at_utc": payload.get("generated_at_utc"),
        "metrics": {
            "state_brier": payload.get("state_brier"),
            "tb_count": payload.get("tb_count"),
            "true_pair_line_pricing": payload.get("true_pair_line_pricing"),
            "dk_tb15_over_gate": payload.get("dk_tb15_over_gate"),
            "blend_gate": payload.get("blend_gate"),
        },
        "usage": "live_tb15_state_probability_scorer",
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.1f}%"
    except Exception:
        return "-"


def build(lookback_days: int = 365, pg_dsn: str = PG_DSN) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, lookback_days))
    with psycopg2.connect(pg_dsn) as conn:
        if not _table_exists(conn, "features.mlb_hitter_player_game_training"):
            player_games = pd.DataFrame()
        else:
            player_games = _query_df(conn, SQL_PLAYER_GAMES, {"cutoff": cutoff})
        if _table_exists(conn, "features.mlb_prop_market_training_examples"):
            offers = _query_df(conn, SQL_TB_OFFERS, {"cutoff": cutoff})
        else:
            offers = pd.DataFrame()
    if player_games.empty:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "status": "no_rows",
            "rows": 0,
        }
    else:
        df = _prepare(player_games)
        base_probs = _baseline_probs_frame(df)
        direct_probs, expected_values, meta = _walk_forward_state_probs(df)
        valid = direct_probs.notna().all(axis=1)
        expected_direct_tb = _expected_tb(direct_probs.loc[valid], expected_values.loc[valid])
        state_brier_base = _state_brier(df.loc[valid, "tb_state"], base_probs.loc[valid])
        state_brier_direct = _state_brier(df.loc[valid, "tb_state"], direct_probs.loc[valid])
        base_count = _count_metrics(df.loc[valid, "actual_total_bases"], df.loc[valid, "model_pred_total_bases"])
        direct_count = _count_metrics(df.loc[valid, "actual_total_bases"], expected_direct_tb)
        joined = _join_offers(df.loc[valid], offers, base_probs.loc[valid], direct_probs.loc[valid])
        line_rows = _line_metrics(joined) if not joined.empty else []
        dk_tb15_gate = _dk_tb15_over_gate(line_rows)
        base_brier = _overall_side_brier(joined, "base_p_over") if not joined.empty else None
        direct_brier = _overall_side_brier(joined, "direct_p_over") if not joined.empty else None
        calibrated_brier = _overall_side_brier(joined, "direct_p_over_calibrated") if not joined.empty else None
        state_gain = (
            float(state_brier_base["brier"] - state_brier_direct["brier"])
            if state_brier_base.get("brier") is not None and state_brier_direct.get("brier") is not None
            else None
        )
        count_gain = (
            float(base_count["mae"] - direct_count["mae"])
            if base_count.get("mae") is not None and direct_count.get("mae") is not None
            else None
        )
        line_gain = (
            float(base_brier - calibrated_brier)
            if base_brier is not None and calibrated_brier is not None
            else None
        )
        blend_gate = _blend_gate(
            df=df.loc[valid],
            offers=offers,
            labels=df.loc[valid, "tb_state"],
            base_probs=base_probs.loc[valid],
            direct_probs=direct_probs.loc[valid],
            direct_expected_tb=expected_direct_tb,
            base_count=base_count,
            base_state_brier=state_brier_base,
            base_line_brier=base_brier,
        )
        selected_blend = blend_gate.get("selected") or {}
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "status": "ready",
            "usage": "challenger_diagnostic_only",
            "rows": int(len(df)),
            "evaluation_rows": int(valid.sum()),
            "dates": int(df["game_date_et"].nunique()),
            "fold_meta": meta,
            "states": list(TB_STATES),
            "state_brier": {
                "baseline": state_brier_base,
                "direct": state_brier_direct,
                "gain": state_gain,
            },
            "tb_count": {
                "baseline": base_count,
                "direct_state_expected": direct_count,
                "mae_gain": count_gain,
            },
            "true_pair_line_pricing": {
                "rows": int(len(joined)),
                "baseline_brier": base_brier,
                "direct_brier": direct_brier,
                "calibrated_brier": calibrated_brier,
                "calibrated_brier_gain_vs_baseline": line_gain,
                "selected_blend_alpha": selected_blend.get("alpha"),
                "selected_blend_brier": selected_blend.get("line_brier"),
                "selected_blend_brier_gain_vs_baseline": selected_blend.get("line_brier_gain"),
                "calibration_offsets": joined.attrs.get("calibration_offsets", []) if not joined.empty else [],
                "by_line_book_side": line_rows,
            },
            "dk_tb15_over_gate": dk_tb15_gate,
            "blend_gate": blend_gate,
            "tail_rates": {
                "actual_four_plus_hr": float(df.loc[valid, "tb_state"].eq("four_plus_hr").mean()) if valid.any() else None,
                "actual_four_plus_non_hr": float(df.loc[valid, "tb_state"].eq("four_plus_non_hr").mean()) if valid.any() else None,
                "baseline_four_plus_hr": float(base_probs.loc[valid, "four_plus_hr"].mean()) if valid.any() else None,
                "baseline_four_plus_non_hr": float(base_probs.loc[valid, "four_plus_non_hr"].mean()) if valid.any() else None,
                "direct_four_plus_hr": float(direct_probs.loc[valid, "four_plus_hr"].mean()) if valid.any() else None,
                "direct_four_plus_non_hr": float(direct_probs.loc[valid, "four_plus_non_hr"].mean()) if valid.any() else None,
            },
            "accepted": bool(blend_gate.get("accepted") and dk_tb15_gate.get("passed")),
        }
        payload["reason"] = (
            "blended_state_count_and_dk_tb15_true_pair_over_improved"
            if payload["accepted"]
            else (
                dk_tb15_gate.get("reason")
                if blend_gate.get("accepted")
                else blend_gate.get("reason", "diagnostic_only_until_state_count_and_line_brier_all_improve")
            )
        )
        production_artifact = _fit_production_artifact(
            df,
            selected_blend=selected_blend,
            payload=payload,
        )
        if production_artifact is not None:
            model_path = _MODEL_DIR / _PRODUCTION_MODEL_FILE
            atomic_write_via(
                model_path,
                lambda temp: joblib.dump(production_artifact, temp),
                attempts=6,
                retry_all_errors=True,
            )
            payload["production_model_path"] = str(model_path)
            payload["production_usage"] = production_artifact.get("usage")
            payload["usage"] = production_artifact.get("usage")
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_MODEL_DIR / "prop_tb_tail_repair_challenger.json", payload)
    report_path = _REPORT_DIR / "mlb_prop_tb_tail_repair_challenger_latest.md"
    atomic_write_text(report_path, _render(payload))
    payload["report_path"] = str(report_path)
    return payload


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB TB Tail Repair Challenger",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        (
            "Usage: live DK TB 1.5 same-book scorer is eligible."
            if payload.get("accepted") and payload.get("production_model_path")
            else "Usage: challenger diagnostic only; production artifacts are unchanged."
        ),
        "",
        f"- Rows: {payload.get('rows', 0)}",
        f"- Evaluation rows: {payload.get('evaluation_rows', 0)}",
        f"- Dates: {payload.get('dates', 0)}",
        f"- Accepted: {payload.get('accepted', False)}",
        f"- Reason: {payload.get('reason', '-')}",
        "",
        "## State / Count Gates",
        "",
        f"- State Brier baseline/direct/gain: {_fmt(((payload.get('state_brier') or {}).get('baseline') or {}).get('brier'), 5)} / "
        f"{_fmt(((payload.get('state_brier') or {}).get('direct') or {}).get('brier'), 5)} / "
        f"{_fmt((payload.get('state_brier') or {}).get('gain'), 5)}",
        f"- TB MAE baseline/direct/gain: {_fmt(((payload.get('tb_count') or {}).get('baseline') or {}).get('mae'))} / "
        f"{_fmt(((payload.get('tb_count') or {}).get('direct_state_expected') or {}).get('mae'))} / "
        f"{_fmt((payload.get('tb_count') or {}).get('mae_gain'))}",
        "",
        "## Tail Rates",
        "",
        "| Tail | Actual | Baseline | Direct |",
        "|---|---:|---:|---:|",
    ]
    tail = payload.get("tail_rates") or {}
    lines.append(
        f"| HR-driven 4+ | {_pct(tail.get('actual_four_plus_hr'))} | "
        f"{_pct(tail.get('baseline_four_plus_hr'))} | {_pct(tail.get('direct_four_plus_hr'))} |"
    )
    lines.append(
        f"| Non-HR 4+ | {_pct(tail.get('actual_four_plus_non_hr'))} | "
        f"{_pct(tail.get('baseline_four_plus_non_hr'))} | {_pct(tail.get('direct_four_plus_non_hr'))} |"
    )
    pricing = payload.get("true_pair_line_pricing") or {}
    lines.extend([
        "",
        "## True-Pair TB Line Pricing",
        "",
        f"- True-pair offer rows: {pricing.get('rows', 0)}",
        f"- Brier baseline/direct/calibrated: {_fmt(pricing.get('baseline_brier'), 5)} / "
        f"{_fmt(pricing.get('direct_brier'), 5)} / {_fmt(pricing.get('calibrated_brier'), 5)}",
        f"- Calibrated Brier gain vs baseline: {_fmt(pricing.get('calibrated_brier_gain_vs_baseline'), 5)}",
        f"- Selected blend alpha / Brier gain: {_fmt(pricing.get('selected_blend_alpha'), 2)} / "
        f"{_fmt(pricing.get('selected_blend_brier_gain_vs_baseline'), 5)}",
        "",
        "| Line | Book | Side | Rows | Base Brier | Direct Brier | Cal Brier | Base Cal Err | Direct Cal Err | Cal Cal Err |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in pricing.get("by_line_book_side") or []:
        lines.append(
            f"| {row.get('line_key')} | {row.get('bookmaker_key')} | {row.get('side')} | "
            f"{row.get('rows', 0)} | {_fmt(row.get('base_brier'), 5)} | "
            f"{_fmt(row.get('direct_brier'), 5)} | {_fmt(row.get('calibrated_brier'), 5)} | "
            f"{_pct(row.get('base_calibration_error'))} | {_pct(row.get('direct_calibration_error'))} | "
            f"{_pct(row.get('calibrated_calibration_error'))} |"
        )
    dk_gate = payload.get("dk_tb15_over_gate") or {}
    lines.extend([
        "",
        "## DK TB 1.5 Over Production Gate",
        "",
        f"- Passed: {bool(dk_gate.get('passed'))}",
        f"- Reason: {dk_gate.get('reason', '-')}",
        f"- Rows: {dk_gate.get('rows', 0)} / required {dk_gate.get('required_rows', _DK_TB15_MIN_TRUE_PAIR_ROWS)}",
        f"- Brier baseline/direct/calibrated: {_fmt(dk_gate.get('base_brier'), 5)} / "
        f"{_fmt(dk_gate.get('direct_brier'), 5)} / {_fmt(dk_gate.get('calibrated_brier'), 5)}",
        f"- Calibration error baseline/calibrated: {_pct(dk_gate.get('base_calibration_error'))} / "
        f"{_pct(dk_gate.get('calibrated_calibration_error'))}",
    ])
    blend = payload.get("blend_gate") or {}
    if blend:
        lines.extend([
            "",
            "## Baseline + TB-Tail Blend Gate",
            "",
            f"- Accepted: {bool(blend.get('accepted'))}",
            f"- Reason: {blend.get('reason') or '-'}",
            "",
            "| Alpha | State Brier | State Gain | TB MAE | MAE Gain | Line Rows | Line Brier | Line Gain | Accepted |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for row in blend.get("records") or []:
            lines.append(
                f"| {_fmt(row.get('alpha'), 2)} | {_fmt(row.get('state_brier'), 5)} | "
                f"{_fmt(row.get('state_brier_gain'), 5)} | {_fmt(row.get('tb_mae'))} | "
                f"{_fmt(row.get('tb_mae_gain'))} | {row.get('line_brier_rows', 0)} | "
                f"{_fmt(row.get('line_brier'), 5)} | {_fmt(row.get('line_brier_gain'), 5)} | "
                f"{bool(row.get('accepted'))} |"
            )
    lines.extend([
        "",
        "## Line-Specific Calibration Offsets",
        "",
        "| Date | Line | Prior Rows | Prior Pred | Prior Actual | Offset |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in (pricing.get("calibration_offsets") or [])[-30:]:
        lines.append(
            f"| {row.get('date')} | {row.get('line_key')} | {row.get('prior_rows')} | "
            f"{_pct(row.get('prior_pred_rate'))} | {_pct(row.get('prior_actual_rate'))} | "
            f"{_fmt(row.get('offset'))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TB tail repair challenger report")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--pg-dsn", default=PG_DSN)
    args = parser.parse_args()
    payload = build(args.lookback_days, args.pg_dsn)
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows"),
        "accepted": payload.get("accepted", False),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
