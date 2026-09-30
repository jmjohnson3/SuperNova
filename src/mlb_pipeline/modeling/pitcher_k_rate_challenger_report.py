"""Offline pitcher K-per-BF challenger report.

This module evaluates a challenger strikeout-rate head on one row per
pitcher-game. It writes diagnostics only and does not replace production
prediction artifacts.
"""
from __future__ import annotations

import argparse
import math
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_SHRINKAGE_ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
_RESIDUAL_V2_ALPHAS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)
_RESIDUAL_V3_ALPHAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
_MIN_MAE_GAIN = 0.01
_MIN_BRIER_GAIN = 0.0005
_MAX_BIAS_WORSENING = 0.05
_FEATURES = [
    "projected_bf",
    "projected_pitch_count",
    "pitcher_starts",
    "is_home",
    "opp_team_k_pct_10",
    "opp_team_obp_10",
    "opp_team_slg_10",
    "opp_bp_ip_last_3",
    "opp_bp_ip_last_7",
    "team_implied_runs",
    "opponent_implied_runs",
    "game_total_line",
    "minutes_to_first_pitch_at_lock",
]
_RESIDUAL_V2_FEATURES = [
    "baseline_k",
    "baseline_k_rate",
    "projected_bf",
    "projected_pitch_count",
    "pitcher_starts",
    "is_home",
    "opp_team_k_pct_10",
    "opp_team_obp_10",
    "opp_team_slg_10",
    "opp_bp_ip_last_3",
    "opp_bp_ip_last_7",
    "team_implied_runs",
    "opponent_implied_runs",
    "game_total_line",
    "minutes_to_first_pitch_at_lock",
    "pitch_count_per_bf",
    "opponent_k_delta",
]


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


def _load_rows(conn, cutoff) -> pd.DataFrame:
    if not _table_exists(conn, "features.mlb_prop_market_training_examples"):
        return pd.DataFrame()
    return _query_df(
        conn,
        """
        WITH ranked AS (
            SELECT DISTINCT ON (game_date_et, game_slug, player_id)
                   game_date_et, game_slug, player_id, player_name,
                   pred_count::float AS baseline_k,
                   projected_bf::float,
                   actual_bf::float,
                   actual_value::float AS actual_k,
                   projected_pitch_count::float,
                   actual_pitch_count_proxy::float AS actual_pitch_count,
                   pitcher_starts::float,
                   is_home::float,
                   opp_team_k_pct_10::float,
                   opp_team_obp_10::float,
                   opp_team_slg_10::float,
                   opp_bp_ip_last_3::float,
                   opp_bp_ip_last_7::float,
                   team_implied_runs::float,
                   opponent_implied_runs::float,
                   game_total_line::float,
                   minutes_to_first_pitch_at_lock::float
            FROM features.mlb_prop_market_training_examples
            WHERE game_date_et >= %(cutoff)s
              AND market = 'pitcher_strikeouts'
              AND pred_count IS NOT NULL
              AND projected_bf IS NOT NULL
              AND projected_bf > 0
              AND actual_bf IS NOT NULL
              AND actual_bf > 0
              AND actual_value IS NOT NULL
            ORDER BY game_date_et, game_slug, player_id, example_updated_at DESC, id DESC
        )
        SELECT * FROM ranked
        ORDER BY game_date_et, game_slug, player_id
        """,
        {"cutoff": cutoff},
    )


def _load_offer_rows(conn, cutoff) -> pd.DataFrame:
    if not _table_exists(conn, "features.mlb_prop_market_training_examples"):
        return pd.DataFrame()
    return _query_df(
        conn,
        """
        SELECT game_date_et, game_slug, player_id,
               side, market_line::float AS market_line,
               won, COALESCE(push, false) AS push,
               bookmaker_key, prop_offer_id
        FROM features.mlb_prop_market_training_examples
        WHERE game_date_et >= %(cutoff)s
          AND market = 'pitcher_strikeouts'
          AND market_line IS NOT NULL
          AND won IS NOT NULL
          AND COALESCE(push, false) IS FALSE
        ORDER BY game_date_et, game_slug, player_id, market_line, side, bookmaker_key
        """,
        {"cutoff": cutoff},
    )


def _count_metrics(actual: pd.Series, pred: pd.Series) -> dict[str, Any]:
    valid = pd.DataFrame({"actual": actual, "pred": pred}).dropna()
    if valid.empty:
        return {"rows": 0}
    error = valid["pred"] - valid["actual"]
    return {
        "rows": int(len(valid)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
    }


def _poisson_cdf(mean: Any, threshold: int) -> float | None:
    try:
        lam = float(mean)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lam):
        return None
    lam = max(1e-6, min(20.0, lam))
    if threshold < 0:
        return 0.0
    term = math.exp(-lam)
    cdf = term
    for k in range(1, threshold + 1):
        term *= lam / k
        cdf += term
    return max(1e-6, min(1.0 - 1e-6, cdf))


def _poisson_over_probability(mean: Any, line: Any) -> float | None:
    try:
        line_value = float(line)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(line_value):
        return None
    cdf = _poisson_cdf(mean, max(0, int(math.floor(line_value))))
    if cdf is None:
        return None
    return max(1e-6, min(1.0 - 1e-6, 1.0 - cdf))


def _side_probability_from_count(mean: Any, line: Any, side: Any) -> float | None:
    try:
        line_value = float(line)
    except (TypeError, ValueError):
        return None
    if str(side or "").lower() == "over":
        return _poisson_over_probability(mean, line_value)
    under_threshold = int(round(line_value) - 1) if abs(line_value - round(line_value)) <= 1e-9 else int(math.floor(line_value))
    return _poisson_cdf(mean, under_threshold)


def _weighted_brier(values: pd.Series, probabilities: pd.Series, weights: pd.Series) -> dict[str, Any]:
    work = pd.DataFrame({
        "target": pd.to_numeric(values, errors="coerce"),
        "probability": pd.to_numeric(probabilities, errors="coerce").clip(1e-6, 1.0 - 1e-6),
        "weight": pd.to_numeric(weights, errors="coerce").fillna(1.0),
    }).dropna(subset=["target", "probability"])
    work = work[work["weight"] > 0]
    if work.empty:
        return {"rows": 0, "brier": None}
    loss = np.square(work["target"] - work["probability"])
    return {
        "rows": int(len(work)),
        "brier": float(np.average(loss, weights=work["weight"])),
    }


def _line_brier_metrics(player_games: pd.DataFrame, offers: pd.DataFrame, pred: pd.Series) -> dict[str, Any]:
    if offers.empty:
        return {"rows": 0, "brier": None}
    keys = ["game_date_et", "game_slug", "player_id"]
    pred_frame = player_games.loc[pred.notna(), keys].copy()
    if pred_frame.empty:
        return {"rows": 0, "brier": None}
    pred_frame["candidate_k"] = pd.to_numeric(pred.loc[pred_frame.index], errors="coerce")
    joined = offers.merge(pred_frame, on=keys, how="inner")
    if joined.empty:
        return {"rows": 0, "brier": None}
    joined["probability"] = [
        _side_probability_from_count(mean, line, side)
        for mean, line, side in zip(joined["candidate_k"], joined["market_line"], joined["side"])
    ]
    joined["target"] = joined["won"].astype(bool).astype(float)
    group_size = joined.groupby(keys)["target"].transform("size").clip(lower=1)
    joined["offer_group_weight"] = 1.0 / group_size
    out = _weighted_brier(joined["target"], joined["probability"], joined["offer_group_weight"])
    out["player_games"] = int(joined[keys].drop_duplicates().shape[0])
    return out


def _candidate_record(
    *,
    alpha: float,
    actual: pd.Series,
    baseline_pred: pd.Series,
    candidate_pred: pd.Series,
    baseline_metrics: dict[str, Any],
    baseline_brier: dict[str, Any],
    player_games: pd.DataFrame,
    offers: pd.DataFrame,
) -> dict[str, Any]:
    count = _count_metrics(actual, candidate_pred)
    brier = _line_brier_metrics(player_games, offers, candidate_pred)
    mae_gain = (
        float(baseline_metrics["mae"] - count["mae"])
        if baseline_metrics.get("mae") is not None and count.get("mae") is not None
        else None
    )
    brier_gain = (
        float(baseline_brier["brier"] - brier["brier"])
        if baseline_brier.get("brier") is not None and brier.get("brier") is not None
        else None
    )
    return {
        "alpha": float(alpha),
        "rows": count.get("rows", 0),
        "mae": count.get("mae"),
        "rmse": count.get("rmse"),
        "bias": count.get("bias"),
        "mae_gain": mae_gain,
        "line_brier_rows": brier.get("rows", 0),
        "line_brier_player_games": brier.get("player_games", 0),
        "line_brier": brier.get("brier"),
        "line_brier_gain": brier_gain,
        "mean_shift_vs_baseline": float((candidate_pred - baseline_pred).mean()),
    }


def choose_conservative_blend(candidate_records: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next((rec for rec in candidate_records if abs(float(rec.get("alpha") or 0.0)) <= 1e-12), None)
    if baseline is None:
        return {"alpha": 0.0, "accepted": False, "reason": "missing_baseline_candidate"}
    accepted: list[dict[str, Any]] = []
    baseline_abs_bias = abs(float(baseline.get("bias") or 0.0))
    for rec in candidate_records:
        alpha = float(rec.get("alpha") or 0.0)
        if alpha <= 0:
            continue
        mae_gain = rec.get("mae_gain")
        brier_gain = rec.get("line_brier_gain")
        if mae_gain is None or brier_gain is None:
            continue
        bias = abs(float(rec.get("bias") or 0.0))
        if (
            mae_gain >= _MIN_MAE_GAIN
            and brier_gain >= _MIN_BRIER_GAIN
            and bias <= baseline_abs_bias + _MAX_BIAS_WORSENING
        ):
            accepted.append(rec)
    if not accepted:
        reasons = []
        positive_alpha = [rec for rec in candidate_records if float(rec.get("alpha") or 0.0) > 0]
        if not any((rec.get("mae_gain") is not None and rec["mae_gain"] >= _MIN_MAE_GAIN) for rec in positive_alpha):
            reasons.append("mae_not_improved")
        if not any((rec.get("line_brier_gain") is not None and rec["line_brier_gain"] >= _MIN_BRIER_GAIN) for rec in positive_alpha):
            reasons.append("line_brier_not_improved")
        if not reasons:
            reasons.append("bias_or_threshold_gate_failed")
        return {"alpha": 0.0, "accepted": False, "reason": ",".join(reasons)}
    best = sorted(
        accepted,
        key=lambda rec: (
            -float(rec.get("line_brier_gain") or 0.0),
            -float(rec.get("mae_gain") or 0.0),
            float(rec.get("alpha") or 0.0),
        ),
    )[0]
    return {"alpha": float(best["alpha"]), "accepted": True, "reason": "mae_and_line_brier_improved"}


def _selected_record(meta: dict[str, Any]) -> dict[str, Any]:
    alpha = float((meta or {}).get("selected_alpha") or 0.0)
    for rec in (meta or {}).get("candidate_records") or []:
        if abs(float(rec.get("alpha") or 0.0) - alpha) <= 1e-12:
            return rec
    return {}


def _fit_predict_challenger(df: pd.DataFrame, min_train_rows: int = 80) -> tuple[pd.Series, dict[str, Any]]:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
    except Exception as exc:
        return pd.Series(np.nan, index=df.index), {"enabled": False, "reason": f"sklearn_unavailable:{exc}"}

    work = df.copy()
    for col in _FEATURES:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    work["actual_k_rate"] = (work["actual_k"] / work["actual_bf"]).clip(0.0, 1.0)
    dates = sorted(pd.to_datetime(work["game_date_et"]).dt.date.unique())
    pred = pd.Series(np.nan, index=work.index, dtype=float)
    folds: list[dict[str, Any]] = []
    for holdout_date in dates:
        train_mask = pd.to_datetime(work["game_date_et"]).dt.date < holdout_date
        test_mask = pd.to_datetime(work["game_date_et"]).dt.date == holdout_date
        if int(train_mask.sum()) < min_train_rows or int(test_mask.sum()) == 0:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.035,
                max_leaf_nodes=16,
                l2_regularization=0.05,
                random_state=17,
            ),
        )
        model.fit(work.loc[train_mask, _FEATURES], work.loc[train_mask, "actual_k_rate"])
        fold_rate = np.clip(model.predict(work.loc[test_mask, _FEATURES]), 0.0, 1.0)
        pred.loc[test_mask] = fold_rate * pd.to_numeric(work.loc[test_mask, "projected_bf"], errors="coerce")
        folds.append({
            "holdout_date": str(holdout_date),
            "train_rows": int(train_mask.sum()),
            "holdout_rows": int(test_mask.sum()),
        })
    return pred, {"enabled": bool(folds), "folds": folds}


def _fit_predict_residual_v2(df: pd.DataFrame, min_train_rows: int = 80) -> tuple[pd.Series, dict[str, Any]]:
    """Small residual model: learn only when baseline misses by context.

    The first boosted challenger tried to replace the K rate.  V2 keeps the
    baseline as the anchor and predicts a bounded count residual from a smaller
    feature set, then the blend gate can shrink it almost completely away.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
    except Exception as exc:
        return pd.Series(np.nan, index=df.index), {"enabled": False, "reason": f"sklearn_unavailable:{exc}"}

    work = df.copy()
    for col in _FEATURES:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    work["baseline_k"] = pd.to_numeric(work["baseline_k"], errors="coerce")
    work["actual_k"] = pd.to_numeric(work["actual_k"], errors="coerce")
    work["projected_bf"] = pd.to_numeric(work["projected_bf"], errors="coerce")
    work["baseline_k_rate"] = work["baseline_k"] / work["projected_bf"].clip(lower=1.0)
    work["pitch_count_per_bf"] = (
        pd.to_numeric(work["projected_pitch_count"], errors="coerce")
        / work["projected_bf"].clip(lower=1.0)
    )
    opp_k = pd.to_numeric(work["opp_team_k_pct_10"], errors="coerce")
    work["opponent_k_delta"] = opp_k - float(opp_k.mean(skipna=True) if opp_k.notna().any() else 0.225)
    work["baseline_residual_k"] = (work["actual_k"] - work["baseline_k"]).clip(-5.0, 5.0)
    dates = sorted(pd.to_datetime(work["game_date_et"]).dt.date.unique())
    pred = pd.Series(np.nan, index=work.index, dtype=float)
    folds: list[dict[str, Any]] = []
    feature_list = [col for col in _RESIDUAL_V2_FEATURES if col in work.columns]
    for holdout_date in dates:
        train_mask = pd.to_datetime(work["game_date_et"]).dt.date < holdout_date
        test_mask = pd.to_datetime(work["game_date_et"]).dt.date == holdout_date
        if int(train_mask.sum()) < min_train_rows or int(test_mask.sum()) == 0:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=90,
                learning_rate=0.025,
                max_leaf_nodes=8,
                min_samples_leaf=18,
                l2_regularization=0.65,
                random_state=23,
            ),
        )
        model.fit(work.loc[train_mask, feature_list], work.loc[train_mask, "baseline_residual_k"])
        residual = np.clip(model.predict(work.loc[test_mask, feature_list]), -3.0, 3.0)
        pred.loc[test_mask] = (
            pd.to_numeric(work.loc[test_mask, "baseline_k"], errors="coerce") + residual
        ).clip(lower=0.0, upper=20.0)
        folds.append({
            "holdout_date": str(holdout_date),
            "train_rows": int(train_mask.sum()),
            "holdout_rows": int(test_mask.sum()),
        })
    return pred, {
        "enabled": bool(folds),
        "model": "small_baseline_residual_hist_gradient_boosting",
        "features": feature_list,
        "folds": folds,
        "target": "actual_k_minus_baseline_k",
        "residual_clip": [-3.0, 3.0],
    }


def _series_bucket(values: pd.Series, low: float, high: float, labels: tuple[str, str, str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [numeric <= low, numeric >= high, numeric.notna()],
            [labels[0], labels[2], labels[1]],
            default="missing",
        ),
        index=values.index,
    )


def _shrunk_residual_map(train: pd.DataFrame, key: str, shrink_rows: float) -> tuple[dict[str, float], float]:
    residual = pd.to_numeric(train["baseline_residual_k"], errors="coerce")
    global_mean = float(residual.mean()) if residual.notna().any() else 0.0
    rows: dict[str, float] = {}
    for value, group in train.groupby(key, dropna=False):
        group_resid = pd.to_numeric(group["baseline_residual_k"], errors="coerce").dropna()
        if group_resid.empty:
            continue
        n = float(len(group_resid))
        weight = n / (n + shrink_rows)
        rows[str(value)] = float(weight * group_resid.mean() + (1.0 - weight) * global_mean)
    return rows, global_mean


def _map_residual(test: pd.DataFrame, key: str, mapping: dict[str, float], fallback: float) -> pd.Series:
    return test[key].astype(str).map(mapping).fillna(float(fallback)).astype(float)


def _fit_predict_residual_v3(df: pd.DataFrame, min_train_rows: int = 80) -> tuple[pd.Series, dict[str, Any]]:
    """Empirical-Bayes K residual repair.

    V3 avoids replacing the projection with another model.  It only learns
    historical residual tendencies, shrunk heavily toward the global baseline,
    for pitcher identity and broad context buckets.
    """
    work = df.copy()
    work["baseline_k"] = pd.to_numeric(work["baseline_k"], errors="coerce")
    work["actual_k"] = pd.to_numeric(work["actual_k"], errors="coerce")
    work["projected_bf"] = pd.to_numeric(work["projected_bf"], errors="coerce")
    work["projected_pitch_count"] = pd.to_numeric(work["projected_pitch_count"], errors="coerce")
    work["pitch_count_per_bf"] = work["projected_pitch_count"] / work["projected_bf"].clip(lower=1.0)
    work["baseline_residual_k"] = (work["actual_k"] - work["baseline_k"]).clip(-4.0, 4.0)
    work["opp_k_bucket"] = _series_bucket(
        work.get("opp_team_k_pct_10"), 0.20, 0.25, ("low_opp_k", "mid_opp_k", "high_opp_k")
    )
    work["bf_bucket"] = _series_bucket(
        work["projected_bf"], 19.0, 24.0, ("short_bf", "normal_bf", "long_bf")
    )
    work["pitch_bucket"] = _series_bucket(
        work["projected_pitch_count"], 78.0, 96.0, ("short_leash", "normal_leash", "long_leash")
    )
    work["leash_context_bucket"] = work["pitch_bucket"].astype(str) + "|" + work["bf_bucket"].astype(str)
    work["pitch_efficiency_bucket"] = _series_bucket(
        work["pitch_count_per_bf"], 3.75, 4.25, ("efficient_pitch_budget", "normal_pitch_budget", "laboring_pitch_budget")
    )
    run_diff = (
        pd.to_numeric(work.get("team_implied_runs"), errors="coerce")
        - pd.to_numeric(work.get("opponent_implied_runs"), errors="coerce")
    )
    work["game_script_bucket"] = pd.Series(
        np.select(
            [run_diff >= 0.75, run_diff <= -0.75, run_diff.notna()],
            ["favorite_script", "dog_script", "neutral_script"],
            default="script_missing",
        ),
        index=work.index,
    )
    opponent_damage = (
        pd.to_numeric(work.get("opp_team_obp_10"), errors="coerce").fillna(0.315)
        + pd.to_numeric(work.get("opp_team_slg_10"), errors="coerce").fillna(0.400)
    )
    work["opponent_damage_bucket"] = _series_bucket(
        opponent_damage,
        0.690,
        0.760,
        ("weak_opp_contact", "normal_opp_contact", "strong_opp_contact"),
    )
    work["baseline_rate_bucket"] = _series_bucket(
        work["baseline_k"] / work["projected_bf"].clip(lower=1.0),
        0.19,
        0.28,
        ("low_k_skill", "mid_k_skill", "high_k_skill"),
    )
    dates = sorted(pd.to_datetime(work["game_date_et"]).dt.date.unique())
    pred = pd.Series(np.nan, index=work.index, dtype=float)
    folds: list[dict[str, Any]] = []
    for holdout_date in dates:
        train_mask = pd.to_datetime(work["game_date_et"]).dt.date < holdout_date
        test_mask = pd.to_datetime(work["game_date_et"]).dt.date == holdout_date
        if int(train_mask.sum()) < min_train_rows or int(test_mask.sum()) == 0:
            continue
        train = work.loc[train_mask].copy()
        test = work.loc[test_mask].copy()
        pitcher_map, global_mean = _shrunk_residual_map(train, "player_id", 5.0)
        opp_map, _ = _shrunk_residual_map(train, "opp_k_bucket", 25.0)
        bf_map, _ = _shrunk_residual_map(train, "bf_bucket", 25.0)
        pitch_map, _ = _shrunk_residual_map(train, "pitch_bucket", 25.0)
        rate_map, _ = _shrunk_residual_map(train, "baseline_rate_bucket", 25.0)
        leash_map, _ = _shrunk_residual_map(train, "leash_context_bucket", 35.0)
        efficiency_map, _ = _shrunk_residual_map(train, "pitch_efficiency_bucket", 35.0)
        script_map, _ = _shrunk_residual_map(train, "game_script_bucket", 45.0)
        damage_map, _ = _shrunk_residual_map(train, "opponent_damage_bucket", 45.0)
        residual = (
            0.36 * _map_residual(test, "player_id", pitcher_map, global_mean)
            + 0.15 * _map_residual(test, "opp_k_bucket", opp_map, global_mean)
            + 0.11 * _map_residual(test, "bf_bucket", bf_map, global_mean)
            + 0.10 * _map_residual(test, "pitch_bucket", pitch_map, global_mean)
            + 0.10 * _map_residual(test, "baseline_rate_bucket", rate_map, global_mean)
            + 0.08 * _map_residual(test, "leash_context_bucket", leash_map, global_mean)
            + 0.04 * _map_residual(test, "pitch_efficiency_bucket", efficiency_map, global_mean)
            + 0.03 * _map_residual(test, "game_script_bucket", script_map, global_mean)
            + 0.03 * _map_residual(test, "opponent_damage_bucket", damage_map, global_mean)
        ).clip(-1.50, 1.50)
        pred.loc[test.index] = (test["baseline_k"] + residual).clip(lower=0.0, upper=20.0)
        folds.append({
            "holdout_date": str(holdout_date),
            "train_rows": int(train_mask.sum()),
            "holdout_rows": int(test_mask.sum()),
            "global_residual": global_mean,
        })
    return pred, {
        "enabled": bool(folds),
        "model": "empirical_bayes_baseline_residual_v4_leash_context",
        "target": "actual_k_minus_baseline_k",
        "folds": folds,
        "residual_clip": [-1.50, 1.50],
        "shrinkage": {
            "pitcher_rows": 5.0,
            "core_context_rows": 25.0,
            "leash_context_rows": 35.0,
            "script_context_rows": 45.0,
        },
        "context_buckets": {
            "opponent_k": "opp_team_k_pct_10",
            "bf": "projected_bf",
            "pitch_count": "projected_pitch_count",
            "leash": "projected_pitch_count|projected_bf",
            "pitch_efficiency": "projected_pitch_count/projected_bf",
            "game_script": "team_implied_runs_minus_opponent_implied_runs",
            "opponent_contact": "opp_team_obp_10_plus_opp_team_slg_10",
        },
    }


def _select_conservative_challenger(
    df: pd.DataFrame,
    boosted_pred: pd.Series,
    offers: pd.DataFrame,
    *,
    alphas: tuple[float, ...] = _SHRINKAGE_ALPHAS,
) -> tuple[pd.Series, dict[str, Any]]:
    valid = boosted_pred.notna() & pd.to_numeric(df["baseline_k"], errors="coerce").notna()
    selected = pd.Series(np.nan, index=df.index, dtype=float)
    if not valid.any():
        return selected, {
            "accepted": False,
            "selected_alpha": 0.0,
            "reason": "no_oof_boosted_predictions",
            "candidate_records": [],
        }
    work = df.loc[valid].copy()
    baseline_pred = pd.to_numeric(work["baseline_k"], errors="coerce")
    boosted = pd.to_numeric(boosted_pred.loc[valid], errors="coerce")
    actual = pd.to_numeric(work["actual_k"], errors="coerce")
    baseline_metrics = _count_metrics(actual, baseline_pred)
    baseline_brier = _line_brier_metrics(work, offers, baseline_pred)
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        candidate = baseline_pred + alpha * (boosted - baseline_pred)
        candidate = candidate.clip(lower=0.0, upper=20.0)
        records.append(_candidate_record(
            alpha=alpha,
            actual=actual,
            baseline_pred=baseline_pred,
            candidate_pred=candidate,
            baseline_metrics=baseline_metrics,
            baseline_brier=baseline_brier,
            player_games=work,
            offers=offers,
        ))
    decision = choose_conservative_blend(records)
    alpha = float(decision.get("alpha") or 0.0)
    selected.loc[valid] = (baseline_pred + alpha * (boosted - baseline_pred)).clip(lower=0.0, upper=20.0)
    return selected, {
        "accepted": bool(decision.get("accepted")),
        "selected_alpha": alpha,
        "reason": decision.get("reason"),
        "candidate_records": records,
        "baseline_line_brier": baseline_brier,
    }


def _bucket(value: pd.Series, thresholds: tuple[float, float], labels: tuple[str, str, str]) -> pd.Series:
    numeric = pd.to_numeric(value, errors="coerce")
    return pd.Series(
        np.select(
            [numeric <= thresholds[0], numeric >= thresholds[1], numeric.notna()],
            [labels[0], labels[2], labels[1]],
            default="missing",
        ),
        index=value.index,
    )


def _slice_rows(df: pd.DataFrame, column: str, min_rows: int) -> list[dict[str, Any]]:
    rows = []
    for key, group in df.groupby(column, dropna=False):
        group = group.loc[group["challenger_k"].notna()]
        if len(group) < min_rows:
            continue
        baseline = _count_metrics(group["actual_k"], group["baseline_k"])
        challenger = _count_metrics(group["actual_k"], group["challenger_k"])
        rows.append({
            column: "missing" if pd.isna(key) else str(key),
            "rows": int(len(group)),
            "dates": int(pd.to_datetime(group["game_date_et"]).dt.date.nunique()),
            "baseline_mae": baseline.get("mae"),
            "challenger_mae": challenger.get("mae"),
            "mae_gain": (
                float(baseline["mae"] - challenger["mae"])
                if baseline.get("mae") is not None and challenger.get("mae") is not None
                else None
            ),
            "baseline_bias": baseline.get("bias"),
            "challenger_bias": challenger.get("bias"),
        })
    return sorted(rows, key=lambda row: float(row.get("mae_gain") or -999.0), reverse=True)


def build(lookback_days: int = 365, min_rows: int = 20, pg_dsn: str = PG_DSN) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, lookback_days))
    with psycopg2.connect(pg_dsn) as conn:
        df = _load_rows(conn, cutoff)
        offer_rows = _load_offer_rows(conn, cutoff)
    if df.empty:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "status": "no_rows",
            "usage": "challenger_diagnostic_only",
            "rows": 0,
        }
    else:
        df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
        if not offer_rows.empty:
            offer_rows["game_date_et"] = pd.to_datetime(offer_rows["game_date_et"]).dt.date
        raw_boosted_k, raw_boosted_meta = _fit_predict_challenger(df)
        boosted_k, boosted_meta = _select_conservative_challenger(
            df,
            raw_boosted_k,
            offer_rows,
            alphas=_SHRINKAGE_ALPHAS,
        )
        residual_v2_k, residual_v2_meta = _fit_predict_residual_v2(df)
        residual_k, residual_meta = _select_conservative_challenger(
            df,
            residual_v2_k,
            offer_rows,
            alphas=_RESIDUAL_V2_ALPHAS,
        )
        residual_v3_k, residual_v3_meta = _fit_predict_residual_v3(df)
        residual_eb_k, residual_eb_meta = _select_conservative_challenger(
            df,
            residual_v3_k,
            offer_rows,
            alphas=_RESIDUAL_V3_ALPHAS,
        )
        challenger_options = [
            ("boosted_rate_v1", boosted_k, boosted_meta),
            ("baseline_residual_v2", residual_k, residual_meta),
            ("empirical_bayes_residual_v3", residual_eb_k, residual_eb_meta),
        ]
        accepted_options = [
            option for option in challenger_options
            if bool((option[2] or {}).get("accepted"))
        ]
        if accepted_options:
            selected_name, challenger_k, challenger_meta = sorted(
                accepted_options,
                key=lambda option: (
                    float(_selected_record(option[2]).get("line_brier_gain") or -999.0),
                    float(_selected_record(option[2]).get("mae_gain") or -999.0),
                ),
                reverse=True,
            )[0]
        else:
            selected_name, challenger_k, challenger_meta = "baseline", boosted_k, boosted_meta
            challenger_meta = {
                **(challenger_meta or {}),
                "accepted": False,
                "selected_alpha": 0.0,
                "reason": "no_challenger_variant_improved",
            }
        df["raw_boosted_k"] = raw_boosted_k
        df["residual_v2_k"] = residual_v2_k
        df["residual_v3_k"] = residual_v3_k
        df["challenger_k"] = challenger_k
        df["bf_error"] = pd.to_numeric(df["projected_bf"], errors="coerce") - pd.to_numeric(df["actual_bf"], errors="coerce")
        df["pitch_count_error"] = pd.to_numeric(df["projected_pitch_count"], errors="coerce") - pd.to_numeric(df["actual_pitch_count"], errors="coerce")
        df["bf_error_bucket"] = _bucket(df["bf_error"], (-3.0, 3.0), ("under_bf_by_3_plus", "near_bf", "over_bf_by_3_plus"))
        df["pitch_count_error_bucket"] = _bucket(df["pitch_count_error"], (-12.0, 12.0), ("under_pitches_by_12_plus", "near_pitches", "over_pitches_by_12_plus"))
        df["opp_k_bucket"] = _bucket(df["opp_team_k_pct_10"], (0.20, 0.25), ("low_opp_k", "mid_opp_k", "high_opp_k"))
        eval_mask = df["challenger_k"].notna()
        eval_df = df.loc[eval_mask].copy()
        baseline_all_rows = _count_metrics(df["actual_k"], df["baseline_k"])
        baseline = _count_metrics(eval_df["actual_k"], eval_df["baseline_k"])
        raw_boosted = _count_metrics(eval_df["actual_k"], eval_df["raw_boosted_k"])
        residual_v2 = _count_metrics(eval_df["actual_k"], eval_df["residual_v2_k"])
        residual_v3 = _count_metrics(eval_df["actual_k"], eval_df["residual_v3_k"])
        challenger = _count_metrics(eval_df["actual_k"], eval_df["challenger_k"])
        baseline_line_brier = _line_brier_metrics(eval_df, offer_rows, pd.to_numeric(eval_df["baseline_k"], errors="coerce"))
        raw_boosted_line_brier = _line_brier_metrics(eval_df, offer_rows, eval_df["raw_boosted_k"])
        residual_v2_line_brier = _line_brier_metrics(eval_df, offer_rows, eval_df["residual_v2_k"])
        residual_v3_line_brier = _line_brier_metrics(eval_df, offer_rows, eval_df["residual_v3_k"])
        challenger_line_brier = _line_brier_metrics(eval_df, offer_rows, eval_df["challenger_k"])
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "status": "ready",
            "usage": "challenger_diagnostic_only",
            "rows": int(len(df)),
            "offer_rows": int(len(offer_rows)),
            "dates": int(df["game_date_et"].nunique()),
            "evaluation_rows": int(len(eval_df)),
            "baseline_all_rows": baseline_all_rows,
            "baseline": baseline,
            "raw_boosted": raw_boosted,
            "residual_v2": residual_v2,
            "residual_v3": residual_v3,
            "challenger": challenger,
            "line_brier": {
                "baseline": baseline_line_brier,
                "raw_boosted": raw_boosted_line_brier,
                "residual_v2": residual_v2_line_brier,
                "residual_v3": residual_v3_line_brier,
                "challenger": challenger_line_brier,
            },
            "mae_gain": (
                float(baseline["mae"] - challenger["mae"])
                if baseline.get("mae") is not None and challenger.get("mae") is not None
                else None
            ),
            "line_brier_gain": (
                float(baseline_line_brier["brier"] - challenger_line_brier["brier"])
                if baseline_line_brier.get("brier") is not None and challenger_line_brier.get("brier") is not None
                else None
            ),
            "selected_challenger_variant": selected_name,
            "raw_boosted_meta": raw_boosted_meta,
            "boosted_rate_v1_meta": boosted_meta,
            "residual_v2_meta": residual_v2_meta,
            "residual_v2_selection_meta": residual_meta,
            "residual_v3_meta": residual_v3_meta,
            "residual_v3_selection_meta": residual_eb_meta,
            "challenger_meta": challenger_meta,
            "by_bf_error_bucket": _slice_rows(df, "bf_error_bucket", min_rows),
            "by_pitch_count_error_bucket": _slice_rows(df, "pitch_count_error_bucket", min_rows),
            "by_opponent_k_bucket": _slice_rows(df, "opp_k_bucket", min_rows),
        }
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_MODEL_DIR / "pitcher_k_rate_challenger_report.json", payload)
    report_path = _REPORT_DIR / "mlb_pitcher_k_rate_challenger_latest.md"
    atomic_write_text(report_path, _render(payload))
    payload["report_path"] = str(report_path)
    return payload


def _num(value: Any, digits: int = 3) -> str:
    try:
        return "-" if value is None else f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Pitcher K-Per-BF Challenger",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Status: **{payload['status']}**",
        "Usage: challenger diagnostic only; production artifacts are unchanged.",
        "",
        f"- Rows: {payload.get('rows', 0)}",
        f"- OOF evaluation rows: {payload.get('evaluation_rows', 0)}",
        f"- Offer rows for line-level Brier: {payload.get('offer_rows', 0)}",
        f"- Dates: {payload.get('dates', 0)}",
        f"- Baseline K MAE, all loaded rows: {_num((payload.get('baseline_all_rows') or {}).get('mae'))}",
        f"- Baseline K MAE: {_num((payload.get('baseline') or {}).get('mae'))}",
        f"- Raw boosted K MAE: {_num((payload.get('raw_boosted') or {}).get('mae'))}",
        f"- Residual v2 K MAE: {_num((payload.get('residual_v2') or {}).get('mae'))}",
        f"- Residual v3 K MAE: {_num((payload.get('residual_v3') or {}).get('mae'))}",
        f"- Challenger K MAE: {_num((payload.get('challenger') or {}).get('mae'))}",
        f"- Selected challenger variant: {payload.get('selected_challenger_variant', '-')}",
        f"- MAE gain: {_num(payload.get('mae_gain'))}",
        f"- Line-level Brier gain: {_num(payload.get('line_brier_gain'), 5)}",
        f"- Selected alpha: {_num((payload.get('challenger_meta') or {}).get('selected_alpha'))}",
        f"- Accepted: {bool((payload.get('challenger_meta') or {}).get('accepted'))}",
        f"- Gate reason: {(payload.get('challenger_meta') or {}).get('reason') or '-'}",
        "",
        "## Conservative Blend Gate",
        "",
        "| Alpha | Rows | MAE | MAE Gain | Bias | Line Brier Rows | Line Brier | Brier Gain | Mean Shift |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in (payload.get("challenger_meta") or {}).get("candidate_records") or []:
        lines.append(
            f"| {_num(rec.get('alpha'))} | {rec.get('rows', 0)} | {_num(rec.get('mae'))} | "
            f"{_num(rec.get('mae_gain'))} | {_num(rec.get('bias'))} | {rec.get('line_brier_rows', 0)} | "
            f"{_num(rec.get('line_brier'), 5)} | {_num(rec.get('line_brier_gain'), 5)} | "
            f"{_num(rec.get('mean_shift_vs_baseline'))} |"
        )
    lines.append("")
    residual_meta = payload.get("residual_v2_selection_meta") or {}
    if residual_meta:
        lines.extend([
            "## Residual v2 Gate",
            "",
            "| Alpha | Rows | MAE | MAE Gain | Bias | Line Brier Rows | Line Brier | Brier Gain | Mean Shift |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for rec in residual_meta.get("candidate_records") or []:
            lines.append(
                f"| {_num(rec.get('alpha'))} | {rec.get('rows', 0)} | {_num(rec.get('mae'))} | "
                f"{_num(rec.get('mae_gain'))} | {_num(rec.get('bias'))} | {rec.get('line_brier_rows', 0)} | "
                f"{_num(rec.get('line_brier'), 5)} | {_num(rec.get('line_brier_gain'), 5)} | "
                f"{_num(rec.get('mean_shift_vs_baseline'))} |"
            )
        lines.extend([
            "",
            f"- Residual v2 accepted: {bool(residual_meta.get('accepted'))}",
            f"- Residual v2 reason: {residual_meta.get('reason') or '-'}",
            "",
        ])
    residual_v3_meta = payload.get("residual_v3_selection_meta") or {}
    if residual_v3_meta:
        lines.extend([
            "## Empirical-Bayes Residual v3 Gate",
            "",
            "| Alpha | Rows | MAE | MAE Gain | Bias | Line Brier Rows | Line Brier | Brier Gain | Mean Shift |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for rec in residual_v3_meta.get("candidate_records") or []:
            lines.append(
                f"| {_num(rec.get('alpha'))} | {rec.get('rows', 0)} | {_num(rec.get('mae'))} | "
                f"{_num(rec.get('mae_gain'))} | {_num(rec.get('bias'))} | {rec.get('line_brier_rows', 0)} | "
                f"{_num(rec.get('line_brier'), 5)} | {_num(rec.get('line_brier_gain'), 5)} | "
                f"{_num(rec.get('mean_shift_vs_baseline'))} |"
            )
        lines.extend([
            "",
            f"- Residual v3 accepted: {bool(residual_v3_meta.get('accepted'))}",
            f"- Residual v3 reason: {residual_v3_meta.get('reason') or '-'}",
            "",
        ])
    lines.extend([
        "## Slices",
        "",
    ])
    for key, title in (
        ("by_bf_error_bucket", "BF Error"),
        ("by_pitch_count_error_bucket", "Pitch Count Error"),
        ("by_opponent_k_bucket", "Opponent K Profile"),
    ):
        lines.extend([
            f"## {title}",
            "",
            "| Bucket | Rows | Dates | Base MAE | Challenger MAE | Gain | Base Bias | Challenger Bias |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in payload.get(key) or []:
            bucket = row.get(key.replace("by_", "").replace("_bucket", "_bucket")) or row.get("bf_error_bucket") or row.get("pitch_count_error_bucket") or row.get("opp_k_bucket")
            lines.append(
                f"| {bucket} | {row.get('rows', 0)} | {row.get('dates', 0)} | "
                f"{_num(row.get('baseline_mae'))} | {_num(row.get('challenger_mae'))} | "
                f"{_num(row.get('mae_gain'))} | {_num(row.get('baseline_bias'))} | {_num(row.get('challenger_bias'))} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline pitcher K-per-BF challenger report")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--min-rows", type=int, default=20)
    parser.add_argument("--pg-dsn", default=PG_DSN)
    args = parser.parse_args()
    payload = build(args.lookback_days, args.min_rows, args.pg_dsn)
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows"),
        "mae_gain": payload.get("mae_gain"),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
