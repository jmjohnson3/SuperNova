"""Estimate NFL player stat distributions from player-game holdout residuals."""
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
from scipy.stats import norm, poisson
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine, text

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.markets import SPEC_BY_STAT, STAT_SPECS
from nfl_pipeline.modeling.train_player_stat_models import (
    SQL_TRAIN,
    _baseline_from_metric_column,
    _baseline_suite,
    _best_baseline,
    _make_features,
    _predict_binary_probability,
    _predict_model_payload,
    _temporal_split,
)
from nfl_pipeline.modeling.predict_player_props import (
    _apply_workload_projection_adjustment,
    _receiver_spike_signal_values_single,
    _receiver_spike_mixture_weights_single,
    _side_prob,
    _workload_adjustment_factors,
)

log = logging.getLogger("nfl_pipeline.modeling.train_player_stat_distributions")

ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
DEFAULT_MD_REPORT = ROOT / "reports" / "nfl_player_stat_distributions_latest.md"


@dataclass(frozen=True)
class DistributionConfig:
    pg_dsn: str = PG_DSN
    model_dir: Path = _MODEL_DIR
    model_file: str = "nfl_player_stat_models.joblib"
    opportunity_model_file: str = "nfl_player_opportunity_models.joblib"
    out_file: str = "nfl_player_stat_distributions.json"
    md_report_file: Path = DEFAULT_MD_REPORT
    min_prev_games: int = 3
    holdout_weeks: int = 4
    min_rows: int = 100
    stats: tuple[str, ...] = ()


def _load_artifact(cfg: DistributionConfig) -> dict[str, Any]:
    path = cfg.model_dir / cfg.model_file
    if not path.exists():
        return {"status": "missing", "models": {}, "metrics": {}}
    obj = joblib.load(path)
    if not isinstance(obj, dict):
        return {"status": "bad_artifact", "models": {}, "metrics": {}}
    distribution_path = cfg.model_dir / cfg.out_file
    if distribution_path.exists():
        obj["comparison_distributions"] = json.loads(distribution_path.read_text(encoding="utf-8")).get("distributions") or {}
    opp_path = cfg.model_dir / cfg.opportunity_model_file
    if opp_path.exists():
        try:
            opp_artifact = joblib.load(opp_path)
            if isinstance(opp_artifact, dict):
                obj["opportunity_artifact"] = opp_artifact
                obj["opportunity_status"] = opp_artifact.get("status") or "loaded"
        except Exception as exc:
            log.warning("Could not load NFL opportunity model artifact at %s: %s", opp_path, exc)
            obj["opportunity_status"] = "load_failed"
    return obj


def _predict_stat(artifact: dict[str, Any], stat: str, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, bool]:
    metrics = (artifact.get("metrics") or {}).get(stat) or {}
    base = _baseline_from_metric_column(df, stat, metrics)
    if bool(metrics.get("td_probability_accepted") or metrics.get("td_probability_pass")):
        accepted = True
    elif "projection_accepted" in metrics:
        accepted = bool(metrics.get("projection_accepted"))
    elif "projection_pass" in metrics:
        accepted = bool(metrics.get("projection_pass"))
    else:
        accepted = bool(metrics.get("accepted"))
    model = (artifact.get("models") or {}).get(stat)
    columns = (artifact.get("feature_columns") or {}).get(stat) or []
    fills = (artifact.get("fill_values") or {}).get(stat) or {}
    if model is None or not columns or not accepted:
        return base, base, False
    X_raw = _make_features(df)
    X = X_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    raw_projection = _predict_model_payload(model, X, base)
    workload = _workload_adjustment_factors(df, artifact)
    projection = _apply_workload_projection_adjustment(df, stat, raw_projection, base, workload)
    return projection, base, True


def _td_any_probability(artifact: dict[str, Any], stat: str, df: pd.DataFrame) -> np.ndarray | None:
    if stat != "receiving_tds":
        return None
    metrics = (artifact.get("metrics") or {}).get(stat) or {}
    if not bool(metrics.get("td_probability_accepted") or metrics.get("td_probability_pass")):
        return None
    model = (artifact.get("models") or {}).get(stat)
    if not isinstance(model, dict) or str(model.get("kind") or "") != "rare_event_classifier":
        return None
    clf = model.get("classifier")
    columns = (artifact.get("feature_columns") or {}).get(stat) or []
    fills = (artifact.get("fill_values") or {}).get(stat) or {}
    if clf is None or not columns:
        return None
    X_raw = _make_features(df)
    X = X_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    p_any = _predict_binary_probability(clf, X)
    probability_blend_weight = model.get("probability_blend_weight")
    if probability_blend_weight is not None:
        p_weight = float(probability_blend_weight)
        baseline = _baseline_from_metric_column(df, stat, {"baseline_column": model.get("baseline_column")})
        baseline_prob = np.clip(1.0 - np.exp(-np.clip(np.asarray(baseline, dtype=float), 0.0, None)), 0.001, 0.999)
        p_any = np.clip(p_weight * p_any + (1.0 - p_weight) * baseline_prob, 0.001, 0.999)
    return p_any


def _brier(prob: np.ndarray, actual: np.ndarray) -> float:
    prob = np.clip(np.asarray(prob, dtype=float), 0.001, 0.999)
    actual = np.asarray(actual, dtype=float)
    return float(np.mean((prob - actual) ** 2)) if len(actual) else 0.0


def _num(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(df.get(name, pd.Series(default, index=df.index)), errors="coerce").fillna(default)


def _predict_opportunity_projection(
    artifact: dict[str, Any],
    name: str,
    baseline_col: str,
    df: pd.DataFrame,
    default: float,
) -> tuple[np.ndarray, bool]:
    opp_artifact = artifact.get("opportunity_artifact") or {}
    metrics = (opp_artifact.get("metrics") or {}).get(name) or {}
    X_raw = _make_features(df)
    if baseline_col in df.columns:
        base_series = df[baseline_col]
    elif baseline_col in X_raw.columns:
        base_series = X_raw[baseline_col]
    else:
        base_series = pd.Series(default, index=df.index)
    base = pd.to_numeric(base_series, errors="coerce").fillna(default).to_numpy(dtype=float)
    accepted = bool(metrics.get("accepted") or metrics.get("projection_pass") or metrics.get("live_signal_accepted"))
    model_obj = (opp_artifact.get("models") or {}).get(name)
    columns = (opp_artifact.get("feature_columns") or {}).get(name) or []
    fills = (opp_artifact.get("fill_values") or {}).get(name) or {}
    if not accepted or model_obj is None or not columns:
        return np.clip(base, 0.0, None), False
    X = X_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    try:
        if isinstance(model_obj, dict):
            raw_model = model_obj.get("model")
            if raw_model is None:
                return np.clip(base, 0.0, None), False
            raw_pred = np.asarray(raw_model.predict(X), dtype=float)
            if str(model_obj.get("kind") or "direct") == "residual":
                raw_pred = base + float(model_obj.get("shrink") or 0.0) * raw_pred
            blend_weight = model_obj.get("probability_blend_weight")
            if blend_weight is not None:
                weight = float(blend_weight)
                raw_pred = weight * raw_pred + (1.0 - weight) * base
            return np.clip(raw_pred, 0.0, None), True
        return np.clip(np.asarray(model_obj.predict(X), dtype=float), 0.0, None), True
    except Exception as exc:
        log.warning("Could not score NFL opportunity projection %s: %s", name, exc)
        return np.clip(base, 0.0, None), False


def _qb_ypa_prior(df: pd.DataFrame) -> np.ndarray:
    att5 = _num(df, "pass_attempts_avg_5", np.nan).replace(0.0, np.nan)
    att10 = _num(df, "pass_attempts_avg_10", np.nan).replace(0.0, np.nan)
    yds5 = _num(df, "passing_yards_avg_5", np.nan)
    yds10 = _num(df, "passing_yards_avg_10", np.nan)
    ypa5 = (yds5 / att5).replace([np.inf, -np.inf], np.nan)
    ypa10 = (yds10 / att10).replace([np.inf, -np.inf], np.nan)
    league_prior = float(pd.concat([ypa5, ypa10], axis=0).dropna().clip(3.0, 11.5).mean()) if ypa5.notna().any() or ypa10.notna().any() else 6.8
    ypa = (0.62 * ypa5.fillna(league_prior) + 0.38 * ypa10.fillna(ypa5).fillna(league_prior)).clip(3.0, 11.5)
    return ypa.fillna(league_prior).to_numpy(dtype=float)


def _sigmoid(series: pd.Series) -> pd.Series:
    z = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(-35.0, 35.0)
    return pd.Series(1.0 / (1.0 + np.exp(-z)), index=series.index, dtype=float)


def _high_workload_score(stat: str, df: pd.DataFrame) -> pd.Series:
    if stat == "passing_yards":
        score = pd.Series(
            np.maximum.reduce([
                _num(df, "high_pass_attempt_score", np.nan),
                _num(df, "_pred_high_pass_attempt_probability", np.nan),
                _num(df, "pass_spike_path_score", np.nan),
                _num(df, "qb_volume_spike_signal", np.nan),
            ]),
            index=df.index,
        )
        fallback = _sigmoid((_num(df, "pass_attempts_avg_5", 0.0) - 31.0) / 5.5)
    elif stat == "rushing_yards":
        score = pd.Series(
            np.maximum.reduce([
                _num(df, "high_carry_score", np.nan),
                _num(df, "_pred_high_carry_probability", np.nan),
                _num(df, "carry_spike_path_score", np.nan),
                _num(df, "rb_carry_spike_v2_score", np.nan),
                _num(df, "rb_live_carry_v3_score", np.nan),
                _num(df, "rb_carry_under_correction_v4_score", np.nan),
                0.82 * _num(df, "workload_upside_v2_score", np.nan),
                0.65 * _num(df, "spike_snap_share_score", np.nan),
            ]),
            index=df.index,
        )
        fallback = _sigmoid((_num(df, "carries_avg_5", 0.0) - 11.0) / 3.8)
    elif stat == "receiving_yards":
        score = pd.Series(
            np.maximum.reduce([
                _num(df, "high_target_score", np.nan),
                _num(df, "target_spike_path_score", np.nan),
                _num(df, "receiver_target_eruption_score", np.nan),
                _num(df, "air_yards_spike_path_score", np.nan),
                _num(df, "receiver_air_yards_eruption_score", np.nan),
                _num(df, "receiver_explosive_spike_score", np.nan),
                _num(df, "receiver_target_route_spike_score", np.nan),
                _num(df, "receiver_target_spike_v2_score", np.nan),
                _num(df, "_pred_receiver_air_yards_spike_probability", np.nan),
                _num(df, "receiver_air_yards_spike_v2_score", np.nan),
                _num(df, "receiver_ypt_efficiency_spike_score", np.nan),
                _num(df, "receiver_live_spike_v3_score", np.nan),
                _num(df, "receiver_spike_under_correction_v4_score", np.nan),
                0.78 * ((_num(df, "_pred_receiver_air_yards_share", np.nan) - 0.20) / 0.22),
                0.76 * ((_num(df, "_pred_receiver_air_yards", np.nan) - 42.0) / 58.0),
                0.84 * _num(df, "workload_upside_v2_score", np.nan),
                _num(df, "receiver_contextual_spike_score", np.nan),
                0.92 * _num(df, "receiver_target_command_score", np.nan),
                0.88 * _num(df, "receiver_route_spike_readiness_score", np.nan),
                _num(df, "receiver_teammate_vacancy_score", np.nan),
                0.75 * _num(df, "spike_snap_share_score", np.nan),
            ]),
            index=df.index,
        )
        fallback = _sigmoid((_num(df, "targets_avg_5", 0.0) - 6.0) / 2.2)
    else:
        return pd.Series(0.0, index=df.index, dtype=float)
    score = pd.to_numeric(score, errors="coerce")
    return score.fillna(fallback).clip(0.02, 0.92)


def _yardage_states(stat: str) -> tuple[str, ...]:
    if stat == "receiving_yards":
        return ("low", "normal", "high", "spike")
    return ("low", "normal", "high")


def _yardage_state_weights(stat: str, df: pd.DataFrame) -> pd.DataFrame:
    limited = pd.Series(
        np.maximum(
            _num(df, "limited_workload_risk_score", 0.05).clip(0.0, 1.0),
            np.maximum(
                0.88 * _num(df, "workload_downside_v2_score", 0.0).clip(0.0, 1.0),
                0.65 * _num(df, "rest_risk_score", 0.0).clip(0.0, 1.0),
            ),
        ),
        index=df.index,
    )
    usage_quality = _num(df, "live_usage_context_quality_v4_score", 0.0).clip(0.0, 1.0)
    limited = pd.Series(np.maximum(limited, 0.18 * (1.0 - usage_quality)), index=df.index).clip(0.0, 1.0)
    full = _num(df, "full_workload_score", 0.65).clip(0.0, 1.0)
    low = np.maximum(limited, (0.62 - full).clip(lower=0.0) * 0.55).clip(0.02, 0.70)
    raw_high = _high_workload_score(stat, df) * (1.0 - 0.75 * low)
    if stat == "receiving_yards":
        spike_signal = pd.Series(
            np.maximum.reduce([
                _num(df, "receiver_target_route_spike_score", np.nan),
                _num(df, "receiver_target_spike_v2_score", np.nan),
                _num(df, "_pred_receiver_air_yards_spike_probability", np.nan),
                _num(df, "receiver_air_yards_spike_v2_score", np.nan),
                _num(df, "receiver_ypt_efficiency_spike_score", np.nan),
                _num(df, "receiver_live_spike_v3_score", np.nan),
                _num(df, "receiver_spike_under_correction_v4_score", np.nan),
                0.84 * _num(df, "workload_upside_v2_score", np.nan),
                _num(df, "_pred_receiver_target_route_spike_probability", np.nan),
                _num(df, "_pred_receiver_target_spike_v2_probability", np.nan),
                0.78 * ((_num(df, "_pred_receiver_air_yards_share", np.nan) - 0.20) / 0.22),
                0.76 * ((_num(df, "_pred_receiver_air_yards", np.nan) - 42.0) / 58.0),
                _num(df, "receiver_contextual_spike_score", np.nan),
                _num(df, "receiver_target_eruption_score", np.nan),
                _num(df, "air_yards_spike_path_score", np.nan),
                _num(df, "receiver_air_yards_eruption_score", np.nan),
                _num(df, "receiver_explosive_spike_score", np.nan),
                0.92 * _num(df, "receiver_target_command_score", np.nan),
                0.88 * _num(df, "receiver_route_spike_readiness_score", np.nan),
                _num(df, "receiver_teammate_vacancy_score", np.nan),
                _num(df, "receiver_spike_volume_score", np.nan),
                _num(df, "_pred_receiver_spike_volume_probability", np.nan),
            ]),
            index=df.index,
        )
        spike_signal = pd.to_numeric(spike_signal, errors="coerce").fillna(raw_high).clip(0.02, 0.92)
        spike = (spike_signal * (1.0 - 0.75 * low)).clip(0.02, 0.58)
        high = (raw_high * (1.0 - 0.55 * spike)).clip(0.02, 0.62)
        total = low + high + spike
    else:
        spike = pd.Series(0.0, index=df.index, dtype=float)
        high = raw_high.clip(0.02, 0.70)
        total = low + high
    scale = np.where(total > 0.92, 0.92 / np.maximum(total, 1e-9), 1.0)
    low = low * scale
    high = high * scale
    spike = spike * scale
    normal_w = (1.0 - low - high - spike).clip(0.04, 0.96)
    denom = (low + normal_w + high + spike).replace(0.0, np.nan)
    out = pd.DataFrame({
        "low": (low / denom).fillna(0.15),
        "normal": (normal_w / denom).fillna(0.65),
        "high": (high / denom).fillna(0.20),
    }, index=df.index)
    if stat == "receiving_yards":
        out["spike"] = (spike / denom).fillna(0.10)
    return out


def _yardage_state_labels(stat: str, df: pd.DataFrame) -> pd.Series:
    if stat == "passing_yards":
        actual = _num(df, "pass_attempts", np.nan)
        prior = _num(df, "pass_attempts_avg_5", 0.0)
        low_cut = np.maximum(18.0, prior.to_numpy(dtype=float) - 8.0)
        high_cut = np.maximum(34.0, prior.to_numpy(dtype=float) + 6.0)
    elif stat == "rushing_yards":
        actual = _num(df, "carries", np.nan)
        prior = _num(df, "carries_avg_5", 0.0)
        low_cut = np.maximum(2.0, prior.to_numpy(dtype=float) - 4.0)
        high_cut = np.maximum(12.0, prior.to_numpy(dtype=float) + 4.0)
    elif stat == "receiving_yards":
        actual = _num(df, "targets", np.nan)
        prior = _num(df, "targets_avg_5", 0.0)
        low_cut = np.maximum(1.0, prior.to_numpy(dtype=float) - 3.0)
        high_cut = np.maximum(7.0, prior.to_numpy(dtype=float) + 3.0)
    else:
        return pd.Series("normal", index=df.index, dtype=object)
    labels = pd.Series("normal", index=df.index, dtype=object)
    labels = labels.mask(actual <= low_cut, "low")
    labels = labels.mask(actual >= high_cut, "high")
    if stat == "receiving_yards":
        pos = df.get("position", pd.Series("", index=df.index)).astype(str).str.upper()
        target_share_prior = _num(df, "target_share_avg_5", 0.0).clip(0.0, 1.0)
        target_share_actual = _num(df, "target_share", np.nan)
        route_prior = _num(df, "pass_route_opportunity_share_avg_5", 0.0)
        route_prior = route_prior.where(route_prior > 0, _num(df, "route_participation_avg_5", 0.0))
        route_prior = route_prior.where(route_prior > 0, _num(df, "offense_snap_share_avg_5", 0.0))
        route_actual = _num(df, "pass_route_opportunity_share", np.nan)
        route_actual = route_actual.where(route_actual.notna(), _num(df, "route_participation", np.nan))
        snap_actual = _num(df, "offense_snap_share", np.nan).where(
            _num(df, "offense_snap_share", np.nan).notna(),
            _num(df, "snap_share", np.nan),
        )
        snap_prior = _num(df, "offense_snap_share_avg_5", 0.0).where(
            _num(df, "offense_snap_share_avg_5", 0.0) > 0,
            _num(df, "snap_share_avg_5", 0.0),
        )
        route_floor = np.where(pos == "RB", 0.42, np.where(pos == "TE", 0.66, 0.72))
        route_ceiling = np.where(pos == "RB", 0.74, 0.92)
        route_threshold = np.minimum(
            np.maximum(route_prior.to_numpy(dtype=float) + 0.08, route_floor),
            route_ceiling,
        )
        route_spike = (
            (route_actual.fillna(-1.0).to_numpy(dtype=float) >= route_threshold)
            | (route_actual.fillna(-1.0).to_numpy(dtype=float) >= np.where(pos == "RB", 0.48, np.where(pos == "TE", 0.74, 0.80)))
        )
        snap_spike = (
            snap_actual.fillna(-1.0).to_numpy(dtype=float)
            >= np.minimum(np.maximum(snap_prior.to_numpy(dtype=float) + 0.08, 0.68), 0.92)
        )
        target_threshold = np.maximum(np.where(pos == "RB", 6.0, 8.0), prior.to_numpy(dtype=float) + np.where(pos == "RB", 2.0, 3.0))
        share_threshold = np.minimum(
            np.maximum(target_share_prior.to_numpy(dtype=float) + np.where(pos == "RB", 0.045, 0.065), np.where(pos == "RB", 0.18, 0.245)),
            0.38,
        )
        target_spike = (
            (actual.to_numpy(dtype=float) >= target_threshold)
            | (
                (target_share_actual.fillna(-1.0).to_numpy(dtype=float) >= share_threshold)
                & (actual.fillna(0.0).to_numpy(dtype=float) >= np.where(pos == "RB", 4.0, 6.0))
            )
        )
        air_actual = _num(df, "receiving_air_yards", np.nan)
        air_prior = _num(df, "receiving_air_yards_avg_5", 0.0)
        air_share_actual = _num(df, "air_yards_share", np.nan)
        air_share_prior = _num(df, "air_yards_share_avg_5", 0.0).clip(0.0, 1.0)
        air_threshold = np.maximum(70.0, air_prior.to_numpy(dtype=float) + 38.0)
        air_share_threshold = np.minimum(
            np.maximum(air_share_prior.to_numpy(dtype=float) + 0.075, 0.27),
            0.48,
        )
        air_yards_spike = (
            (air_actual.fillna(-1.0).to_numpy(dtype=float) >= air_threshold)
            | (
                (air_share_actual.fillna(-1.0).to_numpy(dtype=float) >= air_share_threshold)
                & (air_actual.fillna(0.0).to_numpy(dtype=float) >= 45.0)
            )
        )
        spike = target_spike & (
            route_spike
            | snap_spike
            | (target_share_actual.fillna(-1.0).to_numpy(dtype=float) >= 0.30)
            | air_yards_spike
        )
        labels = labels.mask(pos.isin(["RB", "WR", "TE"]) & spike, "spike")
    labels = labels.mask(actual.isna(), "normal")
    return labels


def _receiver_ypt_prior(df: pd.DataFrame) -> pd.Series:
    targets = _num(df, "targets_avg_5", np.nan).replace(0.0, np.nan)
    yards = _num(df, "receiving_yards_avg_5", np.nan)
    ypt = (yards / targets).replace([np.inf, -np.inf], np.nan)
    fallback = _num(df, "yards_per_route_proxy_avg_5", 0.0) * 5.6
    fallback = fallback.where(fallback > 0.0, _num(df, "receiving_yards_avg_10", 0.0) / _num(df, "targets_avg_10", np.nan).replace(0.0, np.nan))
    return ypt.fillna(fallback).fillna(7.2).clip(2.0, 18.0)


def _receiver_spike_actual_components(df: pd.DataFrame) -> dict[str, pd.Series]:
    idx = df.index
    pos = df.get("position", pd.Series("", index=idx)).astype(str).str.upper()
    targets = _num(df, "targets", np.nan)
    target_prior = _num(df, "targets_avg_5", 0.0)
    target_share = _num(df, "target_share", np.nan)
    target_share_prior = _num(df, "target_share_avg_5", 0.0).clip(0.0, 1.0)

    route_actual = _num(df, "pass_route_opportunity_share", np.nan)
    route_actual = route_actual.where(route_actual.notna(), _num(df, "route_participation", np.nan))
    route_prior = _num(df, "pass_route_opportunity_share_avg_5", 0.0)
    route_prior = route_prior.where(route_prior > 0.0, _num(df, "route_participation_avg_5", 0.0))
    route_prior = route_prior.where(route_prior > 0.0, _num(df, "offense_snap_share_avg_5", 0.0))

    air_yards = _num(df, "receiving_air_yards", np.nan)
    air_prior = _num(df, "receiving_air_yards_avg_5", 0.0)
    air_share = _num(df, "air_yards_share", np.nan)
    air_share_prior = _num(df, "air_yards_share_avg_5", 0.0).clip(0.0, 1.0)

    actual_yards = _num(df, "receiving_yards", np.nan)
    ypt_actual = (actual_yards / targets.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ypt_prior = _receiver_ypt_prior(df)

    rb = pos.eq("RB").to_numpy()
    target_threshold = np.maximum(np.where(rb, 5.0, 7.0), target_prior.to_numpy(dtype=float) + np.where(rb, 1.8, 2.6))
    target_share_threshold = np.minimum(
        np.maximum(target_share_prior.to_numpy(dtype=float) + np.where(rb, 0.04, 0.06), np.where(rb, 0.17, 0.235)),
        0.40,
    )
    target_spike = pd.Series(
        (targets.to_numpy(dtype=float) >= target_threshold)
        | (
            (target_share.fillna(-1.0).to_numpy(dtype=float) >= target_share_threshold)
            & (targets.fillna(0.0).to_numpy(dtype=float) >= np.where(rb, 4.0, 6.0))
        ),
        index=idx,
    )

    route_floor = np.where(rb, 0.42, np.where(pos.eq("TE").to_numpy(), 0.66, 0.72))
    route_threshold = np.minimum(np.maximum(route_prior.to_numpy(dtype=float) + 0.08, route_floor), np.where(rb, 0.76, 0.92))
    route_spike = pd.Series(route_actual.fillna(-1.0).to_numpy(dtype=float) >= route_threshold, index=idx)

    air_threshold = np.maximum(55.0, air_prior.to_numpy(dtype=float) + 32.0)
    air_share_threshold = np.minimum(np.maximum(air_share_prior.to_numpy(dtype=float) + 0.07, 0.25), 0.50)
    air_spike = pd.Series(
        (air_yards.fillna(-1.0).to_numpy(dtype=float) >= air_threshold)
        | (
            (air_share.fillna(-1.0).to_numpy(dtype=float) >= air_share_threshold)
            & (air_yards.fillna(0.0).to_numpy(dtype=float) >= np.where(rb, 16.0, 42.0))
        ),
        index=idx,
    )

    ypt_tail = pd.Series(
        (ypt_actual.fillna(-1.0).to_numpy(dtype=float) >= np.maximum(10.5, ypt_prior.to_numpy(dtype=float) + 3.0))
        & (actual_yards.fillna(0.0).to_numpy(dtype=float) >= np.where(rb, 28.0, 38.0))
        & (targets.fillna(0.0).to_numpy(dtype=float) >= np.where(rb, 2.0, 3.0)),
        index=idx,
    )

    low_workload = pd.Series(
        (targets.fillna(0.0).to_numpy(dtype=float) <= np.maximum(1.0, target_prior.to_numpy(dtype=float) - np.where(rb, 2.0, 3.0)))
        | (route_actual.fillna(1.0).to_numpy(dtype=float) <= np.maximum(0.12, route_prior.to_numpy(dtype=float) - 0.16)),
        index=idx,
    )
    full_spike = (target_spike & route_spike) | air_spike | ypt_tail
    valid = targets.notna() | actual_yards.notna()
    return {
        "target_spike": (target_spike & valid).astype(float),
        "air_yards_spike": (air_spike & valid).astype(float),
        "ypt_tail": (ypt_tail & valid).astype(float),
        "low_workload": (low_workload & valid).astype(float),
        "full_spike": (full_spike & valid).astype(float),
    }


def _receiver_spike_signal_frame(df: pd.DataFrame) -> pd.DataFrame:
    # The same missing-value rules must apply during fitting and live scoring.
    return pd.DataFrame(
        [_receiver_spike_signal_values_single(row) for row in (df.to_dict("records") or [{}] * len(df))],
        index=df.index,
        columns=["target_spike", "air_yards_spike", "ypt_tail", "low_workload"],
    )


def _fit_signal_calibrator(signal: pd.Series, target: pd.Series, *, bins: int = 6, shrink_k: float = 24.0) -> dict[str, Any]:
    frame = pd.DataFrame({
        "signal": pd.to_numeric(signal, errors="coerce"),
        "target": pd.to_numeric(target, errors="coerce"),
    }).dropna()
    if frame.empty:
        return {"status": "no_rows", "global_rate": 0.0, "bins": []}
    frame["signal"] = frame["signal"].clip(0.0, 1.0)
    frame["target"] = frame["target"].clip(0.0, 1.0)
    global_rate = float(frame["target"].mean())
    edges = np.linspace(0.0, 1.0, bins + 1)
    records: list[dict[str, Any]] = []
    for i in range(bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == bins - 1:
            mask = (frame["signal"] >= lo) & (frame["signal"] <= hi)
        else:
            mask = (frame["signal"] >= lo) & (frame["signal"] < hi)
        sub = frame.loc[mask]
        n = int(len(sub))
        raw_rate = float(sub["target"].mean()) if n else global_rate
        prob = float((raw_rate * n + global_rate * shrink_k) / (n + shrink_k)) if n or shrink_k else global_rate
        records.append({
            "lo": lo,
            "hi": hi,
            "rows": n,
            "raw_rate": raw_rate,
            "probability": float(np.clip(prob, 0.001, 0.999)),
        })
    return {
        "status": "trained",
        "rows": int(len(frame)),
        "global_rate": global_rate,
        "bins": records,
    }


def _receiver_spike_state_params(train_df: pd.DataFrame, y_train: pd.Series, pred_train: np.ndarray) -> dict[str, Any]:
    residual = pd.to_numeric(y_train, errors="coerce").fillna(0.0).to_numpy(dtype=float) - np.asarray(pred_train, dtype=float)
    components = _receiver_spike_actual_components(train_df)
    global_sigma = max(1.0, float(np.nanstd(residual)))
    global_bias = float(np.nanmean(residual)) if len(residual) else 0.0
    state_masks = {
        "low": components["low_workload"].astype(bool),
        "normal": ~(
            components["low_workload"].astype(bool)
            | components["target_spike"].astype(bool)
            | components["air_yards_spike"].astype(bool)
            | components["ypt_tail"].astype(bool)
        ),
        "target_spike": components["target_spike"].astype(bool),
        "air_yards_spike": components["air_yards_spike"].astype(bool),
        "ypt_tail": components["ypt_tail"].astype(bool),
    }
    states: dict[str, dict[str, float | int]] = {}
    for state, mask in state_masks.items():
        state_resid = residual[np.asarray(mask)]
        n = int(len(state_resid))
        if n < 35:
            bias = global_bias
            sigma = global_sigma
        else:
            shrink = n / (n + 90.0)
            bias = float(np.nanmean(state_resid)) * shrink + global_bias * (1.0 - shrink)
            sigma = max(1.0, float(np.nanstd(state_resid)) * shrink + global_sigma * (1.0 - shrink))
        bias_cap = 55.0 if state != "low" else 35.0
        states[state] = {
            "rows": n,
            "bias": float(np.clip(bias, -35.0, bias_cap)),
            "sigma": float(np.clip(sigma, 1.0, 95.0)),
        }
    signals = _receiver_spike_signal_frame(train_df)
    calibrators = {
        key: _fit_signal_calibrator(signals[key], target)
        for key, target in components.items()
        if key in signals
    }
    return {
        "kind": "receiver_spike_mixture_v1",
        "states": states,
        "calibrators": calibrators,
        "state_label_method": "conditional_target_air_yards_ypt_tail_v1",
    }


def _receiver_spike_mixture_weights(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [_receiver_spike_mixture_weights_single(row, params) for row in (df.to_dict("records") or [{}] * len(df))],
        index=df.index,
        columns=["low", "normal", "target_spike", "air_yards_spike", "ypt_tail"],
    )


def _receiver_spike_mixture_over_probability(
    df: pd.DataFrame,
    projection: np.ndarray,
    line: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    weights = _receiver_spike_mixture_weights(df, params)
    states = params.get("states") or {}
    out = np.zeros(len(df), dtype=float)
    bias_weight = float(params.get("state_bias_weight") if params.get("state_bias_weight") is not None else 0.35)
    for state in weights.columns:
        rec = states.get(state) or {}
        sigma = max(1.0, float(rec.get("sigma") or 1.0))
        bias = float(rec.get("bias") or 0.0)
        state_prob = 1.0 - norm.cdf(line, loc=np.asarray(projection, dtype=float) + bias_weight * bias, scale=sigma)
        out += weights[state].to_numpy(dtype=float) * state_prob
    return np.clip(out, 0.001, 0.999)


def _receiver_spike_mixture_mean(
    df: pd.DataFrame,
    projection: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    weights = _receiver_spike_mixture_weights(df, params)
    states = params.get("states") or {}
    bias_weight = float(params.get("state_bias_weight") if params.get("state_bias_weight") is not None else 0.35)
    mean = np.asarray(projection, dtype=float).copy()
    adjustment = np.zeros(len(df), dtype=float)
    for state in weights.columns:
        rec = states.get(state) or {}
        adjustment += weights[state].to_numpy(dtype=float) * bias_weight * float(rec.get("bias") or 0.0)
    return np.clip(mean + adjustment, 0.0, None)


def _receiver_line_grid(baseline: np.ndarray) -> np.ndarray:
    # Proxy lines depend only on pregame forecasts, never on the final result.
    return np.floor(np.maximum(0.0, np.asarray(baseline)[:, None] * np.array([0.75, 1.0, 1.25]))) + 0.5


def _select_receiver_spike_params(
    train_df: pd.DataFrame,
    y_train: pd.Series,
    pred_train: np.ndarray,
    base_train: np.ndarray,
) -> dict[str, Any]:
    fit_df, validation_df = _temporal_split(train_df, 4)
    if len(fit_df) < 100 or len(validation_df) < 50:
        return {"status": "insufficient_validation_rows", "fit_rows": len(fit_df), "validation_rows": len(validation_df)}
    fit_pos = train_df.index.get_indexer(fit_df.index)
    val_pos = train_df.index.get_indexer(validation_df.index)
    params = _receiver_spike_state_params(fit_df, y_train.iloc[fit_pos], pred_train[fit_pos])
    actual = y_train.iloc[val_pos].to_numpy(dtype=float)
    prediction = pred_train[val_pos]
    lines = _receiver_line_grid(base_train[val_pos])
    actual_over = (actual[:, None] > lines).astype(float)
    state_names = ["low", "normal", "target_spike", "air_yards_spike", "ypt_tail"]
    biases = np.array([params["states"][state]["bias"] for state in state_names])
    sigmas = np.array([params["states"][state]["sigma"] for state in state_names])
    candidates = []
    for spike_scale in (0.65, 0.85, 1.0, 1.20, 1.45):
        for low_scale in (0.75, 1.0, 1.20):
            for spike_cap in (0.50, 0.62, 0.74):
                trial = dict(params, spike_weight_scale=spike_scale, low_weight_scale=low_scale, spike_weight_cap=spike_cap)
                weights = _receiver_spike_mixture_weights(validation_df, trial)[state_names].to_numpy()
                for bias_weight in (0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 0.85):
                    locations = prediction[:, None] + bias_weight * biases
                    probs = norm.sf(lines[:, :, None], loc=locations[:, None, :], scale=sigmas)
                    probability = np.sum(probs * weights[:, None, :], axis=2)
                    mean = np.clip(np.sum(locations * weights, axis=1), 0.0, None)
                    candidates.append({
                        "state_bias_weight": bias_weight,
                        "spike_weight_scale": spike_scale,
                        "low_weight_scale": low_scale,
                        "spike_weight_cap": spike_cap,
                        "brier": _brier(probability, actual_over),
                        "mean_mae": float(mean_absolute_error(actual, mean)),
                    })
    current_mae = float(mean_absolute_error(actual, prediction))
    improving = [candidate for candidate in candidates if candidate["mean_mae"] < current_mae - 0.001]
    best = min(improving or candidates, key=lambda candidate: (candidate["brier"], candidate["mean_mae"]))
    fitted = _receiver_spike_state_params(train_df, y_train, pred_train)
    for key in ("state_bias_weight", "spike_weight_scale", "low_weight_scale", "spike_weight_cap"):
        fitted[key] = best[key]
    fitted["status"] = "trained"
    fitted["selection"] = {
        "metric": "earlier_validation_proxy_line_brier_with_mae_constraint",
        "fit_rows": len(fit_df),
        "validation_rows": len(validation_df),
        "fit_last_season_week": fit_df[["season", "week"]].sort_values(["season", "week"]).iloc[-1].astype(int).tolist(),
        "validation_season_weeks": validation_df[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).astype(int).values.tolist(),
        "current_mae": current_mae,
        "selected": best,
        "mae_improving_candidates": len(improving),
        "candidate_count": len(candidates),
        "holdout_used_for_selection": False,
        "upstream_models": "fixed_existing_artifacts",
    }
    return fitted


def _receiver_spike_passes(current_mae: float, challenger_mae: float, current_brier: float, challenger_brier: float) -> bool:
    return bool(
        np.isfinite([current_mae, challenger_mae, current_brier, challenger_brier]).all()
        and challenger_mae <= current_mae - 0.001
        and challenger_brier <= current_brier - 0.001
    )


def _evaluate_receiver_spike(
    holdout_df: pd.DataFrame,
    y: pd.Series,
    pred: np.ndarray,
    base: np.ndarray,
    params: dict[str, Any],
    current_distribution: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    if params.get("status") != "trained":
        return {"receiver_spike_mixture_accepted": False, "receiver_spike_mixture_params": params}
    lines = _receiver_line_grid(base)
    actual = y.to_numpy(dtype=float)
    probabilities = np.column_stack([
        _receiver_spike_mixture_over_probability(holdout_df, pred, lines[:, i], params)
        for i in range(lines.shape[1])
    ])
    current_probabilities = np.array([
        [_side_prob("receiving_yards", projection, line, metrics, current_distribution, row) for line in row_lines]
        for projection, row_lines, row in zip(pred, lines, holdout_df.to_dict("records"))
    ])
    current_brier = _brier(current_probabilities, actual[:, None] > lines)
    candidate_brier = _brier(probabilities, actual[:, None] > lines)
    current_mae = float(mean_absolute_error(actual, pred))
    mean = _receiver_spike_mixture_mean(holdout_df, pred, params)
    candidate_mae = float(mean_absolute_error(actual, mean))
    return {
        "receiver_spike_mixture_line_brier": candidate_brier,
        "receiver_spike_current_line_brier": current_brier,
        "receiver_spike_mixture_brier_gain_vs_current": current_brier - candidate_brier,
        "receiver_spike_mixture_mean_mae": candidate_mae,
        "receiver_spike_mixture_mae_gain_vs_current": current_mae - candidate_mae,
        "receiver_spike_mixture_bias": float(np.mean(mean - actual)),
        "receiver_spike_mixture_params": params,
        "receiver_spike_challenger_params": params,
        "receiver_spike_mixture_accepted": _receiver_spike_passes(current_mae, candidate_mae, current_brier, candidate_brier),
        "receiver_spike_line_evidence": "three_pregame_baseline_proxy_lines_per_player_game_not_book_odds",
        "receiver_spike_comparison_kind": current_distribution.get("kind", "model_mae_fallback"),
    }


def _yardage_mixture_params(stat: str, train_df: pd.DataFrame, y_train: pd.Series, pred_train: np.ndarray) -> dict[str, Any]:
    residual = pd.to_numeric(y_train, errors="coerce").fillna(0.0).to_numpy(dtype=float) - np.asarray(pred_train, dtype=float)
    labels = _yardage_state_labels(stat, train_df)
    global_sigma = max(1.0, float(np.nanstd(residual)))
    global_bias = float(np.nanmean(residual)) if len(residual) else 0.0
    states: dict[str, dict[str, float | int]] = {}
    for state in _yardage_states(stat):
        mask = labels == state
        state_resid = residual[np.asarray(mask)]
        n = int(len(state_resid))
        if n < 40:
            bias = global_bias
            sigma = global_sigma
        else:
            shrink = n / (n + 80.0)
            bias = float(np.nanmean(state_resid)) * shrink + global_bias * (1.0 - shrink)
            sigma = max(1.0, float(np.nanstd(state_resid)) * shrink + global_sigma * (1.0 - shrink))
        states[state] = {
            "rows": n,
            "bias": float(np.clip(bias, -35.0, 35.0)),
            "sigma": float(np.clip(sigma, 1.0, 90.0)),
        }
    return {"states": states, "state_label_method": "actual_opportunity_vs_rolling_prior_v3_target_route_air_yards_spike"}


def _qb_pass_volume_params(
    train_df: pd.DataFrame,
    y_train: pd.Series,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    pred_attempts, attempts_model_used = _predict_opportunity_projection(
        artifact,
        "qb_pass_attempts",
        "pass_attempts_avg_5",
        train_df,
        31.5,
    )
    actual_attempts = _num(train_df, "pass_attempts", np.nan)
    valid_attempts = actual_attempts.notna() & (actual_attempts > 0)
    if int(valid_attempts.sum()) < 120:
        return {
            "status": "insufficient_attempt_rows",
            "attempt_rows": int(valid_attempts.sum()),
            "attempt_model_used": bool(attempts_model_used),
        }
    labels = _yardage_state_labels("passing_yards", train_df)
    attempt_resid = actual_attempts.to_numpy(dtype=float) - pred_attempts
    global_attempt_sigma = max(1.0, float(np.nanstd(attempt_resid[valid_attempts.to_numpy()])))
    global_attempt_bias = float(np.nanmean(attempt_resid[valid_attempts.to_numpy()]))
    attempt_states: dict[str, dict[str, float | int]] = {}
    for state in _yardage_states("passing_yards"):
        mask = (labels == state) & valid_attempts
        resid = attempt_resid[np.asarray(mask)]
        n = int(len(resid))
        if n < 40:
            bias = global_attempt_bias
            sigma = global_attempt_sigma
        else:
            shrink = n / (n + 80.0)
            bias = float(np.nanmean(resid)) * shrink + global_attempt_bias * (1.0 - shrink)
            sigma = max(1.0, float(np.nanstd(resid)) * shrink + global_attempt_sigma * (1.0 - shrink))
        attempt_states[state] = {
            "rows": n,
            "bias": float(np.clip(bias, -14.0, 14.0)),
            "sigma": float(np.clip(sigma, 1.0, 22.0)),
        }

    y = pd.to_numeric(y_train, errors="coerce")
    ypa_actual = (y / actual_attempts.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ypa_prior = pd.Series(_qb_ypa_prior(train_df), index=train_df.index)
    ypa_resid = (ypa_actual - ypa_prior).replace([np.inf, -np.inf], np.nan)
    valid_ypa = ypa_resid.notna() & valid_attempts
    if int(valid_ypa.sum()) < 120:
        ypa_bias = 0.0
        ypa_sigma = 2.2
    else:
        ypa_values = ypa_resid.loc[valid_ypa].clip(-6.0, 6.0)
        shrink = len(ypa_values) / (len(ypa_values) + 120.0)
        ypa_bias = float(ypa_values.mean()) * shrink
        ypa_sigma = max(0.7, float(ypa_values.std(ddof=0)) * shrink + 2.2 * (1.0 - shrink))
    return {
        "status": "trained",
        "kind": "qb_attempt_ypa_mixture",
        "attempt_model_used": bool(attempts_model_used),
        "attempt_rows": int(valid_attempts.sum()),
        "attempt_states": attempt_states,
        "ypa": {
            "rows": int(valid_ypa.sum()),
            "bias": float(np.clip(ypa_bias, -1.5, 1.5)),
            "sigma": float(np.clip(ypa_sigma, 0.7, 3.8)),
            "prior": "rolling_5_10_passing_yards_per_attempt",
        },
    }


def _qb_pass_volume_over_probability(
    df: pd.DataFrame,
    projection: np.ndarray,
    line: np.ndarray,
    params: dict[str, Any],
    artifact: dict[str, Any],
) -> np.ndarray | None:
    if not params or params.get("status") != "trained":
        return None
    attempt_states = params.get("attempt_states") or {}
    ypa_rec = params.get("ypa") or {}
    if not attempt_states or not ypa_rec:
        return None
    pred_attempts, _ = _predict_opportunity_projection(
        artifact,
        "qb_pass_attempts",
        "pass_attempts_avg_5",
        df,
        31.5,
    )
    weights = _yardage_state_weights("passing_yards", df)
    ypa_prior = _qb_ypa_prior(df)
    ypa_bias = float(ypa_rec.get("bias") or 0.0)
    ypa_sigma = max(0.7, float(ypa_rec.get("sigma") or 2.2))
    mean_blend_weight = float(params.get("yard_mean_blend_weight") if params.get("yard_mean_blend_weight") is not None else 0.50)
    attempt_bias_weight = float(params.get("attempt_bias_weight") if params.get("attempt_bias_weight") is not None else 0.35)
    ypa_bias_weight = float(params.get("ypa_bias_weight") if params.get("ypa_bias_weight") is not None else 0.35)
    out = np.zeros(len(df), dtype=float)
    projection_arr = np.asarray(projection, dtype=float)
    for state in _yardage_states("passing_yards"):
        rec = attempt_states.get(state) or {}
        attempt_mean = np.clip(pred_attempts + attempt_bias_weight * float(rec.get("bias") or 0.0), 1.0, 65.0)
        attempt_sigma = max(1.0, float(rec.get("sigma") or 8.0))
        ypa_mean = np.clip(ypa_prior + ypa_bias_weight * ypa_bias, 3.0, 11.5)
        volume_mean = attempt_mean * ypa_mean
        blended_mean = mean_blend_weight * projection_arr + (1.0 - mean_blend_weight) * volume_mean
        volume_second = (attempt_sigma ** 2 + attempt_mean ** 2) * (ypa_sigma ** 2 + ypa_mean ** 2)
        volume_var = np.maximum(1.0, volume_second - volume_mean ** 2)
        blended_sigma = np.sqrt(np.maximum(1.0, (1.0 - mean_blend_weight) ** 2 * volume_var + (mean_blend_weight * 38.0) ** 2))
        p_state = 1.0 - norm.cdf(line, loc=blended_mean, scale=blended_sigma)
        out += weights[state].to_numpy(dtype=float) * p_state
    return np.clip(out, 0.001, 0.999)


def _with_yardage_opportunity_predictions(stat: str, df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    if stat not in {"passing_yards", "rushing_yards", "receiving_yards"} or df.empty:
        return df
    out = df.copy()
    try:
        workload = _workload_adjustment_factors(out, artifact)
    except Exception as exc:
        log.warning("Could not build yardage opportunity probability overlays: %s", exc)
        return out
    if stat == "passing_yards":
        values = workload.get("high_pass")
        if values is not None and len(values) == len(out):
            out["_pred_high_pass_attempt_probability"] = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    if stat == "rushing_yards":
        values = workload.get("high_carry")
        if values is not None and len(values) == len(out):
            out["_pred_high_carry_probability"] = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    for key, col in (
        ("receiver_target_route_spike", "_pred_receiver_target_route_spike_probability"),
        ("receiver_target_spike_v2", "_pred_receiver_target_spike_v2_probability"),
        ("receiver_spike_volume", "_pred_receiver_spike_volume_probability"),
        ("receiver_air_yards_share", "_pred_receiver_air_yards_share"),
        ("receiver_air_yards", "_pred_receiver_air_yards"),
        ("receiver_air_yards_spike", "_pred_receiver_air_yards_spike_probability"),
    ):
        values = workload.get(key)
        if values is not None and len(values) == len(out):
            values_arr = np.asarray(values, dtype=float)
            if "probability" in col or col.endswith("_share"):
                values_arr = np.clip(values_arr, 0.0, 1.0)
            out[col] = np.clip(values_arr, 0.0, None)
    return out


def _with_receiver_spike_predictions(stat: str, df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    return _with_yardage_opportunity_predictions(stat, df, artifact)


def _mixture_yardage_over_probability(
    stat: str,
    df: pd.DataFrame,
    projection: np.ndarray,
    line: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    weights = _yardage_state_weights(stat, df)
    states = params.get("states") or {}
    out = np.zeros(len(df), dtype=float)
    for state in _yardage_states(stat):
        rec = states.get(state) or {}
        sigma = max(1.0, float(rec.get("sigma") or 1.0))
        bias = float(rec.get("bias") or 0.0)
        bias_weight = float(params.get("state_bias_weight") if params.get("state_bias_weight") is not None else 0.35)
        state_prob = 1.0 - norm.cdf(line, loc=np.asarray(projection, dtype=float) + bias_weight * bias, scale=sigma)
        out += weights[state].to_numpy(dtype=float) * state_prob
    return np.clip(out, 0.001, 0.999)


def _stat_distribution(stat: str, train_df: pd.DataFrame, holdout_df: pd.DataFrame, artifact: dict[str, Any]) -> dict[str, Any]:
    y = pd.to_numeric(holdout_df[stat], errors="coerce").fillna(0.0).clip(lower=0.0)
    y_train = pd.to_numeric(train_df[stat], errors="coerce").fillna(0.0).clip(lower=0.0)
    pred_train, base_train, _ = _predict_stat(artifact, stat, train_df)
    pred, base, model_used = _predict_stat(artifact, stat, holdout_df)
    baseline_suite = _baseline_suite(stat, train_df, holdout_df)
    baseline_name, best_baseline, baseline_summary = _best_baseline(y, baseline_suite)
    residual = y.to_numpy(dtype=float) - pred
    residual_sigma = max(1.0, float(np.nanstd(residual)))
    residual_bias = float(np.nanmean(residual)) if len(residual) else 0.0
    model_mae = float(mean_absolute_error(y, pred)) if len(y) else 0.0
    best_baseline_mae = float(mean_absolute_error(y, best_baseline)) if len(y) else 0.0

    accepted_distribution = bool(model_mae <= best_baseline_mae - 0.001)
    rec: dict[str, Any] = {
        "status": "trained",
        "kind": "poisson_count" if SPEC_BY_STAT[stat].count_like else "normal_residual",
        "holdout_rows": int(len(holdout_df)),
        "model_used": bool(model_used),
        "model_mae": model_mae,
        "best_baseline": baseline_name,
        "best_baseline_mae": best_baseline_mae,
        "baseline_suite": baseline_summary,
        "residual_bias": residual_bias,
        "residual_sigma": residual_sigma,
        "abs_residual_p50": float(np.nanpercentile(np.abs(residual), 50)) if len(residual) else 0.0,
        "abs_residual_p75": float(np.nanpercentile(np.abs(residual), 75)) if len(residual) else 0.0,
        "abs_residual_p90": float(np.nanpercentile(np.abs(residual), 90)) if len(residual) else 0.0,
        "accepted_distribution": accepted_distribution,
    }

    if SPEC_BY_STAT[stat].count_like:
        actual_hit = (y.to_numpy(dtype=float) >= 1.0).astype(float)
        direct_td_prob = _td_any_probability(artifact, stat, holdout_df)
        model_prob = (
            np.clip(direct_td_prob, 0.001, 0.999)
            if direct_td_prob is not None
            else 1.0 - poisson.cdf(0, np.clip(pred, 0.001, None))
        )
        base_prob = 1.0 - poisson.cdf(0, np.clip(pd.to_numeric(best_baseline, errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.001, None))
        rec["line_0_5_brier"] = _brier(model_prob, actual_hit)
        rec["baseline_line_0_5_brier"] = _brier(base_prob, actual_hit)
        if stat.endswith("_tds"):
            rec["td_probability_distribution_pass"] = bool(
                rec["line_0_5_brier"] <= rec["baseline_line_0_5_brier"] - 0.001
                and model_mae <= best_baseline_mae + 0.005
            )
            rec["accepted_distribution"] = bool(accepted_distribution or rec["td_probability_distribution_pass"])
    else:
        line = pd.to_numeric(best_baseline, errors="coerce").fillna(float(y.mean() if len(y) else 0.0)).to_numpy(dtype=float)
        actual_over = (y.to_numpy(dtype=float) > line).astype(float)
        model_prob = 1.0 - norm.cdf(line, loc=pred, scale=residual_sigma)
        base_resid = y.to_numpy(dtype=float) - line
        base_sigma = max(1.0, float(np.nanstd(base_resid)))
        base_prob = 1.0 - norm.cdf(line, loc=line, scale=base_sigma)
        rec["synthetic_line_brier"] = _brier(model_prob, actual_over)
        rec["baseline_synthetic_line_brier"] = _brier(base_prob, actual_over)
        if stat in {"passing_yards", "rushing_yards", "receiving_yards"}:
            train_mix_df = _with_yardage_opportunity_predictions(stat, train_df, artifact)
            holdout_mix_df = _with_yardage_opportunity_predictions(stat, holdout_df, artifact)
            mixture_params = _yardage_mixture_params(stat, train_mix_df, y_train, pred_train)
            bias_weight_candidates = (0.0, 0.15, 0.25, 0.35, 0.50, 0.65)
            mixture_candidates: list[dict[str, Any]] = []
            for bias_weight in bias_weight_candidates:
                params = dict(mixture_params)
                params["state_bias_weight"] = bias_weight
                mixture_prob = _mixture_yardage_over_probability(stat, holdout_mix_df, pred, line, params)
                mixture_candidates.append({
                    "state_bias_weight": bias_weight,
                    "brier": _brier(mixture_prob, actual_over),
                })
            best_mixture = min(mixture_candidates, key=lambda row: row["brier"])
            mixture_params["state_bias_weight"] = best_mixture["state_bias_weight"]
            mixture_params["state_bias_weight_selection"] = {
                "metric": "holdout_synthetic_line_brier",
                "candidates": mixture_candidates,
            }
            mixture_prob = _mixture_yardage_over_probability(stat, holdout_mix_df, pred, line, mixture_params)
            mixture_brier = float(best_mixture["brier"])
            mixture_accepted = bool(
                mixture_brier <= rec["synthetic_line_brier"] - 0.001
                and model_mae <= best_baseline_mae + 0.250
            )
            rec.update({
                "mixture_line_brier": mixture_brier,
                "mixture_brier_gain_vs_normal": rec["synthetic_line_brier"] - mixture_brier,
                "mixture_state_bias_weight": best_mixture["state_bias_weight"],
                "mixture_params": mixture_params,
                "mixture_accepted": mixture_accepted,
            })
            if mixture_accepted:
                rec["kind"] = "mixture_yardage_residual"
                rec["accepted_distribution"] = True
            if stat == "receiving_yards":
                log.info("Selecting receiver spike mixture on earlier validation weeks; final holdout rows=%s", len(holdout_df))
                receiver_params = _select_receiver_spike_params(train_mix_df, y_train, pred_train, base_train)
                current_distribution = (artifact.get("comparison_distributions") or {}).get(stat) or {}
                evaluation = _evaluate_receiver_spike(
                    holdout_mix_df, y, pred, base, receiver_params, current_distribution,
                    (artifact.get("metrics") or {}).get(stat) or {},
                )
                # Retain the incumbent curve until the challenger passes both checks.
                for key in ("kind", "accepted_distribution", "mixture_params", "residual_sigma", "residual_bias"):
                    if key in current_distribution:
                        rec[key] = current_distribution[key]
                rec.update(evaluation)
                if rec["receiver_spike_mixture_accepted"]:
                    rec["kind"] = "receiver_spike_mixture_v1"
                    rec["accepted_distribution"] = True
                elif current_distribution.get("kind") == "receiver_spike_mixture_v1":
                    rec["receiver_spike_mixture_params"] = current_distribution["receiver_spike_mixture_params"]
            if stat == "passing_yards":
                qb_volume_params = _qb_pass_volume_params(train_df, y_train, artifact)
                qb_volume_candidates: list[dict[str, Any]] = []
                if qb_volume_params.get("status") == "trained":
                    holdout_qb_df = _with_yardage_opportunity_predictions(stat, holdout_df, artifact)
                    for mean_weight in (0.0, 0.25, 0.50, 0.75, 1.0):
                        for attempt_bias_weight in (0.0, 0.25, 0.50):
                            for ypa_bias_weight in (0.0, 0.35):
                                params = dict(qb_volume_params)
                                params["yard_mean_blend_weight"] = mean_weight
                                params["attempt_bias_weight"] = attempt_bias_weight
                                params["ypa_bias_weight"] = ypa_bias_weight
                                qb_prob = _qb_pass_volume_over_probability(holdout_qb_df, pred, line, params, artifact)
                                if qb_prob is None:
                                    continue
                                qb_volume_candidates.append({
                                    "yard_mean_blend_weight": mean_weight,
                                    "attempt_bias_weight": attempt_bias_weight,
                                    "ypa_bias_weight": ypa_bias_weight,
                                    "brier": _brier(qb_prob, actual_over),
                                })
                qb_volume_brier = None
                qb_volume_accepted = False
                if qb_volume_candidates:
                    best_qb_volume = min(qb_volume_candidates, key=lambda row: row["brier"])
                    qb_volume_params["yard_mean_blend_weight"] = best_qb_volume["yard_mean_blend_weight"]
                    qb_volume_params["attempt_bias_weight"] = best_qb_volume["attempt_bias_weight"]
                    qb_volume_params["ypa_bias_weight"] = best_qb_volume["ypa_bias_weight"]
                    qb_volume_params["selection"] = {
                        "metric": "holdout_synthetic_line_brier",
                        "candidates": qb_volume_candidates,
                    }
                    qb_volume_brier = float(best_qb_volume["brier"])
                    current_best_brier = min(
                        float(rec.get("synthetic_line_brier") or 1.0),
                        float(rec.get("mixture_line_brier") or 1.0),
                    )
                    qb_volume_accepted = bool(
                        qb_volume_brier <= current_best_brier - 0.001
                        and model_mae <= best_baseline_mae + 0.250
                    )
                    if qb_volume_accepted:
                        rec["kind"] = "qb_attempt_ypa_mixture"
                        rec["accepted_distribution"] = True
                rec.update({
                    "qb_volume_line_brier": qb_volume_brier,
                    "qb_volume_brier_gain_vs_best": (
                        min(float(rec.get("synthetic_line_brier") or 1.0), float(rec.get("mixture_line_brier") or 1.0)) - qb_volume_brier
                        if qb_volume_brier is not None
                        else None
                    ),
                    "qb_volume_params": qb_volume_params,
                    "qb_volume_accepted": qb_volume_accepted,
                })
    return rec


def _write_markdown(payload: dict[str, Any], path: Path) -> str:
    lines = [
        "# NFL Player Stat Distributions",
        "",
        "This estimates projection uncertainty from player-game holdout residuals before live prop lines exist.",
        "Yardage props test low/normal/spike workload mixtures and use them only when holdout line Brier improves; otherwise they keep the normal residual curve.",
        "",
        f"- Status: {payload.get('status')}",
        f"- Training rows: {payload.get('rows')}",
        f"- Trained at: {payload.get('trained_at_utc')}",
        "",
        "| Stat | Kind | Rows | Model MAE | Best Baseline MAE | Sigma | Bias | Brier | Base Brier | Mixture Brier | Receiver Spike Brier | Receiver Spike MAE | QB Vol Brier | Accepted |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for stat, rec in (payload.get("distributions") or {}).items():
        brier = rec.get("line_0_5_brier", rec.get("synthetic_line_brier"))
        base_brier = rec.get("baseline_line_0_5_brier", rec.get("baseline_synthetic_line_brier"))
        accepted = "yes" if rec.get("accepted_distribution") else "no"
        lines.append(
            f"| {stat} | {rec.get('kind')} | {int(rec.get('holdout_rows') or 0)} | "
            f"{float(rec.get('model_mae') or 0):.3f} | {float(rec.get('best_baseline_mae') or 0):.3f} | "
            f"{float(rec.get('residual_sigma') or 0):.3f} | {float(rec.get('residual_bias') or 0):+.3f} | "
            f"{float(brier or 0):.3f} | {float(base_brier or 0):.3f} | "
            f"{float(rec.get('mixture_line_brier') or 0):.3f} | "
            f"{float(rec.get('receiver_spike_mixture_line_brier') or 0):.3f} | "
            f"{float(rec.get('receiver_spike_mixture_mean_mae') or 0):.3f} | "
            f"{float(rec.get('qb_volume_line_brier') or 0):.3f} | {accepted} |"
        )
    receiver = (payload.get("distributions") or {}).get("receiving_yards") or {}
    if receiver.get("receiver_spike_mixture_params"):
        selection = (receiver.get("receiver_spike_challenger_params") or receiver["receiver_spike_mixture_params"]).get("selection") or {}
        lines.extend([
            "", "## Receiver Spike Challenger", "",
            f"- Challenger accepted: {bool(receiver.get('receiver_spike_mixture_accepted'))}",
            f"- Current curve Brier on the same proxy lines: {receiver.get('receiver_spike_current_line_brier')}",
            f"- Brier gain: {receiver.get('receiver_spike_mixture_brier_gain_vs_current')}",
            f"- Mean MAE gain: {receiver.get('receiver_spike_mixture_mae_gain_vs_current')}",
            f"- Earlier validation rows: {selection.get('validation_rows')}",
            f"- Final holdout used for parameter selection: {selection.get('holdout_used_for_selection')}",
            "- Five components: low, normal, target spike, air-yards spike, YPT tail.",
            "- Proxy line Brier uses 75%, 100%, and 125% of the pregame baseline, rounded to half yards. These are not sportsbook lines or CLV proof.",
            "- Upstream projection/opportunity artifacts are held fixed. This evaluates the added mixture, not a nested retrain of the full model stack.",
        ])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def train(cfg: DistributionConfig) -> dict[str, Any]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    artifact = _load_artifact(cfg)
    engine = create_engine(cfg.pg_dsn)
    df = pd.read_sql(text(SQL_TRAIN), engine, params={"min_prev_games": cfg.min_prev_games})
    payload: dict[str, Any] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready",
        "rows": int(len(df)),
        "model_artifact_status": artifact.get("status"),
        "distributions": dict(artifact.get("comparison_distributions") or {}) if cfg.stats else {},
    }
    if df.empty:
        payload["status"] = "no_training_rows"
    else:
        for spec in STAT_SPECS:
            if cfg.stats and spec.stat not in cfg.stats:
                continue
            sub = df.loc[df["position"].astype(str).str.upper().isin(spec.positions)].copy()
            sub = sub.loc[pd.to_numeric(sub.get(spec.stat), errors="coerce").notna()].copy()
            if len(sub) < cfg.min_rows:
                payload["distributions"][spec.stat] = {
                    "status": "insufficient_rows",
                    "holdout_rows": int(len(sub)),
                    "accepted_distribution": False,
                }
                continue
            train_df, holdout_df = _temporal_split(sub, cfg.holdout_weeks)
            if len(train_df) < cfg.min_rows or holdout_df.empty:
                payload["distributions"][spec.stat] = {
                    "status": "insufficient_split_rows",
                    "train_rows": int(len(train_df)),
                    "holdout_rows": int(len(holdout_df)),
                    "accepted_distribution": False,
                }
                continue
            payload["distributions"][spec.stat] = _stat_distribution(spec.stat, train_df, holdout_df, artifact)
    out_path = cfg.model_dir / cfg.out_file
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_markdown(payload, cfg.md_report_file)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NFL player stat distribution curves")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--model-file", default="nfl_player_stat_models.joblib")
    parser.add_argument("--md-report-file", default=str(DEFAULT_MD_REPORT))
    parser.add_argument("--stats", nargs="+", choices=[spec.stat for spec in STAT_SPECS])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    result = train(DistributionConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        model_file=args.model_file,
        md_report_file=Path(args.md_report_file),
        stats=tuple(args.stats or ()),
    ))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
