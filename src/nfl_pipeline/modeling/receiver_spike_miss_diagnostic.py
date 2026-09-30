"""Diagnose NFL receiving-yards spike misses by component."""
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
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine, text

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.modeling.predict_player_props import (
    _apply_workload_projection_adjustment,
    _workload_adjustment_factors,
)
from nfl_pipeline.modeling.train_player_stat_models import (
    SQL_TRAIN,
    _baseline_from_metric_column,
    _baseline_suite,
    _best_baseline,
    _make_features,
    _predict_model_payload,
    _temporal_split,
)

log = logging.getLogger("nfl_pipeline.modeling.receiver_spike_miss_diagnostic")

ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
DEFAULT_JSON = _MODEL_DIR / "nfl_receiver_spike_miss_diagnostic.json"
DEFAULT_MD = ROOT / "reports" / "nfl_receiver_spike_miss_diagnostic_latest.md"


@dataclass(frozen=True)
class ReceiverSpikeMissConfig:
    pg_dsn: str = PG_DSN
    model_dir: Path = _MODEL_DIR
    model_file: str = "nfl_player_stat_models.joblib"
    opportunity_model_file: str = "nfl_player_opportunity_models.joblib"
    out_file: Path = DEFAULT_JSON
    md_report_file: Path = DEFAULT_MD
    min_prev_games: int = 3
    holdout_weeks: int = 4
    min_rows: int = 100
    big_miss_yards: float = 22.0


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _num(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(df.get(name, pd.Series(default, index=df.index)), errors="coerce").fillna(default)


def _first_present(df: pd.DataFrame, names: tuple[str, ...], default: float = 0.0) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for name in names:
        if name in df.columns:
            out = out.fillna(pd.to_numeric(df[name], errors="coerce"))
    return out.fillna(default)


def _load_artifact(cfg: ReceiverSpikeMissConfig) -> dict[str, Any]:
    path = cfg.model_dir / cfg.model_file
    if not path.exists():
        return {"status": "missing", "models": {}, "metrics": {}}
    obj = joblib.load(path)
    if not isinstance(obj, dict):
        return {"status": "bad_artifact", "models": {}, "metrics": {}}
    opp_path = cfg.model_dir / cfg.opportunity_model_file
    obj["opportunity_status"] = "missing"
    if opp_path.exists():
        try:
            opp_artifact = joblib.load(opp_path)
            if isinstance(opp_artifact, dict):
                obj["opportunity_artifact"] = opp_artifact
                obj["opportunity_status"] = opp_artifact.get("status") or "loaded"
            else:
                obj["opportunity_status"] = "bad_artifact"
        except Exception as exc:
            log.warning("Could not load NFL opportunity artifact at %s: %s", opp_path, exc)
            obj["opportunity_status"] = "load_failed"
    return obj


def _projection_layer_accepted(metrics: dict[str, Any]) -> bool:
    if "projection_accepted" in metrics:
        return bool(metrics.get("projection_accepted"))
    if "projection_pass" in metrics:
        return bool(metrics.get("projection_pass"))
    return bool(metrics.get("accepted"))


def _predict_receiving_yards(
    artifact: dict[str, Any],
    holdout_df: pd.DataFrame,
    baseline: pd.Series,
) -> tuple[np.ndarray, bool, bool]:
    stat = "receiving_yards"
    model = (artifact.get("models") or {}).get(stat)
    metrics = (artifact.get("metrics") or {}).get(stat) or {}
    accepted = _projection_layer_accepted(metrics)
    columns = (artifact.get("feature_columns") or {}).get(stat) or []
    fills = (artifact.get("fill_values") or {}).get(stat) or {}
    base = pd.to_numeric(baseline, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if model is None or not columns or not accepted:
        return np.clip(base, 0.0, None), False, False
    X_raw = _make_features(holdout_df)
    X = X_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    raw_projection = _predict_model_payload(model, X, base)
    workload = _workload_adjustment_factors(holdout_df, artifact)
    adjusted = _apply_workload_projection_adjustment(holdout_df, stat, raw_projection, base, workload)
    return adjusted, True, bool(np.any(np.abs(adjusted - raw_projection) > 1e-6))


def _predict_opportunity_count(
    artifact: dict[str, Any],
    df: pd.DataFrame,
    name: str,
    baseline_col: str,
    default: float,
) -> tuple[np.ndarray, bool]:
    base = _first_present(df, (baseline_col,), default=default).to_numpy(dtype=float)
    opp_artifact = artifact.get("opportunity_artifact") or {}
    metrics = (opp_artifact.get("metrics") or {}).get(name) or {}
    if not bool(metrics.get("accepted") or metrics.get("projection_pass")):
        return np.clip(base, 0.0, None), False
    model_obj = (opp_artifact.get("models") or {}).get(name)
    columns = (opp_artifact.get("feature_columns") or {}).get(name) or []
    fills = (opp_artifact.get("fill_values") or {}).get(name) or {}
    if model_obj is None or not columns:
        return np.clip(base, 0.0, None), False
    X_raw = _make_features(df)
    X = X_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    try:
        if isinstance(model_obj, dict):
            raw_model = model_obj.get("model")
            if raw_model is None:
                return np.clip(base, 0.0, None), False
            pred = np.asarray(raw_model.predict(X), dtype=float)
            if str(model_obj.get("kind") or "direct") == "residual":
                shrink = float(model_obj.get("shrink") or 0.0)
                pred = base + shrink * pred
            return np.clip(pred, 0.0, None), True
        return np.clip(np.asarray(model_obj.predict(X), dtype=float), 0.0, None), True
    except Exception as exc:
        log.warning("Could not predict %s for receiver diagnostic: %s", name, exc)
        return np.clip(base, 0.0, None), False


def _predict_spike_probability(
    artifact: dict[str, Any],
    df: pd.DataFrame,
    name: str,
    baseline_col: str,
    default: float,
) -> tuple[np.ndarray, bool]:
    base = _first_present(df, (baseline_col,), default=default).clip(0.0, 1.0).to_numpy(dtype=float)
    opp_artifact = artifact.get("opportunity_artifact") or {}
    metrics = (opp_artifact.get("metrics") or {}).get(name) or {}
    if not bool(metrics.get("accepted") or metrics.get("projection_pass") or metrics.get("upside_signal_pass")):
        return base, False
    model_obj = (opp_artifact.get("models") or {}).get(name)
    columns = (opp_artifact.get("feature_columns") or {}).get(name) or []
    fills = (opp_artifact.get("fill_values") or {}).get(name) or {}
    if model_obj is None or not columns:
        return base, False
    X_raw = _make_features(df)
    X = X_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    try:
        if isinstance(model_obj, dict):
            use_upside = bool(metrics.get("upside_signal_pass") and model_obj.get("upside_model") is not None)
            raw_model = model_obj.get("upside_model") if use_upside else model_obj.get("model")
            if raw_model is None:
                return base, False
            pred = np.asarray(raw_model.predict(X), dtype=float)
            kind_key = "upside_kind" if use_upside else "kind"
            if str(model_obj.get(kind_key) or "direct") == "residual":
                shrink_key = "upside_shrink" if use_upside else "shrink"
                pred = base + float(model_obj.get(shrink_key) or 0.0) * pred
            blend_key = "upside_probability_blend_weight" if use_upside else "probability_blend_weight"
            if model_obj.get(blend_key) is not None:
                weight = float(model_obj.get(blend_key) or 1.0)
                pred = weight * pred + (1.0 - weight) * base
            return np.clip(pred, 0.0, 1.0), True
        return np.clip(np.asarray(model_obj.predict(X), dtype=float), 0.0, 1.0), True
    except Exception as exc:
        log.warning("Could not predict %s for receiver diagnostic: %s", name, exc)
        return base, False


def _classify_miss(row: pd.Series) -> str:
    error = float(row.get("model_error") or 0.0)
    if error < 0.0:
        if float(row.get("target_error") or 0.0) >= 2.5:
            return "target_projection_under"
        if float(row.get("route_error") or 0.0) >= 0.14:
            return "route_snap_projection_under"
        if float(row.get("game_script_error") or 0.0) >= 7.0:
            return "game_script_pass_volume_under"
        if float(row.get("air_yards_error") or 0.0) >= 28.0:
            return "air_yards_role_under"
        if float(row.get("ypt_error_contribution") or 0.0) >= 16.0:
            return "yards_per_target_efficiency_under"
        if float(row.get("receiver_contextual_spike_score") or 0.0) >= 0.62:
            return "context_spike_not_lifted_enough"
    if error > 0.0:
        if float(row.get("target_error") or 0.0) <= -2.5:
            return "target_projection_over"
        if float(row.get("route_error") or 0.0) <= -0.14:
            return "route_snap_projection_over"
        if float(row.get("limited_workload_risk_score") or 0.0) >= 0.45:
            return "depth_injury_limited_usage"
        if float(row.get("game_script_error") or 0.0) <= -7.0:
            return "game_script_pass_volume_over"
        if float(row.get("ypt_error_contribution") or 0.0) <= -16.0:
            return "yards_per_target_efficiency_over"
    if float(row.get("target_route_spike_probability") or 0.0) >= 0.60 and error < 0.0:
        return "spike_probability_underweighted"
    if float(row.get("receiver_teammate_vacancy_score") or 0.0) >= 0.35:
        return "teammate_vacancy_uncertain"
    return "player_rate_or_noise"


def _group_metrics(df: pd.DataFrame, keys: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for key_values, sub in df.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        rows.append({
            **{key: value for key, value in zip(keys, key_values)},
            "rows": int(len(sub)),
            "model_mae": float(np.mean(np.abs(sub["model_error"]))),
            "baseline_mae": float(np.mean(np.abs(sub["baseline_error"]))),
            "gain_vs_baseline": float(np.mean(np.abs(sub["baseline_error"])) - np.mean(np.abs(sub["model_error"]))),
            "bias": float(np.mean(sub["projection"] - sub["actual"])),
            "avg_target_error": float(np.mean(sub["target_error"])),
            "avg_route_error": float(np.mean(sub["route_error"])),
            "avg_air_yards_error": float(np.mean(sub["air_yards_error"])),
            "avg_ypt_error_contribution": float(np.mean(sub["ypt_error_contribution"])),
            "avg_spike_probability": float(np.mean(sub["target_route_spike_probability"])),
        })
    rows.sort(key=lambda row: (row["rows"], row["model_mae"]), reverse=True)
    return rows


def _build_receiver_audit(df: pd.DataFrame, artifact: dict[str, Any], cfg: ReceiverSpikeMissConfig) -> dict[str, Any]:
    sub = df.loc[df["position"].astype(str).str.upper().isin(["WR", "TE", "RB"])].copy()
    sub = sub.loc[pd.to_numeric(sub.get("receiving_yards"), errors="coerce").notna()].copy()
    if len(sub) < cfg.min_rows:
        return {"status": "insufficient_rows", "rows": int(len(sub))}
    train_df, holdout_df = _temporal_split(sub, cfg.holdout_weeks)
    if len(train_df) < cfg.min_rows or holdout_df.empty:
        return {
            "status": "insufficient_split_rows",
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
        }
    y = pd.to_numeric(holdout_df["receiving_yards"], errors="coerce").fillna(0.0).clip(lower=0.0)
    baseline_name, best_baseline, baseline_summary = _best_baseline(
        y,
        _baseline_suite("receiving_yards", train_df, holdout_df),
    )
    metrics = (artifact.get("metrics") or {}).get("receiving_yards") or {}
    if metrics:
        baseline_name = str(metrics.get("baseline_column") or baseline_name)
        baseline = pd.Series(_baseline_from_metric_column(holdout_df, "receiving_yards", metrics), index=holdout_df.index, dtype=float)
    else:
        baseline = best_baseline
    projection, model_used, workload_applied = _predict_receiving_yards(artifact, holdout_df, baseline)
    target_projection, target_model_used = _predict_opportunity_count(
        artifact,
        holdout_df,
        "receiver_targets",
        "targets_avg_5",
        3.0,
    )
    spike_prob, spike_model_used = _predict_spike_probability(
        artifact,
        holdout_df,
        "receiver_target_route_spike_probability",
        "receiver_target_route_spike_score",
        0.16,
    )
    spike_volume_prob, spike_volume_model_used = _predict_spike_probability(
        artifact,
        holdout_df,
        "receiver_spike_volume_probability",
        "receiver_spike_volume_score",
        0.18,
    )

    audit_df = holdout_df.copy()
    audit_df["actual"] = y.to_numpy(dtype=float)
    audit_df["projection"] = projection
    audit_df["baseline_projection"] = pd.to_numeric(baseline, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    audit_df["model_error"] = audit_df["projection"] - audit_df["actual"]
    audit_df["baseline_error"] = audit_df["baseline_projection"] - audit_df["actual"]
    audit_df["actual_targets"] = _num(audit_df, "targets", 0.0)
    audit_df["predicted_targets"] = target_projection
    audit_df["target_prior"] = _num(audit_df, "targets_avg_5", 0.0)
    audit_df["target_error"] = audit_df["actual_targets"] - audit_df["predicted_targets"]
    audit_df["target_route_spike_probability"] = spike_prob
    audit_df["receiver_spike_volume_probability"] = spike_volume_prob

    route_actual = _first_present(
        audit_df,
        ("pass_route_opportunity_share", "route_participation", "offense_snap_share", "snap_share"),
        default=np.nan,
    )
    route_prior = _first_present(
        audit_df,
        (
            "pass_route_opportunity_share_avg_5",
            "route_participation_avg_5",
            "offense_snap_share_avg_5",
            "snap_share_avg_5",
        ),
        default=0.0,
    )
    audit_df["actual_route_share"] = route_actual
    audit_df["route_prior"] = route_prior
    audit_df["route_error"] = route_actual.fillna(route_prior) - route_prior

    actual_air = _num(audit_df, "receiving_air_yards", 0.0)
    prior_air = _num(audit_df, "receiving_air_yards_avg_5", 0.0)
    audit_df["air_yards_error"] = actual_air - prior_air
    actual_ypt = (audit_df["actual"] / audit_df["actual_targets"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    prior_ypt = (
        _num(audit_df, "receiving_yards_avg_5", 0.0)
        / _num(audit_df, "targets_avg_5", 0.0).replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    prior_ypt = prior_ypt.fillna(_num(audit_df, "yards_per_route_proxy_avg_5", 8.4)).fillna(8.4).clip(2.0, 22.0)
    audit_df["actual_yards_per_target"] = actual_ypt.fillna(0.0)
    audit_df["prior_yards_per_target"] = prior_ypt
    audit_df["ypt_error_contribution"] = (actual_ypt.fillna(prior_ypt) - prior_ypt) * audit_df["actual_targets"]
    audit_df["expected_team_pass_attempts"] = _first_present(
        audit_df,
        ("team_player_pass_attempts_avg5_sum", "team_pass_attempts_avg_5", "pass_attempts_avg_5"),
        default=0.0,
    )
    game_team_actual_pass = audit_df.groupby(["game_id", "team_abbr"], dropna=False)["pass_attempts"].transform("sum")
    audit_df["actual_team_pass_attempts"] = pd.to_numeric(game_team_actual_pass, errors="coerce").fillna(0.0)
    audit_df["game_script_error"] = audit_df["actual_team_pass_attempts"] - audit_df["expected_team_pass_attempts"]
    audit_df["receiver_contextual_spike_score"] = _num(audit_df, "receiver_contextual_spike_score", 0.0)
    audit_df["receiver_teammate_vacancy_score"] = _num(audit_df, "receiver_teammate_vacancy_score", 0.0)
    audit_df["limited_workload_risk_score"] = _num(audit_df, "limited_workload_risk_score", 0.0)
    audit_df["miss_cause"] = [_classify_miss(row) for _, row in audit_df.iterrows()]
    audit_df["abs_error"] = audit_df["model_error"].abs()
    audit_df["missed_spike"] = (audit_df["actual"] - audit_df["projection"]) >= cfg.big_miss_yards
    audit_df["model_big_miss"] = audit_df["abs_error"] >= cfg.big_miss_yards

    big_misses = audit_df.loc[audit_df["model_big_miss"]].copy()
    rec: dict[str, Any] = {
        "status": "ready",
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "big_miss_yards": float(cfg.big_miss_yards),
        "model_used": bool(model_used),
        "workload_adjustment_applied": bool(workload_applied),
        "target_model_used": bool(target_model_used),
        "spike_model_used": bool(spike_model_used),
        "spike_volume_model_used": bool(spike_volume_model_used),
        "baseline": baseline_name,
        "best_holdout_baseline": baseline_name,
        "baseline_suite": baseline_summary,
        "model_mae": float(mean_absolute_error(audit_df["actual"], audit_df["projection"])),
        "baseline_mae": float(mean_absolute_error(audit_df["actual"], audit_df["baseline_projection"])),
        "bias": float(np.mean(audit_df["projection"] - audit_df["actual"])),
        "missed_spike_rows": int(audit_df["missed_spike"].sum()),
        "big_miss_rows": int(len(big_misses)),
        "target_projection_mae": float(np.mean(np.abs(audit_df["target_error"]))),
        "target_projection_bias": float(np.mean(audit_df["predicted_targets"] - audit_df["actual_targets"])),
        "route_share_abs_error": float(np.nanmean(np.abs(audit_df["route_error"]))),
        "by_miss_cause": _group_metrics(big_misses if not big_misses.empty else audit_df, ["miss_cause"])[:30],
        "by_position": _group_metrics(audit_df, ["position"])[:12],
        "by_depth_rank": _group_metrics(audit_df.assign(
            depth_rank_bucket=pd.cut(
                _num(audit_df, "depth_pos_rank", 99.0),
                bins=[0, 1.25, 2.25, 4.25, 99],
                labels=["rank_1", "rank_2", "rank_3_4", "rank_5_plus"],
            ).astype(object).fillna("missing"),
        ), ["depth_rank_bucket"])[:12],
        "top_misses": [],
        "missed_spike_rows_detail": [],
    }
    for _, row in audit_df.sort_values("abs_error", ascending=False).head(30).iterrows():
        rec["top_misses"].append(_row_detail(row))
    for _, row in audit_df.loc[audit_df["missed_spike"]].sort_values("model_error").head(25).iterrows():
        rec["missed_spike_rows_detail"].append(_row_detail(row))
    return rec


def _row_detail(row: pd.Series) -> dict[str, Any]:
    return {
        "season": int(row["season"]) if _safe_float(row.get("season")) is not None else row.get("season"),
        "week": int(row["week"]) if _safe_float(row.get("week")) is not None else row.get("week"),
        "player": row.get("player_name"),
        "team": row.get("team_abbr"),
        "opponent": row.get("opponent_abbr"),
        "position": row.get("position"),
        "actual": _safe_float(row.get("actual")),
        "projection": _safe_float(row.get("projection")),
        "baseline": _safe_float(row.get("baseline_projection")),
        "model_error": _safe_float(row.get("model_error")),
        "miss_cause": row.get("miss_cause"),
        "actual_targets": _safe_float(row.get("actual_targets")),
        "predicted_targets": _safe_float(row.get("predicted_targets")),
        "target_prior": _safe_float(row.get("target_prior")),
        "target_error": _safe_float(row.get("target_error")),
        "actual_route_share": _safe_float(row.get("actual_route_share")),
        "route_prior": _safe_float(row.get("route_prior")),
        "route_error": _safe_float(row.get("route_error")),
        "actual_air_yards": _safe_float(row.get("receiving_air_yards")),
        "air_yards_error": _safe_float(row.get("air_yards_error")),
        "actual_ypt": _safe_float(row.get("actual_yards_per_target")),
        "prior_ypt": _safe_float(row.get("prior_yards_per_target")),
        "ypt_error_contribution": _safe_float(row.get("ypt_error_contribution")),
        "target_route_spike_probability": _safe_float(row.get("target_route_spike_probability")),
        "receiver_spike_volume_probability": _safe_float(row.get("receiver_spike_volume_probability")),
        "contextual_spike_score": _safe_float(row.get("receiver_contextual_spike_score")),
        "teammate_vacancy_score": _safe_float(row.get("receiver_teammate_vacancy_score")),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> str:
    rec = payload.get("receiving_yards") or {}
    lines = [
        "# NFL Receiver Spike Miss Diagnostic",
        "",
        "This audits receiving-yards misses on one row per player-game, before exact prop lines, so duplicated offers cannot fake evidence.",
        "",
        f"- Status: {payload.get('status')}",
        f"- Training rows: {payload.get('rows')}",
        f"- Model artifact status: {payload.get('model_artifact_status')}",
        f"- Opportunity artifact status: {payload.get('opportunity_artifact_status')}",
        f"- Built at: {payload.get('built_at_utc')}",
        "",
        "## Summary",
        "",
        "| Holdout Rows | Model MAE | Baseline MAE | Gain | Bias | Big Miss Rows | Missed Spike Rows | Target MAE | Target Bias | Spike Model |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {int(rec.get('holdout_rows') or 0)} | {_fmt_num(rec.get('model_mae'))} | "
            f"{_fmt_num(rec.get('baseline_mae'))} | {_fmt_num((rec.get('baseline_mae') or 0) - (rec.get('model_mae') or 0))} | "
            f"{_fmt_num(rec.get('bias'))} | {int(rec.get('big_miss_rows') or 0)} | "
            f"{int(rec.get('missed_spike_rows') or 0)} | {_fmt_num(rec.get('target_projection_mae'))} | "
            f"{_fmt_num(rec.get('target_projection_bias'))} | {'yes' if rec.get('spike_model_used') else 'baseline'} |"
        ),
        "",
        "## Big-Miss Causes",
        "",
        "| Cause | Rows | Model MAE | Baseline MAE | Gain | Bias | Target Err | Route Err | Air Err | YPT Err | Spike P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rec.get("by_miss_cause") or []:
        lines.append(
            f"| {row.get('miss_cause')} | {int(row.get('rows') or 0)} | "
            f"{_fmt_num(row.get('model_mae'))} | {_fmt_num(row.get('baseline_mae'))} | "
            f"{_fmt_num(row.get('gain_vs_baseline'))} | {_fmt_num(row.get('bias'))} | "
            f"{_fmt_num(row.get('avg_target_error'))} | {_fmt_num(row.get('avg_route_error'))} | "
            f"{_fmt_num(row.get('avg_air_yards_error'))} | {_fmt_num(row.get('avg_ypt_error_contribution'))} | "
            f"{_fmt_num(row.get('avg_spike_probability'))} |"
        )
    lines.extend([
        "",
        "## Missed Spike Rows",
        "",
        "| Week | Player | Team | Actual | Projection | Baseline | Error | Cause | Tgt Act/Pred | Route Act/Prior | Air Err | YPT Err | Spike P |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in (rec.get("missed_spike_rows_detail") or [])[:15]:
        lines.append(
            f"| {row.get('season')}-{row.get('week')} | {row.get('player')} | {row.get('team')} | "
            f"{_fmt_num(row.get('actual'), 1)} | {_fmt_num(row.get('projection'), 1)} | "
            f"{_fmt_num(row.get('baseline'), 1)} | {_fmt_num(row.get('model_error'), 1)} | "
            f"{row.get('miss_cause')} | {_fmt_num(row.get('actual_targets'), 1)}/{_fmt_num(row.get('predicted_targets'), 1)} | "
            f"{_fmt_num(row.get('actual_route_share'), 2)}/{_fmt_num(row.get('route_prior'), 2)} | "
            f"{_fmt_num(row.get('air_yards_error'), 1)} | {_fmt_num(row.get('ypt_error_contribution'), 1)} | "
            f"{_fmt_num(row.get('target_route_spike_probability'), 2)} |"
        )
    lines.extend([
        "",
        "## Biggest Absolute Misses",
        "",
        "| Week | Player | Team | Actual | Projection | Baseline | Error | Cause | Tgt Act/Pred | Air Err | YPT Err | Vacancy |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ])
    for row in (rec.get("top_misses") or [])[:15]:
        lines.append(
            f"| {row.get('season')}-{row.get('week')} | {row.get('player')} | {row.get('team')} | "
            f"{_fmt_num(row.get('actual'), 1)} | {_fmt_num(row.get('projection'), 1)} | "
            f"{_fmt_num(row.get('baseline'), 1)} | {_fmt_num(row.get('model_error'), 1)} | "
            f"{row.get('miss_cause')} | {_fmt_num(row.get('actual_targets'), 1)}/{_fmt_num(row.get('predicted_targets'), 1)} | "
            f"{_fmt_num(row.get('air_yards_error'), 1)} | {_fmt_num(row.get('ypt_error_contribution'), 1)} | "
            f"{_fmt_num(row.get('teammate_vacancy_score'), 2)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    text_out = "\n".join(lines)
    path.write_text(text_out, encoding="utf-8")
    return text_out


def build_report(cfg: ReceiverSpikeMissConfig) -> dict[str, Any]:
    artifact = _load_artifact(cfg)
    engine = create_engine(cfg.pg_dsn)
    df = pd.read_sql(text(SQL_TRAIN), engine, params={"min_prev_games": cfg.min_prev_games})
    payload: dict[str, Any] = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if not df.empty else "no_training_rows",
        "rows": int(len(df)),
        "model_artifact_status": artifact.get("status"),
        "opportunity_artifact_status": artifact.get("opportunity_status"),
        "receiving_yards": {},
    }
    if not df.empty:
        payload["receiving_yards"] = _build_receiver_audit(df, artifact, cfg)
    cfg.out_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_markdown(payload, cfg.md_report_file)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NFL receiver spike miss diagnostic")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--model-file", default="nfl_player_stat_models.joblib")
    parser.add_argument("--opportunity-model-file", default="nfl_player_opportunity_models.joblib")
    parser.add_argument("--out-file", default=str(DEFAULT_JSON))
    parser.add_argument("--md-report-file", default=str(DEFAULT_MD))
    parser.add_argument("--min-prev-games", type=int, default=3)
    parser.add_argument("--holdout-weeks", type=int, default=4)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--big-miss-yards", type=float, default=22.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    payload = build_report(ReceiverSpikeMissConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        model_file=args.model_file,
        opportunity_model_file=args.opportunity_model_file,
        out_file=Path(args.out_file),
        md_report_file=Path(args.md_report_file),
        min_prev_games=args.min_prev_games,
        holdout_weeks=args.holdout_weeks,
        min_rows=args.min_rows,
        big_miss_yards=args.big_miss_yards,
    ))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
