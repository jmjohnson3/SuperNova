"""Audit NFL player-game projections before betting-line selection."""
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
from nfl_pipeline.markets import SPEC_BY_STAT, STAT_SPECS
from nfl_pipeline.modeling.train_player_stat_models import (
    SQL_TRAIN,
    _baseline_from_metric_column,
    _baseline_suite,
    _best_baseline,
    _make_features,
    _predict_model_payload,
    _temporal_split,
)
from nfl_pipeline.modeling.predict_player_props import (
    _apply_workload_projection_adjustment,
    _workload_adjustment_factors,
)

log = logging.getLogger("nfl_pipeline.modeling.player_projection_audit")

ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
DEFAULT_JSON = _MODEL_DIR / "nfl_player_projection_audit.json"
DEFAULT_MD = ROOT / "reports" / "nfl_player_projection_audit_latest.md"


@dataclass(frozen=True)
class ProjectionAuditConfig:
    pg_dsn: str = PG_DSN
    model_dir: Path = _MODEL_DIR
    model_file: str = "nfl_player_stat_models.joblib"
    opportunity_model_file: str = "nfl_player_opportunity_models.joblib"
    out_file: Path = DEFAULT_JSON
    md_report_file: Path = DEFAULT_MD
    min_prev_games: int = 3
    holdout_weeks: int = 4
    min_rows: int = 100


OPPORTUNITY_PRIORS: dict[str, tuple[str, float]] = {
    "passing_yards": ("pass_attempts", 6.0),
    "passing_tds": ("red_zone_pass_attempts", 2.0),
    "rushing_yards": ("carries", 4.0),
    "rushing_tds": ("red_zone_carries", 2.0),
    "receiving_yards": ("targets", 3.0),
    "receiving_tds": ("red_zone_targets", 2.0),
}


def _load_artifact(cfg: ProjectionAuditConfig) -> dict[str, Any]:
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
            log.warning("Could not load NFL opportunity model artifact at %s: %s", opp_path, exc)
            obj["opportunity_status"] = "load_failed"
    return obj


def _predict_current_layer(
    artifact: dict[str, Any],
    stat: str,
    holdout_df: pd.DataFrame,
    baseline: pd.Series,
) -> tuple[np.ndarray, bool, bool]:
    model = (artifact.get("models") or {}).get(stat)
    metrics = (artifact.get("metrics") or {}).get(stat) or {}
    if "projection_accepted" in metrics:
        accepted = bool(metrics.get("projection_accepted"))
    elif "projection_pass" in metrics:
        accepted = bool(metrics.get("projection_pass"))
    else:
        accepted = bool(metrics.get("accepted"))
    columns = (artifact.get("feature_columns") or {}).get(stat) or []
    fills = (artifact.get("fill_values") or {}).get(stat) or {}
    base = pd.to_numeric(baseline, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if model is None or not columns or not accepted:
        return np.clip(base, 0.0, None), False, False
    X_raw = _make_features(holdout_df)
    X = X_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    raw_projection = _predict_model_payload(model, X, base)
    workload = _workload_adjustment_factors(holdout_df, artifact)
    adjusted_projection = _apply_workload_projection_adjustment(holdout_df, stat, raw_projection, base, workload)
    workload_applied = bool(np.any(np.abs(adjusted_projection - raw_projection) > 1e-6))
    return adjusted_projection, True, workload_applied


def _bucket_depth(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(val) or val <= 0:
        return "missing"
    if val <= 1.25:
        return "rank_1"
    if val <= 2.25:
        return "rank_2"
    if val <= 3.25:
        return "rank_3"
    return "rank_4_plus"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _error_type(row: pd.Series, stat: str, error: float) -> str:
    week = _safe_float(row.get("week"))
    if week is not None and week >= 18 and abs(error) >= 0.5:
        return "week18_rest_or_weird_usage"
    rest_risk = _safe_float(row.get("rest_risk_score"))
    if rest_risk is not None and rest_risk >= 0.55 and abs(error) >= 0.5:
        return "starter_rest_or_limited_usage"
    limited_risk = _safe_float(row.get("limited_workload_risk_score"))
    fragility = _safe_float(row.get("high_usage_fragility_score"))
    full_workload = _safe_float(row.get("full_workload_score"))
    if abs(error) >= 0.5 and limited_risk is not None and limited_risk >= 0.55:
        return "model_flagged_limited_usage"
    if abs(error) >= 0.5 and fragility is not None and fragility >= 0.55:
        return "high_usage_fragility"
    opp_col, threshold = OPPORTUNITY_PRIORS.get(stat, ("", 999.0))
    if not opp_col:
        return "rate_or_noise"
    actual_opp = _safe_float(row.get(opp_col))
    prior_opp = _safe_float(row.get(f"{opp_col}_avg_5"))
    if actual_opp is None or prior_opp is None:
        return "opportunity_unknown"
    opp_error = actual_opp - prior_opp
    if (
        opp_error <= -threshold
        and full_workload is not None
        and full_workload < 0.45
        and abs(error) >= 0.5
    ):
        return "projected_full_workload_failed"
    if abs(opp_error) >= threshold:
        direction = "low" if opp_error < 0 else "high"
        return f"opportunity_{direction}"
    if stat.endswith("_tds"):
        rz_col = "red_zone_pass_attempts" if stat == "passing_tds" else "red_zone_carries" if stat == "rushing_tds" else "red_zone_targets"
        actual_rz = _safe_float(row.get(rz_col))
        prior_rz = _safe_float(row.get(f"{rz_col}_avg_5"))
        if actual_rz is None or prior_rz is None:
            return "td_role_unknown"
        if abs(actual_rz - prior_rz) >= 1.5:
            return "td_role_opportunity"
        return "td_rate_or_variance"
    return "player_rate_or_efficiency" if abs(error) >= 0.0 else "rate_or_noise"


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
        })
    rows.sort(key=lambda row: (row["rows"], abs(row["bias"])), reverse=True)
    return rows


def _stat_audit(stat: str, df: pd.DataFrame, artifact: dict[str, Any], cfg: ProjectionAuditConfig) -> dict[str, Any]:
    spec = SPEC_BY_STAT[stat]
    sub = df.loc[df["position"].astype(str).str.upper().isin(spec.positions)].copy()
    sub = sub.loc[pd.to_numeric(sub.get(stat), errors="coerce").notna()].copy()
    if len(sub) < cfg.min_rows:
        return {"status": "insufficient_rows", "rows": int(len(sub))}
    train_df, holdout_df = _temporal_split(sub, cfg.holdout_weeks)
    if len(train_df) < cfg.min_rows or holdout_df.empty:
        return {
            "status": "insufficient_split_rows",
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
        }
    y = pd.to_numeric(holdout_df[stat], errors="coerce").fillna(0.0).clip(lower=0.0)
    best_baseline_name, best_baseline, baseline_summary = _best_baseline(y, _baseline_suite(stat, train_df, holdout_df))
    metrics = (artifact.get("metrics") or {}).get(stat) or {}
    if metrics:
        baseline_name = str(metrics.get("baseline_column") or "rolling_5")
        baseline = pd.Series(_baseline_from_metric_column(holdout_df, stat, metrics), index=holdout_df.index, dtype=float)
    else:
        baseline_name = best_baseline_name
        baseline = best_baseline
    projection, model_used, workload_applied = _predict_current_layer(artifact, stat, holdout_df, baseline)
    audit_df = holdout_df.copy()
    audit_df["stat"] = stat
    audit_df["actual"] = y.to_numpy(dtype=float)
    audit_df["projection"] = projection
    audit_df["baseline_projection"] = pd.to_numeric(baseline, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    audit_df["model_error"] = audit_df["projection"] - audit_df["actual"]
    audit_df["baseline_error"] = audit_df["baseline_projection"] - audit_df["actual"]
    audit_df["depth_rank_bucket"] = audit_df.get("depth_pos_rank", pd.Series(index=audit_df.index)).map(_bucket_depth)
    audit_df["error_type"] = [
        _error_type(row, stat, float(row["model_error"]))
        for _, row in audit_df.iterrows()
    ]
    rec: dict[str, Any] = {
        "status": "ready",
        "model_used": bool(model_used),
        "baseline": baseline_name,
        "best_holdout_baseline": best_baseline_name,
        "baseline_suite": baseline_summary,
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "model_mae": float(mean_absolute_error(audit_df["actual"], audit_df["projection"])),
        "baseline_mae": float(mean_absolute_error(audit_df["actual"], audit_df["baseline_projection"])),
        "bias": float(np.mean(audit_df["projection"] - audit_df["actual"])),
        "projection_pass": bool(
            mean_absolute_error(audit_df["actual"], audit_df["projection"])
            <= mean_absolute_error(audit_df["actual"], audit_df["baseline_projection"]) - 0.001
        ),
        "workload_adjustment_applied": bool(workload_applied),
        "by_position_depth": _group_metrics(audit_df, ["position", "depth_rank_bucket"])[:30],
        "by_team": _group_metrics(audit_df, ["team_abbr"])[:40],
        "by_week": _group_metrics(audit_df, ["season", "week"])[:40],
        "by_error_type": _group_metrics(audit_df, ["error_type"])[:20],
        "top_misses": [],
    }
    misses = audit_df.assign(abs_error=lambda d: d["model_error"].abs()).sort_values("abs_error", ascending=False).head(25)
    for _, row in misses.iterrows():
        rec["top_misses"].append({
            "season": int(row["season"]) if _safe_float(row.get("season")) is not None else row.get("season"),
            "week": int(row["week"]) if _safe_float(row.get("week")) is not None else row.get("week"),
            "player": row.get("player_name"),
            "team": row.get("team_abbr"),
            "position": row.get("position"),
            "depth_rank": _safe_float(row.get("depth_pos_rank")),
            "actual": _safe_float(row.get("actual")),
            "projection": _safe_float(row.get("projection")),
            "baseline": _safe_float(row.get("baseline_projection")),
            "error_type": row.get("error_type"),
        })
    return rec


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _write_markdown(payload: dict[str, Any], path: Path) -> str:
    lines = [
        "# NFL Player Projection Audit",
        "",
        "This audits one player-game forecast per stat before odds selection, so offer duplication cannot inflate evidence.",
        "",
        f"- Status: {payload.get('status')}",
        f"- Training rows: {payload.get('rows')}",
        f"- Model artifact status: {payload.get('model_artifact_status')}",
        f"- Opportunity artifact status: {payload.get('opportunity_artifact_status')}",
        f"- Built at: {payload.get('built_at_utc')}",
        "",
        "## Stat Summary",
        "",
        "| Stat | Holdout Rows | Model MAE | Baseline MAE | Gain | Bias | Model Used | Workload Adj | Pass |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for stat, rec in (payload.get("stats") or {}).items():
        if not isinstance(rec, dict):
            continue
        lines.append(
            f"| {stat} | {int(rec.get('holdout_rows') or rec.get('rows') or 0)} | "
            f"{_fmt_num(rec.get('model_mae'))} | {_fmt_num(rec.get('baseline_mae'))} | "
            f"{_fmt_num((rec.get('baseline_mae') or 0) - (rec.get('model_mae') or 0))} | "
            f"{_fmt_num(rec.get('bias'))} | {'yes' if rec.get('model_used') else 'baseline'} | "
            f"{'yes' if rec.get('workload_adjustment_applied') else 'no'} | "
            f"{'yes' if rec.get('projection_pass') else 'no'} |"
        )
    lines.extend(["", "## Error Decomposition", ""])
    for stat, rec in (payload.get("stats") or {}).items():
        if not isinstance(rec, dict) or not rec.get("by_error_type"):
            continue
        lines.extend([
            f"### {stat}",
            "",
            "| Error Type | Rows | Model MAE | Baseline MAE | Gain | Bias |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in rec["by_error_type"]:
            lines.append(
                f"| {row.get('error_type')} | {int(row.get('rows') or 0)} | "
                f"{_fmt_num(row.get('model_mae'))} | {_fmt_num(row.get('baseline_mae'))} | "
                f"{_fmt_num(row.get('gain_vs_baseline'))} | {_fmt_num(row.get('bias'))} |"
            )
        lines.append("")
    lines.extend(["## Biggest Misses", ""])
    for stat, rec in (payload.get("stats") or {}).items():
        misses = rec.get("top_misses") if isinstance(rec, dict) else None
        if not misses:
            continue
        lines.extend([
            f"### {stat}",
            "",
            "| Week | Player | Team | Pos | Depth | Actual | Projection | Baseline | Error Type |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ])
        for row in misses[:12]:
            lines.append(
                f"| {row.get('season')}-{row.get('week')} | {row.get('player')} | {row.get('team')} | "
                f"{row.get('position')} | {_fmt_num(row.get('depth_rank'), 1)} | "
                f"{_fmt_num(row.get('actual'))} | {_fmt_num(row.get('projection'))} | "
                f"{_fmt_num(row.get('baseline'))} | {row.get('error_type')} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def build_report(cfg: ProjectionAuditConfig) -> dict[str, Any]:
    artifact = _load_artifact(cfg)
    engine = create_engine(cfg.pg_dsn)
    df = pd.read_sql(text(SQL_TRAIN), engine, params={"min_prev_games": cfg.min_prev_games})
    payload: dict[str, Any] = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if not df.empty else "no_training_rows",
        "rows": int(len(df)),
        "model_artifact_status": artifact.get("status"),
        "opportunity_artifact_status": artifact.get("opportunity_status"),
        "stats": {},
    }
    if not df.empty:
        for spec in STAT_SPECS:
            payload["stats"][spec.stat] = _stat_audit(spec.stat, df, artifact, cfg)
    cfg.out_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_markdown(payload, cfg.md_report_file)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NFL player projection audit")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--model-file", default="nfl_player_stat_models.joblib")
    parser.add_argument("--opportunity-model-file", default="nfl_player_opportunity_models.joblib")
    parser.add_argument("--out-file", default=str(DEFAULT_JSON))
    parser.add_argument("--md-report-file", default=str(DEFAULT_MD))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    payload = build_report(ProjectionAuditConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        model_file=args.model_file,
        opportunity_model_file=args.opportunity_model_file,
        out_file=Path(args.out_file),
        md_report_file=Path(args.md_report_file),
    ))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
