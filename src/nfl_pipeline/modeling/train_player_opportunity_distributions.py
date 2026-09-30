"""Estimate NFL opportunity distributions from player-game holdout residuals."""
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
from nfl_pipeline.modeling.train_player_opportunity_models import (
    OPPORTUNITY_SPECS,
    SQL_TRAIN,
    _align_features,
    _prepare_targets,
)
from nfl_pipeline.modeling.train_player_stat_models import _temporal_split

log = logging.getLogger("nfl_pipeline.modeling.train_player_opportunity_distributions")

ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
DEFAULT_MD_REPORT = ROOT / "reports" / "nfl_player_opportunity_distributions_latest.md"


@dataclass(frozen=True)
class OpportunityDistributionConfig:
    pg_dsn: str = PG_DSN
    model_dir: Path = _MODEL_DIR
    model_file: str = "nfl_player_opportunity_models.joblib"
    out_file: str = "nfl_player_opportunity_distributions.json"
    md_report_file: Path = DEFAULT_MD_REPORT
    min_prev_games: int = 3
    holdout_weeks: int = 4
    min_rows: int = 100


def _load_artifact(cfg: OpportunityDistributionConfig) -> dict[str, Any]:
    path = cfg.model_dir / cfg.model_file
    if not path.exists():
        return {"status": "missing", "models": {}, "metrics": {}}
    obj = joblib.load(path)
    return obj if isinstance(obj, dict) else {"status": "bad_artifact", "models": {}, "metrics": {}}


def _clean_series(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).clip(lower=0.0)


def _predict_opportunity(
    artifact: dict[str, Any],
    spec_name: str,
    target: str,
    baseline_col: str,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, bool]:
    train_target = _clean_series(train_df.get(target, pd.Series(dtype=float)))
    default = float(train_target.mean()) if len(train_target) else 0.0
    base = _clean_series(holdout_df.get(baseline_col, pd.Series(index=holdout_df.index)), default).to_numpy(dtype=float)
    metrics = (artifact.get("metrics") or {}).get(spec_name) or {}
    accepted = bool(metrics.get("accepted") or metrics.get("projection_pass"))
    model_obj = (artifact.get("models") or {}).get(spec_name)
    if model_obj is None or not accepted:
        return base, base, False
    columns = (artifact.get("feature_columns") or {}).get(spec_name) or []
    fills = (artifact.get("fill_values") or {}).get(spec_name) or {}
    if not columns:
        return base, base, False
    _, X_holdout, _, _ = _align_features(train_df, holdout_df, target_col=target)
    X = X_holdout.reindex(columns=columns).fillna(fills).fillna(0.0)
    if isinstance(model_obj, dict):
        raw_model = model_obj.get("model")
        kind = str(model_obj.get("kind") or "direct")
        shrink = float(model_obj.get("shrink") or 0.0)
        if raw_model is None:
            return base, base, False
        raw_pred = np.asarray(raw_model.predict(X), dtype=float)
        if kind == "residual":
            return np.clip(base + shrink * raw_pred, 0.0, None), base, True
        return np.clip(raw_pred, 0.0, None), base, True
    return np.clip(np.asarray(model_obj.predict(X), dtype=float), 0.0, None), base, True


def _brier(prob: np.ndarray, actual: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=float), 0.001, 0.999)
    y = np.asarray(actual, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(y) else 0.0


def _curve_metrics(y: pd.Series, pred: np.ndarray, baseline: np.ndarray, *, count_like: bool) -> dict[str, Any]:
    y_arr = y.to_numpy(dtype=float)
    resid = y_arr - pred
    base_resid = y_arr - baseline
    finite_y = y_arr[np.isfinite(y_arr)]
    is_binary_target = (
        not count_like
        and len(finite_y) > 0
        and set(np.unique(finite_y)).issubset({0.0, 1.0})
    )
    sigma = max(0.35 if not count_like else 1.0, float(np.nanstd(resid)))
    base_sigma = max(0.35 if not count_like else 1.0, float(np.nanstd(base_resid)))
    line = np.clip(baseline, 0.0, None)
    if is_binary_target:
        actual_over = y_arr.astype(float)
        model_prob = np.clip(pred, 0.001, 0.999)
        base_prob = np.clip(baseline, 0.001, 0.999)
        sigma = float(np.nanstd(resid)) if len(resid) else 0.0
    elif count_like:
        actual_over = (y_arr > line).astype(float)
        model_prob = 1.0 - poisson.cdf(np.floor(line), np.clip(pred, 0.001, None))
        base_prob = 1.0 - poisson.cdf(np.floor(line), np.clip(baseline, 0.001, None))
    else:
        actual_over = (y_arr > line).astype(float)
        model_prob = 1.0 - norm.cdf(line, loc=pred, scale=sigma)
        base_prob = 1.0 - norm.cdf(line, loc=baseline, scale=base_sigma)
    mae = float(mean_absolute_error(y_arr, pred)) if len(y_arr) else 0.0
    base_mae = float(mean_absolute_error(y_arr, baseline)) if len(y_arr) else 0.0
    return {
        "holdout_rows": int(len(y_arr)),
        "mae": mae,
        "baseline_mae": base_mae,
        "mae_gain_vs_baseline": base_mae - mae,
        "residual_sigma": sigma,
        "residual_bias": float(np.nanmean(resid)) if len(resid) else 0.0,
        "synthetic_line_brier": _brier(model_prob, actual_over),
        "baseline_synthetic_line_brier": _brier(base_prob, actual_over),
        "accepted_distribution": bool(mae <= base_mae - 0.001 and _brier(model_prob, actual_over) <= _brier(base_prob, actual_over) + 0.002),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> str:
    lines = [
        "# NFL Player Opportunity Distributions",
        "",
        "These curves describe opportunity uncertainty on one row per player/game before bet-offer pricing.",
        "",
        f"- Status: {payload.get('status')}",
        f"- Training rows: {payload.get('rows')}",
        f"- Trained at: {payload.get('trained_at_utc')}",
        "",
        "| Opportunity | Rows | MAE | Baseline MAE | Gain | Sigma | Bias | Brier | Base Brier | Accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, rec in (payload.get("distributions") or {}).items():
        accepted = "yes" if rec.get("accepted_distribution") else "no"
        lines.append(
            f"| {name} | {int(rec.get('holdout_rows') or 0)} | "
            f"{float(rec.get('mae') or 0):.3f} | {float(rec.get('baseline_mae') or 0):.3f} | "
            f"{float(rec.get('mae_gain_vs_baseline') or 0):+.3f} | "
            f"{float(rec.get('residual_sigma') or 0):.3f} | {float(rec.get('residual_bias') or 0):+.3f} | "
            f"{float(rec.get('synthetic_line_brier') or 0):.3f} | {float(rec.get('baseline_synthetic_line_brier') or 0):.3f} | {accepted} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def train(cfg: OpportunityDistributionConfig) -> dict[str, Any]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    artifact = _load_artifact(cfg)
    engine = create_engine(cfg.pg_dsn)
    df = pd.read_sql(text(SQL_TRAIN), engine, params={"min_prev_games": cfg.min_prev_games})
    df = _prepare_targets(df)
    payload: dict[str, Any] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready",
        "rows": int(len(df)),
        "model_artifact_status": artifact.get("status"),
        "distributions": {},
    }
    if df.empty:
        payload["status"] = "no_training_rows"
    else:
        for spec in OPPORTUNITY_SPECS:
            sub = df.loc[df["position"].astype(str).str.upper().isin(spec.positions)].copy()
            sub = sub.loc[pd.to_numeric(sub.get(spec.target), errors="coerce").notna()].copy()
            if len(sub) < cfg.min_rows:
                payload["distributions"][spec.name] = {
                    "status": "insufficient_rows",
                    "holdout_rows": int(len(sub)),
                    "accepted_distribution": False,
                }
                continue
            train_df, holdout_df = _temporal_split(sub, cfg.holdout_weeks)
            if len(train_df) < cfg.min_rows or holdout_df.empty:
                payload["distributions"][spec.name] = {
                    "status": "insufficient_split_rows",
                    "train_rows": int(len(train_df)),
                    "holdout_rows": int(len(holdout_df)),
                    "accepted_distribution": False,
                }
                continue
            y = _clean_series(holdout_df[spec.target])
            pred, baseline, model_used = _predict_opportunity(
                artifact,
                spec.name,
                spec.target,
                spec.baseline_col,
                train_df,
                holdout_df,
            )
            rec = _curve_metrics(y, pred, baseline, count_like=spec.count_like)
            rec.update({
                "status": "trained",
                "target": spec.target,
                "label": spec.label,
                "model_used": bool(model_used),
                "baseline_column": spec.baseline_col,
            })
            payload["distributions"][spec.name] = rec
            log.info(
                "%s: opportunity MAE %.3f vs baseline %.3f accepted=%s",
                spec.name,
                rec["mae"],
                rec["baseline_mae"],
                rec["accepted_distribution"],
            )
    out_path = cfg.model_dir / cfg.out_file
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_markdown(payload, cfg.md_report_file)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NFL player opportunity distribution curves")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--model-file", default="nfl_player_opportunity_models.joblib")
    parser.add_argument("--md-report-file", default=str(DEFAULT_MD_REPORT))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    result = train(OpportunityDistributionConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        model_file=args.model_file,
        md_report_file=Path(args.md_report_file),
    ))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
