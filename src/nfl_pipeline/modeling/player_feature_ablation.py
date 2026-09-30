"""Ablate NFL player feature groups against true holdout accuracy."""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sqlalchemy import create_engine, text

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.modeling.train_player_opportunity_models import LEAKAGE_TARGET_COLS, _prepare_targets
from nfl_pipeline.modeling.train_player_stat_models import (
    SQL_TRAIN,
    _baseline_suite,
    _best_baseline,
    _make_features,
    _temporal_split,
)

log = logging.getLogger("nfl_pipeline.modeling.player_feature_ablation")

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
DEFAULT_JSON = MODEL_DIR / "nfl_player_feature_ablation.json"
DEFAULT_MD = ROOT / "reports" / "nfl_player_feature_ablation_latest.md"


@dataclass(frozen=True)
class FeatureAblationConfig:
    pg_dsn: str = PG_DSN
    out_file: Path = DEFAULT_JSON
    md_report_file: Path = DEFAULT_MD
    min_prev_games: int = 3
    min_rows: int = 200
    holdout_weeks: int = 4
    random_state: int = 42


TARGETS: tuple[dict[str, Any], ...] = (
    {"name": "qb_yards", "target": "passing_yards", "positions": ("QB",), "kind": "regression"},
    {"name": "rb_rush_yards", "target": "rushing_yards", "positions": ("RB",), "kind": "regression"},
    {"name": "receiving_yards", "target": "receiving_yards", "positions": ("RB", "WR", "TE"), "kind": "regression"},
    {"name": "rush_td", "target": "rushing_tds", "positions": ("RB",), "kind": "regression"},
    {"name": "receiving_td_any", "target": "receiving_tds", "positions": ("WR", "TE"), "kind": "binary_any"},
    {"name": "limited_usage_risk", "target": "actual_limited_usage", "positions": ("QB", "RB", "WR", "TE"), "kind": "binary"},
)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "workload_rest": (
        "full_workload", "limited_workload", "fragility", "rest_risk", "starter_confidence",
        "snap_share", "offense_snap_share", "usage_volatility", "is_week_18", "is_late_season",
        "backup_role", "starter_role_stability", "recent_snap", "spike_snap",
        "projected_starter", "weird_usage", "normal_usage", "role_continuity",
    ),
    "depth_injury": ("depth_", "roster_", "injury_", "practice_status"),
    "route_target_quality": (
        "route", "target", "reception", "air_yards", "wopr", "first_read",
        "yards_per_route", "receiving_yards_after_catch", "receiver_",
    ),
    "rb_carry_role": ("carries", "rushing_yards", "rb_", "yards_per_carry", "spike_carry", "carry_spike_path"),
    "td_red_zone": ("red_zone", "goal_line", "end_zone", "td_", "_tds"),
    "game_market_env": ("team_implied", "opponent_implied", "team_spread", "game_total", "opp_allowed", "is_home", "game_script"),
}


def _fit_regressor(X: pd.DataFrame, y: pd.Series, cfg: FeatureAblationConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=23,
        l2_regularization=0.08,
        random_state=cfg.random_state,
    ).fit(X, y)


def _fit_classifier(X: pd.DataFrame, y: pd.Series, cfg: FeatureAblationConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.12,
        random_state=cfg.random_state,
    ).fit(X, y)


def _drop_group_columns(columns: list[str], group: str) -> list[str]:
    patterns = FEATURE_GROUPS[group]
    kept = [
        col for col in columns
        if not any(pattern in col for pattern in patterns)
    ]
    return kept or columns


def _score(
    kind: str,
    target: str,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    cfg: FeatureAblationConfig,
) -> dict[str, float]:
    if kind == "binary":
        y_train = pd.to_numeric(train_df[target], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(int)
        y_holdout = pd.to_numeric(holdout_df[target], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(int)
        if y_train.nunique() < 2 or y_holdout.nunique() < 1:
            return {"mae": 0.0, "brier": 0.0, "bias": 0.0}
        model = _fit_classifier(X_train, y_train, cfg)
        prob = np.asarray(model.predict_proba(X_holdout), dtype=float)[:, 1].clip(0.001, 0.999)
        return {
            "mae": float(mean_absolute_error(y_holdout, prob)),
            "brier": float(brier_score_loss(y_holdout, prob)),
            "bias": float(np.mean(prob - y_holdout.to_numpy(dtype=float))),
        }
    if kind == "binary_any":
        y_train_count = pd.to_numeric(train_df[target], errors="coerce").fillna(0.0).clip(lower=0.0)
        y_holdout_count = pd.to_numeric(holdout_df[target], errors="coerce").fillna(0.0).clip(lower=0.0)
        y_train = (y_train_count > 0).astype(int)
        y_holdout = (y_holdout_count > 0).astype(int)
        if y_train.nunique() < 2:
            return {"mae": 0.0, "brier": 0.0, "bias": 0.0}
        model = _fit_classifier(X_train, y_train, cfg)
        prob = np.asarray(model.predict_proba(X_holdout), dtype=float)[:, 1].clip(0.001, 0.999)
        return {
            "mae": float(mean_absolute_error(y_holdout_count, prob)),
            "brier": float(brier_score_loss(y_holdout, prob)),
            "bias": float(np.mean(prob - y_holdout.to_numpy(dtype=float))),
        }
    y_train = pd.to_numeric(train_df[target], errors="coerce").fillna(0.0).clip(lower=0.0)
    y_holdout = pd.to_numeric(holdout_df[target], errors="coerce").fillna(0.0).clip(lower=0.0)
    model = _fit_regressor(X_train, y_train, cfg)
    pred = np.clip(np.asarray(model.predict(X_holdout), dtype=float), 0.0, None)
    return {
        "mae": float(mean_absolute_error(y_holdout, pred)),
        "brier": 0.0,
        "bias": float(np.mean(pred - y_holdout.to_numpy(dtype=float))),
    }


def _baseline_score(target: str, kind: str, train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> dict[str, float]:
    if kind in {"binary", "binary_any"}:
        if kind == "binary_any":
            train_y = (pd.to_numeric(train_df[target], errors="coerce").fillna(0.0) > 0).astype(float)
            hold_y = (pd.to_numeric(holdout_df[target], errors="coerce").fillna(0.0) > 0).astype(float)
        else:
            train_y = pd.to_numeric(train_df[target], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            hold_y = pd.to_numeric(holdout_df[target], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        prob = np.full(len(hold_y), float(train_y.mean() if len(train_y) else 0.5))
        return {
            "mae": float(mean_absolute_error(hold_y, prob)),
            "brier": float(brier_score_loss(hold_y, prob.clip(0.001, 0.999))),
            "bias": float(np.mean(prob - hold_y.to_numpy(dtype=float))),
        }
    y_holdout = pd.to_numeric(holdout_df[target], errors="coerce").fillna(0.0).clip(lower=0.0)
    _, baseline, _ = _best_baseline(y_holdout, _baseline_suite(target, train_df, holdout_df))
    base = pd.to_numeric(baseline, errors="coerce").fillna(float(y_holdout.mean() if len(y_holdout) else 0.0))
    return {
        "mae": float(mean_absolute_error(y_holdout, base)),
        "brier": 0.0,
        "bias": float(np.mean(base.to_numpy(dtype=float) - y_holdout.to_numpy(dtype=float))),
    }


def _target_ablation(df: pd.DataFrame, spec: dict[str, Any], cfg: FeatureAblationConfig) -> dict[str, Any]:
    target = str(spec["target"])
    kind = str(spec["kind"])
    sub = df.loc[df["position"].astype(str).str.upper().isin(tuple(spec["positions"]))].copy()
    sub = sub.loc[pd.to_numeric(sub.get(target), errors="coerce").notna()].copy()
    if len(sub) < cfg.min_rows:
        return {"status": "insufficient_rows", "rows": int(len(sub))}
    train_df, holdout_df = _temporal_split(sub, cfg.holdout_weeks)
    if len(train_df) < cfg.min_rows or holdout_df.empty:
        return {"status": "insufficient_split_rows", "train_rows": int(len(train_df)), "holdout_rows": int(len(holdout_df))}
    drop_cols = sorted(LEAKAGE_TARGET_COLS | {target})
    X_train_raw = _make_features(train_df.drop(columns=drop_cols, errors="ignore"))
    X_holdout_raw = _make_features(holdout_df.drop(columns=drop_cols, errors="ignore"))
    columns = list(X_train_raw.columns)
    fills = {
        col: float(value) if math.isfinite(float(value)) else 0.0
        for col, value in X_train_raw.median(numeric_only=True).fillna(0.0).to_dict().items()
    }
    X_train = X_train_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    X_holdout = X_holdout_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    full = _score(kind, target, train_df, holdout_df, X_train, X_holdout, cfg)
    baseline = _baseline_score(target, kind, train_df, holdout_df)
    groups: list[dict[str, Any]] = []
    for group in FEATURE_GROUPS:
        kept = _drop_group_columns(columns, group)
        if len(kept) == len(columns):
            groups.append({"group": group, "status": "no_matching_columns", "columns_removed": 0})
            continue
        score = _score(kind, target, train_df, holdout_df, X_train[kept], X_holdout[kept], cfg)
        groups.append({
            "group": group,
            "status": "trained",
            "columns_removed": len(columns) - len(kept),
            "mae": score["mae"],
            "brier": score["brier"],
            "bias": score["bias"],
            "mae_loss_when_removed": score["mae"] - full["mae"],
            "brier_loss_when_removed": score["brier"] - full["brier"],
            "helpful": bool(score["mae"] >= full["mae"] + 0.001 or score["brier"] >= full["brier"] + 0.001),
            "hurts_full_model": bool(score["mae"] <= full["mae"] - 0.001 or (score["brier"] and score["brier"] <= full["brier"] - 0.001)),
        })
    groups.sort(key=lambda row: (row.get("mae_loss_when_removed") or -999), reverse=True)
    return {
        "status": "ready",
        "target": target,
        "kind": kind,
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "full": full,
        "baseline": baseline,
        "full_gain_vs_baseline_mae": baseline["mae"] - full["mae"],
        "groups": groups,
    }


def build_report(cfg: FeatureAblationConfig) -> dict[str, Any]:
    engine = create_engine(cfg.pg_dsn)
    df = pd.read_sql(text(SQL_TRAIN), engine, params={"min_prev_games": cfg.min_prev_games})
    df = _prepare_targets(df)
    payload: dict[str, Any] = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if not df.empty else "no_training_rows",
        "rows": int(len(df)),
        "targets": {},
    }
    for spec in TARGETS:
        payload["targets"][spec["name"]] = _target_ablation(df, spec, cfg)
        log.info("ablation %s: %s", spec["name"], payload["targets"][spec["name"]].get("status"))
    cfg.out_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_markdown(payload, cfg.md_report_file)
    return payload


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _write_markdown(payload: dict[str, Any], path: Path) -> str:
    lines = [
        "# NFL Player Feature Ablation",
        "",
        "This report removes feature groups one at a time on the holdout split. Positive loss means the group helped the full model; negative loss means the model was better without that group.",
        "",
        f"- Status: {payload.get('status')}",
        f"- Rows: {payload.get('rows')}",
        f"- Built at: {payload.get('built_at_utc')}",
        "",
    ]
    for name, rec in (payload.get("targets") or {}).items():
        if not isinstance(rec, dict):
            continue
        lines.extend([
            f"## {name}",
            "",
            f"- Status: {rec.get('status')}",
            f"- Holdout rows: {rec.get('holdout_rows', 0)}",
            f"- Full MAE: {_fmt((rec.get('full') or {}).get('mae'))} | Baseline MAE: {_fmt((rec.get('baseline') or {}).get('mae'))} | Gain: {_fmt(rec.get('full_gain_vs_baseline_mae'))}",
            "",
            "| Feature Group | Removed Cols | MAE Loss Removed | Brier Loss Removed | Helpful | Hurts Full |",
            "|---|---:|---:|---:|---|---|",
        ])
        for row in rec.get("groups") or []:
            lines.append(
                f"| {row.get('group')} | {int(row.get('columns_removed') or 0)} | "
                f"{_fmt(row.get('mae_loss_when_removed'))} | {_fmt(row.get('brier_loss_when_removed'))} | "
                f"{'yes' if row.get('helpful') else 'no'} | {'yes' if row.get('hurts_full_model') else 'no'} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NFL player feature ablation report")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--out-file", default=str(DEFAULT_JSON))
    parser.add_argument("--md-report-file", default=str(DEFAULT_MD))
    parser.add_argument("--min-rows", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    payload = build_report(FeatureAblationConfig(
        pg_dsn=args.pg_dsn,
        out_file=Path(args.out_file),
        md_report_file=Path(args.md_report_file),
        min_rows=args.min_rows,
    ))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
