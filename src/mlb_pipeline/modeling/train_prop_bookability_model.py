"""Train a shadow MLB prop bookability model.

Target: whether a locked offer later receives a valid same-book exact-line
close snapshot. This catches lines that existed at lock but disappeared,
changed, or could not be verified near first pitch. Same-day rows are only
used after their first pitch has passed, so late games are not mislabeled as
unbookable before their close window has had a chance to occur.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

_NUMERIC = [
    "market_line",
    "abs_price",
    "is_plus_price",
    "model_prob_side",
    "market_prob_side",
    "prob_edge_vs_market",
    "ev",
    "minutes_lock_to_start",
    "lock_hour_utc",
    "confirmed_batting_order",
    "projected_pa",
    "projected_bf",
    "projected_pitch_count",
    "is_home",
    "opp_sp_k_pct_10",
    "opp_bp_era_10",
    "opp_bp_ip_last_3",
    "opp_bp_ip_last_7",
    "pinch_hit_risk",
]

_CATEGORICAL = [
    "market",
    "side",
    "line_surface",
    "line_bucket",
    "price_bucket",
    "bookmaker_key",
    "model_family",
    "edge_type",
    "opp_sp_hand",
]


@dataclass(frozen=True)
class BookabilityConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_bookability_model.json"
    report_file: str = "mlb_prop_bookability_latest.md"
    lookback_days: int = 365
    holdout_days: int = 28
    min_train_rows: int = 250
    min_holdout_rows: int = 60


SQL = """
SELECT
    e.id,
    e.replay_id,
    e.game_date_et,
    e.market,
    e.side,
    COALESCE(e.line_surface, 'unknown') AS line_surface,
    COALESCE(e.line_bucket, 'unknown') AS line_bucket,
    COALESCE(e.price_bucket, 'missing_price') AS price_bucket,
    COALESCE(e.bookmaker_key, 'unknown') AS bookmaker_key,
    COALESCE(e.model_family, 'unknown') AS model_family,
    COALESCE(e.edge_type, 'unknown') AS edge_type,
    e.market_line::float AS market_line,
    ABS(e.market_price::float) AS abs_price,
    CASE WHEN e.market_price::float > 0 THEN 1.0 WHEN e.market_price IS NOT NULL THEN 0.0 ELSE NULL END AS is_plus_price,
    e.model_prob_side::float AS model_prob_side,
    e.market_prob_side::float AS market_prob_side,
    e.prob_edge_vs_market::float AS prob_edge_vs_market,
    e.ev::float AS ev,
    e.confirmed_batting_order::float AS confirmed_batting_order,
    e.projected_pa::float AS projected_pa,
    e.projected_bf::float AS projected_bf,
    e.projected_pitch_count::float AS projected_pitch_count,
    e.is_home::float AS is_home,
    e.opp_sp_hand,
    e.opp_sp_k_pct_10::float AS opp_sp_k_pct_10,
    e.opp_bp_era_10::float AS opp_bp_era_10,
    e.opp_bp_ip_last_3::float AS opp_bp_ip_last_3,
    e.opp_bp_ip_last_7::float AS opp_bp_ip_last_7,
    e.pinch_hit_risk::float AS pinch_hit_risk,
    COALESCE(e.clv_valid, false) AS clv_valid,
    COALESCE(e.clv_unknown_reason, 'valid_close') AS clv_unknown_reason,
    EXTRACT(EPOCH FROM (l.commence_time_utc - l.snapshot_at_utc)) / 60.0 AS minutes_lock_to_start,
    EXTRACT(HOUR FROM l.snapshot_at_utc AT TIME ZONE 'UTC')::float AS lock_hour_utc
FROM features.mlb_prop_market_training_examples e
LEFT JOIN odds.mlb_player_prop_line_snapshots l
  ON l.id = e.lock_snapshot_id
WHERE e.game_date_et >= %(cutoff)s
  AND e.lock_snapshot_id IS NOT NULL
  AND e.market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND e.side IN ('over','under')
  AND (e.clv_valid IS TRUE OR e.clv_unknown_reason IS NOT NULL)
  AND (e.game_date_et < CURRENT_DATE OR l.commence_time_utc <= NOW())
ORDER BY e.game_date_et, e.replay_id
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


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _load(cfg: BookabilityConfig) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, cfg.lookback_days))
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if not _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            return pd.DataFrame()
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in _NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target"] = df["clv_valid"].fillna(False).astype(bool).astype(int)
    for col in _CATEGORICAL:
        df[col] = df[col].fillna("unknown").astype(str)
    return df.replace([np.inf, -np.inf], np.nan)


def _prepare(df: pd.DataFrame, *, means=None, scales=None, cats=None):
    means = dict(means or {})
    scales = dict(scales or {})
    cats = {k: list(v) for k, v in (cats or {}).items()}
    parts = []
    names = []
    out_means = {}
    out_scales = {}
    for name in _NUMERIC:
        s = pd.to_numeric(df.get(name), errors="coerce")
        mean = float(means.get(name, s.mean() if not s.dropna().empty else 0.0))
        filled = s.fillna(mean)
        scale = float(scales.get(name, filled.std(ddof=0) if float(filled.std(ddof=0) or 0.0) > 1e-9 else 1.0))
        if scale <= 1e-9:
            scale = 1.0
        parts.append(((filled - mean) / scale).to_numpy().reshape(-1, 1))
        names.append(name)
        out_means[name] = mean
        out_scales[name] = scale
    out_cats = {}
    for name in _CATEGORICAL:
        values = cats.get(name)
        if values is None:
            values = sorted(str(v) for v in df[name].fillna("unknown").unique())
        out_cats[name] = values
        series = df[name].fillna("unknown").astype(str)
        for value in values:
            parts.append((series == value).astype(float).to_numpy().reshape(-1, 1))
            names.append(f"{name}={value}")
    return np.hstack(parts) if parts else np.zeros((len(df), 0)), names, out_means, out_scales, out_cats


def _split(df: pd.DataFrame, cfg: BookabilityConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
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


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _rate(mask: pd.Series) -> float | None:
    return float(mask.mean()) if len(mask) else None


def _bucket_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    group_cols = ["market", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"]
    for key, group in df.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({
            "bucket": "|".join(str(v) for v in key),
            "rows": int(len(group)),
            "bookable_rate": _mean(group["target"]),
            "stale_rate": float((group["clv_unknown_reason"] == "stale_close_before_lock").mean()),
            "no_valid_rate": float((group["clv_unknown_reason"] == "no_valid_close_snapshot").mean()),
        })
    rows.sort(key=lambda rec: (rec["bookable_rate"] if rec["bookable_rate"] is not None else 1.0, -rec["rows"]))
    return rows


def _prob_bucket(value: Any) -> str:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(p):
        return "unknown"
    p = max(0.0, min(0.999999, p))
    lo = int(p * 10) * 10
    hi = lo + 10
    return f"{lo:02d}-{hi:02d}%"


def _prediction_calibration_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "bookability_pred" not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    work = df.dropna(subset=["bookability_pred", "target"]).copy()
    if work.empty:
        return []
    work["pred_bin"] = work["bookability_pred"].map(_prob_bucket)
    for bucket, group in work.groupby("pred_bin", dropna=False):
        rows.append({
            "bucket": str(bucket),
            "rows": int(len(group)),
            "actual_bookable_rate": _mean(group["target"]),
            "avg_pred_bookable": _mean(group["bookability_pred"]),
            "calibration_error": (
                _mean(group["target"]) - _mean(group["bookability_pred"])
                if _mean(group["target"]) is not None and _mean(group["bookability_pred"]) is not None
                else None
            ),
        })
    rows.sort(key=lambda rec: rec["bucket"])
    return rows


def _group_prediction_gaps(df: pd.DataFrame, *, min_rows: int = 10) -> list[dict[str, Any]]:
    if df.empty or "bookability_pred" not in df.columns:
        return []
    specs = [
        ("market_side", ["market", "side"]),
        ("surface", ["market", "side", "line_surface"]),
        ("book_surface", ["market", "side", "line_surface", "bookmaker_key"]),
        ("exact_bucket", ["market", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"]),
    ]
    rows: list[dict[str, Any]] = []
    for level, cols in specs:
        for key, group in df.groupby(cols, dropna=False):
            if len(group) < min_rows:
                continue
            key_tuple = key if isinstance(key, tuple) else (key,)
            actual = _mean(group["target"])
            pred = _mean(group["bookability_pred"])
            if actual is None or pred is None:
                continue
            error = actual - pred
            rows.append({
                "level": level,
                "bucket": "|".join(str(v) for v in key_tuple),
                "rows": int(len(group)),
                "actual_bookable_rate": actual,
                "avg_pred_bookable": pred,
                "calibration_error": error,
                "note": "model_too_pessimistic" if error > 0.05 else "model_too_optimistic" if error < -0.05 else "calibrated",
            })
    rows.sort(key=lambda rec: (abs(float(rec["calibration_error"] or 0.0)), rec["rows"]), reverse=True)
    return rows


def _reason_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for reason, group in df.groupby("clv_unknown_reason", dropna=False):
        rows.append({
            "reason": str(reason or "unknown"),
            "rows": int(len(group)),
            "bookable_rate": _mean(group["target"]),
            "avg_pred_bookable": _mean(group["bookability_pred"]) if "bookability_pred" in group else None,
        })
    rows.sort(key=lambda rec: rec["rows"], reverse=True)
    return rows


def _empirical_rate_rows(df: pd.DataFrame, *, min_rows: int = 10) -> dict[str, dict[str, dict[str, Any]]]:
    if df.empty:
        return {}
    specs = {
        "exact_bucket": ["market", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"],
        "line_surface_book": ["market", "side", "line_surface", "bookmaker_key"],
        "line_surface": ["market", "side", "line_surface"],
        "market_side_book": ["market", "side", "bookmaker_key"],
        "market_side": ["market", "side"],
    }
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for level, cols in specs.items():
        level_rows: dict[str, dict[str, Any]] = {}
        for key, group in df.groupby(cols, dropna=False):
            if len(group) < min_rows:
                continue
            key_tuple = key if isinstance(key, tuple) else (key,)
            unknown = group.loc[~group["target"].astype(bool), "clv_unknown_reason"].fillna("none").astype(str)
            level_rows["|".join(str(v) for v in key_tuple)] = {
                "level": level,
                "key": "|".join(str(v) for v in key_tuple),
                "rows": int(len(group)),
                "bookable_rate": _mean(group["target"]),
                "stale_rate": _rate(unknown == "stale_close_before_lock"),
                "no_valid_rate": _rate(unknown == "no_valid_close_snapshot"),
                "close_window_miss_rate": _rate(unknown == "close_outside_two_hour_window"),
            }
        out[level] = level_rows
    return out


def _fmt_pct(value: Any, *, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value) * 100:+.1f}%" if signed else f"{float(value) * 100:.1f}%"


def _fmt_num(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _write_report(payload: dict[str, Any], cfg: BookabilityConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / cfg.report_file
    lines = [
        "# MLB Prop Bookability Model",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Rows: {payload.get('rows', 0)}",
        f"Date range: {payload.get('date_min')} to {payload.get('date_max')}",
        f"Status: {payload.get('status')}",
        "",
        "## Holdout",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    holdout = payload.get("holdout") or {}
    for key in ("rows", "actual_bookable_rate", "avg_pred_bookable", "brier_baseline", "brier_model", "log_loss_model", "auc_model", "model_usable"):
        value = holdout.get(key)
        if isinstance(value, bool):
            rendered = "yes" if value else "no"
        else:
            rendered = _fmt_pct(value) if "rate" in key else _fmt_num(value)
        lines.append(f"| {key} | {rendered} |")
    lines.append(f"| selected_scoring_method | {payload.get('selected_scoring_method', '-')} |")
    lines.extend([
        "",
        "## Prediction Calibration",
        "",
        "| Predicted Bucket | Rows | Actual Bookable | Avg Predicted | Error |",
        "|---|---:|---:|---:|---:|",
    ])
    for rec in payload.get("calibration_bins", []):
        lines.append(
            f"| {rec['bucket']} | {rec['rows']} | {_fmt_pct(rec['actual_bookable_rate'])} | "
            f"{_fmt_pct(rec['avg_pred_bookable'])} | {_fmt_pct(rec['calibration_error'], signed=True)} |"
        )
    lines.extend([
        "",
        "## Prediction Gap Audit",
        "",
        "| Level | Bucket | Rows | Actual | Predicted | Error | Note |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for rec in payload.get("prediction_gap_audit", [])[:40]:
        lines.append(
            f"| {rec['level']} | {rec['bucket']} | {rec['rows']} | "
            f"{_fmt_pct(rec['actual_bookable_rate'])} | {_fmt_pct(rec['avg_pred_bookable'])} | "
            f"{_fmt_pct(rec['calibration_error'], signed=True)} | {rec['note']} |"
        )
    lines.extend([
        "",
        "## Close Reason Audit",
        "",
        "`clv_unknown_reason` is label-only and is not used as a training feature.",
        "",
        "| Reason | Rows | Actual Bookable | Avg Predicted |",
        "|---|---:|---:|---:|",
    ])
    for rec in payload.get("close_reason_audit", []):
        lines.append(
            f"| {rec['reason']} | {rec['rows']} | {_fmt_pct(rec['bookable_rate'])} | "
            f"{_fmt_pct(rec.get('avg_pred_bookable'))} |"
        )
    lines.extend([
        "",
        "## Least Bookable Buckets",
        "",
        "| Bucket | Rows | Bookable | Stale | No Valid Close |",
        "|---|---:|---:|---:|---:|",
    ])
    for rec in payload.get("buckets", [])[:40]:
        lines.append(
            f"| {rec['bucket']} | {rec['rows']} | {_fmt_pct(rec['bookable_rate'])} | "
            f"{_fmt_pct(rec['stale_rate'])} | {_fmt_pct(rec['no_valid_rate'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def train(cfg: BookabilityConfig) -> dict[str, Any]:
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
        "buckets": _bucket_rows(df) if not df.empty else [],
        "empirical_bookability_rates": _empirical_rate_rows(df) if not df.empty else {},
    }
    if df.empty or df["target"].nunique() < 2:
        payload["status"] = "insufficient_rows"
        payload["report_path"] = _write_report(payload, cfg)
        (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    train_df, holdout_df, split = _split(df, cfg)
    payload["split_strategy"] = split
    payload["train_rows"] = int(len(train_df))
    payload["holdout_rows"] = int(len(holdout_df))
    if len(train_df) < cfg.min_train_rows or len(holdout_df) < cfg.min_holdout_rows or train_df["target"].nunique() < 2:
        payload["status"] = "insufficient_rows"
        payload["report_path"] = _write_report(payload, cfg)
        (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    X_train, names, means, scales, cats = _prepare(train_df)
    y_train = train_df["target"].astype(int).to_numpy()
    model = LogisticRegression(max_iter=3000, solver="lbfgs")
    model.fit(X_train, y_train)
    X_hold, _, _, _, _ = _prepare(holdout_df, means=means, scales=scales, cats=cats)
    y_hold = holdout_df["target"].astype(int).to_numpy()
    pred = np.clip(model.predict_proba(X_hold)[:, 1], 1e-6, 1 - 1e-6)
    holdout_scored = holdout_df.copy()
    holdout_scored["bookability_pred"] = pred
    base_rate = float(np.clip(y_train.mean(), 1e-6, 1 - 1e-6))
    base = np.repeat(base_rate, len(y_hold))
    payload["holdout"] = {
        "rows": int(len(holdout_df)),
        "actual_bookable_rate": float(y_hold.mean()),
        "avg_pred_bookable": float(pred.mean()),
        "brier_baseline": float(brier_score_loss(y_hold, base)),
        "brier_model": float(brier_score_loss(y_hold, pred)),
        "log_loss_model": float(log_loss(y_hold, pred, labels=[0, 1])),
        "auc_model": float(roc_auc_score(y_hold, pred)) if len(np.unique(y_hold)) == 2 else None,
    }
    payload["holdout"]["model_usable"] = bool(
        payload["holdout"]["brier_model"] <= payload["holdout"]["brier_baseline"]
        and (payload["holdout"]["auc_model"] is None or payload["holdout"]["auc_model"] >= 0.50)
        and abs(payload["holdout"]["avg_pred_bookable"] - payload["holdout"]["actual_bookable_rate"]) <= 0.10
    )
    payload["selected_scoring_method"] = "logistic" if payload["holdout"]["model_usable"] else "empirical_bucket_rate"
    payload["calibration_bins"] = _prediction_calibration_rows(holdout_scored)
    payload["prediction_gap_audit"] = _group_prediction_gaps(holdout_scored)
    payload["close_reason_audit"] = _reason_rows(holdout_scored)
    payload["models"]["global"] = {
        "method": "locked_offer_bookability_logistic",
        "intercept": float(model.intercept_[0]),
        "coef": {name: float(value) for name, value in zip(names, model.coef_[0]) if abs(float(value)) > 1e-12},
        "numeric_means": means,
        "numeric_scales": scales,
        "categorical_values": cats,
        "numeric_features": _NUMERIC,
        "categorical_features": _CATEGORICAL,
    }
    payload["status"] = "ready"
    payload["report_path"] = _write_report(payload, cfg)
    (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB prop bookability shadow model")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    args = parser.parse_args()
    payload = train(BookabilityConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
    ))
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows", 0),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
