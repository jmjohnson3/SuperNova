"""K-under repair gates for micro and bankroll candidate scoring.

K unders can look calibrated on win rate while still failing CLV.  This report
keeps those concepts separate and writes a selector-consumable gate artifact.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

from .prop_training_groups import dedupe_locked_offer_rows
from .side_recalibration import prop_line_bucket

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

_MIN_ROWS = 75
_MIN_CLV_ROWS = 25
_MIN_CLV_BEAT = 0.52
_MIN_AVG_CLV = 0.0
_MAX_CAL_ERROR = 0.08

SQL = """
SELECT
    id,
    replay_id,
    source_created_at,
    game_date_et,
    game_slug,
    player_id,
    player_name,
    market,
    side,
    bookmaker_key,
    market_line::float AS market_line,
    market_price::float AS market_price,
    COALESCE(line_bucket, 'unknown') AS line_bucket,
    COALESCE(price_bucket, 'missing_price') AS price_bucket,
    COALESCE(model_family, 'unknown') AS model_family,
    pred_count::float AS pred_count,
    projected_bf::float AS projected_bf,
    actual_bf::float AS actual_bf,
    projected_pitch_count::float AS projected_pitch_count,
    actual_pitch_count_proxy::float AS actual_pitch_count,
    model_prob_side::float AS model_prob_side,
    CASE WHEN won IS TRUE THEN 1.0 WHEN won IS FALSE THEN 0.0 ELSE NULL END AS target,
    COALESCE(push, false) AS push,
    clv_valid,
    clv_price::float AS clv_price,
    COALESCE(true_pair_flag::float, CASE WHEN pair_quality IN ('same_book','cross_book') THEN 1.0 ELSE 0.0 END) AS true_pair_flag,
    COALESCE(synthetic_pair_flag::float, CASE WHEN pair_quality = 'synthetic' THEN 1.0 ELSE 0.0 END) AS synthetic_pair_flag,
    pair_quality,
    prop_offer_id
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market = 'pitcher_strikeouts'
  AND side = 'under'
  AND model_prob_side IS NOT NULL
  AND won IS NOT NULL
  AND COALESCE(push, false) IS FALSE
ORDER BY game_date_et, source_created_at, id
"""


def _table_exists(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (table_name,))
        return bool(cur.fetchone()[0])


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _brier(target: pd.Series, prob: pd.Series) -> float | None:
    work = pd.DataFrame({"target": target, "prob": prob}).dropna()
    if work.empty:
        return None
    return float(np.square(work["target"] - work["prob"].clip(1e-6, 1.0 - 1e-6)).mean())


def _safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _load_rows(pg_dsn: str, lookback_days: int) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, int(lookback_days)))
    with psycopg2.connect(pg_dsn) as conn:
        if not _table_exists(conn, "features.mlb_prop_market_training_examples"):
            return pd.DataFrame()
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in (
        "market_line", "market_price", "pred_count", "projected_bf", "actual_bf",
        "projected_pitch_count", "actual_pitch_count", "model_prob_side", "target",
        "true_pair_flag", "synthetic_pair_flag", "clv_price",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["bookmaker_key"] = df["bookmaker_key"].fillna("unknown").astype(str).str.lower()
    df["line_bucket"] = [
        lb if isinstance(lb, str) and lb and lb != "unknown" else prop_line_bucket("pitcher_strikeouts", line)
        for lb, line in zip(df["line_bucket"], df["market_line"])
    ]
    df = df.loc[
        df["target"].notna()
        & df["model_prob_side"].between(0.0, 1.0)
        & df["true_pair_flag"].fillna(0.0).ge(0.5)
        & df["synthetic_pair_flag"].fillna(0.0).lt(0.5)
    ].copy()
    df = dedupe_locked_offer_rows(df)
    df["bf_error"] = df["projected_bf"] - df["actual_bf"]
    df["pitch_count_error"] = df["projected_pitch_count"] - df["actual_pitch_count"]
    df["k_error"] = df["pred_count"] - df["market_line"]
    return df


def _metrics(group: pd.DataFrame) -> dict[str, Any]:
    target = group["target"].astype(float)
    prob = group["model_prob_side"].astype(float)
    clv_valid = group["clv_valid"].astype(bool)
    clv_price = pd.to_numeric(group["clv_price"], errors="coerce")
    clv_rows = group.loc[clv_valid].copy()
    cal_error = float(prob.mean() - target.mean())
    brier = _brier(target, prob)
    opportunity = {
        "projected_bf_mae": _safe_mean(group["bf_error"].abs()),
        "projected_bf_bias": _safe_mean(group["bf_error"]),
        "pitch_count_mae": _safe_mean(group["pitch_count_error"].abs()),
        "pitch_count_bias": _safe_mean(group["pitch_count_error"]),
        "avg_projection_vs_line": _safe_mean(group["k_error"]),
    }
    blockers: list[str] = []
    if len(group) < _MIN_ROWS:
        blockers.append(f"rows<{_MIN_ROWS}")
    if len(clv_rows) < _MIN_CLV_ROWS:
        blockers.append(f"clv_rows<{_MIN_CLV_ROWS}")
    clv_beat = float((clv_price.loc[clv_valid] > 0).mean()) if clv_valid.any() else None
    avg_clv = float(clv_price.loc[clv_valid].mean()) if clv_valid.any() else None
    if clv_beat is None or clv_beat < _MIN_CLV_BEAT:
        blockers.append("clv_beat_below_micro_gate")
    if avg_clv is None or avg_clv < _MIN_AVG_CLV:
        blockers.append("avg_clv_below_micro_gate")
    if abs(cal_error) > _MAX_CAL_ERROR:
        blockers.append("calibration_error_above_gate")
    if opportunity["projected_bf_mae"] is not None and opportunity["projected_bf_mae"] > 5.5:
        blockers.append("bf_projection_error_high")
    if opportunity["pitch_count_mae"] is not None and opportunity["pitch_count_mae"] > 20.0:
        blockers.append("pitch_count_projection_error_high")
    return {
        "rows": int(len(group)),
        "dates": int(group["game_date_et"].nunique()),
        "win_rate": float(target.mean()),
        "avg_model_prob": float(prob.mean()),
        "calibration_error": cal_error,
        "brier": brier,
        "clv_rows": int(len(clv_rows)),
        "clv_beat_rate": clv_beat,
        "avg_clv_price": avg_clv,
        "opportunity": opportunity,
        "micro_allowed": not blockers,
        "blockers": blockers,
    }


def build(*, pg_dsn: str = PG_DSN, lookback_days: int = 365) -> dict[str, Any]:
    df = _load_rows(pg_dsn, lookback_days)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if not df.empty else "no_rows",
        "usage": "k_under_micro_repair_gate",
        "lookback_days": int(lookback_days),
        "rows": int(len(df)),
        "thresholds": {
            "min_rows": _MIN_ROWS,
            "min_clv_rows": _MIN_CLV_ROWS,
            "min_clv_beat_rate": _MIN_CLV_BEAT,
            "min_avg_clv_price": _MIN_AVG_CLV,
            "max_abs_calibration_error": _MAX_CAL_ERROR,
        },
        "gates": {},
        "groups": [],
    }
    if df.empty:
        _write_outputs(payload)
        return payload

    groups: list[dict[str, Any]] = []
    gate_levels = {
        "line_book": ["line_bucket", "bookmaker_key"],
        "line": ["line_bucket"],
        "book": ["bookmaker_key"],
        "global": [],
    }
    for level, cols in gate_levels.items():
        iterable = [(("*",), df)] if not cols else df.groupby(cols, dropna=False)
        for values, group in iterable:
            if not isinstance(values, tuple):
                values = (values,)
            if cols == ["line_bucket", "bookmaker_key"]:
                key = f"{values[0]}|{values[1]}"
            elif cols == ["line_bucket"]:
                key = f"{values[0]}|*"
            elif cols == ["bookmaker_key"]:
                key = f"*|{values[0]}"
            else:
                key = "*|*"
            rec = _metrics(group)
            rec.update({
                "level": level,
                "key": str(key),
                "line_bucket": str(values[0]) if cols and cols[0] == "line_bucket" else "*",
                "bookmaker_key": (
                    str(values[1]) if cols == ["line_bucket", "bookmaker_key"]
                    else str(values[0]) if cols == ["bookmaker_key"]
                    else "*"
                ),
            })
            groups.append(rec)
            payload["gates"][str(key)] = {
                "micro_allowed": rec["micro_allowed"],
                "blockers": rec["blockers"],
                "rows": rec["rows"],
                "clv_rows": rec["clv_rows"],
                "clv_beat_rate": rec["clv_beat_rate"],
                "avg_clv_price": rec["avg_clv_price"],
                "calibration_error": rec["calibration_error"],
                "level": level,
            }
    groups.sort(key=lambda row: (row["level"], row["line_bucket"], row["bookmaker_key"]))
    payload["groups"] = groups
    payload["micro_allowed_count"] = sum(1 for row in groups if row["level"] == "line_book" and row["micro_allowed"])
    _write_outputs(payload)
    return payload


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


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Prop K-Under Repair Report",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        "Usage: K-under micro gate; paper/watch output remains visible.",
        "",
        f"- Rows: {payload.get('rows', 0)}",
        f"- Line/book groups allowed for micro: {payload.get('micro_allowed_count', 0)}",
        "",
        "| Level | Key | Rows | Dates | Win | Model P | Cal Err | Brier | CLV Rows | CLV Beat | Avg CLV | BF MAE | PC MAE | Micro Allowed | Blockers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload.get("groups") or []:
        opp = row.get("opportunity") or {}
        lines.append(
            f"| {row.get('level')} | {row.get('key')} | {row.get('rows')} | {row.get('dates')} | "
            f"{_pct(row.get('win_rate'))} | {_pct(row.get('avg_model_prob'))} | "
            f"{_pct(row.get('calibration_error'))} | {_fmt(row.get('brier'), 5)} | "
            f"{row.get('clv_rows')} | {_pct(row.get('clv_beat_rate'))} | {_fmt(row.get('avg_clv_price'))} | "
            f"{_fmt(opp.get('projected_bf_mae'))} | {_fmt(opp.get('pitch_count_mae'))} | "
            f"{bool(row.get('micro_allowed'))} | {', '.join(row.get('blockers') or []) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def _write_outputs(payload: dict[str, Any]) -> tuple[Path, Path]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _MODEL_DIR / "prop_k_under_repair.json"
    report_path = _REPORT_DIR / "mlb_prop_k_under_repair_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, _render(payload))
    payload["json_path"] = str(json_path)
    payload["report_path"] = str(report_path)
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build K-under repair and micro gate report")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    args = parser.parse_args()
    payload = build(pg_dsn=args.pg_dsn, lookback_days=args.lookback_days)
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows"),
        "micro_allowed_count": payload.get("micro_allowed_count", 0),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
