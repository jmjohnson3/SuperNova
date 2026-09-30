"""Focused diagnostic for DraftKings K 4.5-6.0 unders.

This bucket has shown CLV direction but poor ROI, which usually means the
projection/calibration layer is wrong.  This report breaks the misses down by
BF, pitch-count/leash, probability confidence, price bucket, and exact line.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

from .prop_training_groups import dedupe_locked_offer_rows
from .side_recalibration import price_bucket, prop_line_bucket

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

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
    opponent_abbr,
    team_implied_runs::float AS team_implied_runs,
    opponent_implied_runs::float AS opponent_implied_runs,
    game_total_line::float AS game_total_line,
    model_prob_side::float AS model_prob_side,
    actual_value::float AS actual_k,
    CASE WHEN won IS TRUE THEN 1.0 WHEN won IS FALSE THEN 0.0 ELSE NULL END AS target,
    COALESCE(push, false) AS push,
    profit_units::float AS profit_units,
    clv_valid,
    clv_price::float AS clv_price,
    clv_status,
    COALESCE(clv_unknown_reason, '') AS clv_unknown_reason,
    pair_quality,
    COALESCE(true_pair_flag::float, CASE WHEN pair_quality IN ('same_book','cross_book') THEN 1.0 ELSE 0.0 END) AS true_pair_flag,
    COALESCE(synthetic_pair_flag::float, CASE WHEN pair_quality = 'synthetic' THEN 1.0 ELSE 0.0 END) AS synthetic_pair_flag,
    prop_offer_id
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market = 'pitcher_strikeouts'
  AND side = 'under'
  AND LOWER(bookmaker_key) = 'draftkings'
  AND market_line::float >= 4.5
  AND market_line::float < 6.5
  AND model_prob_side IS NOT NULL
  AND won IS NOT NULL
  AND COALESCE(push, false) IS FALSE
ORDER BY game_date_et, source_created_at, id
"""


def _table_exists(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('features.mlb_prop_market_training_examples') IS NOT NULL")
        return bool(cur.fetchone()[0])


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _brier(group: pd.DataFrame) -> float | None:
    work = group[["target", "model_prob_side"]].dropna()
    if work.empty:
        return None
    return float(np.square(work["target"].astype(float) - work["model_prob_side"].astype(float).clip(1e-6, 1.0 - 1e-6)).mean())


def _prob_bucket(prob: Any) -> str:
    p = _float(prob)
    if p is None:
        return "missing_prob"
    if p < 0.50:
        return "p<50"
    if p < 0.55:
        return "p50_55"
    if p < 0.60:
        return "p55_60"
    if p < 0.65:
        return "p60_65"
    return "p65_plus"


def _margin_bucket(margin: Any) -> str:
    value = _float(margin)
    if value is None:
        return "missing_margin"
    if value < 0:
        return "model_projects_over_line"
    if value < 0.5:
        return "under_margin_0_0.49"
    if value < 1.0:
        return "under_margin_0.5_0.99"
    return "under_margin_1_plus"


def _bf_error_bucket(error: Any) -> str:
    value = _float(error)
    if value is None:
        return "missing_bf_error"
    # projected BF - actual BF; negative means starter faced more batters than expected.
    if value <= -4:
        return "actual_bf_4plus_high"
    if value <= -2:
        return "actual_bf_2_4_high"
    if value < 2:
        return "bf_close"
    if value < 4:
        return "actual_bf_2_4_low"
    return "actual_bf_4plus_low"


def _load_rows(pg_dsn: str, lookback_days: int) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, int(lookback_days)))
    with psycopg2.connect(pg_dsn) as conn:
        if not _table_exists(conn):
            return pd.DataFrame()
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    numeric_cols = (
        "market_line", "market_price", "pred_count", "projected_bf", "actual_bf",
        "projected_pitch_count", "actual_pitch_count", "team_implied_runs",
        "opponent_implied_runs", "game_total_line", "model_prob_side",
        "actual_k", "target", "profit_units", "clv_price", "true_pair_flag",
        "synthetic_pair_flag",
    )
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["bookmaker_key"] = df["bookmaker_key"].fillna("unknown").astype(str).str.lower()
    df["line_bucket"] = [
        bucket if isinstance(bucket, str) and bucket and bucket != "unknown" else prop_line_bucket("pitcher_strikeouts", line)
        for bucket, line in zip(df["line_bucket"], df["market_line"])
    ]
    df["price_bucket"] = [
        bucket if isinstance(bucket, str) and bucket and bucket != "missing_price" else price_bucket(price)
        for bucket, price in zip(df["price_bucket"], df["market_price"])
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
    df["projection_under_margin"] = df["market_line"] - df["pred_count"]
    df["actual_under_margin"] = df["market_line"] - df["actual_k"]
    df["prob_bucket"] = df["model_prob_side"].map(_prob_bucket)
    df["projection_margin_bucket"] = df["projection_under_margin"].map(_margin_bucket)
    df["bf_error_bucket"] = df["bf_error"].map(_bf_error_bucket)
    df["clv_valid_bool"] = df["clv_valid"].fillna(False).astype(bool)
    return df


def _metrics(group: pd.DataFrame) -> dict[str, Any]:
    rows = int(len(group))
    target = group["target"].astype(float) if rows else pd.Series(dtype=float)
    prob = group["model_prob_side"].astype(float) if rows else pd.Series(dtype=float)
    clv_valid = group["clv_valid_bool"] if rows else pd.Series(dtype=bool)
    clv_prices = pd.to_numeric(group.loc[clv_valid, "clv_price"], errors="coerce").dropna() if rows else pd.Series(dtype=float)
    losses = group.loc[group["target"].eq(0.0)]
    wins = group.loc[group["target"].eq(1.0)]
    return {
        "rows": rows,
        "dates": int(group["game_date_et"].nunique()) if rows else 0,
        "win_rate": float(target.mean()) if rows else None,
        "avg_model_prob": float(prob.mean()) if rows else None,
        "calibration_error": float(prob.mean() - target.mean()) if rows else None,
        "brier": _brier(group) if rows else None,
        "roi": float(group["profit_units"].mean()) if rows and group["profit_units"].notna().any() else None,
        "clv_rows": int(len(clv_prices)),
        "clv_beat_rate": float((clv_prices > 0).mean()) if len(clv_prices) else None,
        "avg_clv_price": float(clv_prices.mean()) if len(clv_prices) else None,
        "projected_bf_mae": _safe_mean(group["bf_error"].abs()),
        "projected_bf_bias": _safe_mean(group["bf_error"]),
        "pitch_count_mae": _safe_mean(group["pitch_count_error"].abs()),
        "pitch_count_bias": _safe_mean(group["pitch_count_error"]),
        "avg_projection_under_margin": _safe_mean(group["projection_under_margin"]),
        "loss_projected_bf_bias": _safe_mean(losses["bf_error"]) if len(losses) else None,
        "win_projected_bf_bias": _safe_mean(wins["bf_error"]) if len(wins) else None,
        "loss_pitch_count_bias": _safe_mean(losses["pitch_count_error"]) if len(losses) else None,
        "win_pitch_count_bias": _safe_mean(wins["pitch_count_error"]) if len(wins) else None,
        "loss_projection_under_margin": _safe_mean(losses["projection_under_margin"]) if len(losses) else None,
        "win_projection_under_margin": _safe_mean(wins["projection_under_margin"]) if len(wins) else None,
    }


def _group_rows(df: pd.DataFrame, columns: list[str], *, top_n: int = 60) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    iterable = [(("*",), df)] if not columns else df.groupby(columns, dropna=False)
    for key, group in iterable:
        values = key if isinstance(key, tuple) else (key,)
        rec = _metrics(group)
        rec.update({column: str(value or "unknown") for column, value in zip(columns, values)})
        rows.append(rec)
    rows.sort(key=lambda row: (
        -int(row.get("rows") or 0),
        float(row.get("roi") or -999.0),
    ))
    return rows[:top_n]


def _diagnosis(metrics: dict[str, Any], df: pd.DataFrame) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    roi = _float(metrics.get("roi"))
    clv_beat = _float(metrics.get("clv_beat_rate"))
    cal = _float(metrics.get("calibration_error"))
    loss_bf = _float(metrics.get("loss_projected_bf_bias"))
    win_bf = _float(metrics.get("win_projected_bf_bias"))
    loss_pc = _float(metrics.get("loss_pitch_count_bias"))
    win_pc = _float(metrics.get("win_pitch_count_bias"))
    if roi is not None and roi < 0 and clv_beat is not None and clv_beat >= 0.52:
        notes.append({
            "area": "projection_or_probability",
            "evidence": "CLV direction is acceptable while ROI is negative",
            "next_action": "repair K-rate/leash and under calibration before increasing micro exposure",
        })
    if cal is not None and cal > 0.05:
        notes.append({
            "area": "overconfident_under_probability",
            "evidence": f"model probability exceeds actual under hit rate by {cal:.3f}",
            "next_action": "shrink K-under probabilities toward bucket priors, especially around K 4.5-6.0",
        })
    elif cal is not None and cal < -0.05:
        notes.append({
            "area": "underconfident_under_probability",
            "evidence": f"model probability trails actual under hit rate by {abs(cal):.3f}",
            "next_action": "check whether price/line filters are rejecting otherwise good under projections",
        })
    if loss_bf is not None and win_bf is not None and loss_bf < win_bf - 1.0:
        notes.append({
            "area": "bf_leash_error",
            "evidence": "losing unders have much more actual BF than projected",
            "next_action": "add heavier leash/recent workload/game-script features and penalize unders when BF uncertainty is high",
        })
    if loss_pc is not None and win_pc is not None and loss_pc < win_pc - 8.0:
        notes.append({
            "area": "pitch_count_error",
            "evidence": "losing unders have materially higher pitch counts than projected",
            "next_action": "rebuild pitch-count distribution before trusting K 4.5-6.0 unders",
        })
    if not notes and not df.empty:
        common = Counter(df.loc[df["target"].eq(0.0), "prob_bucket"].astype(str)).most_common(1)
        notes.append({
            "area": "bucket_specific_review",
            "evidence": f"largest losing probability bucket: {common[0][0] if common else 'unknown'}",
            "next_action": "inspect the worst rows below; repair the dominant line/price/probability pocket",
        })
    return notes


def _worst_rows(df: pd.DataFrame, limit: int = 30) -> list[dict[str, Any]]:
    if df.empty:
        return []
    losses = df.loc[df["target"].eq(0.0)].copy()
    if losses.empty:
        return []
    losses["badness"] = (
        losses["model_prob_side"].fillna(0.0)
        + losses["projection_under_margin"].fillna(0.0).clip(lower=0.0) * 0.08
        - losses["clv_price"].fillna(0.0).clip(lower=-10.0, upper=10.0) * 0.005
    )
    losses = losses.sort_values("badness", ascending=False)
    out: list[dict[str, Any]] = []
    for _, row in losses.head(limit).iterrows():
        out.append({
            "date": row.get("game_date_et"),
            "player_name": row.get("player_name"),
            "line": _float(row.get("market_line")),
            "price": _float(row.get("market_price")),
            "pred_k": _float(row.get("pred_count")),
            "actual_k": _float(row.get("actual_k")),
            "model_prob_under": _float(row.get("model_prob_side")),
            "projected_bf": _float(row.get("projected_bf")),
            "actual_bf": _float(row.get("actual_bf")),
            "projected_pitch_count": _float(row.get("projected_pitch_count")),
            "actual_pitch_count": _float(row.get("actual_pitch_count")),
            "clv_price": _float(row.get("clv_price")),
            "price_bucket": row.get("price_bucket"),
            "prob_bucket": row.get("prob_bucket"),
        })
    return out


def build(*, pg_dsn: str = PG_DSN, lookback_days: int = 365) -> dict[str, Any]:
    df = _load_rows(pg_dsn, lookback_days)
    overall = _metrics(df) if not df.empty else {}
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if not df.empty else "no_rows",
        "lookback_days": int(lookback_days),
        "target": "draftkings|pitcher_strikeouts|under|K 4.5-6.0",
        "rows": int(len(df)),
        "overall": overall,
        "diagnosis": _diagnosis(overall, df) if not df.empty else [],
        "by_line": _group_rows(df, ["market_line"]) if not df.empty else [],
        "by_price_bucket": _group_rows(df, ["price_bucket"]) if not df.empty else [],
        "by_probability_bucket": _group_rows(df, ["prob_bucket"]) if not df.empty else [],
        "by_projection_margin": _group_rows(df, ["projection_margin_bucket"]) if not df.empty else [],
        "by_bf_error_bucket": _group_rows(df, ["bf_error_bucket"]) if not df.empty else [],
        "worst_losses": _worst_rows(df),
    }
    _write_outputs(payload)
    return payload


def _fmt(value: Any, digits: int = 3) -> str:
    numeric = _float(value)
    return "-" if numeric is None else f"{numeric:.{digits}f}"


def _pct(value: Any) -> str:
    numeric = _float(value)
    return "-" if numeric is None else f"{numeric * 100.0:.1f}%"


def _render_group_table(title: str, rows: list[dict[str, Any]], key_col: str) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        f"| {key_col.replace('_', ' ').title()} | Rows | Win | Model P | Cal Err | ROI | CLV Beat | Avg CLV | BF MAE | PC MAE | Loss BF Bias | Loss PC Bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get(key_col)} | {row.get('rows')} | {_pct(row.get('win_rate'))} | "
            f"{_pct(row.get('avg_model_prob'))} | {_pct(row.get('calibration_error'))} | "
            f"{_pct(row.get('roi'))} | {_pct(row.get('clv_beat_rate'))} | {_fmt(row.get('avg_clv_price'))} | "
            f"{_fmt(row.get('projected_bf_mae'))} | {_fmt(row.get('pitch_count_mae'))} | "
            f"{_fmt(row.get('loss_projected_bf_bias'))} | {_fmt(row.get('loss_pitch_count_bias'))} |"
        )
    return lines


def _render(payload: dict[str, Any]) -> str:
    overall = payload.get("overall") or {}
    lines = [
        "# MLB Prop DK K 4.5-6.0 Under Repair Diagnostic",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Target: `{payload.get('target')}`",
        f"Status: **{payload.get('status')}**",
        "",
        f"- Rows: {payload.get('rows', 0)}",
        f"- Win rate: {_pct(overall.get('win_rate'))}",
        f"- Model P: {_pct(overall.get('avg_model_prob'))}",
        f"- Calibration error: {_pct(overall.get('calibration_error'))}",
        f"- ROI: {_pct(overall.get('roi'))}",
        f"- CLV beat: {_pct(overall.get('clv_beat_rate'))}",
        f"- Avg CLV: {_fmt(overall.get('avg_clv_price'))}",
        f"- BF MAE: {_fmt(overall.get('projected_bf_mae'))}",
        f"- Pitch-count MAE: {_fmt(overall.get('pitch_count_mae'))}",
        "",
        "## Diagnosis",
        "",
        "| Area | Evidence | Next Action |",
        "|---|---|---|",
    ]
    for row in payload.get("diagnosis") or []:
        lines.append(f"| {row['area']} | {row['evidence']} | {row['next_action']} |")
    lines.extend(_render_group_table("By Line", payload.get("by_line") or [], "market_line"))
    lines.extend(_render_group_table("By Price Bucket", payload.get("by_price_bucket") or [], "price_bucket"))
    lines.extend(_render_group_table("By Probability Bucket", payload.get("by_probability_bucket") or [], "prob_bucket"))
    lines.extend(_render_group_table("By Projection Margin", payload.get("by_projection_margin") or [], "projection_margin_bucket"))
    lines.extend(_render_group_table("By BF Error Bucket", payload.get("by_bf_error_bucket") or [], "bf_error_bucket"))
    lines.extend([
        "",
        "## Worst Losing Unders",
        "",
        "| Date | Player | Line | Price | Pred K | Actual K | P Under | BF Proj | BF Actual | PC Proj | PC Actual | CLV | Price Bucket | Prob Bucket |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in payload.get("worst_losses") or []:
        lines.append(
            f"| {row.get('date')} | {row.get('player_name')} | {_fmt(row.get('line'), 1)} | {_fmt(row.get('price'), 0)} | "
            f"{_fmt(row.get('pred_k'))} | {_fmt(row.get('actual_k'))} | {_pct(row.get('model_prob_under'))} | "
            f"{_fmt(row.get('projected_bf'))} | {_fmt(row.get('actual_bf'))} | "
            f"{_fmt(row.get('projected_pitch_count'))} | {_fmt(row.get('actual_pitch_count'))} | "
            f"{_fmt(row.get('clv_price'))} | {row.get('price_bucket')} | {row.get('prob_bucket')} |"
        )
    return "\n".join(lines) + "\n"


def _write_outputs(payload: dict[str, Any]) -> tuple[Path, Path]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _MODEL_DIR / "prop_k_under_46_repair_diagnostic.json"
    report_path = _REPORT_DIR / "mlb_prop_k_under_46_repair_diagnostic_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, _render(payload))
    payload["json_path"] = str(json_path)
    payload["report_path"] = str(report_path)
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build focused DK K 4.5-6.0 under repair diagnostic")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    args = parser.parse_args()
    payload = build(pg_dsn=args.pg_dsn, lookback_days=args.lookback_days)
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows"),
        "roi": (payload.get("overall") or {}).get("roi"),
        "clv_beat_rate": (payload.get("overall") or {}).get("clv_beat_rate"),
        "report_path": payload.get("report_path"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
