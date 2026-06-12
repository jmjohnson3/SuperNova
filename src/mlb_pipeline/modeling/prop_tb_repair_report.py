"""Total-bases repair report.

TB props need their own diagnostic because the failure can come from PA,
hit probability, extra-base structure, HR contribution, market pricing, or
bookability.  This report keeps those pieces separate so TB is repaired
deliberately instead of reopened as one broad market.
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
import psycopg2.extras
from sklearn.metrics import brier_score_loss

from .prop_market_training import ensure_prop_market_training_schema
from .side_recalibration import price_bucket, prop_line_bucket, prop_line_surface

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass(frozen=True)
class TBRepairConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    lookback_days: int = 365
    min_bucket_rows: int = 20
    top_n: int = 30
    report_file: str = "mlb_prop_tb_repair_latest.md"
    json_out: str = "prop_tb_repair_report.json"


SQL_WITH_PLAYER_GAME = """
SELECT
    e.game_date_et,
    e.game_slug,
    e.player_id,
    e.player_name,
    e.team_abbr,
    e.side,
    COALESCE(e.bookmaker_key, 'unknown') AS bookmaker_key,
    e.market_line::float AS market_line,
    e.market_price::float AS market_price,
    e.paired_price::float AS paired_price,
    COALESCE(e.pair_quality, 'unknown') AS pair_quality,
    e.model_family,
    e.pred_count::float AS pred_count,
    e.model_prob_side::float AS model_prob_side,
    e.market_prob_side::float AS market_prob_side,
    e.prob_edge_vs_market::float AS prob_edge_vs_market,
    e.ev::float AS ev,
    e.actual_value::float AS actual_total_bases,
    h.actual_pa::float AS actual_pa,
    h.actual_hits::float AS actual_hits,
    h.actual_home_runs::float AS actual_home_runs,
    h.projected_pa::float AS player_game_projected_pa,
    e.projected_pa::float AS offer_projected_pa,
    CASE WHEN e.won IS TRUE THEN 1 WHEN e.won IS FALSE THEN 0 ELSE NULL END AS won,
    COALESCE(e.push, false) AS push,
    e.profit_units::float AS profit_units,
    COALESCE(e.clv_valid, false) AS clv_valid,
    e.clv_price::float AS clv_price,
    CASE WHEN e.beat_clv_price IS TRUE THEN 1 WHEN e.beat_clv_price IS FALSE THEN 0 ELSE NULL END AS beat_clv_price,
    e.clv_status,
    e.clv_unknown_reason,
    e.confirmed_batting_order::float AS confirmed_batting_order
FROM features.mlb_prop_market_training_examples e
LEFT JOIN features.mlb_hitter_player_game_training h
  ON h.game_slug = e.game_slug
 AND h.player_id = e.player_id
WHERE e.game_date_et >= %(cutoff)s
  AND e.market = 'batter_total_bases'
  AND e.side IN ('over','under')
  AND e.market_line IS NOT NULL
  AND e.won IS NOT NULL
ORDER BY e.game_date_et, e.side, e.bookmaker_key, e.market_line
"""


SQL_EXAMPLES_ONLY = """
SELECT
    e.game_date_et,
    e.game_slug,
    e.player_id,
    e.player_name,
    e.team_abbr,
    e.side,
    COALESCE(e.bookmaker_key, 'unknown') AS bookmaker_key,
    e.market_line::float AS market_line,
    e.market_price::float AS market_price,
    e.paired_price::float AS paired_price,
    COALESCE(e.pair_quality, 'unknown') AS pair_quality,
    e.model_family,
    e.pred_count::float AS pred_count,
    e.model_prob_side::float AS model_prob_side,
    e.market_prob_side::float AS market_prob_side,
    e.prob_edge_vs_market::float AS prob_edge_vs_market,
    e.ev::float AS ev,
    e.actual_value::float AS actual_total_bases,
    e.actual_pa::float AS actual_pa,
    NULL::float AS actual_hits,
    NULL::float AS actual_home_runs,
    NULL::float AS player_game_projected_pa,
    e.projected_pa::float AS offer_projected_pa,
    CASE WHEN e.won IS TRUE THEN 1 WHEN e.won IS FALSE THEN 0 ELSE NULL END AS won,
    COALESCE(e.push, false) AS push,
    e.profit_units::float AS profit_units,
    COALESCE(e.clv_valid, false) AS clv_valid,
    e.clv_price::float AS clv_price,
    CASE WHEN e.beat_clv_price IS TRUE THEN 1 WHEN e.beat_clv_price IS FALSE THEN 0 ELSE NULL END AS beat_clv_price,
    e.clv_status,
    e.clv_unknown_reason,
    e.confirmed_batting_order::float AS confirmed_batting_order
FROM features.mlb_prop_market_training_examples e
WHERE e.game_date_et >= %(cutoff)s
  AND e.market = 'batter_total_bases'
  AND e.side IN ('over','under')
  AND e.market_line IS NOT NULL
  AND e.won IS NOT NULL
ORDER BY e.game_date_et, e.side, e.bookmaker_key, e.market_line
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
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return pd.DataFrame([dict(row) for row in cur.fetchall()])


def _clean_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _safe_brier(y: pd.Series, p: pd.Series) -> float | None:
    yy = pd.to_numeric(y, errors="coerce")
    pp = pd.to_numeric(p, errors="coerce")
    mask = yy.notna() & pp.notna()
    if not mask.any():
        return None
    return float(brier_score_loss(yy[mask].astype(int), pp[mask].clip(0.001, 0.999)))


def _load(cfg: TBRepairConfig) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, cfg.lookback_days))
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_prop_market_training_schema(conn)
        if not _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            return pd.DataFrame()
        sql = SQL_WITH_PLAYER_GAME if _table_exists(conn, "features", "mlb_hitter_player_game_training") else SQL_EXAMPLES_ONLY
        df = _query_df(conn, sql, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in (
        "market_line", "market_price", "paired_price", "pred_count",
        "model_prob_side", "market_prob_side", "prob_edge_vs_market", "ev",
        "actual_total_bases", "actual_pa", "actual_hits", "actual_home_runs",
        "player_game_projected_pa", "offer_projected_pa", "won", "profit_units",
        "clv_price", "beat_clv_price", "confirmed_batting_order",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["push"] = df["push"].fillna(False).astype(bool)
    df["clv_valid"] = df["clv_valid"].fillna(False).astype(bool)
    df["line_bucket"] = [prop_line_bucket("batter_total_bases", line) for line in df["market_line"]]
    df["line_surface"] = [
        prop_line_surface("batter_total_bases", side, line)
        for side, line in zip(df["side"], df["market_line"])
    ]
    df["price_bucket"] = df["market_price"].map(price_bucket)
    df["projected_pa"] = df["player_game_projected_pa"].combine_first(df["offer_projected_pa"])
    df["pa_error"] = df["projected_pa"] - df["actual_pa"]
    df["tb_error"] = df["pred_count"] - df["actual_total_bases"]
    df["line_result_margin"] = np.where(
        df["side"].eq("over"),
        df["actual_total_bases"] - df["market_line"],
        df["market_line"] - df["actual_total_bases"],
    )
    df["pa_bucket"] = pd.cut(
        df["actual_pa"],
        bins=[-0.1, 2.0, 3.0, 4.0, 9.0],
        labels=["0-2 PA", "3 PA", "4 PA", "5+ PA"],
    ).astype(str)
    df.loc[df["actual_pa"].isna(), "pa_bucket"] = "PA unknown"
    df["tb_shape"] = np.select(
        [
            df["actual_total_bases"].fillna(-1).eq(0),
            df["actual_home_runs"].fillna(0).ge(1),
            df["actual_total_bases"].fillna(0).ge(4),
            df["actual_total_bases"].fillna(0).between(2, 3, inclusive="both"),
            df["actual_total_bases"].fillna(0).eq(1),
        ],
        ["0 TB", "HR driven", "4+ TB non-HR/unknown", "2-3 TB", "1 TB"],
        default="TB unknown",
    )
    return df.replace([np.inf, -np.inf], np.nan)


def _bucket_summary(group: pd.DataFrame) -> dict[str, Any]:
    rows = len(group)
    won = pd.to_numeric(group["won"], errors="coerce")
    valid_clv = group[group["clv_valid"]].copy()
    model_brier = _safe_brier(won, group["model_prob_side"])
    market_brier = _safe_brier(won, group["market_prob_side"])
    pa_mae = _safe_mean(group["pa_error"].abs()) if "pa_error" in group else None
    tb_bias = _safe_mean(group["tb_error"])
    clv_beat = _safe_mean(valid_clv["beat_clv_price"]) if not valid_clv.empty else None
    avg_clv = _safe_mean(valid_clv["clv_price"]) if not valid_clv.empty else None
    issues: list[str] = []
    if rows < 50:
        issues.append("sample_small")
    if str(group["line_surface"].iloc[0]) == "alt_tail":
        issues.append("alt_tail_requires_separate_proof")
    if tb_bias is not None and tb_bias > 0.15:
        issues.append("tb_projection_high")
    if tb_bias is not None and tb_bias < -0.15:
        issues.append("tb_projection_low")
    if pa_mae is not None and pa_mae > 0.80:
        issues.append("pa_projection_error")
    if market_brier is not None and model_brier is not None and market_brier < model_brier:
        issues.append("market_beats_model_brier")
    if avg_clv is not None and avg_clv <= 0.0:
        issues.append("negative_or_flat_clv")
    if clv_beat is not None and clv_beat < 0.52:
        issues.append("weak_clv_beat")
    if not valid_clv.empty and len(valid_clv) / max(rows, 1) < 0.5:
        issues.append("low_valid_clv_coverage")
    if valid_clv.empty:
        issues.append("no_valid_clv")

    return {
        "side": str(group["side"].iloc[0]),
        "line_surface": str(group["line_surface"].iloc[0]),
        "line_bucket": str(group["line_bucket"].iloc[0]),
        "price_bucket": str(group["price_bucket"].iloc[0]),
        "bookmaker_key": str(group["bookmaker_key"].iloc[0]),
        "pair_quality": str(group["pair_quality"].mode().iloc[0]) if not group["pair_quality"].mode().empty else "unknown",
        "rows": int(rows),
        "dates": int(group["game_date_et"].nunique()),
        "win_rate": _safe_mean(won),
        "roi_units": _safe_mean(group["profit_units"]),
        "avg_model_prob": _safe_mean(group["model_prob_side"]),
        "avg_market_prob": _safe_mean(group["market_prob_side"]),
        "model_brier": model_brier,
        "market_brier": market_brier,
        "avg_pred_tb": _safe_mean(group["pred_count"]),
        "avg_actual_tb": _safe_mean(group["actual_total_bases"]),
        "tb_bias_pred_minus_actual": tb_bias,
        "tb_mae": _safe_mean(group["tb_error"].abs()),
        "pa_mae": pa_mae,
        "low_pa_actual_rate": float((pd.to_numeric(group["actual_pa"], errors="coerce") <= 2).mean()),
        "confirmed_lineup_rate": float(group["confirmed_batting_order"].notna().mean()),
        "valid_clv_rows": int(len(valid_clv)),
        "avg_clv_price": avg_clv,
        "clv_beat_rate": clv_beat,
        "issues": issues,
    }


def _summaries(df: pd.DataFrame, cfg: TBRepairConfig) -> dict[str, Any]:
    exact = []
    group_cols = ["side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"]
    for _, group in df.groupby(group_cols, dropna=False):
        exact.append(_bucket_summary(group))
    exact.sort(
        key=lambda r: (
            len(r["issues"]),
            999 if r.get("model_brier") is None else float(r["model_brier"]),
            -int(r["rows"]),
        )
    )

    by_shape = []
    for cols in (["side", "tb_shape"], ["side", "pa_bucket"], ["side", "pair_quality"]):
        for key, group in df.groupby(cols, dropna=False):
            rec = _bucket_summary(group)
            rec["group"] = " | ".join(map(str, key if isinstance(key, tuple) else (key,)))
            by_shape.append(rec)
    by_shape.sort(key=lambda r: (str(r.get("group")), -int(r["rows"])))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_rows": int(len(df)),
        "date_min": str(min(df["game_date_et"])) if not df.empty else None,
        "date_max": str(max(df["game_date_et"])) if not df.empty else None,
        "exact_buckets": exact,
        "diagnostic_groups": by_shape,
        "top_repair_targets": [
            r for r in exact
            if int(r["rows"]) >= cfg.min_bucket_rows
            and ("market_beats_model_brier" in r["issues"] or "tb_projection_high" in r["issues"] or "pa_projection_error" in r["issues"])
        ][: cfg.top_n],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    v = _clean_float(value)
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _write(payload: dict[str, Any], cfg: TBRepairConfig) -> None:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    (cfg.model_dir / cfg.json_out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / cfg.report_file
    lines: list[str] = [
        "# MLB TB Prop Repair Report",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        f"Rows: {payload.get('source_rows', 0)} | Dates: {payload.get('date_min')} to {payload.get('date_max')}",
        "",
        "## Top Repair Targets",
        "",
    ]
    targets = payload.get("top_repair_targets") or []
    if not targets:
        lines.append("No TB buckets met the repair-target row threshold yet.")
    else:
        lines.append("| Bucket | Rows | ROI | CLV | Brier M/Mkt | TB Bias | PA MAE | Issues |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for rec in targets[: cfg.top_n]:
            bucket = " | ".join([
                rec["side"], rec["line_surface"], rec["line_bucket"], rec["price_bucket"], rec["bookmaker_key"],
            ])
            lines.append(
                f"| {bucket} | {rec['rows']} | {_fmt(rec.get('roi_units'))} | "
                f"{_fmt(rec.get('avg_clv_price'))} | {_fmt(rec.get('model_brier'))}/{_fmt(rec.get('market_brier'))} | "
                f"{_fmt(rec.get('tb_bias_pred_minus_actual'))} | {_fmt(rec.get('pa_mae'))} | "
                f"{', '.join(rec.get('issues') or [])} |"
            )

    lines.extend(["", "## Exact Buckets", ""])
    exact = payload.get("exact_buckets") or []
    if not exact:
        lines.append("No exact TB buckets available.")
    else:
        lines.append("| Bucket | Rows | Dates | Win | ROI | CLV Beat | Avg CLV | Pred/Actual TB | PA MAE | Issues |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for rec in exact[: cfg.top_n]:
            bucket = " | ".join([
                rec["side"], rec["line_surface"], rec["line_bucket"], rec["price_bucket"], rec["bookmaker_key"],
            ])
            lines.append(
                f"| {bucket} | {rec['rows']} | {rec['dates']} | {_fmt(rec.get('win_rate'))} | "
                f"{_fmt(rec.get('roi_units'))} | {_fmt(rec.get('clv_beat_rate'))} | {_fmt(rec.get('avg_clv_price'))} | "
                f"{_fmt(rec.get('avg_pred_tb'))}/{_fmt(rec.get('avg_actual_tb'))} | {_fmt(rec.get('pa_mae'))} | "
                f"{', '.join(rec.get('issues') or [])} |"
            )

    lines.extend(["", "## Diagnostic Groups", ""])
    groups = payload.get("diagnostic_groups") or []
    if not groups:
        lines.append("No TB diagnostic groups available.")
    else:
        lines.append("| Group | Rows | Win | ROI | TB Bias | PA MAE | Model/Mkt Brier | Issues |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for rec in groups[: cfg.top_n]:
            lines.append(
                f"| {rec.get('group')} | {rec['rows']} | {_fmt(rec.get('win_rate'))} | {_fmt(rec.get('roi_units'))} | "
                f"{_fmt(rec.get('tb_bias_pred_minus_actual'))} | {_fmt(rec.get('pa_mae'))} | "
                f"{_fmt(rec.get('model_brier'))}/{_fmt(rec.get('market_brier'))} | "
                f"{', '.join(rec.get('issues') or [])} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(cfg: TBRepairConfig) -> dict[str, Any]:
    df = _load(cfg)
    if df.empty:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_rows": 0,
            "status": "no_rows",
            "exact_buckets": [],
            "diagnostic_groups": [],
            "top_repair_targets": [],
        }
        _write(payload, cfg)
        return payload
    payload = _summaries(df, cfg)
    payload["status"] = "ok"
    _write(payload, cfg)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TB-specific prop repair report")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--min-bucket-rows", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()
    payload = build_report(
        TBRepairConfig(
            lookback_days=args.lookback_days,
            min_bucket_rows=args.min_bucket_rows,
            top_n=args.top_n,
        )
    )
    print(f"TB repair report status={payload.get('status')} rows={payload.get('source_rows', 0)}")


if __name__ == "__main__":
    main()
