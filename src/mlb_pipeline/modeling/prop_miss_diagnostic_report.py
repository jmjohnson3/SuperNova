"""Diagnose why locked MLB prop picks missed.

This report separates projection accuracy from betting accuracy and labels
each losing locked prop with likely failure modes:
projection, side probability, pricing conversion, bookability, calibration
bucket, and weak market/side.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
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

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass(frozen=True)
class MissDiagnosticConfig:
    pg_dsn: str = _PG_DSN
    lookback_days: int = 365
    out: str = "mlb_prop_miss_diagnostic_latest.md"
    json_out: str = "prop_miss_diagnostic_report.json"
    min_bucket_rows: int = 20
    projection_miss_threshold: float = 1.0
    overconfidence_threshold: float = 0.58
    min_market_side_roi: float = 0.0


SQL = """
SELECT
    id,
    run_id,
    replay_id,
    game_date_et,
    game_slug,
    player_id,
    player_name,
    team_abbr,
    market,
    side,
    COALESCE(bookmaker_key, 'unknown') AS bookmaker_key,
    COALESCE(line_surface, 'unknown') AS line_surface,
    COALESCE(line_bucket, 'unknown') AS line_bucket,
    COALESCE(price_bucket, 'missing_price') AS price_bucket,
    COALESCE(model_family, 'unknown') AS model_family,
    market_line::float AS market_line,
    market_price::float AS market_price,
    market_prob_side::float AS market_prob_side,
    model_prob_side::float AS model_prob_side,
    prob_edge_vs_market::float AS prob_edge_vs_market,
    ev::float AS ev,
    pred_count::float AS pred_count,
    pred_value::float AS pred_value,
    actual_value::float AS actual_value,
    CASE WHEN won IS TRUE THEN 1 WHEN won IS FALSE THEN 0 ELSE NULL END AS won,
    COALESCE(push, false) AS push,
    profit_units::float AS profit_units,
    clv_valid,
    clv_status,
    clv_unknown_reason,
    clv_price::float AS clv_price,
    CASE WHEN beat_clv_price IS TRUE THEN 1 WHEN beat_clv_price IS FALSE THEN 0 ELSE NULL END AS beat_clv_price,
    confirmed_batting_order::float AS confirmed_batting_order,
    projected_pa::float AS projected_pa,
    projected_bf::float AS projected_bf,
    projected_pitch_count::float AS projected_pitch_count,
    actual_pa::float AS actual_pa,
    actual_bf::float AS actual_bf,
    actual_pitch_count_proxy::float AS actual_pitch_count_proxy,
    is_home::float AS is_home,
    opp_sp_hand,
    opp_sp_k_pct_10::float AS opp_sp_k_pct_10,
    opp_bp_era_10::float AS opp_bp_era_10,
    opp_team_k_pct_10::float AS opp_team_k_pct_10,
    batter_vs_hand_hits_avg_10::float AS batter_vs_hand_hits_avg_10,
    batter_vs_hand_tb_avg_10::float AS batter_vs_hand_tb_avg_10,
    batter_vs_hand_hr_avg_10::float AS batter_vs_hand_hr_avg_10,
    pinch_hit_risk::float AS pinch_hit_risk
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND side IN ('over','under')
  AND model_prob_side IS NOT NULL
  AND market_line IS NOT NULL
  AND actual_value IS NOT NULL
  AND won IS NOT NULL
ORDER BY game_date_et, market, side, player_name
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


def _load(cfg: MissDiagnosticConfig) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, cfg.lookback_days))
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_prop_market_training_schema(conn)
        if not _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            return pd.DataFrame()
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    numeric = [
        "market_line", "market_price", "market_prob_side", "model_prob_side",
        "prob_edge_vs_market", "ev", "pred_count", "pred_value", "actual_value",
        "won", "profit_units", "clv_price", "beat_clv_price",
        "projected_pa", "projected_bf", "projected_pitch_count",
        "actual_pa", "actual_bf", "actual_pitch_count_proxy",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["push"] = df["push"].fillna(False).astype(bool)
    return df.replace([np.inf, -np.inf], np.nan)


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _brier(group: pd.DataFrame) -> float | None:
    work = group.loc[~group["push"]].dropna(subset=["model_prob_side", "won"])
    if work.empty or work["won"].nunique() < 2:
        return None
    return float(brier_score_loss(work["won"].astype(int), work["model_prob_side"].clip(1e-6, 1 - 1e-6)))


def _f(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _poisson_cdf(k: int, lam: float) -> float:
    lam = max(1e-6, min(40.0, float(lam)))
    term = math.exp(-lam)
    total = term
    for i in range(1, max(0, int(k)) + 1):
        term *= lam / i
        total += term
    return max(0.0, min(1.0, total))


def _distribution_side_prob(row: pd.Series) -> float | None:
    pred = _f(row.get("pred_count"))
    line = _f(row.get("market_line"))
    if pred is None or line is None:
        return None
    p_over = 1.0 - _poisson_cdf(math.floor(line), pred)
    p_over = max(1e-6, min(1.0 - 1e-6, p_over))
    return p_over if row.get("side") == "over" else 1.0 - p_over


def _opportunity_miss(row: pd.Series) -> bool:
    market = str(row.get("market") or "")
    if market in {"batter_hits", "batter_total_bases", "batter_home_runs"}:
        projected_pa = _f(row.get("projected_pa"))
        actual_pa = _f(row.get("actual_pa"))
        if projected_pa is None or actual_pa is None:
            return False
        if abs(actual_pa - projected_pa) >= 1.25:
            return True
        if projected_pa >= 3.8 and actual_pa <= 2.0:
            return True
    if market == "pitcher_strikeouts":
        projected_bf = _f(row.get("projected_bf"))
        actual_bf = _f(row.get("actual_bf"))
        if projected_bf is not None and actual_bf is not None and abs(actual_bf - projected_bf) >= 5.0:
            return True
        projected_pitch_count = _f(row.get("projected_pitch_count"))
        actual_pitch_count = _f(row.get("actual_pitch_count_proxy"))
        if (
            projected_pitch_count is not None
            and actual_pitch_count is not None
            and abs(actual_pitch_count - projected_pitch_count) >= 18.0
        ):
            return True
    return False


def _group_metrics(df: pd.DataFrame, cols: list[str], cfg: MissDiagnosticConfig) -> dict[tuple, dict[str, Any]]:
    out: dict[tuple, dict[str, Any]] = {}
    for key, group in df.groupby(cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        settled = group.loc[~group["push"]]
        rows = int(len(settled))
        if rows < cfg.min_bucket_rows:
            continue
        avg_prob = _mean(settled["model_prob_side"])
        win_rate = _mean(settled["won"])
        roi = _mean(group["profit_units"])
        clv = group.loc[group["clv_valid"].fillna(False).astype(bool)]
        out[key] = {
            "rows": rows,
            "win_rate": win_rate,
            "avg_prob": avg_prob,
            "calibration_error": None if win_rate is None or avg_prob is None else win_rate - avg_prob,
            "roi": roi,
            "brier": _brier(group),
            "clv_rows": int(len(clv)),
            "clv_beat_rate": _mean(clv["beat_clv_price"]) if not clv.empty else None,
            "avg_clv_price": _mean(clv["clv_price"]) if not clv.empty else None,
            "mae": _mean((group["actual_value"] - group["pred_count"]).abs()) if "pred_count" in group else None,
            "rmse": float(np.sqrt(np.nanmean((group["actual_value"] - group["pred_count"]) ** 2))) if "pred_count" in group else None,
        }
    return out


def _classify(row: pd.Series, bucket: dict[str, Any] | None, market_side: dict[str, Any] | None, cfg: MissDiagnosticConfig) -> list[str]:
    reasons: list[str] = []
    if bool(row.get("push")) or int(row.get("won") or 0) == 1:
        return reasons

    opportunity_miss = _opportunity_miss(row)
    if opportunity_miss:
        reasons.append("bad_opportunity_projection")

    side = str(row.get("side") or "")
    pred_count = row.get("pred_count")
    actual = row.get("actual_value")
    line = row.get("market_line")
    if pd.notna(pred_count) and pd.notna(actual) and pd.notna(line):
        projected_side = "over" if float(pred_count) > float(line) else "under"
        actual_side = "over" if float(actual) > float(line) else "under"
        if projected_side == side and actual_side != side:
            reasons.append("bad_player_projection")
        elif abs(float(actual) - float(pred_count)) >= cfg.projection_miss_threshold:
            reasons.append("bad_player_rate_projection" if not opportunity_miss else "large_projection_error")

    model_prob = row.get("model_prob_side")
    if pd.notna(model_prob) and float(model_prob) >= cfg.overconfidence_threshold:
        reasons.append("bad_side_probability")
    dist_prob = _distribution_side_prob(row)
    if (
        dist_prob is not None
        and pd.notna(model_prob)
        and abs(float(model_prob) - float(dist_prob)) >= 0.15
        and float(model_prob) >= cfg.overconfidence_threshold
    ):
        reasons.append("bad_distribution_pricing")

    market_prob = row.get("market_prob_side")
    edge = row.get("prob_edge_vs_market")
    ev = row.get("ev")
    if pd.notna(ev) and float(ev) > 0 and (pd.isna(edge) or float(edge) <= 0.0):
        reasons.append("bad_line_price_edge")
    if pd.notna(model_prob) and pd.notna(market_prob) and float(model_prob) < float(market_prob):
        reasons.append("model_worse_than_market_price")

    raw_clv_valid = row.get("clv_valid")
    clv_valid = False if pd.isna(raw_clv_valid) else bool(raw_clv_valid)
    unknown = str(row.get("clv_unknown_reason") or "")
    if not clv_valid and unknown:
        reasons.append("bad_clv_bookability")
    elif pd.notna(row.get("clv_price")) and float(row.get("clv_price")) < 0:
        reasons.append("lost_clv")

    if bucket:
        cal = bucket.get("calibration_error")
        if cal is not None and abs(float(cal)) > 0.05:
            reasons.append("bad_calibration_bucket")
        if bucket.get("roi") is not None and float(bucket["roi"]) < cfg.min_market_side_roi:
            reasons.append("bad_bucket_roi")

    if market_side and market_side.get("roi") is not None and float(market_side["roi"]) < cfg.min_market_side_roi:
        reasons.append("weak_market_bucket")

    return sorted(dict.fromkeys(reasons or ["unclassified_miss"]))


def _metrics_payload(df: pd.DataFrame, cfg: MissDiagnosticConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    if df.empty:
        return {"rows": 0}, df
    exact_cols = ["market", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"]
    market_side_cols = ["market", "side"]
    exact = _group_metrics(df, exact_cols, cfg)
    market_side = _group_metrics(df, market_side_cols, cfg)
    reason_rows = []
    reason_counter: Counter[str] = Counter()
    reason_market_side: Counter[str] = Counter()
    work = df.copy()
    labels: list[str] = []
    for _, row in work.iterrows():
        exact_key = tuple(row[col] for col in exact_cols)
        market_side_key = tuple(row[col] for col in market_side_cols)
        reasons = _classify(row, exact.get(exact_key), market_side.get(market_side_key), cfg)
        labels.append(",".join(reasons))
        if int(row.get("won") or 0) == 0 and not bool(row.get("push")):
            for reason in reasons:
                reason_counter[reason] += 1
                reason_market_side[f"{reason}|{row.get('market')}|{row.get('side')}"] += 1
            reason_rows.append({
                "date": str(row.get("game_date_et")),
                "player": row.get("player_name"),
                "market": row.get("market"),
                "side": row.get("side"),
                "book": row.get("bookmaker_key"),
                "line": row.get("market_line"),
                "price": row.get("market_price"),
                "pred_count": row.get("pred_count"),
                "actual": row.get("actual_value"),
                "projected_pa": row.get("projected_pa"),
                "actual_pa": row.get("actual_pa"),
                "projected_bf": row.get("projected_bf"),
                "actual_bf": row.get("actual_bf"),
                "model_prob": row.get("model_prob_side"),
                "market_prob": row.get("market_prob_side"),
                "clv_reason": row.get("clv_unknown_reason"),
                "reasons": reasons,
            })
    work["miss_reasons"] = labels

    def summarize(cols: list[str]) -> list[dict[str, Any]]:
        rows = []
        for key, group in df.groupby(cols, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            settled = group.loc[~group["push"]]
            clv = group.loc[group["clv_valid"].fillna(False).astype(bool)]
            rows.append({
                "key": "|".join(str(v) for v in key_tuple),
                "rows": int(len(settled)),
                "win_rate": _mean(settled["won"]),
                "avg_prob": _mean(settled["model_prob_side"]),
                "brier": _brier(group),
                "roi": _mean(group["profit_units"]),
                "mae": _mean((group["actual_value"] - group["pred_count"]).abs()),
                "rmse": float(np.sqrt(np.nanmean((group["actual_value"] - group["pred_count"]) ** 2))),
                "clv_rows": int(len(clv)),
                "clv_beat_rate": _mean(clv["beat_clv_price"]) if not clv.empty else None,
                "avg_clv_price": _mean(clv["clv_price"]) if not clv.empty else None,
            })
        rows.sort(key=lambda rec: (rec["rows"], abs(rec.get("roi") or 0.0)), reverse=True)
        return rows

    return {
        "rows": int(len(df)),
        "date_min": str(min(df["game_date_et"])),
        "date_max": str(max(df["game_date_et"])),
        "unique_dates": int(df["game_date_et"].nunique()),
        "miss_reasons": dict(reason_counter.most_common()),
        "miss_reasons_by_market_side": [
            {
                "reason": key.split("|", 2)[0],
                "market_side": key.split("|", 1)[1],
                "misses": count,
            }
            for key, count in reason_market_side.most_common(40)
        ],
        "market_side": summarize(["market", "side"]),
        "market_side_line_book": summarize(exact_cols),
        "examples": reason_rows[:50],
    }, work


def _fmt_pct(value: Any, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    v = float(value) * 100.0
    return f"{v:+.1f}%" if signed else f"{v:.1f}%"


def _fmt_num(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"


def _write_report(payload: dict[str, Any], cfg: MissDiagnosticConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / cfg.out
    lines = [
        "# MLB Prop Miss Diagnostic",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')}",
        f"Rows: {payload.get('rows', 0)}",
        f"Date range: {payload.get('date_min')} to {payload.get('date_max')}",
        f"Unique dates: {payload.get('unique_dates', 0)}",
        "",
        "## Miss Reason Counts",
        "",
        "| Reason | Misses |",
        "|---|---:|",
    ]
    for reason, count in payload.get("miss_reasons", {}).items():
        lines.append(f"| {reason} | {count} |")

    lines.extend([
        "",
        "## Miss Reasons By Market/Side",
        "",
        "| Reason | Market/Side | Misses |",
        "|---|---|---:|",
    ])
    for rec in payload.get("miss_reasons_by_market_side", [])[:40]:
        lines.append(f"| {rec['reason']} | {rec['market_side']} | {rec['misses']} |")

    lines.extend([
        "",
        "## Projection vs Betting Accuracy by Market/Side",
        "",
        "| Bucket | Rows | Win | Avg Prob | Brier | ROI | MAE | RMSE | CLV Rows | CLV Beat | Avg CLV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rec in payload.get("market_side", [])[:30]:
        lines.append(
            f"| {rec['key']} | {rec['rows']} | {_fmt_pct(rec['win_rate'])} | "
            f"{_fmt_pct(rec['avg_prob'])} | {_fmt_num(rec['brier'])} | {_fmt_pct(rec['roi'])} | "
            f"{_fmt_num(rec['mae'])} | {_fmt_num(rec.get('rmse'))} | {rec['clv_rows']} | {_fmt_pct(rec['clv_beat_rate'])} | "
            f"{_fmt_num(rec['avg_clv_price'], 2, signed=True)} |"
        )

    lines.extend([
        "",
        "## Weak Exact Buckets",
        "",
        "| Bucket | Rows | Win | Avg Prob | Brier | ROI | MAE | RMSE | CLV Rows | CLV Beat | Avg CLV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    weak = sorted(
        payload.get("market_side_line_book", []),
        key=lambda rec: (rec.get("roi") if rec.get("roi") is not None else 9.0),
    )
    for rec in weak[:30]:
        lines.append(
            f"| {rec['key']} | {rec['rows']} | {_fmt_pct(rec['win_rate'])} | "
            f"{_fmt_pct(rec['avg_prob'])} | {_fmt_num(rec['brier'])} | {_fmt_pct(rec['roi'])} | "
            f"{_fmt_num(rec['mae'])} | {_fmt_num(rec.get('rmse'))} | {rec['clv_rows']} | {_fmt_pct(rec['clv_beat_rate'])} | "
            f"{_fmt_num(rec['avg_clv_price'], 2, signed=True)} |"
        )

    lines.extend([
        "",
        "## Recent Losing Examples",
        "",
        "| Date | Player | Bet | Price | Pred | Actual | PA | BF | Model | Market | CLV Reason | Labels |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in payload.get("examples", [])[:30]:
        bet = f"{row['market']} {row['side']} {row['line']} {row['book']}"
        labels = ", ".join(row.get("reasons") or [])
        pa = ""
        if row.get("projected_pa") is not None or row.get("actual_pa") is not None:
            pa = f"{_fmt_num(row.get('projected_pa'), 1)}->{_fmt_num(row.get('actual_pa'), 1)}"
        bf = ""
        if row.get("projected_bf") is not None or row.get("actual_bf") is not None:
            bf = f"{_fmt_num(row.get('projected_bf'), 1)}->{_fmt_num(row.get('actual_bf'), 1)}"
        lines.append(
            f"| {row['date']} | {row['player']} | {bet} | {row['price']} | "
            f"{_fmt_num(row['pred_count'])} | {_fmt_num(row['actual'])} | "
            f"{pa} | {bf} | {_fmt_pct(row['model_prob'])} | {_fmt_pct(row['market_prob'])} | "
            f"{row.get('clv_reason') or ''} | {labels} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def build_report(cfg: MissDiagnosticConfig) -> dict[str, Any]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = _load(cfg)
    payload, _ = _metrics_payload(df, cfg) if not df.empty else ({"rows": 0, "status": "no_rows"}, df)
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    payload["status"] = "ready" if payload.get("rows", 0) else "no_rows"
    payload["report_path"] = _write_report(payload, cfg)
    (_MODEL_DIR / cfg.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop miss diagnostic report")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--out", default="mlb_prop_miss_diagnostic_latest.md")
    parser.add_argument("--json-out", default="prop_miss_diagnostic_report.json")
    args = parser.parse_args()
    payload = build_report(MissDiagnosticConfig(
        pg_dsn=args.pg_dsn,
        lookback_days=args.lookback_days,
        out=args.out,
        json_out=args.json_out,
    ))
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows", 0),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
