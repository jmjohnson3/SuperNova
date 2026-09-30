"""Calibrate $1 micro-projection probabilities from locked micro results.

This artifact is intentionally separate from the broader prop distribution and
bucket-promotion reports. It answers a narrower live-betting question: when the
micro lane says a prop is 70-85%, how much should we distrust that confidence
until the exact bucket has proven it with real locked $1 outcomes?
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

_ET = ZoneInfo("America/New_York")
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

_PRIOR_STRENGTH = {
    "exact": 30.0,
    "approved_model": 30.0,
    "book_line": 35.0,
    "line_price": 35.0,
    "line": 45.0,
    "market_side": 60.0,
    "global": 40.0,
}


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    return math.log(p / (1.0 - p))


def _pct(value: Any, digits: int = 1, *, signed: bool = False) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    sign = "+" if signed else ""
    return f"{numeric:{sign}.{digits}%}"


def _num(value: Any, digits: int = 3, *, signed: bool = False) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    sign = "+" if signed else ""
    return f"{numeric:{sign}.{digits}f}"


def _micro_where() -> str:
    return """
      source = 'prop'
      AND (
        LOWER(COALESCE(model_tier, '')) = 'micro_projection'
        OR LOWER(COALESCE(model_meta->>'selector_tier', '')) = 'micro_projection'
      )
    """


def _fetch_rows(conn, start_date: date, end_date: date) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT
                game_date_et,
                player_name,
                team_abbr,
                market,
                side,
                COALESCE(bookmaker_key, 'unknown') AS bookmaker_key,
                COALESCE(NULLIF(model_meta->>'model_family', ''), 'unknown') AS model_family,
                market_line,
                market_price,
                model_prob,
                ev,
                result_status,
                won,
                push,
                profit_units,
                stake_usd,
                clv_valid,
                clv_price,
                COALESCE(NULLIF(model_meta->>'line_bucket', ''), CASE
                    WHEN market = 'pitcher_strikeouts' AND market_line < 4.5 THEN 'K <4.5'
                    WHEN market = 'pitcher_strikeouts' AND market_line < 6.5 THEN 'K 4.5-6.0'
                    WHEN market = 'pitcher_strikeouts' AND market_line < 8.5 THEN 'K 6.5-8.0'
                    WHEN market = 'pitcher_strikeouts' THEN 'K 8.5+'
                    WHEN market = 'batter_total_bases' AND market_line < 1.0 THEN 'TB 0.5'
                    WHEN market = 'batter_total_bases' AND market_line < 2.0 THEN 'TB 1.5'
                    WHEN market = 'batter_total_bases' AND market_line < 3.0 THEN 'TB 2.5'
                    WHEN market = 'batter_total_bases' AND market_line < 4.0 THEN 'TB 3.5'
                    WHEN market = 'batter_total_bases' THEN 'TB 4.5+'
                    WHEN market = 'batter_hits' AND market_line < 1.0 THEN 'H 0.5'
                    WHEN market = 'batter_hits' AND market_line < 2.0 THEN 'H 1.5'
                    WHEN market = 'batter_hits' AND market_line < 3.0 THEN 'H 2.5'
                    WHEN market = 'batter_hits' THEN 'H 3.5+'
                    WHEN market = 'batter_home_runs' AND market_line < 1.0 THEN 'HR 0.5'
                    WHEN market = 'batter_home_runs' THEN 'HR 1.5+'
                    ELSE 'other'
                END) AS line_bucket,
                COALESCE(NULLIF(model_meta->>'price_bucket', ''), CASE
                    WHEN market_price IS NULL THEN 'missing_price'
                    WHEN market_price > 0 AND market_price < 150 THEN 'plus_100_149'
                    WHEN market_price > 0 AND market_price < 250 THEN 'plus_150_249'
                    WHEN market_price > 0 AND market_price < 500 THEN 'plus_250_499'
                    WHEN market_price > 0 THEN 'plus_500_plus'
                    WHEN market_price >= -129 THEN 'fair_lay'
                    WHEN market_price >= -149 THEN 'lay_130_149'
                    WHEN market_price >= -180 THEN 'lay_150_180'
                    ELSE 'heavy_lay'
                END) AS price_bucket
            FROM bets.mlb_model_pick_ledger
            WHERE {_micro_where()}
              AND game_date_et BETWEEN %(start_date)s AND %(end_date)s
              AND result_status = 'graded'
              AND COALESCE(push, false) IS FALSE
              AND won IS NOT NULL
              AND model_prob IS NOT NULL
            ORDER BY game_date_et DESC
            """,
            {"start_date": start_date, "end_date": end_date},
        )
        return [dict(row) for row in cur.fetchall()]


def _group_keys(row: dict[str, Any]) -> list[tuple[str, str]]:
    market = str(row.get("market") or "unknown")
    side = str(row.get("side") or "unknown").lower()
    line_bucket = str(row.get("line_bucket") or "unknown")
    price_bucket = str(row.get("price_bucket") or "unknown")
    book = str(row.get("bookmaker_key") or "unknown").lower()
    model_family = str(row.get("model_family") or "unknown").lower()
    return [
        ("exact", "|".join(["exact", market, side, line_bucket, price_bucket, book])),
        ("approved_model", "|".join(["approved_model", market, side, line_bucket, book, model_family])),
        ("book_line", "|".join(["book_line", market, side, line_bucket, book])),
        ("line_price", "|".join(["line_price", market, side, line_bucket, price_bucket])),
        ("line", "|".join(["line", market, side, line_bucket])),
        ("market_side", "|".join(["market_side", market, side])),
        ("global", "global|*"),
    ]


def _summarize(rows: list[dict[str, Any]], *, level: str, prior_rate: float) -> dict[str, Any]:
    graded = len(rows)
    wins = sum(1 for row in rows if _bool(row.get("won")))
    losses = graded - wins
    stake = sum((_float(row.get("stake_usd")) or 1.0) for row in rows)
    units = sum((_float(row.get("profit_units")) or 0.0) for row in rows)
    probs = [_float(row.get("model_prob")) for row in rows if _float(row.get("model_prob")) is not None]
    clv_rows = [row for row in rows if _bool(row.get("clv_valid"))]
    avg_prob = sum(probs) / len(probs) if probs else None
    win_rate = wins / graded if graded else None
    roi = units / stake if stake > 0 else None
    clv_beat = (
        sum(1 for row in clv_rows if (_float(row.get("clv_price")) or 0.0) > 0.0) / len(clv_rows)
        if clv_rows
        else None
    )
    avg_clv = (
        sum((_float(row.get("clv_price")) or 0.0) for row in clv_rows) / len(clv_rows)
        if clv_rows
        else None
    )
    prior_strength = _PRIOR_STRENGTH.get(level, 45.0)
    posterior = ((wins if graded else 0.0) + prior_rate * prior_strength) / max(1.0, graded + prior_strength)
    shrink_weight = 0.0 if graded <= 0 else min(0.85, graded / (graded + prior_strength))
    calibration_error = (avg_prob - win_rate) if avg_prob is not None and win_rate is not None else None
    proven = bool(
        graded >= 50
        and roi is not None and roi > 0.0
        and clv_beat is not None and clv_beat >= 0.55
        and avg_clv is not None and avg_clv > 0.0
        and calibration_error is not None and abs(calibration_error) <= 0.05
    )
    if proven:
        max_probability = 0.88
        shrink_weight = min(shrink_weight, 0.15)
    else:
        max_probability = min(0.68, max(0.58, posterior + 0.18))
    return {
        "level": level,
        "enabled": graded > 0,
        "proven": proven,
        "graded": graded,
        "wins": wins,
        "losses": losses,
        "record": f"{wins}-{losses}-0",
        "avg_model_prob": avg_prob,
        "win_rate": win_rate,
        "calibration_error": calibration_error,
        "roi": roi,
        "clv_rows": len(clv_rows),
        "clv_beat_rate": clv_beat,
        "avg_clv_price": avg_clv,
        "prior_rate": prior_rate,
        "prior_strength": prior_strength,
        "target_probability": posterior,
        "logit_offset": (_logit(posterior) - _logit(avg_prob)) if avg_prob is not None else None,
        "shrink_weight": shrink_weight,
        "max_probability": max_probability,
        "method": "one_way_empirical_bayes_shrink_to_micro_results",
    }


def build_payload(*, pg_dsn: str = PG_DSN, end_date: date | None = None, lookback_days: int = 60) -> dict[str, Any]:
    end = end_date or datetime.now(tz=_ET).date()
    start = end - timedelta(days=max(1, int(lookback_days)) - 1)
    with psycopg2.connect(pg_dsn) as conn:
        rows = _fetch_rows(conn, start, end)
    global_win = sum(1 for row in rows if _bool(row.get("won"))) / len(rows) if rows else 0.52
    prior_rate = max(0.35, min(0.55, global_win))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    levels: dict[str, str] = {}
    for row in rows:
        for level, key in _group_keys(row):
            grouped[key].append(row)
            levels[key] = level
    groups = {
        key: _summarize(group_rows, level=levels.get(key, "unknown"), prior_rate=prior_rate)
        for key, group_rows in grouped.items()
    }
    global_group = groups.get("global|*") or _summarize(rows, level="global", prior_rate=prior_rate)
    enabled = bool(len(rows) >= 10)
    return {
        "generated_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if rows else "no_graded_micro_rows",
        "enabled": enabled,
        "usage": "shrink_micro_projection_probabilities_before_ev_and_min_price",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "lookback_days": int(lookback_days),
        "graded_rows": len(rows),
        "global_prior_rate": prior_rate,
        "global": global_group,
        "groups": groups,
    }


def render(payload: dict[str, Any]) -> str:
    global_row = payload.get("global") or {}
    lines = [
        "# MLB Prop Micro Probability Calibrator",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        f"Enabled: **{bool(payload.get('enabled'))}**",
        f"Window: {payload.get('start_date')} through {payload.get('end_date')}",
        "",
        "## Global Micro Truth",
        "",
        f"- Graded rows: {global_row.get('graded', 0)}",
        f"- Record: {global_row.get('record', '0-0-0')}",
        f"- Avg model probability: {_pct(global_row.get('avg_model_prob'))}",
        f"- Win rate: {_pct(global_row.get('win_rate'))}",
        f"- Calibration error: {_pct(global_row.get('calibration_error'), signed=True)}",
        f"- ROI: {_pct(global_row.get('roi'), signed=True)}",
        f"- CLV beat: {_pct(global_row.get('clv_beat_rate'))}",
        f"- Avg CLV: {_num(global_row.get('avg_clv_price'), signed=True)}",
        f"- High-probability cap unless proven: {_pct(global_row.get('max_probability'))}",
        "",
        "## Groups",
        "",
        "| Key | Level | Graded | Record | Avg P | Win% | Cal Err | ROI | CLV Beat | Target | Shrink | Cap | Proven |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    sorted_groups = sorted(
        (payload.get("groups") or {}).items(),
        key=lambda item: (-int((item[1] or {}).get("graded") or 0), item[0]),
    )
    for key, row in sorted_groups[:80]:
        lines.append(
            f"| {key} | {row.get('level')} | {row.get('graded')} | {row.get('record')} | "
            f"{_pct(row.get('avg_model_prob'))} | {_pct(row.get('win_rate'))} | "
            f"{_pct(row.get('calibration_error'), signed=True)} | {_pct(row.get('roi'), signed=True)} | "
            f"{_pct(row.get('clv_beat_rate'))} | {_pct(row.get('target_probability'))} | "
            f"{_pct(row.get('shrink_weight'))} | {_pct(row.get('max_probability'))} | {bool(row.get('proven'))} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], *, model_dir: Path = _MODEL_DIR, report_dir: Path = _REPORT_DIR) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = model_dir / "prop_micro_probability_calibrator.json"
    report_path = report_dir / "mlb_prop_micro_probability_calibrator_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, render(payload))
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop $1 micro probability calibrator")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--date", default=None, help="End date YYYY-MM-DD ET")
    parser.add_argument("--lookback-days", type=int, default=60)
    args = parser.parse_args()
    end = date.fromisoformat(args.date) if args.date else datetime.now(tz=_ET).date()
    payload = build_payload(pg_dsn=args.pg_dsn, end_date=end, lookback_days=args.lookback_days)
    json_path, report_path = write_outputs(payload)
    print(json.dumps({
        "status": payload.get("status"),
        "enabled": payload.get("enabled"),
        "graded_rows": payload.get("graded_rows"),
        "json_path": str(json_path),
        "report_path": str(report_path),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
