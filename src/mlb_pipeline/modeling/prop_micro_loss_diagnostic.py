"""Explain why locked $1 micro-projection prop plays are losing or missing CLV."""
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


def _pct(value: Any, digits: int = 1, *, signed: bool = False) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    fmt = f"{{:{'+' if signed else ''}.{digits}%}}"
    return fmt.format(numeric)


def _num(value: Any, digits: int = 3, *, signed: bool = False) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    sign = "+" if signed else ""
    return f"{numeric:{sign}.{digits}f}"


def _drivers(row: dict[str, Any]) -> list[str]:
    meta = row.get("model_meta") or {}
    reasons_text = str(meta.get("selector_reasons") or "")
    reasons = {part.strip() for part in reasons_text.split(";") if part.strip()}
    drivers: list[str] = []
    selector_tier = str(meta.get("selector_tier") or "").lower()
    candidate = _bool(meta.get("micro_projection_candidate"))
    if selector_tier != "micro_projection" or not candidate:
        drivers.append("selector_no_bet_locked")
    clv_prob = _float(meta.get("clv_beat_prob"))
    if clv_prob is None or clv_prob < 0.52 or "clv_model_not_confirming" in reasons:
        drivers.append("clv_model_weak")
    if _bool(row.get("clv_valid")):
        clv_price = _float(row.get("clv_price"))
        if clv_price is not None and clv_price <= 0:
            drivers.append("did_not_beat_close")
    else:
        drivers.append("clv_unknown_or_invalid")
    if (
        "no_bet_negative_roi" in reasons
        or "no_bet_bad_clv" in reasons
        or str(meta.get("residual_bucket_decision") or "").startswith("no_bet")
        or str(meta.get("distribution_bucket_decision") or "").startswith("no_bet")
    ):
        drivers.append("bucket_policy_no_bet")
    bucket_roi = _float(meta.get("bucket_roi"))
    if bucket_roi is not None and bucket_roi < 0:
        drivers.append("bucket_roi_negative")
    bucket_clv = _float(meta.get("bucket_clv_beat_rate"))
    if bucket_clv is not None and bucket_clv < 0.50:
        drivers.append("bucket_clv_weak")
    bucket_avg_clv = _float(meta.get("bucket_avg_clv"))
    if bucket_avg_clv is not None and bucket_avg_clv < 0:
        drivers.append("bucket_avg_clv_negative")
    if "opportunity_not_confirming" in reasons or "micro_projection_opportunity_weak" in reasons:
        drivers.append("opportunity_weak")
    if "tb_hr_line_production_gate_failed" in reasons:
        drivers.append("tb_hr_line_gate_failed")
    if "micro_projection_bookability_weak" in reasons or "bookability_not_confirming" in reasons:
        drivers.append("bookability_weak")
    if "micro_projection_close_capture_weak" in reasons or "close_capture_not_confirming" in reasons:
        drivers.append("close_capture_weak")
    pair_quality = str(meta.get("pair_quality") or "").lower()
    if pair_quality and pair_quality != "same_book":
        drivers.append(f"pair_quality_{pair_quality}")
    if row.get("result_status") == "graded" and row.get("push") is not True:
        won = _bool(row.get("won"))
        model_prob = _float(row.get("model_prob"))
        if model_prob is not None:
            actual_prob = 1.0 if won else 0.0
            if model_prob - actual_prob >= 0.15:
                drivers.append("projection_overconfident")
    return sorted(set(drivers)) or ["no_specific_driver"]


def _summarize(rows: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        keys = key_fn(row)
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            groups[str(key)].append(row)
    out = []
    for key, group in groups.items():
        graded = [row for row in group if row.get("result_status") == "graded"]
        wins = sum(1 for row in graded if _bool(row.get("won")) and not _bool(row.get("push")))
        losses = sum(1 for row in graded if not _bool(row.get("won")) and not _bool(row.get("push")))
        pushes = sum(1 for row in graded if _bool(row.get("push")))
        stake = sum((_float(row.get("stake_usd")) or 1.0) for row in graded)
        units = sum((_float(row.get("profit_units")) or 0.0) for row in graded)
        clv_rows = [row for row in graded if _bool(row.get("clv_valid"))]
        avg_model_prob = [
            _float(row.get("model_prob"))
            for row in graded
            if _float(row.get("model_prob")) is not None and not _bool(row.get("push"))
        ]
        out.append({
            "key": key,
            "rows": len(group),
            "graded": len(graded),
            "pending": sum(1 for row in group if row.get("result_status") == "pending"),
            "record": f"{wins}-{losses}-{pushes}",
            "units": units,
            "roi": units / stake if stake > 0 else None,
            "win_rate": wins / max(1, wins + losses) if wins + losses else None,
            "avg_model_prob": sum(avg_model_prob) / len(avg_model_prob) if avg_model_prob else None,
            "calibration_error": (
                (sum(avg_model_prob) / len(avg_model_prob)) - (wins / max(1, wins + losses))
                if avg_model_prob and wins + losses
                else None
            ),
            "clv_rows": len(clv_rows),
            "clv_beat_rate": (
                sum(1 for row in clv_rows if (_float(row.get("clv_price")) or 0.0) > 0) / len(clv_rows)
                if clv_rows
                else None
            ),
            "avg_clv_price": (
                sum((_float(row.get("clv_price")) or 0.0) for row in clv_rows) / len(clv_rows)
                if clv_rows
                else None
            ),
        })
    out.sort(key=lambda row: (-int(row["graded"]), float(row["roi"] or -999.0), row["key"]))
    return out


def _fetch_rows(conn, start_date: date, end_date: date) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                game_date_et,
                inserted_at_utc,
                player_name,
                market,
                stat,
                side,
                bookmaker_key,
                market_line,
                market_price,
                minimum_acceptable_price,
                model_prob,
                ev,
                model_tier,
                model_meta,
                result_status,
                won,
                push,
                profit_units,
                actual_value,
                closing_price,
                clv_price,
                clv_valid,
                clv_status,
                clv_unknown_reason,
                stake_usd
            FROM bets.mlb_model_pick_ledger
            WHERE source = 'prop'
              AND LOWER(COALESCE(model_tier, '')) = 'micro_projection'
              AND game_date_et BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY game_date_et DESC, inserted_at_utc DESC
            """,
            {"start_date": start_date, "end_date": end_date},
        )
        rows = [dict(row) for row in cur.fetchall()]
    for row in rows:
        row["drivers"] = _drivers(row)
    return rows


def build_payload(*, pg_dsn: str = PG_DSN, end_date: date | None = None, lookback_days: int = 45) -> dict[str, Any]:
    end = end_date or datetime.now(tz=_ET).date()
    start = end - timedelta(days=max(1, int(lookback_days)) - 1)
    with psycopg2.connect(pg_dsn) as conn:
        rows = _fetch_rows(conn, start, end)
    return {
        "generated_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds"),
        "status": "ready",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "lookback_days": int(lookback_days),
        "rows": len(rows),
        "by_driver": _summarize(rows, lambda row: row.get("drivers") or ["no_specific_driver"]),
        "by_market_side": _summarize(rows, lambda row: f"{row.get('market')} | {row.get('side')}"),
        "by_book": _summarize(rows, lambda row: f"{row.get('market')} | {row.get('side')} | {row.get('bookmaker_key')}"),
        "recent_rows": rows[:40],
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Prop Micro Loss Diagnostic",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Window: {payload.get('start_date')} through {payload.get('end_date')}",
        f"Locked micro rows: {payload.get('rows', 0)}",
        "",
        "## Loss / CLV Drivers",
        "",
        "| Driver | Rows | Graded | Pending | Record | ROI | Cal Err | CLV Rows | CLV Beat | Avg CLV |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("by_driver") or []:
        lines.append(
            f"| {row.get('key')} | {row.get('rows')} | {row.get('graded')} | {row.get('pending')} | "
            f"{row.get('record')} | {_pct(row.get('roi'), signed=True)} | {_pct(row.get('calibration_error'), signed=True)} | "
            f"{row.get('clv_rows')} | {_pct(row.get('clv_beat_rate'))} | {_num(row.get('avg_clv_price'), 3, signed=True)} |"
        )
    lines.extend([
        "",
        "## By Market / Side",
        "",
        "| Market / Side | Rows | Graded | Record | ROI | Win Rate | Avg Model P | Cal Err | CLV Beat |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in payload.get("by_market_side") or []:
        lines.append(
            f"| {row.get('key')} | {row.get('rows')} | {row.get('graded')} | {row.get('record')} | "
            f"{_pct(row.get('roi'), signed=True)} | {_pct(row.get('win_rate'))} | {_pct(row.get('avg_model_prob'))} | "
            f"{_pct(row.get('calibration_error'), signed=True)} | {_pct(row.get('clv_beat_rate'))} |"
        )
    lines.extend([
        "",
        "## Recent Locked Micro Rows",
        "",
        "| Date | Player | Market | Side | Line | Price | Result | Actual | CLV | Model P | Drivers |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---|",
    ])
    for row in payload.get("recent_rows") or []:
        result = row.get("result_status")
        if row.get("result_status") == "graded":
            result = "W" if _bool(row.get("won")) else ("P" if _bool(row.get("push")) else "L")
        lines.append(
            f"| {row.get('game_date_et')} | {row.get('player_name')} | {row.get('market')} | {row.get('side')} | "
            f"{_num(row.get('market_line'), 1)} | {_num(row.get('market_price'), 0, signed=True)} | {result} | "
            f"{_num(row.get('actual_value'), 1)} | {_num(row.get('clv_price'), 2, signed=True)} | "
            f"{_pct(row.get('model_prob'))} | {', '.join(row.get('drivers') or [])} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], *, model_dir: Path = _MODEL_DIR, report_dir: Path = _REPORT_DIR) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = model_dir / "prop_micro_loss_diagnostic.json"
    report_path = report_dir / "mlb_prop_micro_loss_diagnostic_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, render(payload))
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose locked MLB prop $1 micro losses and CLV misses")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--date", default=None, help="End date YYYY-MM-DD ET")
    parser.add_argument("--lookback-days", type=int, default=45)
    args = parser.parse_args()
    end = date.fromisoformat(args.date) if args.date else datetime.now(tz=_ET).date()
    payload = build_payload(pg_dsn=args.pg_dsn, end_date=end, lookback_days=args.lookback_days)
    json_path, report_path = write_outputs(payload)
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows", 0),
        "json_path": str(json_path),
        "report_path": str(report_path),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
