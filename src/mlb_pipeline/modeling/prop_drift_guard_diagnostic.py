"""Diagnostic report for MLB prop live-price drift guards.

The drift guard should answer one narrow question: is the current book price
still good enough for the probability source used to rank the play?
"""
from __future__ import annotations

import argparse
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN as _PG_DSN

from .predict_player_props import _load_prop_lines
from .prop_candidate_engine import candidate_from_prediction_row
from .prop_offer_snapshots import minimum_american_price
from .prop_replay import ev_per_unit
from .prop_shadow_selector import (
    SelectorContext,
    ShadowSelectorConfig,
    score_prediction_row,
)

_ET = ZoneInfo("America/New_York")
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fmt_pct(value: Any, *, signed: bool = False) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    return f"{numeric * 100.0:+.1f}%" if signed else f"{numeric * 100.0:.1f}%"


def _fmt_price(value: Any) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    return f"{numeric:+.0f}"


def _fmt_num(value: Any, digits: int = 3) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{digits}f}"


def _active_rows(conn, game_date: date) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                p.*,
                g.start_ts_utc AS game_start_ts_utc
            FROM bets.mlb_prop_predictions p
            LEFT JOIN raw.mlb_games g
              ON g.game_slug = p.game_slug
             AND g.game_date_et = p.game_date_et
            WHERE p.game_date_et = %(game_date)s
              AND COALESCE(p.is_active, true) IS TRUE
              AND p.bet_side IN ('over', 'under')
            ORDER BY p.stat, p.player_name, p.book_line, p.bet_side, p.bookmaker_key
            """,
            {"game_date": game_date},
        )
        return [dict(row) for row in cur.fetchall()]


def _guard_result(
    *,
    current_price: float | None,
    minimum_price: int | None,
    ev_current: float | None,
    required_ev: float,
    stale_status: str | None = None,
) -> str:
    if stale_status:
        return stale_status
    if current_price is None:
        return "missing_current_price"
    if minimum_price is None:
        return "missing_minimum_price"
    if current_price < minimum_price:
        return "price_below_minimum"
    if ev_current is None:
        return "missing_current_ev"
    if ev_current + 1e-9 < required_ev:
        return "current_ev_below_required"
    return "bettable_now"


def _stale_status(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    stale_after = (
        row.get("stale_after_utc")
        or row.get("game_start_ts_utc")
        or row.get("start_ts_utc")
        or row.get("commence_time_utc")
    )
    if stale_after is None:
        return "stale_after_missing", None, None
    try:
        ts = datetime.fromisoformat(str(stale_after).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=ZoneInfo("UTC"))
        ts_utc = ts.astimezone(ZoneInfo("UTC"))
    except Exception:
        return "stale_after_invalid", None, None
    if datetime.now(ZoneInfo("UTC")) > ts_utc:
        status = "stale_after_expired"
    else:
        status = None
    label = ts_utc.astimezone(_ET).strftime("%I:%M %p ET").lstrip("0")
    return status, ts_utc.isoformat(timespec="seconds"), label


def build_payload(
    *,
    game_date: date,
    pg_dsn: str = _PG_DSN,
    model_dir: Path = _MODEL_DIR,
) -> dict[str, Any]:
    cfg = ShadowSelectorConfig(pg_dsn=pg_dsn, report_date=game_date, model_dir=model_dir)
    ctx = SelectorContext(model_dir)
    with psycopg2.connect(pg_dsn) as conn:
        prop_lines = _load_prop_lines(conn, game_date)
        rows = _active_rows(conn, game_date)

    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        item = candidate_from_prediction_row(row, prop_lines)
        if item is None:
            continue
        selector_row = dict(row)
        selector_row["price_drift_ok"] = item.get("price_drift_ok")
        if item.get("current_price") is not None:
            selector_row["bet_price"] = item.get("current_price")
        try:
            selector = score_prediction_row(selector_row, ctx=ctx, cfg=cfg)
        except Exception as exc:
            selector = {"selector_reasons": [f"selector_error:{type(exc).__name__}"]}

        tier = str(selector.get("selector_tier") or row.get("bankroll_tier") or "").lower()
        is_micro = tier == "micro_projection"
        prob = (
            _float(selector.get("micro_projection_prob_side"))
            if is_micro
            else _float(selector.get("selector_prob_side"))
        )
        source = (
            selector.get("micro_projection_prob_source")
            if is_micro
            else selector.get("policy_variant") or "selector"
        )
        if prob is None:
            p_over = _float(row.get("pred_prob_over"))
            side = str(row.get("bet_side") or "").lower()
            prob = p_over if side == "over" else (1.0 - p_over if p_over is not None else None)
            source = source or "model_prob_side"
        required_ev = cfg.micro_projection_min_ev if is_micro else cfg.min_ev
        if is_micro:
            row_required_ev = _float(selector.get("micro_projection_required_ev"))
            if row_required_ev is not None:
                required_ev = row_required_ev
        current_price = _float(item.get("current_price"))
        if current_price is None:
            current_price = _float(row.get("bet_price"))
        stored_min = _float(row.get("minimum_acceptable_price"))
        guard_min = minimum_american_price(prob, required_ev)
        ev_current = ev_per_unit(prob, current_price)
        stale_status, stale_after_utc, stale_after_label = _stale_status(row)
        stored_result = _guard_result(
            current_price=current_price,
            minimum_price=int(stored_min) if stored_min is not None else None,
            ev_current=ev_current,
            required_ev=required_ev,
            stale_status=stale_status,
        )
        guard_result = _guard_result(
            current_price=current_price,
            minimum_price=guard_min,
            ev_current=ev_current,
            required_ev=required_ev,
            stale_status=stale_status,
        )
        diagnostics.append({
            "player": row.get("player_name"),
            "team": row.get("team_abbr"),
            "market": row.get("stat"),
            "side": row.get("bet_side"),
            "line": _float(row.get("book_line")),
            "book": row.get("bookmaker_key"),
            "selector_tier": tier or None,
            "probability_source": source,
            "model_probability": prob,
            "current_price": current_price,
            "stored_minimum_price": stored_min,
            "guard_minimum_price": guard_min,
            "required_ev": required_ev,
            "current_ev": ev_current,
            "stale_after_utc": stale_after_utc,
            "stale_after_label": stale_after_label,
            "stored_drift_result": stored_result,
            "drift_result": guard_result,
            "stored_guard_disagrees": stored_result != guard_result,
            "stored_minimum_disagrees": (
                stored_min is not None
                and guard_min is not None
                and int(round(stored_min)) != int(guard_min)
            ),
            "selector_reasons": selector.get("selector_reasons") or [],
        })

    diagnostics.sort(
        key=lambda row: (
            0 if row.get("selector_tier") == "micro_projection" else 1,
            0 if row.get("drift_result") == "bettable_now" else 1,
            -(_float(row.get("current_ev")) or -999.0),
        )
    )
    return {
        "generated_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds"),
        "game_date": str(game_date),
        "active_rows": len(rows),
        "diagnostic_rows": len(diagnostics),
        "micro_rows": sum(1 for row in diagnostics if row.get("selector_tier") == "micro_projection"),
        "micro_bettable_now": sum(
            1 for row in diagnostics
            if row.get("selector_tier") == "micro_projection"
            and row.get("drift_result") == "bettable_now"
        ),
        "stored_guard_disagreements": sum(1 for row in diagnostics if row.get("stored_guard_disagrees")),
        "stored_minimum_disagreements": sum(1 for row in diagnostics if row.get("stored_minimum_disagrees")),
        "rows": diagnostics,
    }


def write_reports(payload: dict[str, Any], *, top_n: int = 80) -> tuple[Path, Path]:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _MODEL_DIR / "prop_drift_guard_diagnostic.json"
    md_path = _REPORT_DIR / "mlb_prop_drift_guard_diagnostic_latest.md"
    rows = list(payload.get("rows") or [])
    shown = rows[: max(1, top_n)]
    lines = [
        "# MLB Prop Drift Guard Diagnostic",
        "",
        f"Date: {payload.get('game_date')}",
        f"Active rows: {payload.get('active_rows')}",
        f"Diagnostic rows: {payload.get('diagnostic_rows')}",
        f"Micro rows: {payload.get('micro_rows')}",
        f"Micro bettable now: {payload.get('micro_bettable_now')}",
        f"Stored guard disagreements: {payload.get('stored_guard_disagreements')}",
        f"Stored minimum price disagreements: {payload.get('stored_minimum_disagreements')}",
        "",
        "| Player | Market | Side | Line | Book | Tier | Prob | Source | Cur | Stored Min | Guard Min | Current EV | Stale After | Result | Stored Result |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in shown:
        lines.append(
            "| "
            + " | ".join([
                str(row.get("player") or ""),
                str(row.get("market") or ""),
                str(row.get("side") or ""),
                _fmt_num(row.get("line"), 1),
                str(row.get("book") or ""),
                str(row.get("selector_tier") or ""),
                _fmt_pct(row.get("model_probability")),
                str(row.get("probability_source") or ""),
                _fmt_price(row.get("current_price")),
                _fmt_price(row.get("stored_minimum_price")),
                _fmt_price(row.get("guard_minimum_price")),
                _fmt_pct(row.get("current_ev"), signed=True),
                str(row.get("stale_after_label") or ""),
                str(row.get("drift_result") or ""),
                str(row.get("stored_drift_result") or ""),
            ])
            + " |"
        )
    if len(rows) > len(shown):
        lines.extend(["", f"_Showing {len(shown)} of {len(rows)} rows._"])
    atomic_write_json(json_path, payload)
    atomic_write_text(md_path, "\n".join(lines) + "\n")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop drift-guard diagnostic report")
    parser.add_argument("--date", default=None)
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--top-n", type=int, default=80)
    args = parser.parse_args()
    game_date = date.fromisoformat(args.date) if args.date else datetime.now(_ET).date()
    payload = build_payload(
        game_date=game_date,
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
    )
    json_path, md_path = write_reports(payload, top_n=args.top_n)
    print({
        "status": "ok",
        "game_date": str(game_date),
        "micro_rows": payload.get("micro_rows"),
        "micro_bettable_now": payload.get("micro_bettable_now"),
        "json_path": str(json_path),
        "report_path": str(md_path),
    })


if __name__ == "__main__":
    main()
