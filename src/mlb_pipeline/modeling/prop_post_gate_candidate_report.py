"""Show which active props would qualify after the patched micro/bankroll gates."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

from .predict_player_props import _load_prop_lines
from .prop_candidate_engine import _current_exact_offer
from .prop_shadow_selector import SelectorContext, ShadowSelectorConfig, score_prediction_row
from .bankroll_ledger import _normalize_name

_ET = ZoneInfo("America/New_York")
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

_MICRO_BLOCKER_PREFIXES = (
    "micro_projection_",
    "clv_model_not_confirming",
    "clv_models_disagree_or_weak",
    "expected_clv_not_positive",
    "bucket_roi_negative",
    "bucket_clv_beat_low",
    "bucket_avg_clv_negative",
    "no_bet_",
    "distribution_bucket_no_bet",
    "tb_hr_line_production_gate_failed",
)


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _active_prediction_rows(conn, game_date: date) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT p.*, g.start_ts_utc AS start_ts_utc
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


def _drift_current(row: dict[str, Any], prop_lines) -> dict[str, Any]:
    side = str(row.get("bet_side") or row.get("side") or "").lower()
    line_data = (prop_lines or {}).get((_normalize_name(row.get("player_name") or ""), row.get("stat")), {})
    offer = _current_exact_offer(
        line_data,
        prop_offer_id=row.get("prop_offer_id"),
        side=side,
        line=row.get("book_line"),
        bookmaker_key=row.get("bookmaker_key"),
    )
    if not offer:
        return {"current_offer_available": False}
    return {
        "current_offer_available": True,
        "current_price": _float(offer.get("price")),
        "current_link_present": bool(offer.get("link")),
        "current_bookmaker_key": offer.get("bookmaker_key"),
    }


def _blockers(reasons: list[str]) -> list[str]:
    out = []
    for reason in reasons:
        text = str(reason)
        if text.startswith(_MICRO_BLOCKER_PREFIXES) or text in _MICRO_BLOCKER_PREFIXES:
            out.append(text)
    return sorted(set(out)) or sorted(set(str(r) for r in reasons[:4]))


def _load_player_game_bankroll_proof(model_dir: Path = _MODEL_DIR) -> dict[str, dict[str, Any]]:
    path = model_dir / "prop_player_game_bankroll_model_proof.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        payload = {}
    return {
        str(row.get("stat")): row
        for row in payload.get("rows") or []
        if row.get("stat")
    }


def _score_closeness(row: dict[str, Any]) -> float:
    blockers = row.get("post_gate_blockers") or []
    selector_score = _float(row.get("selector_score")) or -2.0
    clv = _float(row.get("clv_beat_prob"))
    ev = _float(row.get("micro_projection_ev")) or -1.0
    return (
        -5.0 * len(blockers)
        + selector_score
        + 0.75 * (clv if clv is not None else 0.0)
        + 0.35 * ev
    )


def build(*, pg_dsn: str = PG_DSN, game_date: date | None = None) -> dict[str, Any]:
    et_date = game_date or datetime.now(tz=_ET).date()
    with psycopg2.connect(pg_dsn) as conn:
        prop_lines = _load_prop_lines(conn, et_date)
        rows = _active_prediction_rows(conn, et_date)
    ctx = SelectorContext(_MODEL_DIR)
    cfg = ShadowSelectorConfig(pg_dsn=pg_dsn, report_date=et_date, model_dir=_MODEL_DIR)
    player_game_proof = _load_player_game_bankroll_proof(_MODEL_DIR)
    scored: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        drift = _drift_current(row, prop_lines)
        if drift.get("current_price") is not None:
            row["bet_price"] = drift["current_price"]
        row["price_drift_ok"] = bool(drift.get("current_offer_available", True))
        score = score_prediction_row(row, ctx=ctx, cfg=cfg)
        reasons = [str(r) for r in (score.get("selector_reasons") or [])]
        market = str(score.get("market") or row.get("stat") or "")
        proof_rec = player_game_proof.get(market) or {}
        projection_proof_allowed = bool(
            proof_rec.get("projection_micro_allowed")
            or proof_rec.get("projection_eligible")
        )
        bankroll_training_allowed = bool(proof_rec.get("exact_line_bankroll_training_allowed"))
        positive_projection = (
            (_float(score.get("micro_projection_ev")) or -999.0) >= cfg.micro_projection_min_ev
            and (_float(score.get("micro_projection_edge")) or -999.0) >= cfg.micro_projection_min_prob_edge
        )
        post_gate_blockers = _blockers(reasons)
        if player_game_proof and not projection_proof_allowed:
            post_gate_blockers.append("player_game_projection_proof_not_passed")
            post_gate_blockers.extend(str(value) for value in (proof_rec.get("projection_blockers") or []))
            post_gate_blockers = sorted(set(post_gate_blockers))
        rec = {
            **score,
            **drift,
            "positive_projection_micro_edge": positive_projection,
            "player_game_projection_allowed": projection_proof_allowed,
            "player_game_bankroll_training_allowed": bankroll_training_allowed,
            "player_game_bankroll_blockers": proof_rec.get("blockers") or [],
            "post_gate_blockers": post_gate_blockers,
        }
        rec["closeness_score"] = _score_closeness(rec)
        scored.append(rec)
    micro = [row for row in scored if row.get("selector_tier") == "micro_projection"]
    real = [row for row in scored if row.get("selector_real_candidate")]
    almost = [
        row for row in scored
        if row.get("positive_projection_micro_edge")
        and row.get("selector_tier") != "micro_projection"
        and not row.get("selector_real_candidate")
    ]
    almost.sort(key=lambda row: row.get("closeness_score", -999.0), reverse=True)
    blocker_counts = Counter()
    for row in almost:
        blocker_counts.update(row.get("post_gate_blockers") or [])
    payload = {
        "generated_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds"),
        "status": "ready",
        "game_date": et_date.isoformat(),
        "active_rows": len(scored),
        "real_candidate_rows": len(real),
        "micro_projection_rows": len(micro),
        "positive_projection_rows": sum(1 for row in scored if row.get("positive_projection_micro_edge")),
        "post_gate_near_miss_rows": len(almost),
        "blocker_counts": dict(blocker_counts.most_common()),
        "micro_rows": micro[:40],
        "near_misses": almost[:80],
    }
    _write_outputs(payload)
    return payload


def _fmt(value: Any, digits: int = 3, *, signed: bool = False) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    sign = "+" if signed else ""
    return f"{numeric:{sign}.{digits}f}"


def _pct(value: Any, *, signed: bool = False) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    sign = "+" if signed else ""
    return f"{numeric:{sign}.1%}"


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Prop Post-Gate Candidate Report",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Date: {payload.get('game_date')}",
        "",
        f"- Active rows: {payload.get('active_rows', 0)}",
        f"- Real candidates: {payload.get('real_candidate_rows', 0)}",
        f"- Micro projection rows after patched gates: {payload.get('micro_projection_rows', 0)}",
        f"- Positive projection rows before patched gates: {payload.get('positive_projection_rows', 0)}",
        f"- Post-gate near misses: {payload.get('post_gate_near_miss_rows', 0)}",
        "",
        "## Top Blockers",
        "",
        "| Blocker | Rows |",
        "|---|---:|",
    ]
    for blocker, count in (payload.get("blocker_counts") or {}).items():
        lines.append(f"| {blocker} | {count} |")
    lines.extend([
        "",
        "## Micro Rows After Patched Gates",
        "",
        "| Player | Market | Side | Line | Book | Price | P | EV | CLV | Micro CLV | Stat Proof | Score |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|",
    ])
    if payload.get("micro_rows"):
        for row in payload.get("micro_rows") or []:
            lines.append(
                f"| {row.get('player_name')} | {row.get('market')} | {row.get('side')} | {_fmt(row.get('line'), 1)} | "
                f"{row.get('bookmaker_key')} | {_fmt(row.get('price'), 0, signed=True)} | "
                f"{_pct(row.get('micro_projection_prob_side'))} | {_pct(row.get('micro_projection_ev'), signed=True)} | "
                f"{_pct(row.get('clv_beat_prob'))} | {_pct(row.get('micro_clv_beat_prob'))} | "
                f"{bool(row.get('player_game_projection_allowed'))} | {_fmt(row.get('selector_score'))} |"
            )
    else:
        lines.append("| _No rows_ |  |  |  |  |  |  |  |  |  |  |  |")
    lines.extend([
        "",
        "## Near Misses",
        "",
        "| Player | Market | Side | Line | Book | Price | Micro P | Micro EV | CLV | Micro CLV | Stat Proof | Score | Blockers |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---|",
    ])
    for row in payload.get("near_misses") or []:
        lines.append(
            f"| {row.get('player_name')} | {row.get('market')} | {row.get('side')} | {_fmt(row.get('line'), 1)} | "
            f"{row.get('bookmaker_key')} | {_fmt(row.get('price'), 0, signed=True)} | "
            f"{_pct(row.get('micro_projection_prob_side'))} | {_pct(row.get('micro_projection_ev'), signed=True)} | "
            f"{_pct(row.get('clv_beat_prob'))} | {_pct(row.get('micro_clv_beat_prob'))} | "
            f"{bool(row.get('player_game_projection_allowed'))} | {_fmt(row.get('selector_score'))} | "
            f"{', '.join(row.get('post_gate_blockers') or []) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def _write_outputs(payload: dict[str, Any]) -> tuple[Path, Path]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _MODEL_DIR / "prop_post_gate_candidate_report.json"
    report_path = _REPORT_DIR / "mlb_prop_post_gate_candidate_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, _render(payload))
    payload["json_path"] = str(json_path)
    payload["report_path"] = str(report_path)
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build post-gate MLB prop candidate near-miss report")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    game_date = date.fromisoformat(args.date) if args.date else datetime.now(tz=_ET).date()
    payload = build(pg_dsn=args.pg_dsn, game_date=game_date)
    print(json.dumps({
        "status": payload.get("status"),
        "active_rows": payload.get("active_rows"),
        "micro_projection_rows": payload.get("micro_projection_rows"),
        "near_misses": payload.get("post_gate_near_miss_rows"),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
