"""Daily bettable-now scan for approved MLB prop micro trials.

This script is intentionally narrower than the shadow selector. The selector
scores every prop row; this scan asks whether an already-approved/near-approved
bucket is actually bettable at the current price right now.
"""
from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN as _PG_DSN

from .predict_player_props import _load_prop_lines
from .prop_candidate_engine import candidate_from_prediction_row, normalize_name
from .prop_drift_guard_diagnostic import _active_rows, _guard_result, _stale_status
from .prop_offer_snapshots import minimum_american_price
from .prop_replay import ev_per_unit
from .prop_shadow_selector import SelectorContext, ShadowSelectorConfig, exact_bucket_key, score_prediction_row
from .side_recalibration import prop_line_bucket

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


def _fmt_num(value: Any, digits: int = 1) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{digits}f}"


def _safe_text(value: Any, *, limit: int = 180) -> str:
    text = str(value or "").replace("|", "/").replace("\n", " ")
    return text[:limit]


def _same_line(a: Any, b: Any) -> bool:
    av = _float(a)
    bv = _float(b)
    return av is not None and bv is not None and abs(av - bv) <= 1e-9


def _load_focus_current_offers(conn, game_date: date) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('features.mlb_prop_offer_links')")
        if cur.fetchone()[0] is None:
            return {"by_id": {}, "by_key": {}}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                id,
                source_row_id,
                player_name_norm,
                stat,
                side,
                line::float AS line,
                LOWER(bookmaker_key) AS bookmaker_key,
                price,
                link,
                is_linkable,
                fetched_at_utc,
                updated_at_utc,
                refreshed_at_utc
            FROM features.mlb_prop_offer_links
            WHERE as_of_date = %(game_date)s
              AND (
                    (stat = 'pitcher_strikeouts' AND side = 'under' AND LOWER(bookmaker_key) = 'draftkings')
                 OR (stat = 'batter_total_bases' AND side = 'over' AND LOWER(bookmaker_key) = 'draftkings' AND ABS(line::float - 1.5) <= 1e-9)
                 OR (stat = 'batter_home_runs' AND side = 'over' AND ABS(line::float - 0.5) <= 1e-9)
              )
            """,
            {"game_date": game_date},
        )
        cols = [desc[0] for desc in cur.description]
        rows = [dict(zip(cols, values)) for values in cur.fetchall()]
    by_id: dict[str, dict[str, Any]] = {}
    by_key: dict[tuple[str, str, str, float, str], dict[str, Any]] = {}
    for offer in rows:
        if offer.get("id") is not None:
            by_id[str(offer["id"])] = offer
        key = (
            str(offer.get("player_name_norm") or ""),
            str(offer.get("stat") or ""),
            str(offer.get("side") or "").lower(),
            round(float(offer.get("line") or 0.0), 3),
            str(offer.get("bookmaker_key") or "").lower(),
        )
        current = by_key.get(key)
        # For American odds, the numerically larger price is better for the bettor
        # on either side (+150 beats +130; -120 beats -140).
        if current is None or (_float(offer.get("price")) or -99999.0) > (_float(current.get("price")) or -99999.0):
            by_key[key] = offer
    return {"by_id": by_id, "by_key": by_key}


def _current_offer_for_row(row: dict[str, Any], current_offers: dict[str, Any] | None) -> dict[str, Any] | None:
    current_offers = current_offers or {}
    prop_offer_id = row.get("prop_offer_id")
    by_id = current_offers.get("by_id") or {}
    if prop_offer_id is not None:
        exact = by_id.get(str(prop_offer_id))
        if exact is not None:
            return exact
    line = _float(row.get("book_line"))
    if line is None:
        return None
    key = (
        normalize_name(str(row.get("player_name") or "")),
        str(row.get("stat") or ""),
        str(row.get("bet_side") or "").lower(),
        round(float(line), 3),
        str(row.get("bookmaker_key") or "").lower(),
    )
    offer = (current_offers.get("by_key") or {}).get(key)
    if offer and _same_line(offer.get("line"), row.get("book_line")):
        return offer
    return None


def _fast_item_from_prediction(
    row: dict[str, Any],
    *,
    current_offer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_offer = current_offer or {}
    return {
        "link": current_offer.get("link") or row.get("bet_link"),
        "current_price": _float(
            current_offer.get("price")
            if current_offer.get("price") is not None
            else row.get("bet_price")
        ),
        "price_drift_ok": True,
        "current_offer_id": current_offer.get("id"),
        "current_offer_source_row_id": current_offer.get("source_row_id"),
    }


def _is_focus_source_row(row: dict[str, Any]) -> bool:
    market = str(row.get("stat") or "")
    side = str(row.get("bet_side") or "").lower()
    book = str(row.get("bookmaker_key") or "").lower()
    line = _float(row.get("book_line"))
    if market == "pitcher_strikeouts" and side == "under" and book == "draftkings":
        return True
    if (
        market == "batter_total_bases"
        and side == "over"
        and book == "draftkings"
        and line is not None
        and abs(line - 1.5) <= 1e-9
    ):
        return True
    if (
        market == "batter_home_runs"
        and side == "over"
        and line is not None
        and abs(line - 0.5) <= 1e-9
    ):
        return True
    return False


def _market_label(market: str) -> str:
    return {
        "pitcher_strikeouts": "K",
        "batter_hits": "H",
        "batter_total_bases": "TB",
        "batter_home_runs": "HR",
    }.get(market, market)


def _approved_or_near_status(row: dict[str, Any], selector: dict[str, Any]) -> tuple[str, str]:
    market = str(selector.get("market") or row.get("stat") or "")
    side = str(selector.get("side") or row.get("bet_side") or "").lower()
    book = str(selector.get("bookmaker_key") or row.get("bookmaker_key") or "").lower()
    pair_quality = str(selector.get("pair_quality") or row.get("pair_quality") or "").lower()
    surface = str(selector.get("line_surface") or row.get("line_surface") or "unknown")
    line = _float(selector.get("line") if selector.get("line") is not None else row.get("book_line"))
    line_bucket = str(selector.get("line_bucket") or row.get("line_bucket") or prop_line_bucket(market, line))
    model_family = str(row.get("model_family") or "unknown").lower()

    if _bool(selector.get("micro_approved_model")):
        return "approved_micro", str(selector.get("micro_approved_model_reason") or "approved_model")

    if (
        market == "pitcher_strikeouts"
        and side == "under"
        and book == "draftkings"
        and pair_quality == "same_book"
        and surface == "common"
    ):
        if _bool(selector.get("k_under_repair_micro_allowed")):
            return "approved_micro", "dk_k_under_repair_gate"
        return "near_approved", "dk_k_under_waiting_on_repair_gate"

    if (
        market == "batter_total_bases"
        and side == "over"
        and book == "draftkings"
        and pair_quality == "same_book"
        and surface == "common"
        and line is not None
        and abs(line - 1.5) <= 1e-9
    ):
        if model_family == "tb_tail_state":
            return "near_approved", "dk_tb15_tb_tail_state_not_micro_candidate"
        return "near_approved", "dk_tb15_waiting_on_tb_tail_state_family"

    if (
        market == "batter_home_runs"
        and side == "over"
        and pair_quality == "same_book"
        and surface == "common"
        and line_bucket == "HR 0.5"
    ):
        return "near_approved", "hr05_true_pair_evidence_watch"

    return "not_focus", "not_approved_or_near_approved_bucket"


def _stored_selector_from_prediction(row: dict[str, Any]) -> dict[str, Any] | None:
    if (
        row.get("selector_tier") is None
        and row.get("micro_projection_prob_side") is None
        and row.get("pair_quality") is None
    ):
        return None
    selector = {
        "market": row.get("stat"),
        "side": row.get("bet_side"),
        "line": row.get("book_line"),
        "price": row.get("bet_price"),
        "bookmaker_key": row.get("bookmaker_key"),
        "bucket_key": row.get("bucket_key") or exact_bucket_key(row),
        "line_bucket": row.get("line_bucket"),
        "price_bucket": row.get("price_bucket"),
    }
    copy_keys = [
        "selector_tier",
        "selector_score",
        "selector_reasons",
        "selector_ev",
        "selector_prob_side",
        "policy_variant",
        "pair_quality",
        "market_prob_source",
        "micro_projection_candidate",
        "micro_projection_prob_side",
        "micro_projection_prob_source",
        "micro_projection_edge",
        "micro_projection_ev",
        "micro_projection_required_ev",
        "micro_approved_model",
        "micro_approved_model_key",
        "micro_approved_model_reason",
        "micro_relaxed_trial_lane",
        "micro_truth_filter_status",
        "micro_truth_filter_reason",
        "micro_truth_filter_record",
        "micro_truth_filter_graded",
        "micro_truth_filter_roi",
        "micro_truth_filter_clv_rows",
        "micro_truth_filter_clv_beat_rate",
        "micro_truth_filter_avg_clv",
        "k_under_repair_gate_key",
        "k_under_repair_micro_allowed",
        "k_under_repair_blockers",
    ]
    for key in copy_keys:
        selector[key] = row.get(key)
    return selector


def _scan_status(
    *,
    approval_status: str,
    selector: dict[str, Any],
    link: str | None,
    current_price: float | None,
    current_ev: float | None,
    minimum_price: int | None,
    required_ev: float,
    stale_status: str | None,
) -> str:
    if approval_status != "approved_micro":
        return approval_status
    if str(selector.get("pair_quality") or "").lower() != "same_book":
        return "not_true_paired"
    if not link:
        return "bet_link_missing"
    if current_price is None:
        return "current_price_missing"
    if current_ev is None or current_ev <= 0.0:
        return "current_ev_not_positive"
    guard = _guard_result(
        current_price=current_price,
        minimum_price=minimum_price,
        ev_current=current_ev,
        required_ev=required_ev,
        stale_status=stale_status,
    )
    if guard != "bettable_now":
        return guard
    return "bettable_now"


def _score_active_row(
    row: dict[str, Any],
    *,
    prop_lines: dict | None,
    current_offers: dict[str, Any] | None,
    ctx: SelectorContext,
    cfg: ShadowSelectorConfig,
    force_rescore: bool = False,
) -> dict[str, Any] | None:
    if force_rescore:
        item = candidate_from_prediction_row(row, prop_lines or {})
        if item is None:
            return None
    else:
        item = _fast_item_from_prediction(
            row,
            current_offer=_current_offer_for_row(row, current_offers),
        )
    selector = None if force_rescore else _stored_selector_from_prediction(row)
    if selector is None:
        selector_row = dict(row)
        selector_row["price_drift_ok"] = item.get("price_drift_ok")
        if item.get("current_price") is not None:
            selector_row["bet_price"] = item.get("current_price")
        try:
            selector = score_prediction_row(selector_row, ctx=ctx, cfg=cfg)
        except Exception as exc:
            selector = {
                "selector_tier": None,
                "selector_reasons": [f"selector_error:{type(exc).__name__}"],
                "pair_quality": row.get("pair_quality"),
            }

    approval_status, approval_reason = _approved_or_near_status(row, selector)
    if approval_status == "not_focus":
        return None

    prob = _float(selector.get("micro_projection_prob_side"))
    if prob is None:
        prob = _float(selector.get("selector_prob_side"))
    required_ev = _float(selector.get("micro_projection_required_ev"))
    if required_ev is None:
        required_ev = 0.0 if approval_status == "approved_micro" else cfg.micro_projection_min_ev
    current_price = _float(item.get("current_price"))
    if current_price is None:
        current_price = _float(row.get("bet_price"))
    minimum_price = minimum_american_price(prob, required_ev)
    current_ev = ev_per_unit(prob, current_price)
    stale_status, stale_after_utc, stale_after_label = _stale_status(row)
    status = _scan_status(
        approval_status=approval_status,
        selector=selector,
        link=item.get("link"),
        current_price=current_price,
        current_ev=current_ev,
        minimum_price=minimum_price,
        required_ev=required_ev,
        stale_status=stale_status,
    )
    market = str(selector.get("market") or row.get("stat") or "")
    line = _float(selector.get("line") if selector.get("line") is not None else row.get("book_line"))
    side = str(selector.get("side") or row.get("bet_side") or "").lower()
    bucket_key = selector.get("bucket_key") or exact_bucket_key(row)
    rec = {
        "player": row.get("player_name"),
        "team": row.get("team_abbr"),
        "prediction_key": row.get("prediction_key"),
        "prop_offer_id": row.get("prop_offer_id"),
        "current_offer_id": item.get("current_offer_id"),
        "current_offer_source_row_id": item.get("current_offer_source_row_id"),
        "market": market,
        "market_label": _market_label(market),
        "side": side,
        "line": line,
        "book": str(selector.get("bookmaker_key") or row.get("bookmaker_key") or "").lower(),
        "bucket_key": bucket_key,
        "line_bucket": selector.get("line_bucket"),
        "price_bucket": selector.get("price_bucket"),
        "model_family": row.get("model_family"),
        "approval_status": approval_status,
        "approval_reason": approval_reason,
        "scan_status": status,
        "selector_tier": selector.get("selector_tier"),
        "micro_projection_candidate": _bool(selector.get("micro_projection_candidate")),
        "micro_approved_model": _bool(selector.get("micro_approved_model")),
        "micro_approved_model_key": selector.get("micro_approved_model_key"),
        "micro_approved_model_reason": selector.get("micro_approved_model_reason"),
        "micro_relaxed_trial_lane": _bool(selector.get("micro_relaxed_trial_lane")),
        "micro_probability": prob,
        "micro_probability_source": selector.get("micro_projection_prob_source") or selector.get("policy_variant"),
        "micro_edge": _float(selector.get("micro_projection_edge")),
        "current_price": current_price,
        "minimum_acceptable_price": minimum_price,
        "required_ev": required_ev,
        "current_ev": current_ev,
        "drift_guard_result": status if status != approval_status else None,
        "stale_after_utc": stale_after_utc,
        "stale_after_label": stale_after_label,
        "has_bet_link": bool(item.get("link")),
        "bet_link": item.get("link"),
        "pair_quality": selector.get("pair_quality"),
        "market_prob_source": selector.get("market_prob_source"),
        "k_under_repair_gate_key": selector.get("k_under_repair_gate_key"),
        "k_under_repair_micro_allowed": _bool(selector.get("k_under_repair_micro_allowed")),
        "k_under_repair_blockers": selector.get("k_under_repair_blockers"),
        "micro_truth_filter_status": selector.get("micro_truth_filter_status"),
        "micro_truth_filter_reason": selector.get("micro_truth_filter_reason"),
        "micro_truth_filter_record": selector.get("micro_truth_filter_record"),
        "micro_truth_filter_graded": selector.get("micro_truth_filter_graded"),
        "micro_truth_filter_roi": selector.get("micro_truth_filter_roi"),
        "micro_truth_filter_clv_rows": selector.get("micro_truth_filter_clv_rows"),
        "micro_truth_filter_clv_beat_rate": selector.get("micro_truth_filter_clv_beat_rate"),
        "micro_truth_filter_avg_clv": selector.get("micro_truth_filter_avg_clv"),
        "selector_score": selector.get("selector_score"),
        "selector_reasons": selector.get("selector_reasons") or [],
    }
    return rec


def build_payload(
    *,
    game_date: date,
    pg_dsn: str = _PG_DSN,
    model_dir: Path = _MODEL_DIR,
    max_props: int = 5,
    force_rescore: bool = False,
) -> dict[str, Any]:
    max_props = max(0, min(5, int(max_props)))
    cfg = ShadowSelectorConfig(pg_dsn=pg_dsn, report_date=game_date, model_dir=model_dir)
    ctx = SelectorContext(model_dir)
    with psycopg2.connect(pg_dsn) as conn:
        prop_lines = _load_prop_lines(conn, game_date) if force_rescore else None
        current_offers = _load_focus_current_offers(conn, game_date) if not force_rescore else None
        rows = _active_rows(conn, game_date)

    focus_source_rows = [row for row in rows if _is_focus_source_row(row)]
    scanned = [
        rec for row in focus_source_rows
        if (
            rec := _score_active_row(
                row,
                prop_lines=prop_lines,
                current_offers=current_offers,
                ctx=ctx,
                cfg=cfg,
                force_rescore=force_rescore,
            )
        ) is not None
    ]
    bettable = [row for row in scanned if row.get("scan_status") == "bettable_now"]
    bettable.sort(
        key=lambda row: (
            _float(row.get("current_ev")) or -999.0,
            _float(row.get("micro_edge")) or -999.0,
            _float(row.get("micro_probability")) or -999.0,
        ),
        reverse=True,
    )
    cap_ids = {id(row): rank + 1 for rank, row in enumerate(bettable[:max_props])}
    all_bettable_ids = {id(row): rank + 1 for rank, row in enumerate(bettable)}
    for row in scanned:
        rank = all_bettable_ids.get(id(row))
        row["micro_bettable_rank"] = rank
        row["inside_micro_cap"] = id(row) in cap_ids
        if row.get("scan_status") == "bettable_now" and rank and rank > max_props:
            row["scan_status"] = "outside_micro_cap"

    status_counts = Counter(str(row.get("scan_status") or "unknown") for row in scanned)
    approval_counts = Counter(str(row.get("approval_status") or "unknown") for row in scanned)
    return {
        "generated_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds"),
        "game_date": game_date.isoformat(),
        "active_prediction_rows": len(rows),
        "focus_source_rows": len(focus_source_rows),
        "force_rescore": force_rescore,
        "focus_rows": len(scanned),
        "approved_rows": sum(1 for row in scanned if row.get("approval_status") == "approved_micro"),
        "near_approved_rows": sum(1 for row in scanned if row.get("approval_status") == "near_approved"),
        "bettable_now_before_cap": len(bettable),
        "bettable_now_inside_cap": sum(1 for row in scanned if row.get("inside_micro_cap")),
        "max_props": max_props,
        "status_counts": dict(status_counts.most_common()),
        "approval_counts": dict(approval_counts.most_common()),
        "rows": scanned,
    }


def _render_table(rows: list[dict[str, Any]], *, include_link: bool = False, top_n: int = 40) -> list[str]:
    lines = [
        "| Player | Market | Side | Line | Book | Model | Prob | Cur | Min | EV | Record | ROI | CLV Beat | Avg CLV | Status | Why |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for row in rows[:top_n]:
        link = ""
        if include_link and row.get("bet_link"):
            link = f" [bet](<{row.get('bet_link')}>)"
        lines.append(
            "| "
            + " | ".join([
                _safe_text(f"{row.get('player') or ''}{link}", limit=120),
                _safe_text(row.get("market_label") or row.get("market")),
                _safe_text(row.get("side")),
                _fmt_num(row.get("line"), 1),
                _safe_text(row.get("book")),
                _safe_text(row.get("model_family")),
                _fmt_pct(row.get("micro_probability")),
                _fmt_price(row.get("current_price")),
                _fmt_price(row.get("minimum_acceptable_price")),
                _fmt_pct(row.get("current_ev"), signed=True),
                _safe_text(row.get("micro_truth_filter_record"), limit=40),
                _fmt_pct(row.get("micro_truth_filter_roi"), signed=True),
                _fmt_pct(row.get("micro_truth_filter_clv_beat_rate")),
                _fmt_price(row.get("micro_truth_filter_avg_clv")),
                _safe_text(row.get("scan_status"), limit=60),
                _safe_text(row.get("approval_reason") or row.get("selector_reasons"), limit=140),
            ])
            + " |"
        )
    return lines


def write_reports(payload: dict[str, Any], *, top_n: int = 80) -> tuple[Path, Path]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _MODEL_DIR / "prop_bettable_now_scan.json"
    report_path = _REPORT_DIR / "mlb_prop_bettable_now_scan_latest.md"
    rows = list(payload.get("rows") or [])
    bettable = [row for row in rows if row.get("inside_micro_cap")]
    approved = [row for row in rows if row.get("approval_status") == "approved_micro" and not row.get("inside_micro_cap")]
    near = [row for row in rows if row.get("approval_status") == "near_approved"]
    approved.sort(
        key=lambda row: (
            0 if row.get("scan_status") == "outside_micro_cap" else 1,
            -(_float(row.get("current_ev")) or -999.0),
        )
    )
    near.sort(key=lambda row: -(_float(row.get("current_ev")) or -999.0))
    lines = [
        "# MLB Prop Bettable-Now Scan",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Date: {payload.get('game_date')}",
        "",
        f"- Active prop prediction rows: {payload.get('active_prediction_rows')}",
        f"- Focus source rows scanned: {payload.get('focus_source_rows')}",
        f"- Force selector rescore: {payload.get('force_rescore')}",
        f"- Approved/near-approved focus rows: {payload.get('focus_rows')}",
        f"- Approved micro rows: {payload.get('approved_rows')}",
        f"- Near-approved rows: {payload.get('near_approved_rows')}",
        f"- Bettable now before cap: {payload.get('bettable_now_before_cap')}",
        f"- Inside $1 micro cap: {payload.get('bettable_now_inside_cap')} of {payload.get('max_props')}",
        "",
        "## Status Counts",
        "",
        "| Status | Rows |",
        "|---|---:|",
    ]
    for status, count in (payload.get("status_counts") or {}).items():
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## $1 Micro Test - Bettable Now", ""])
    if bettable:
        lines.extend(_render_table(bettable, include_link=True, top_n=top_n))
    else:
        lines.append("No approved micro-test props are bettable at the current price.")
    lines.extend(["", "## Approved But Not Bettable", ""])
    if approved:
        lines.extend(_render_table(approved, include_link=False, top_n=top_n))
    else:
        lines.append("No approved micro rows were blocked by price/link/stale checks.")
    lines.extend(["", "## Near-Approved Watch", ""])
    if near:
        lines.extend(_render_table(near, include_link=False, top_n=top_n))
    else:
        lines.append("No near-approved watch rows.")

    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, "\n".join(lines) + "\n")
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan approved/near-approved MLB prop buckets for bettable-now micro plays")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--max-props", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--force-rescore", action="store_true")
    args = parser.parse_args()
    game_date = date.fromisoformat(args.date) if args.date else datetime.now(_ET).date()
    max_props = args.max_props
    if max_props is None:
        max_props = int(os.getenv("MLB_PROJECTION_MICRO_MAX_PROPS", "5"))
    payload = build_payload(
        game_date=game_date,
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        max_props=max_props,
        force_rescore=args.force_rescore,
    )
    json_path, report_path = write_reports(payload, top_n=args.top_n)
    print({
        "status": "ok",
        "game_date": payload.get("game_date"),
        "approved_rows": payload.get("approved_rows"),
        "near_approved_rows": payload.get("near_approved_rows"),
        "bettable_now_before_cap": payload.get("bettable_now_before_cap"),
        "bettable_now_inside_cap": payload.get("bettable_now_inside_cap"),
        "json_path": str(json_path),
        "report_path": str(report_path),
    })


if __name__ == "__main__":
    main()
