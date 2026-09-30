"""Lock currently bettable $1 micro-projection prop rows into the model ledger."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

from .ai_pick_engine import (
    ai_pick_to_prop_prediction_row,
    load_ai_pick_rows,
    refresh_ai_pick_engine,
)
from .bankroll_ledger import _cfg_thresholds, _clean_int, _normalize_name, _pick_key
from .model_pick_ledger import (
    _apply_selector_for_micro_ledger,
    _attach_prop_game_start_times,
    _clean_bool,
    _clean_float,
    _prop_price_drift_status,
    insert_model_pick_rows,
    insert_prop_model_pick_ledger,
)
from .predict_player_props import PredictConfig, _load_prop_lines
from .external_pick_ledger import attach_external_agreement
from .prop_offer_snapshots import minimum_american_price
from .prop_shadow_selector import SelectorContext, ShadowSelectorConfig, exact_bucket_key

_ET = ZoneInfo("America/New_York")
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_TARGET_MICRO_BUCKETS = {
    "pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings",
    "batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings",
    "batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings",
}


def _active_prediction_rows(conn, game_date: date) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                p.*,
                g.start_ts_utc AS start_ts_utc
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
        rows = [dict(row) for row in cur.fetchall()]
    try:
        attach_external_agreement(conn, rows, game_date=game_date)
    except Exception:
        pass
    return rows


def _ledger_summary(conn, game_date: date) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                COALESCE(model_tier, 'unknown') AS model_tier,
                COUNT(*)::int AS rows,
                COALESCE(SUM(stake_usd), 0)::float AS stake_usd
            FROM bets.mlb_model_pick_ledger
            WHERE source = 'prop'
              AND game_date_et = %(game_date)s
            GROUP BY COALESCE(model_tier, 'unknown')
            ORDER BY model_tier
            """,
            {"game_date": game_date},
        )
        return [dict(row) for row in cur.fetchall()]


def _micro_ledger_keys(conn, game_date: date) -> dict[str, set[Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT prop_offer_id, prediction_key
            FROM bets.mlb_model_pick_ledger
            WHERE source = 'prop'
              AND game_date_et = %(game_date)s
              AND (
                LOWER(COALESCE(model_tier, '')) = 'micro_projection'
                OR LOWER(COALESCE(model_meta->>'selector_tier', '')) = 'micro_projection'
              )
            """,
            {"game_date": game_date},
        )
        rows = [dict(row) for row in cur.fetchall()]
    return {
        "prop_offer_ids": {row["prop_offer_id"] for row in rows if row.get("prop_offer_id") is not None},
        "prediction_keys": {row["prediction_key"] for row in rows if row.get("prediction_key")},
    }


def _ai_meta(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("model_meta")
    return meta if isinstance(meta, dict) else {}


def _ai_bucket_key(row: dict[str, Any]) -> str | None:
    meta = _ai_meta(row)
    bucket = meta.get("bucket_key")
    if bucket:
        return str(bucket)
    try:
        return exact_bucket_key(ai_pick_to_prop_prediction_row(row))
    except Exception:
        return None


def _ai_score(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _clean_float(row.get("ranking_score")) or -999.0,
        _clean_float(row.get("current_ev")) or _clean_float(row.get("ev")) or -999.0,
        _clean_float(row.get("model_prob")) or -999.0,
    )


def _warning_text(row: dict[str, Any]) -> str:
    parts = [
        *(str(item) for item in (row.get("blockers") or []) if str(item).strip()),
        *(str(item) for item in (row.get("reasons") or []) if str(item).strip()),
        "micro_projection_not_bankroll_proven",
        "ai_pick_engine_source",
    ]
    return "; ".join(dict.fromkeys(parts))


def _ledger_row_from_ai_pick(row: dict[str, Any], *, cfg: PredictConfig) -> dict[str, Any] | None:
    side = str(row.get("side") or "").lower()
    if side not in {"over", "under"}:
        return None
    line = _clean_float(row.get("line"))
    if line is None:
        return None
    meta = _ai_meta(row)
    game_date = row.get("game_date_et")
    slug = row.get("game_slug")
    stat = row.get("stat") or row.get("market")
    book = str(row.get("book") or "").lower() or None
    price = _clean_float(row.get("current_price"))
    if price is None:
        price = _clean_float(row.get("locked_price"))
    prop_offer_id = _clean_int(row.get("prop_offer_id"))
    prop_offer_source_row_id = _clean_int(row.get("prop_offer_source_row_id"))
    pick_identity = f"micro_projection:{prop_offer_id or row.get('prediction_key') or row.get('link') or row.get('pick_key')}"
    player_name = row.get("player_name") or ""
    return {
        "pick_key": _pick_key(
            "mlb", "model", "prop", game_date, slug,
            row.get("player_id"), stat, side, line, book, pick_identity,
        ),
        "source": "prop",
        "game_date_et": game_date,
        "game_slug": slug,
        "market": stat,
        "stat": stat,
        "prediction_key": row.get("prediction_key") or row.get("source_prediction_id"),
        "prop_offer_id": prop_offer_id,
        "prop_offer_source_row_id": prop_offer_source_row_id,
        "side": side,
        "label": f"{player_name} {stat} {side} {line}",
        "team_abbr": row.get("team_abbr"),
        "opponent_abbr": None,
        "home_team_abbr": row.get("home_team_abbr"),
        "away_team_abbr": row.get("away_team_abbr"),
        "player_id": _clean_int(row.get("player_id")),
        "player_name": player_name,
        "player_name_norm": _normalize_name(player_name),
        "bookmaker_key": book,
        "market_line": line,
        "bet_line": line,
        "market_price": price,
        "link": row.get("link"),
        "pred_value": _clean_float(row.get("pred_value")),
        "pred_count": _clean_float(row.get("pred_value")),
        "model_prob": _clean_float(row.get("model_prob")),
        "edge": _clean_float(row.get("edge")),
        "edge_type": meta.get("edge_type") or "ai_pick_engine",
        "ev": _clean_float(row.get("current_ev")) if row.get("current_ev") is not None else _clean_float(row.get("ev")),
        "kelly_fraction": None,
        "model_tier": "micro_projection",
        "warning_reasons": _warning_text(row),
        "stake_pct": 0.0,
        "stake_usd": _clean_float(row.get("minimum_stake_usd")) or _clean_float(getattr(cfg, "projection_micro_stake_usd", None)) or 1.0,
        "minimum_acceptable_price": _clean_float(row.get("minimum_acceptable_price")),
        "locked_at_utc": row.get("generated_at_utc") or datetime.now(tz=ZoneInfo("UTC")),
        "model_meta": {
            **meta,
            "ai_pick_key": row.get("pick_key"),
            "ai_pick_engine_source": True,
            "ai_pick_tier": row.get("recommendation_tier"),
            "ai_pick_status": row.get("pick_status"),
            "ai_pick_ranking_score": row.get("ranking_score"),
            "ai_pick_blockers": row.get("blockers") or [],
            "ai_pick_reasons": row.get("reasons") or [],
            "selector_tier": "micro_projection",
            "micro_projection_candidate": True,
            "micro_projection_prob_side": row.get("model_prob"),
            "micro_projection_ev": row.get("current_ev") if row.get("current_ev") is not None else row.get("ev"),
            "micro_current_ev": row.get("current_ev") if row.get("current_ev") is not None else row.get("ev"),
            "micro_projection_stake_usd": _clean_float(row.get("minimum_stake_usd")) or _clean_float(getattr(cfg, "projection_micro_stake_usd", None)) or 1.0,
            "micro_projection_price_drift_ok": True,
            "micro_projection_price_drift_reason": None,
            "price_bucket": meta.get("price_bucket"),
            "bookmaker_key": book,
            "prediction_key": row.get("prediction_key") or row.get("source_prediction_id"),
            "prop_offer_id": prop_offer_id,
            "prop_offer_source_row_id": prop_offer_source_row_id,
        },
        "thresholds": _cfg_thresholds(cfg),
    }


def _build_ai_micro_lock_audit(
    *,
    game_date: date,
    ai_rows: list[dict[str, Any]],
    locked_rows: list[dict[str, Any]],
    cfg: PredictConfig,
    ledger_keys: dict[str, set[Any]] | None = None,
    locked_rows_inserted: int | None = None,
    ledger_before: list[dict] | None = None,
    ledger_after: list[dict] | None = None,
) -> dict[str, Any]:
    ledger_keys = ledger_keys or {"prop_offer_ids": set(), "prediction_keys": set()}
    micro_cap = int(getattr(cfg, "projection_micro_max_props", 5) or 0)
    micro_stake = _clean_float(getattr(cfg, "projection_micro_stake_usd", None)) or 1.0
    micro_min_ev = _clean_float(getattr(cfg, "projection_micro_min_ev", None)) or 0.03
    locked_ai_keys = {row.get("pick_key") for row in locked_rows}
    lockable = [
        row for row in ai_rows
        if str(row.get("recommendation_tier") or "").lower() == "micro"
        and _clean_bool(row.get("qualifies_now"))
        and str(row.get("pick_status") or "").lower() == "bettable_now"
    ]
    lockable.sort(key=_ai_score, reverse=True)
    ranks = {row.get("pick_key"): idx + 1 for idx, row in enumerate(lockable)}
    cap_keys = {row.get("pick_key") for row in lockable[:micro_cap]} if micro_cap > 0 else set()

    audited: list[dict[str, Any]] = []
    for row in ai_rows:
        if str(row.get("recommendation_tier") or "").lower() != "micro":
            continue
        meta = _ai_meta(row)
        bucket_key = _ai_bucket_key(row)
        blockers = list(row.get("blockers") or [])
        qualifies = _clean_bool(row.get("qualifies_now"))
        rank = ranks.get(row.get("pick_key"))
        status = "lockable_now" if qualifies and str(row.get("pick_status") or "").lower() == "bettable_now" else (blockers[0] if blockers else "ai_pick_not_bettable_now")
        if status == "lockable_now" and micro_cap <= 0:
            status = "micro_cap_zero"
        elif status == "lockable_now" and row.get("pick_key") not in cap_keys:
            status = "outside_micro_cap"
        ledger_locked = (
            row.get("prop_offer_id") in ledger_keys["prop_offer_ids"]
            or row.get("prediction_key") in ledger_keys["prediction_keys"]
            or row.get("pick_key") in locked_ai_keys
        )
        audited.append({
            "player_name": row.get("player_name"),
            "team_abbr": row.get("team_abbr"),
            "game_slug": row.get("game_slug"),
            "prediction_key": row.get("prediction_key") or row.get("source_prediction_id"),
            "prop_offer_id": row.get("prop_offer_id"),
            "market": row.get("stat") or row.get("market"),
            "side": row.get("side"),
            "bookmaker_key": row.get("book"),
            "line": _clean_float(row.get("line")),
            "bucket_key": bucket_key,
            "is_target_micro_bucket": bucket_key in _TARGET_MICRO_BUCKETS,
            "selector_tier": "micro_projection",
            "micro_projection_candidate": qualifies,
            "micro_trial_ready": meta.get("micro_trial_ready"),
            "micro_trial_blockers": meta.get("micro_trial_blockers"),
            "micro_projection_prob_side": _clean_float(row.get("model_prob")),
            "micro_projection_raw_prob_side": _clean_float(meta.get("micro_projection_raw_prob_side")),
            "micro_projection_prob_source": meta.get("micro_projection_prob_source") or row.get("model_family"),
            "micro_probability_calibration_key": meta.get("micro_probability_calibration_key"),
            "micro_probability_calibration_status": meta.get("micro_probability_calibration_status"),
            "micro_probability_calibration_target": _clean_float(meta.get("micro_probability_calibration_target")),
            "micro_probability_calibration_cap": _clean_float(meta.get("micro_probability_calibration_cap")),
            "micro_probability_calibration_shrink": _clean_float(meta.get("micro_probability_calibration_shrink")),
            "micro_tb15_high_pa_power_cap_applied": _clean_bool(meta.get("micro_tb15_high_pa_power_cap_applied")),
            "micro_tb15_high_pa_power_cap": _clean_float(meta.get("micro_tb15_high_pa_power_cap")),
            "micro_tb15_high_pa_power_cap_reason": meta.get("micro_tb15_high_pa_power_cap_reason"),
            "micro_tb15_high_pa_power_flags": meta.get("micro_tb15_high_pa_power_flags"),
            "micro_projection_ev": _clean_float(row.get("current_ev")) if row.get("current_ev") is not None else _clean_float(row.get("ev")),
            "micro_projection_edge": _clean_float(row.get("edge")),
            "micro_projection_required_ev": _clean_float(meta.get("micro_projection_required_ev")) or micro_min_ev,
            "micro_projection_required_prob_edge": _clean_float(meta.get("micro_projection_required_prob_edge")),
            "micro_approved_model": _clean_bool(meta.get("micro_approved_model")),
            "micro_approved_model_key": meta.get("micro_approved_model_key"),
            "micro_approved_model_reason": meta.get("micro_approved_model_reason"),
            "micro_external_agreement": _clean_bool(meta.get("micro_external_agreement")),
            "micro_external_agreement_reason": meta.get("micro_external_agreement_reason"),
            "micro_external_agreement_blockers": meta.get("micro_external_agreement_blockers"),
            "external_agreement_count": _clean_float(meta.get("external_agreement_count")),
            "external_disagreement_count": _clean_float(meta.get("external_disagreement_count")),
            "external_agreement_strength": _clean_float(meta.get("external_agreement_strength")),
            "external_match_level": meta.get("external_match_level"),
            "external_platforms": meta.get("external_platforms"),
            "external_best_grade": meta.get("external_best_grade"),
            "external_max_ev": _clean_float(meta.get("external_max_ev")),
            "external_max_probability": _clean_float(meta.get("external_max_probability")),
            "micro_relaxed_trial_lane": _clean_bool(meta.get("micro_relaxed_trial_lane")),
            "micro_truth_filter_status": meta.get("micro_truth_filter_status"),
            "micro_truth_filter_reason": meta.get("micro_truth_filter_reason"),
            "micro_truth_filter_key": meta.get("micro_truth_filter_key"),
            "micro_truth_filter_graded": _clean_float(meta.get("micro_truth_filter_graded")),
            "micro_truth_filter_record": meta.get("micro_truth_filter_record"),
            "micro_truth_filter_roi": _clean_float(meta.get("micro_truth_filter_roi")),
            "micro_truth_filter_clv_rows": _clean_float(meta.get("micro_truth_filter_clv_rows")),
            "micro_truth_filter_clv_beat_rate": _clean_float(meta.get("micro_truth_filter_clv_beat_rate")),
            "micro_truth_filter_avg_clv": _clean_float(meta.get("micro_truth_filter_avg_clv")),
            "minimum_acceptable_price": _clean_float(row.get("minimum_acceptable_price")),
            "current_price": _clean_float(row.get("current_price")),
            "current_ev": _clean_float(row.get("current_ev")) if row.get("current_ev") is not None else _clean_float(row.get("ev")),
            "drift_ok": qualifies,
            "drift_reason": None if qualifies else (blockers[0] if blockers else None),
            "has_bet_link": bool(row.get("link")),
            "audit_status": status,
            "selector_score": _clean_float(row.get("ranking_score")),
            "clv_beat_prob": _clean_float(meta.get("clv_beat_prob")),
            "exact_bucket_clv_beat_prob": _clean_float(meta.get("exact_bucket_clv_beat_prob")),
            "exact_bucket_avg_clv": _clean_float(meta.get("exact_bucket_avg_clv")),
            "bookable_prob": _clean_float(meta.get("bookable_prob")),
            "close_capture_prob": _clean_float(meta.get("close_capture_prob")),
            "k_under_repair_gate_key": meta.get("k_under_repair_gate_key"),
            "k_under_repair_micro_allowed": _clean_bool(meta.get("k_under_repair_micro_allowed")),
            "k_under_repair_blockers": meta.get("k_under_repair_blockers"),
            "selector_reasons": "; ".join(list(row.get("blockers") or []) + list(row.get("reasons") or [])),
            "ledger_locked": ledger_locked,
            "micro_lock_rank": rank,
            "inside_micro_cap": row.get("pick_key") in cap_keys,
        })

    status_counts = Counter(row["audit_status"] for row in audited)
    target_rows = [row for row in audited if row["is_target_micro_bucket"]]
    payload = {
        "generated_at_utc": datetime.now(tz=ZoneInfo("UTC")).isoformat(timespec="seconds"),
        "game_date": game_date.isoformat(),
        "source_label": "AI pick engine prop rows",
        "target_micro_buckets": sorted(_TARGET_MICRO_BUCKETS),
        "selector_error": None,
        "active_prediction_rows": len(ai_rows),
        "micro_cap": micro_cap,
        "micro_stake_usd": micro_stake,
        "micro_min_ev": micro_min_ev,
        "locked_rows_inserted": locked_rows_inserted,
        "ledger_before": ledger_before or [],
        "ledger_after": ledger_after or [],
        "status_counts": dict(status_counts.most_common()),
        "target_rows": len(target_rows),
        "target_micro_candidates": sum(1 for row in target_rows if row["micro_projection_candidate"]),
        "target_lockable_now": sum(1 for row in target_rows if row["audit_status"] == "lockable_now"),
        "target_inside_cap": sum(1 for row in target_rows if row["inside_micro_cap"]),
        "target_ledger_locked": sum(1 for row in target_rows if row["ledger_locked"]),
        "micro_projection_rows": len(audited),
        "micro_external_agreement_rows": sum(1 for row in audited if row.get("micro_external_agreement")),
        "micro_lockable_now": sum(1 for row in audited if row["audit_status"] == "lockable_now"),
        "micro_inside_cap": sum(1 for row in audited if row["inside_micro_cap"]),
        "micro_ledger_locked": sum(1 for row in audited if row["ledger_locked"]),
        "rows": audited,
    }
    return payload


def _ev_per_unit(prob: Any, price: Any) -> float | None:
    p = _clean_float(prob)
    px = _clean_float(price)
    if p is None or px is None or px == 0:
        return None
    payout = px / 100.0 if px > 0 else 100.0 / abs(px)
    return p * payout - (1.0 - p)


def _fmt(value: Any, digits: int = 3) -> str:
    numeric = _clean_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{digits}f}"


def _fmt_price(value: Any) -> str:
    numeric = _clean_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:+.0f}"


def _audit_status(
    row: dict[str, Any],
    *,
    drift_ok: bool,
    drift_reason: str,
    current_price: float | None,
    current_ev: float | None,
    link: str | None,
) -> str:
    if str(row.get("selector_tier") or "").lower() != "micro_projection":
        return "selector_not_micro_projection"
    if not _clean_bool(row.get("micro_projection_candidate")):
        return "selector_micro_candidate_false"
    micro_ev = _clean_float(row.get("micro_projection_ev"))
    micro_prob = _clean_float(row.get("micro_projection_prob_side"))
    if micro_ev is None or micro_ev <= 0.0:
        return "micro_ev_not_positive"
    if micro_prob is None:
        return "micro_probability_missing"
    if not drift_ok:
        return drift_reason or "price_drift_failed"
    if current_price is None:
        return "current_price_missing"
    if current_ev is None or current_ev <= 0.0:
        return "current_ev_not_positive"
    if not link:
        return "bet_link_missing"
    return "lockable_now"


def _build_micro_lock_audit(
    conn,
    *,
    game_date: date,
    rows: list[dict[str, Any]],
    prop_lines,
    cfg: PredictConfig,
    ledger_keys: dict[str, set[Any]] | None = None,
    locked_rows_inserted: int | None = None,
    ledger_before: list[dict] | None = None,
    ledger_after: list[dict] | None = None,
) -> dict[str, Any]:
    ledger_keys = ledger_keys or {"prop_offer_ids": set(), "prediction_keys": set()}
    micro_cap = int(getattr(cfg, "projection_micro_max_props", 5) or 0)
    micro_min_ev = _clean_float(getattr(cfg, "projection_micro_min_ev", None))
    if micro_min_ev is None:
        micro_min_ev = 0.03
    micro_stake = _clean_float(getattr(cfg, "projection_micro_stake_usd", None)) or 1.0
    selector_ctx = selector_cfg = None
    try:
        selector_ctx = SelectorContext(getattr(cfg, "model_dir", None) or _MODEL_DIR)
        selector_cfg = ShadowSelectorConfig(
            pg_dsn=getattr(cfg, "pg_dsn", None) or "",
            report_date=game_date,
            model_dir=getattr(cfg, "model_dir", None) or selector_ctx.model_dir,
            min_ev=_clean_float(getattr(cfg, "min_ev", None)) or 0.02,
        )
    except Exception as exc:
        selector_error = f"{type(exc).__name__}: {exc}"
    else:
        selector_error = None

    work_rows = [dict(row) for row in rows]
    _attach_prop_game_start_times(conn, work_rows)
    audited: list[dict[str, Any]] = []
    lockable_indexes: list[tuple[tuple[float, ...], int]] = []
    for index, row in enumerate(work_rows):
        if selector_ctx is not None and selector_cfg is not None:
            _apply_selector_for_micro_ledger(
                row,
                prop_lines=prop_lines,
                selector_ctx=selector_ctx,
                selector_cfg=selector_cfg,
            )
        bucket_key = row.get("bucket_key") or exact_bucket_key(row)
        micro_prob = _clean_float(row.get("micro_projection_prob_side"))
        micro_required_ev = _clean_float(row.get("micro_projection_required_ev"))
        if micro_required_ev is None:
            micro_required_ev = micro_min_ev
        if micro_prob is not None:
            row["minimum_acceptable_price"] = minimum_american_price(micro_prob, micro_required_ev)
        drift_ok, drift_reason, current_price, link, bookmaker = _prop_price_drift_status(row, prop_lines)
        current_ev = _ev_per_unit(micro_prob, current_price)
        status = _audit_status(
            row,
            drift_ok=drift_ok,
            drift_reason=drift_reason,
            current_price=current_price,
            current_ev=current_ev,
            link=link,
        )
        is_target = bucket_key in _TARGET_MICRO_BUCKETS
        ledger_locked = (
            row.get("prop_offer_id") in ledger_keys["prop_offer_ids"]
            or row.get("prediction_key") in ledger_keys["prediction_keys"]
        )
        rec = {
            "player_name": row.get("player_name"),
            "team_abbr": row.get("team_abbr"),
            "game_slug": row.get("game_slug"),
            "prediction_key": row.get("prediction_key"),
            "prop_offer_id": row.get("prop_offer_id"),
            "market": row.get("stat"),
            "side": row.get("bet_side"),
            "bookmaker_key": (bookmaker or row.get("bookmaker_key") or "").lower() or None,
            "line": _clean_float(row.get("book_line")),
            "bucket_key": bucket_key,
            "is_target_micro_bucket": is_target,
            "selector_tier": row.get("selector_tier"),
            "micro_projection_candidate": _clean_bool(row.get("micro_projection_candidate")),
            "micro_trial_ready": _clean_bool(row.get("micro_trial_ready")),
            "micro_trial_blockers": row.get("micro_trial_blockers"),
            "micro_projection_prob_side": micro_prob,
            "micro_projection_raw_prob_side": _clean_float(row.get("micro_projection_raw_prob_side")),
            "micro_projection_prob_source": row.get("micro_projection_prob_source"),
            "micro_probability_calibration_key": row.get("micro_probability_calibration_key"),
            "micro_probability_calibration_status": row.get("micro_probability_calibration_status"),
            "micro_probability_calibration_target": _clean_float(row.get("micro_probability_calibration_target")),
            "micro_probability_calibration_cap": _clean_float(row.get("micro_probability_calibration_cap")),
            "micro_probability_calibration_shrink": _clean_float(row.get("micro_probability_calibration_shrink")),
            "micro_tb15_high_pa_power_cap_applied": _clean_bool(row.get("micro_tb15_high_pa_power_cap_applied")),
            "micro_tb15_high_pa_power_cap": _clean_float(row.get("micro_tb15_high_pa_power_cap")),
            "micro_tb15_high_pa_power_cap_reason": row.get("micro_tb15_high_pa_power_cap_reason"),
            "micro_tb15_high_pa_power_flags": row.get("micro_tb15_high_pa_power_flags"),
            "micro_projection_ev": _clean_float(row.get("micro_projection_ev")),
            "micro_projection_edge": _clean_float(row.get("micro_projection_edge")),
            "micro_projection_required_ev": micro_required_ev,
            "micro_projection_required_prob_edge": _clean_float(row.get("micro_projection_required_prob_edge")),
            "micro_approved_model": _clean_bool(row.get("micro_approved_model")),
            "micro_approved_model_key": row.get("micro_approved_model_key"),
            "micro_approved_model_reason": row.get("micro_approved_model_reason"),
            "micro_external_agreement": _clean_bool(row.get("micro_external_agreement")),
            "micro_external_agreement_reason": row.get("micro_external_agreement_reason"),
            "micro_external_agreement_blockers": row.get("micro_external_agreement_blockers"),
            "external_agreement_count": _clean_float(row.get("external_agreement_count")),
            "external_disagreement_count": _clean_float(row.get("external_disagreement_count")),
            "external_agreement_strength": _clean_float(row.get("external_agreement_strength")),
            "external_match_level": row.get("external_match_level"),
            "external_platforms": row.get("external_platforms"),
            "external_best_grade": row.get("external_best_grade"),
            "external_max_ev": _clean_float(row.get("external_max_ev")),
            "external_max_probability": _clean_float(row.get("external_max_probability")),
            "micro_relaxed_trial_lane": _clean_bool(row.get("micro_relaxed_trial_lane")),
            "micro_truth_filter_status": row.get("micro_truth_filter_status"),
            "micro_truth_filter_reason": row.get("micro_truth_filter_reason"),
            "micro_truth_filter_key": row.get("micro_truth_filter_key"),
            "micro_truth_filter_graded": _clean_float(row.get("micro_truth_filter_graded")),
            "micro_truth_filter_record": row.get("micro_truth_filter_record"),
            "micro_truth_filter_roi": _clean_float(row.get("micro_truth_filter_roi")),
            "micro_truth_filter_clv_rows": _clean_float(row.get("micro_truth_filter_clv_rows")),
            "micro_truth_filter_clv_beat_rate": _clean_float(row.get("micro_truth_filter_clv_beat_rate")),
            "micro_truth_filter_avg_clv": _clean_float(row.get("micro_truth_filter_avg_clv")),
            "minimum_acceptable_price": _clean_float(row.get("minimum_acceptable_price")),
            "current_price": current_price,
            "current_ev": current_ev,
            "drift_ok": drift_ok,
            "drift_reason": drift_reason or None,
            "has_bet_link": bool(link),
            "audit_status": status,
            "selector_score": _clean_float(row.get("selector_score")),
            "clv_beat_prob": _clean_float(row.get("clv_beat_prob")),
            "exact_bucket_clv_beat_prob": _clean_float(row.get("exact_bucket_clv_beat_prob")),
            "exact_bucket_avg_clv": _clean_float(row.get("exact_bucket_avg_clv")),
            "bookable_prob": _clean_float(row.get("bookable_prob")),
            "close_capture_prob": _clean_float(row.get("close_capture_prob")),
            "k_under_repair_gate_key": row.get("k_under_repair_gate_key"),
            "k_under_repair_micro_allowed": _clean_bool(row.get("k_under_repair_micro_allowed")),
            "k_under_repair_blockers": row.get("k_under_repair_blockers"),
            "selector_reasons": row.get("selector_reasons"),
            "ledger_locked": ledger_locked,
        }
        if status == "lockable_now":
            score = (
                _clean_float(row.get("selector_score")) or -999.0,
                _clean_float(row.get("clv_beat_prob")) or -999.0,
                _clean_float(row.get("bucket_clv_beat_rate")) or -999.0,
                _clean_float(row.get("bookable_prob")) or -999.0,
                current_ev or -999.0,
                _clean_float(row.get("micro_projection_edge")) or -999.0,
                micro_prob or -999.0,
            )
            lockable_indexes.append((score, index))
        audited.append(rec)

    lockable_indexes.sort(key=lambda item: item[0], reverse=True)
    inside_cap = {idx: rank + 1 for rank, (_score, idx) in enumerate(lockable_indexes[:micro_cap])}
    all_lockable = {idx: rank + 1 for rank, (_score, idx) in enumerate(lockable_indexes)}
    for idx, rec in enumerate(audited):
        rank = all_lockable.get(idx)
        rec["micro_lock_rank"] = rank
        rec["inside_micro_cap"] = bool(idx in inside_cap)
        if rec["audit_status"] == "lockable_now" and micro_cap <= 0:
            rec["audit_status"] = "micro_cap_zero"
        elif rec["audit_status"] == "lockable_now" and rank is not None and rank > micro_cap:
            rec["audit_status"] = "outside_micro_cap"

    status_counts = Counter(row["audit_status"] for row in audited)
    target_rows = [row for row in audited if row["is_target_micro_bucket"]]
    micro_rows = [row for row in audited if str(row.get("selector_tier") or "").lower() == "micro_projection"]
    payload = {
        "generated_at_utc": datetime.now(tz=ZoneInfo("UTC")).isoformat(timespec="seconds"),
        "game_date": game_date.isoformat(),
        "target_micro_buckets": sorted(_TARGET_MICRO_BUCKETS),
        "selector_error": selector_error,
        "active_prediction_rows": len(rows),
        "micro_cap": micro_cap,
        "micro_stake_usd": micro_stake,
        "micro_min_ev": micro_min_ev,
        "locked_rows_inserted": locked_rows_inserted,
        "ledger_before": ledger_before or [],
        "ledger_after": ledger_after or [],
        "status_counts": dict(status_counts.most_common()),
        "target_rows": len(target_rows),
        "target_micro_candidates": sum(1 for row in target_rows if row["micro_projection_candidate"]),
        "target_lockable_now": sum(1 for row in target_rows if row["audit_status"] == "lockable_now"),
        "target_inside_cap": sum(1 for row in target_rows if row["inside_micro_cap"]),
        "target_ledger_locked": sum(1 for row in target_rows if row["ledger_locked"]),
        "micro_projection_rows": len(micro_rows),
        "micro_external_agreement_rows": sum(1 for row in audited if row.get("micro_external_agreement")),
        "micro_lockable_now": sum(1 for row in audited if row["audit_status"] == "lockable_now"),
        "micro_inside_cap": sum(1 for row in audited if row["inside_micro_cap"]),
        "micro_ledger_locked": sum(1 for row in audited if row["ledger_locked"]),
        "rows": audited,
    }
    return payload


def _render_audit(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Prop Micro Lock Audit",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Date: {payload.get('game_date')}",
        f"Target buckets: `{'; '.join(payload.get('target_micro_buckets') or [])}`",
        "",
        f"- {payload.get('source_label') or 'Active prop prediction rows'}: {payload.get('active_prediction_rows', 0)}",
        f"- Selector micro rows: {payload.get('micro_projection_rows', 0)}",
        f"- External-agreement micro rows: {payload.get('micro_external_agreement_rows', 0)}",
        f"- Lockable now before cap: {payload.get('micro_lockable_now', 0)}",
        f"- Inside ${payload.get('micro_stake_usd', 1):.0f} micro cap: {payload.get('micro_inside_cap', 0)}",
        f"- Locked micro ledger rows today: {payload.get('micro_ledger_locked', 0)}",
        f"- Newly inserted rows this run: {payload.get('locked_rows_inserted')}",
        "",
        "## Status Counts",
        "",
        "| Status | Rows |",
        "|---|---:|",
    ]
    for status, count in (payload.get("status_counts") or {}).items():
        lines.append(f"| {status} | {count} |")
    lines.extend([
        "",
        "## Target Bucket",
        "",
        f"- Target rows: {payload.get('target_rows', 0)}",
        f"- Target selector candidates: {payload.get('target_micro_candidates', 0)}",
        f"- Target inside cap: {payload.get('target_inside_cap', 0)}",
        f"- Target locked in ledger: {payload.get('target_ledger_locked', 0)}",
        "",
        "| Player | Line | Current | Min | Prob | EV | External | Status | Rank | Locked | Reasons |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|---|---|",
    ])
    target = [row for row in payload.get("rows") or [] if row.get("is_target_micro_bucket")]
    target.sort(key=lambda row: (
        0 if row.get("inside_micro_cap") else 1,
        0 if row.get("audit_status") == "lockable_now" else 1,
        -(float(row.get("micro_projection_ev") or -999.0)),
    ))
    for row in target[:50]:
        reasons = str(row.get("selector_reasons") or "").replace("|", "/")
        lines.append(
            f"| {row.get('player_name')} | {_fmt(row.get('line'), 1)} | {_fmt_price(row.get('current_price'))} | "
            f"{_fmt_price(row.get('minimum_acceptable_price'))} | {_fmt(row.get('micro_projection_prob_side'))} | "
            f"{_fmt(row.get('current_ev'))} | {row.get('external_platforms') or '-'} | "
            f"{row.get('audit_status')} | {row.get('micro_lock_rank') or '-'} | "
            f"{bool(row.get('ledger_locked'))} | {reasons[:240]} |"
        )
    lines.extend([
        "",
        "## All Micro Candidates",
        "",
        "| Player | Bucket | Current | Min | Prob | EV | External | Status | Rank | Locked |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---|",
    ])
    candidates = [
        row for row in payload.get("rows") or []
        if str(row.get("selector_tier") or "").lower() == "micro_projection"
        or row.get("micro_projection_candidate")
        or row.get("ledger_locked")
    ]
    candidates.sort(key=lambda row: (
        row.get("micro_lock_rank") if row.get("micro_lock_rank") is not None else 9999,
        0 if row.get("audit_status") == "lockable_now" else 1,
        -(float(row.get("micro_projection_ev") or -999.0)),
    ))
    for row in candidates[:80]:
        lines.append(
            f"| {row.get('player_name')} | `{row.get('bucket_key')}` | {_fmt_price(row.get('current_price'))} | "
            f"{_fmt_price(row.get('minimum_acceptable_price'))} | {_fmt(row.get('micro_projection_prob_side'))} | "
            f"{_fmt(row.get('current_ev'))} | {row.get('external_platforms') or '-'} | "
            f"{row.get('audit_status')} | {row.get('micro_lock_rank') or '-'} | "
            f"{bool(row.get('ledger_locked'))} |"
        )
    if payload.get("selector_error"):
        lines.extend(["", f"> Selector error: {payload.get('selector_error')}"])
    return "\n".join(lines) + "\n"


def _write_audit(payload: dict[str, Any]) -> tuple[Path, Path]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _MODEL_DIR / "prop_micro_lock_audit.json"
    report_path = _REPORT_DIR / "mlb_prop_micro_lock_audit_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, _render_audit(payload))
    return json_path, report_path


def run(game_date: date, *, pg_dsn: str = PG_DSN) -> dict:
    cfg = PredictConfig()
    cfg.et_date = game_date
    cfg.pg_dsn = pg_dsn
    with psycopg2.connect(pg_dsn) as conn:
        prop_lines = _load_prop_lines(conn, game_date)
        before = _ledger_summary(conn, game_date)
        refresh_payload = refresh_ai_pick_engine(conn, game_date, setup_schema=False)
        ai_rows = load_ai_pick_rows(conn, game_date, market_type="prop")
        lockable_ai_rows = [
            row for row in ai_rows
            if str(row.get("recommendation_tier") or "").lower() == "micro"
            and _clean_bool(row.get("qualifies_now"))
            and str(row.get("pick_status") or "").lower() == "bettable_now"
        ]
        lockable_ai_rows.sort(key=_ai_score, reverse=True)
        micro_cap = int(getattr(cfg, "projection_micro_max_props", 5) or 0)
        lockable_ai_rows = lockable_ai_rows[:micro_cap] if micro_cap > 0 else []
        ledger_rows = [
            ledger_row for ledger_row in (
                _ledger_row_from_ai_pick(row, cfg=cfg)
                for row in lockable_ai_rows
            )
            if ledger_row is not None
        ]
        locked = insert_model_pick_rows(conn, ledger_rows, setup_schema=False)
        after = _ledger_summary(conn, game_date)
        ledger_keys = _micro_ledger_keys(conn, game_date)
        audit = _build_ai_micro_lock_audit(
            game_date=game_date,
            ai_rows=ai_rows,
            locked_rows=lockable_ai_rows,
            cfg=cfg,
            ledger_keys=ledger_keys,
            locked_rows_inserted=locked,
            ledger_before=before,
            ledger_after=after,
        )
        audit_json, audit_report = _write_audit(audit)
    return {
        "status": "ok",
        "game_date": game_date.isoformat(),
        "active_prediction_rows": len(ai_rows),
        "ai_pick_engine_saved_rows": refresh_payload.get("saved_rows"),
        "locked_rows_attempted": locked,
        "ledger_before": before,
        "ledger_after": after,
        "micro_lock_audit": {
            "target_buckets": sorted(_TARGET_MICRO_BUCKETS),
            "target_ledger_locked": audit.get("target_ledger_locked"),
            "micro_ledger_locked": audit.get("micro_ledger_locked"),
            "status_counts": audit.get("status_counts"),
            "json_path": str(audit_json),
            "report_path": str(audit_report),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock bettable MLB prop micro-projection ledger rows")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    args = parser.parse_args()
    game_date = date.fromisoformat(args.date) if args.date else datetime.now(tz=_ET).date()
    print(json.dumps(run(game_date, pg_dsn=args.pg_dsn), indent=2, default=str))


if __name__ == "__main__":
    main()
