"""Canonical MLB AI pick engine.

The projection scripts answer "what will happen?"  This module answers
"what should we do with that forecast today?" across games and player props.
It writes one pick table, one CSV, one JSON artifact, and one Markdown report
that Discord/ledgers can consume without each layer reinventing selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

from .external_pick_ledger import attach_external_agreement
from .ai_bet_selection_model import (
    load_ai_bet_selection_artifact,
    score_ai_bet_selection_row,
    score_ai_bet_selection_rows,
)
from .prop_candidate_engine import normalize_name
from .prop_ledger_classification import classify_prop_ledger
from .prop_offer_snapshots import minimum_american_price
from .prop_replay import american_to_prob, ev_per_unit
from .prop_shadow_selector import SelectorContext, ShadowSelectorConfig, score_prediction_row

_ET = ZoneInfo("America/New_York")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = _REPO_ROOT / "reports"
_DEFAULT_CSV_DIR = _REPO_ROOT / "data" / "ai_picks" / "mlb"

_PROP_STATS = {
    "pitcher_strikeouts",
    "batter_hits",
    "batter_hits_runs_rbis",
    "batter_total_bases",
    "batter_home_runs",
}

_ACTIONABLE_TIERS = {"bankroll", "starter", "micro"}
_TIER_PRIORITY = {
    "bankroll": 100,
    "starter": 90,
    "micro": 80,
    "watch": 55,
    "paper_common": 35,
    "paper_game": 32,
    "lottery": 15,
    "one_sided_fanduel": 10,
    "no_bet": 0,
}

_CSV_COLUMNS = (
    "pick_key",
    "game_date_et",
    "market_type",
    "recommendation_tier",
    "pick_status",
    "qualifies_now",
    "ranking_score",
    "game_slug",
    "player_name",
    "team_abbr",
    "home_team_abbr",
    "away_team_abbr",
    "market",
    "stat",
    "side",
    "line",
    "book",
    "locked_price",
    "current_price",
    "minimum_acceptable_price",
    "model_prob",
    "market_prob",
    "edge",
    "ev",
    "current_ev",
    "pred_value",
    "model_family",
    "link",
    "blockers",
    "reasons",
    "external_platforms",
    "external_match_level",
    "source_prediction_id",
    "prop_offer_id",
)


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _side_prob(row: Mapping[str, Any]) -> float | None:
    side = _text(row.get("bet_side") or row.get("side")).lower()
    p_over = _float(row.get("pred_prob_over"))
    if p_over is None:
        return None
    if side == "over":
        return min(max(p_over, 0.0), 1.0)
    if side == "under":
        return min(max(1.0 - p_over, 0.0), 1.0)
    return None


def _fmt_pct(value: Any, digits: int = 1, *, signed: bool = False) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    sign = "+" if signed else ""
    return f"{numeric * 100.0:{sign}.{digits}f}%"


def _fmt_num(value: Any, digits: int = 2) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{digits}f}"


def _fmt_price(value: Any) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    return f"{numeric:+.0f}"


def _safe(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ").strip()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _stable_pick_key(row: Mapping[str, Any]) -> str:
    raw = "|".join(
        _text(row.get(key)).lower()
        for key in (
            "game_date_et",
            "market_type",
            "game_slug",
            "source_prediction_id",
            "player_name",
            "team_abbr",
            "market",
            "stat",
            "side",
            "line",
            "book",
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _price_drift_blockers(
    *,
    model_prob: Any,
    locked_price: Any,
    current_price: Any,
    minimum_price: Any,
    link: Any,
    require_current_offer: bool,
) -> tuple[bool, float | None, float | None, list[str]]:
    """Return drift pass, current EV, min price, blockers.

    American-odds comparison is intentionally numeric: +150 beats +130 and
    -115 beats -125, so the offered/current price must be >= min acceptable.
    """
    blockers: list[str] = []
    p_win = _float(model_prob)
    locked = _float(locked_price)
    current = _float(current_price)
    min_price = _float(minimum_price)
    if min_price is None and p_win is not None:
        min_price = _float(minimum_american_price(p_win, 0.0))
    if current is None:
        current = locked
        if require_current_offer:
            blockers.append("current_exact_offer_missing")
    if not _text(link):
        blockers.append("valid_link_missing")
    current_ev = ev_per_unit(p_win, current) if p_win is not None and current is not None else None
    if min_price is not None and current is not None and current < min_price:
        blockers.append("price_below_minimum")
    if current_ev is None:
        blockers.append("current_ev_unknown")
    elif current_ev <= 0.0:
        blockers.append("current_ev_not_positive")
    return not blockers, current_ev, min_price, blockers


def _tier_from_prop(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    ledger_row = dict(row)
    ledger_row["side"] = row.get("bet_side")
    classified = classify_prop_ledger(ledger_row)
    bankroll_tier = _text(row.get("bankroll_tier")).lower()
    bankroll_candidate = _bool(row.get("bankroll_candidate"))
    reasons: list[str] = []
    if bankroll_tier:
        reasons.append(f"prediction_tier={bankroll_tier}")
    if classified:
        reasons.append(f"ledger_class={classified}")
    if bankroll_candidate and bankroll_tier in {"bankroll", "starter"}:
        return bankroll_tier, reasons
    if classified in {"micro", "micro_projection", "micro_external_agreement"}:
        return "micro", reasons
    if bankroll_tier in {"micro", "micro_projection"}:
        return "micro", reasons
    if classified in {"one_sided_fanduel", "lottery", "watch"}:
        return classified, reasons
    if bankroll_tier == "watch":
        return "watch", reasons
    return "paper_common", reasons


def _ranking_score(row: Mapping[str, Any]) -> float:
    tier = _text(row.get("recommendation_tier")).lower()
    score = float(_TIER_PRIORITY.get(tier, 0))
    meta = row.get("model_meta") if isinstance(row.get("model_meta"), Mapping) else {}
    model_prob = _float(row.get("model_prob"))
    if model_prob is not None:
        score += max(-0.20, min(0.35, model_prob - 0.50)) * 100.0
    current_ev = _float(row.get("current_ev"))
    ev = _float(row.get("ev"))
    if current_ev is not None:
        score += max(-0.25, min(0.35, current_ev)) * 100.0
    elif ev is not None:
        score += max(-0.20, min(0.25, ev)) * 100.0
    if _bool(row.get("external_agreement")):
        score += min(10.0, (_float(row.get("external_agreement_strength")) or 0.0) * 10.0)
    if _bool(row.get("beat_clv_price")) or _bool(row.get("beat_clv_line")):
        score += 5.0
    if row.get("clv_valid") is False:
        score -= 6.0
    ai_ml_score = _float(meta.get("ai_ml_score"))
    if _bool(meta.get("ai_ml_enabled")) and ai_ml_score is not None:
        score += max(-16.0, min(22.0, (ai_ml_score - 50.0) * 0.45))
    ai_ml_ev = _float(meta.get("ai_ml_ev"))
    if _bool(meta.get("ai_ml_enabled")) and ai_ml_ev is not None and ai_ml_ev <= 0.0:
        score -= 8.0
    blockers = row.get("blockers") or []
    if isinstance(blockers, str):
        blockers = [part for part in blockers.split(",") if part]
    score -= min(30.0, 4.0 * len(blockers))
    return round(score, 4)


def _load_current_prop_offers(conn, game_date: date) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('features.mlb_prop_offer_links')")
        if cur.fetchone()[0] is None:
            return {"by_id": {}, "by_key": {}}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id,
                source_row_id,
                player_name_norm,
                stat,
                LOWER(side) AS side,
                line::float AS line,
                LOWER(bookmaker_key) AS bookmaker_key,
                price::float AS price,
                link,
                is_linkable,
                fetched_at_utc,
                updated_at_utc,
                refreshed_at_utc
            FROM features.mlb_prop_offer_links
            WHERE as_of_date = %(game_date)s
              AND stat = ANY(%(stats)s)
            """,
            {"game_date": game_date, "stats": list(_PROP_STATS)},
        )
        offers = [dict(row) for row in cur.fetchall()]
    by_id: dict[str, dict[str, Any]] = {}
    by_key: dict[tuple[str, str, str, float, str], dict[str, Any]] = {}
    for offer in offers:
        if offer.get("id") is not None:
            by_id[str(offer["id"])] = offer
        line = _float(offer.get("line"))
        if line is None:
            continue
        key = (
            _text(offer.get("player_name_norm")),
            _text(offer.get("stat")),
            _text(offer.get("side")).lower(),
            round(line, 3),
            _text(offer.get("bookmaker_key")).lower(),
        )
        current = by_key.get(key)
        if current is None or (_float(offer.get("price")) or -99999.0) > (_float(current.get("price")) or -99999.0):
            by_key[key] = offer
    return {"by_id": by_id, "by_key": by_key}


def _current_offer_for_prop(row: Mapping[str, Any], offers: Mapping[str, Any]) -> dict[str, Any] | None:
    prop_offer_id = row.get("prop_offer_id")
    if prop_offer_id is not None:
        exact = (offers.get("by_id") or {}).get(str(prop_offer_id))
        if exact:
            return exact
    line = _float(row.get("book_line"))
    if line is None:
        return None
    key = (
        normalize_name(_text(row.get("player_name"))),
        _text(row.get("stat")),
        _text(row.get("bet_side")).lower(),
        round(line, 3),
        _text(row.get("bookmaker_key")).lower(),
    )
    return (offers.get("by_key") or {}).get(key)


def _apply_selector_metadata(
    row: dict[str, Any],
    *,
    current_price: float | None,
    ctx: SelectorContext | None,
    cfg: ShadowSelectorConfig | None,
) -> dict[str, Any]:
    """Attach live selector fields used by micro/watch/paper decisions.

    The prediction DB is intentionally narrower than the selector output, so
    the canonical pick table has to rescore rows before choosing tier/prob.
    """
    if ctx is None or cfg is None:
        return row
    score_row = dict(row)
    score_row["price_drift_ok"] = True
    if current_price is not None:
        score_row["bet_price"] = current_price
    try:
        selector = score_prediction_row(score_row, ctx=ctx, cfg=cfg)
    except Exception:
        return row
    for key, value in selector.items():
        row[key] = value
    if selector.get("selector_tier") == "micro_projection" and _bool(selector.get("micro_projection_candidate")):
        row["bankroll_tier"] = "micro_projection"
    return row


def _fetch_prop_predictions(conn, game_date: date) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('bets.mlb_prop_predictions')")
        if cur.fetchone()[0] is None:
            return []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                p.*,
                id,
                game_date_et,
                game_slug,
                player_id,
                player_name,
                team_abbr,
                stat,
                pred_value::float AS pred_value,
                pred_count::float AS pred_count,
                pred_prob_over::float AS pred_prob_over,
                book_line::float AS book_line,
                edge::float AS edge,
                edge_type,
                model_family,
                bet_side,
                line_bucket,
                over_price::float AS over_price,
                under_price::float AS under_price,
                bet_price::float AS bet_price,
                breakeven_prob::float AS breakeven_prob,
                ev::float AS ev,
                LOWER(bookmaker_key) AS bookmaker_key,
                bankroll_tier,
                bankroll_candidate,
                bankroll_reasons,
                stake_pct::float AS stake_pct,
                prediction_key,
                prop_offer_id,
                prop_offer_source_row_id,
                bet_link,
                closing_line::float AS closing_line,
                closing_price::float AS closing_price,
                clv_line::float AS clv_line,
                clv_price::float AS clv_price,
                beat_clv_line,
                beat_clv_price,
                is_active,
                stale_reason,
                lock_snapshot_id,
                locked_at_utc,
                minimum_acceptable_price::float AS minimum_acceptable_price,
                stake_usd::float AS stake_usd,
                clv_valid,
                clv_status,
                clv_unknown_reason,
                run_id,
                opportunity_context
            FROM bets.mlb_prop_predictions p
            WHERE p.game_date_et = %(game_date)s
              AND p.stat = ANY(%(stats)s)
              AND COALESCE(p.is_active, TRUE) = TRUE
            """,
            {"game_date": game_date, "stats": list(_PROP_STATS)},
        )
        return [dict(row) for row in cur.fetchall()]


def _prop_pick_rows(conn, game_date: date) -> list[dict[str, Any]]:
    raw_rows = _fetch_prop_predictions(conn, game_date)
    attach_external_agreement(conn, raw_rows, game_date=game_date)
    offers = _load_current_prop_offers(conn, game_date)
    ai_selection_artifact = load_ai_bet_selection_artifact(_MODEL_DIR)
    selector_ctx: SelectorContext | None = None
    selector_cfg: ShadowSelectorConfig | None = None
    try:
        selector_ctx = SelectorContext(_MODEL_DIR)
        selector_cfg = ShadowSelectorConfig(
            pg_dsn=PG_DSN,
            report_date=game_date,
            model_dir=_MODEL_DIR,
        )
    except Exception:
        selector_ctx = None
        selector_cfg = None
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        current_offer = _current_offer_for_prop(raw, offers)
        current_price = (
            _float(current_offer.get("price"))
            if current_offer is not None
            else _float(raw.get("bet_price"))
        )
        raw = _apply_selector_metadata(
            raw,
            current_price=current_price,
            ctx=selector_ctx,
            cfg=selector_cfg,
        )
        tier, reasons = _tier_from_prop(raw)
        model_prob = (
            _float(raw.get("micro_projection_prob_side"))
            if tier == "micro"
            else _side_prob(raw)
        )
        if model_prob is None:
            model_prob = _side_prob(raw)
        link = (
            current_offer.get("link")
            if current_offer is not None and current_offer.get("link")
            else raw.get("bet_link")
        )
        drift_ok, current_ev, min_price, blockers = _price_drift_blockers(
            model_prob=model_prob,
            locked_price=raw.get("bet_price"),
            current_price=current_price,
            minimum_price=raw.get("minimum_acceptable_price"),
            link=link,
            require_current_offer=tier in _ACTIONABLE_TIERS,
        )
        if _text(raw.get("stale_reason")):
            blockers.append(f"stale={raw.get('stale_reason')}")
        if raw.get("clv_valid") is False and tier in {"bankroll", "starter"}:
            blockers.append("bankroll_clv_invalid")
        if _bool(raw.get("external_agreement")):
            reasons.append(
                "external_agreement="
                + ",".join(str(item) for item in (raw.get("external_platforms") or []))
            )
        if int(raw.get("external_disagreement_count") or 0) > 0:
            reasons.append(f"external_disagreement_count={raw.get('external_disagreement_count')}")
        qualifies_now = tier in _ACTIONABLE_TIERS and drift_ok
        pick_status = "bettable_now" if qualifies_now else ("research" if tier not in _ACTIONABLE_TIERS else "blocked_now")
        row: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc),
            "game_date_et": raw.get("game_date_et") or game_date,
            "sport": "mlb",
            "source": "supernovabets_ai_pick_engine",
            "market_type": "prop",
            "recommendation_tier": tier,
            "pick_status": pick_status,
            "qualifies_now": qualifies_now,
            "game_slug": raw.get("game_slug"),
            "player_id": raw.get("player_id"),
            "player_name": raw.get("player_name"),
            "player_name_norm": normalize_name(_text(raw.get("player_name"))),
            "team_abbr": raw.get("team_abbr"),
            "home_team_abbr": None,
            "away_team_abbr": None,
            "market": raw.get("stat"),
            "stat": raw.get("stat"),
            "side": _text(raw.get("bet_side")).lower(),
            "line": raw.get("book_line"),
            "book": raw.get("bookmaker_key"),
            "locked_price": raw.get("bet_price"),
            "current_price": current_price,
            "minimum_acceptable_price": min_price,
            "model_prob": model_prob,
            "market_prob": raw.get("breakeven_prob") or american_to_prob(raw.get("bet_price")),
            "edge": raw.get("edge"),
            "ev": raw.get("ev"),
            "current_ev": current_ev,
            "pred_value": raw.get("pred_value") if raw.get("pred_value") is not None else raw.get("pred_count"),
            "model_family": raw.get("model_family"),
            "link": link,
            "blockers": sorted(set(blockers)),
            "reasons": sorted(set(filter(None, [*reasons, _text(raw.get("bankroll_reasons"))]))),
            "source_prediction_id": str(raw.get("id") or raw.get("prediction_key") or ""),
            "prediction_key": raw.get("prediction_key"),
            "prop_offer_id": raw.get("prop_offer_id"),
            "prop_offer_source_row_id": raw.get("prop_offer_source_row_id"),
            "minimum_stake_usd": 1.0 if tier == "micro" else raw.get("stake_usd"),
            "model_meta": {
                "pred_count": raw.get("pred_count"),
                "projection_count": raw.get("pred_count"),
                "projection_pred_value": raw.get("pred_value"),
                "line_bucket": raw.get("line_bucket"),
                "line_surface": raw.get("line_surface"),
                "bucket_key": raw.get("bucket_key"),
                "price_bucket": raw.get("price_bucket"),
                "edge_type": raw.get("edge_type"),
                "over_price": raw.get("over_price"),
                "under_price": raw.get("under_price"),
                "selector_tier": raw.get("selector_tier"),
                "selector_score": raw.get("selector_score"),
                "selector_ev": raw.get("selector_ev"),
                "selector_reasons": raw.get("selector_reasons"),
                "selector_real_candidate": raw.get("selector_real_candidate"),
                "pair_quality": raw.get("pair_quality"),
                "market_prob_source": raw.get("market_prob_source"),
                "policy_variant": raw.get("policy_variant"),
                "bucket_trust_status": raw.get("bucket_trust_status"),
                "no_bet_decision": raw.get("no_bet_decision"),
                "micro_projection_candidate": raw.get("micro_projection_candidate"),
                "micro_projection_prob_side": raw.get("micro_projection_prob_side"),
                "micro_projection_raw_prob_side": raw.get("micro_projection_raw_prob_side"),
                "micro_projection_prob_source": raw.get("micro_projection_prob_source"),
                "micro_projection_edge": raw.get("micro_projection_edge"),
                "micro_projection_ev": raw.get("micro_projection_ev"),
                "micro_projection_required_prob_edge": raw.get("micro_projection_required_prob_edge"),
                "micro_projection_required_ev": raw.get("micro_projection_required_ev"),
                "micro_projection_clv_confirms": raw.get("micro_projection_clv_confirms"),
                "micro_projection_bucket_history_confirms": raw.get("micro_projection_bucket_history_confirms"),
                "micro_probability_calibration_key": raw.get("micro_probability_calibration_key"),
                "micro_probability_calibration_status": raw.get("micro_probability_calibration_status"),
                "micro_probability_calibration_target": raw.get("micro_probability_calibration_target"),
                "micro_probability_calibration_cap": raw.get("micro_probability_calibration_cap"),
                "micro_probability_calibration_shrink": raw.get("micro_probability_calibration_shrink"),
                "micro_tb15_high_pa_power_cap_applied": raw.get("micro_tb15_high_pa_power_cap_applied"),
                "micro_tb15_high_pa_power_cap": raw.get("micro_tb15_high_pa_power_cap"),
                "micro_tb15_high_pa_power_cap_reason": raw.get("micro_tb15_high_pa_power_cap_reason"),
                "micro_tb15_high_pa_power_flags": raw.get("micro_tb15_high_pa_power_flags"),
                "micro_approved_model": raw.get("micro_approved_model"),
                "micro_approved_model_key": raw.get("micro_approved_model_key"),
                "micro_approved_model_reason": raw.get("micro_approved_model_reason"),
                "micro_external_agreement": raw.get("micro_external_agreement"),
                "micro_external_agreement_reason": raw.get("micro_external_agreement_reason"),
                "micro_external_agreement_blockers": raw.get("micro_external_agreement_blockers"),
                "micro_relaxed_trial_lane": raw.get("micro_relaxed_trial_lane"),
                "micro_truth_filter_status": raw.get("micro_truth_filter_status"),
                "micro_truth_filter_reason": raw.get("micro_truth_filter_reason"),
                "micro_truth_filter_key": raw.get("micro_truth_filter_key"),
                "micro_truth_filter_graded": raw.get("micro_truth_filter_graded"),
                "micro_truth_filter_record": raw.get("micro_truth_filter_record"),
                "micro_truth_filter_roi": raw.get("micro_truth_filter_roi"),
                "micro_truth_filter_clv_rows": raw.get("micro_truth_filter_clv_rows"),
                "micro_truth_filter_clv_beat_rate": raw.get("micro_truth_filter_clv_beat_rate"),
                "micro_truth_filter_avg_clv": raw.get("micro_truth_filter_avg_clv"),
                "clv_beat_prob": raw.get("clv_beat_prob"),
                "expected_clv_price": raw.get("expected_clv_price"),
                "residual_clv_beat_prob": raw.get("residual_clv_beat_prob"),
                "event_side_line_clv_beat_prob": raw.get("event_side_line_clv_beat_prob"),
                "bookable_prob": raw.get("bookable_prob"),
                "line_available_prob": raw.get("line_available_prob"),
                "close_capture_prob": raw.get("close_capture_prob"),
                "bucket_roi": raw.get("bucket_roi"),
                "bucket_clv_beat_rate": raw.get("bucket_clv_beat_rate"),
                "bucket_avg_clv": raw.get("bucket_avg_clv"),
                "tb15_calibration_key": raw.get("tb15_calibration_key"),
                "micro_trial_ready": raw.get("micro_trial_ready"),
                "micro_trial_blockers": raw.get("micro_trial_blockers"),
                "micro_clv_beat_prob": raw.get("micro_clv_beat_prob"),
                "exact_bucket_clv_beat_prob": raw.get("exact_bucket_clv_beat_prob"),
                "exact_bucket_avg_clv": raw.get("exact_bucket_avg_clv"),
                "exact_bucket_clv_micro_confirmed": raw.get("exact_bucket_clv_micro_confirmed"),
                "k_under_repair_gate_key": raw.get("k_under_repair_gate_key"),
                "k_under_repair_micro_allowed": raw.get("k_under_repair_micro_allowed"),
                "k_under_repair_blockers": raw.get("k_under_repair_blockers"),
                "clv_valid": raw.get("clv_valid"),
                "clv_status": raw.get("clv_status"),
                "clv_unknown_reason": raw.get("clv_unknown_reason"),
                "beat_clv_line": raw.get("beat_clv_line"),
                "beat_clv_price": raw.get("beat_clv_price"),
                "external_agreement": raw.get("external_agreement"),
                "external_agreement_count": raw.get("external_agreement_count"),
                "external_disagreement_count": raw.get("external_disagreement_count"),
                "external_agreement_strength": raw.get("external_agreement_strength"),
                "external_match_level": raw.get("external_match_level"),
                "external_platforms": raw.get("external_platforms") or [],
                "external_best_grade": raw.get("external_best_grade"),
                "external_max_probability": raw.get("external_max_probability"),
                "external_max_ev": raw.get("external_max_ev"),
                "current_offer_id": current_offer.get("id") if current_offer else None,
                "current_offer_source_row_id": current_offer.get("source_row_id") if current_offer else None,
                "run_id": raw.get("run_id"),
                "opportunity_context": raw.get("opportunity_context"),
            },
        }
        row["pick_key"] = _stable_pick_key(row)
        rows.append(row)
    if rows:
        for row, score_meta in zip(rows, score_ai_bet_selection_rows(rows, ai_selection_artifact)):
            row["model_meta"].update(score_meta)
            row["ranking_score"] = _ranking_score({**row, **row["model_meta"]})
    return rows


def _fetch_game_predictions(conn, game_date: date) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('bets.mlb_game_predictions')")
        if cur.fetchone()[0] is None:
            return []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                game_slug,
                game_date_et,
                home_team_abbr,
                away_team_abbr,
                pred_run_diff::float AS pred_run_diff,
                pred_total::float AS pred_total,
                market_run_line::float AS market_run_line,
                market_total::float AS market_total,
                edge_run_line::float AS edge_run_line,
                edge_total::float AS edge_total,
                run_line_bet_side,
                total_bet_side,
                kelly_fraction_rl::float AS kelly_fraction_rl,
                kelly_fraction_total::float AS kelly_fraction_total,
                win_prob_rl::float AS win_prob_rl,
                win_prob_total::float AS win_prob_total,
                market_rl_price::float AS market_rl_price,
                market_total_price::float AS market_total_price,
                p_home_cover_clf::float AS p_home_cover_clf,
                p_total_over_clf::float AS p_total_over_clf,
                edge_run_line_prob::float AS edge_run_line_prob,
                edge_total_prob::float AS edge_total_prob,
                bankroll_tier_rl,
                bankroll_tier_total,
                bankroll_candidate_rl,
                bankroll_candidate_total,
                bankroll_reasons_rl,
                bankroll_reasons_total,
                stake_pct_rl::float AS stake_pct_rl,
                stake_pct_total::float AS stake_pct_total,
                clv_rl_valid,
                clv_rl_status,
                clv_total_valid,
                clv_total_status,
                predicted_at_utc
            FROM bets.mlb_game_predictions
            WHERE game_date_et = %(game_date)s
            """,
            {"game_date": game_date},
        )
        return [dict(row) for row in cur.fetchall()]


def _game_side_label(row: Mapping[str, Any], market: str) -> str:
    if market == "game_run_line":
        side = _text(row.get("run_line_bet_side")).lower()
        if side in {"home", "away"}:
            return f"{side}_run_line"
        return side
    return _text(row.get("total_bet_side")).lower()


def _game_pick_rows(conn, game_date: date) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _fetch_game_predictions(conn, game_date):
        specs = [
            {
                "market": "game_run_line",
                "side": _game_side_label(raw, "game_run_line"),
                "line": raw.get("market_run_line"),
                "price": raw.get("market_rl_price"),
                "model_prob": raw.get("win_prob_rl") or raw.get("p_home_cover_clf"),
                "edge": raw.get("edge_run_line_prob") if raw.get("edge_run_line_prob") is not None else raw.get("edge_run_line"),
                "ev": raw.get("edge_run_line_prob"),
                "tier": raw.get("bankroll_tier_rl"),
                "candidate": raw.get("bankroll_candidate_rl"),
                "reasons": raw.get("bankroll_reasons_rl"),
                "stake_pct": raw.get("stake_pct_rl"),
                "clv_valid": raw.get("clv_rl_valid"),
                "clv_status": raw.get("clv_rl_status"),
            },
            {
                "market": "game_total",
                "side": _game_side_label(raw, "game_total"),
                "line": raw.get("market_total"),
                "price": raw.get("market_total_price"),
                "model_prob": raw.get("win_prob_total") or raw.get("p_total_over_clf"),
                "edge": raw.get("edge_total_prob") if raw.get("edge_total_prob") is not None else raw.get("edge_total"),
                "ev": raw.get("edge_total_prob"),
                "tier": raw.get("bankroll_tier_total"),
                "candidate": raw.get("bankroll_candidate_total"),
                "reasons": raw.get("bankroll_reasons_total"),
                "stake_pct": raw.get("stake_pct_total"),
                "clv_valid": raw.get("clv_total_valid"),
                "clv_status": raw.get("clv_total_status"),
            },
        ]
        for spec in specs:
            if not spec["side"] or spec["line"] is None:
                continue
            model_prob = _float(spec.get("model_prob"))
            current_ev = ev_per_unit(model_prob, spec.get("price")) if model_prob is not None else None
            tier = _text(spec.get("tier")).lower()
            if _bool(spec.get("candidate")) and tier in {"bankroll", "starter"}:
                recommendation_tier = tier
            else:
                recommendation_tier = "paper_game"
            blockers: list[str] = []
            if recommendation_tier in {"bankroll", "starter"}:
                if current_ev is None or current_ev <= 0.0:
                    blockers.append("game_current_ev_not_positive")
                if spec.get("clv_valid") is False:
                    blockers.append("game_clv_invalid")
            qualifies_now = recommendation_tier in {"bankroll", "starter"} and not blockers
            row = {
                "generated_at_utc": datetime.now(timezone.utc),
                "game_date_et": raw.get("game_date_et") or game_date,
                "sport": "mlb",
                "source": "supernovabets_ai_pick_engine",
                "market_type": "game",
                "recommendation_tier": recommendation_tier,
                "pick_status": "bettable_now" if qualifies_now else "research",
                "qualifies_now": qualifies_now,
                "game_slug": raw.get("game_slug"),
                "player_id": None,
                "player_name": None,
                "player_name_norm": None,
                "team_abbr": raw.get("home_team_abbr") if "home" in str(spec["side"]) else raw.get("away_team_abbr"),
                "home_team_abbr": raw.get("home_team_abbr"),
                "away_team_abbr": raw.get("away_team_abbr"),
                "market": spec.get("market"),
                "stat": None,
                "side": spec.get("side"),
                "line": spec.get("line"),
                "book": "consensus",
                "locked_price": spec.get("price"),
                "current_price": spec.get("price"),
                "minimum_acceptable_price": minimum_american_price(model_prob, 0.0) if model_prob is not None else None,
                "model_prob": model_prob,
                "market_prob": american_to_prob(spec.get("price")),
                "edge": spec.get("edge"),
                "ev": spec.get("ev"),
                "current_ev": current_ev,
                "pred_value": raw.get("pred_run_diff") if spec["market"] == "game_run_line" else raw.get("pred_total"),
                "model_family": "game_market",
                "link": None,
                "blockers": blockers,
                "reasons": [reason for reason in [_text(spec.get("reasons"))] if reason],
                "source_prediction_id": f"{raw.get('game_slug')}:{spec['market']}",
                "prediction_key": None,
                "prop_offer_id": None,
                "prop_offer_source_row_id": None,
                "minimum_stake_usd": None,
                "model_meta": {
                    "stake_pct": spec.get("stake_pct"),
                    "clv_valid": spec.get("clv_valid"),
                    "clv_status": spec.get("clv_status"),
                    "predicted_at_utc": raw.get("predicted_at_utc"),
                },
            }
            row["ranking_score"] = _ranking_score(row)
            row["pick_key"] = _stable_pick_key(row)
            rows.append(row)
    return rows


def build_ai_pick_rows(conn, game_date: date) -> list[dict[str, Any]]:
    rows = [*_game_pick_rows(conn, game_date), *_prop_pick_rows(conn, game_date)]
    rows.sort(
        key=lambda row: (
            _TIER_PRIORITY.get(_text(row.get("recommendation_tier")).lower(), 0),
            _bool(row.get("qualifies_now")),
            _float(row.get("ranking_score")) or -999.0,
        ),
        reverse=True,
    )
    return rows


def _summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    by_tier = Counter(_text(row.get("recommendation_tier")).lower() or "unknown" for row in row_list)
    by_status = Counter(_text(row.get("pick_status")).lower() or "unknown" for row in row_list)
    by_market = Counter(_text(row.get("market")).lower() or "unknown" for row in row_list)
    return {
        "rows": len(row_list),
        "bettable_now": sum(1 for row in row_list if _bool(row.get("qualifies_now"))),
        "by_tier": dict(by_tier),
        "by_status": dict(by_status),
        "by_market": dict(by_market),
    }


def build_ai_pick_payload(conn, game_date: date) -> dict[str, Any]:
    rows = build_ai_pick_rows(conn, game_date)
    return {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "game_date": game_date.isoformat(),
        "summary": _summary(rows),
        "rows": rows,
    }


def ensure_ai_pick_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout = '2s'")
        cur.execute(
            """
            CREATE SCHEMA IF NOT EXISTS bets;
            CREATE TABLE IF NOT EXISTS bets.mlb_ai_pick_engine (
                pick_key TEXT PRIMARY KEY,
                generated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                game_date_et DATE NOT NULL,
                sport TEXT NOT NULL DEFAULT 'mlb',
                source TEXT NOT NULL,
                market_type TEXT NOT NULL,
                recommendation_tier TEXT NOT NULL,
                pick_status TEXT NOT NULL,
                qualifies_now BOOLEAN NOT NULL DEFAULT FALSE,
                ranking_score NUMERIC,
                game_slug TEXT,
                player_id BIGINT,
                player_name TEXT,
                player_name_norm TEXT,
                team_abbr TEXT,
                home_team_abbr TEXT,
                away_team_abbr TEXT,
                market TEXT,
                stat TEXT,
                side TEXT,
                line NUMERIC,
                book TEXT,
                locked_price NUMERIC,
                current_price NUMERIC,
                minimum_acceptable_price NUMERIC,
                model_prob NUMERIC,
                market_prob NUMERIC,
                edge NUMERIC,
                ev NUMERIC,
                current_ev NUMERIC,
                pred_value NUMERIC,
                model_family TEXT,
                link TEXT,
                blockers TEXT,
                reasons TEXT,
                source_prediction_id TEXT,
                prediction_key TEXT,
                prop_offer_id BIGINT,
                prop_offer_source_row_id INTEGER,
                minimum_stake_usd NUMERIC,
                model_meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_mlb_ai_pick_engine_date
                ON bets.mlb_ai_pick_engine (game_date_et);
            CREATE INDEX IF NOT EXISTS idx_mlb_ai_pick_engine_tier
                ON bets.mlb_ai_pick_engine (game_date_et, recommendation_tier, qualifies_now);
            """
        )
    conn.commit()


_UPSERT_SQL = """
INSERT INTO bets.mlb_ai_pick_engine (
    pick_key, generated_at_utc, game_date_et, sport, source, market_type,
    recommendation_tier, pick_status, qualifies_now, ranking_score, game_slug,
    player_id, player_name, player_name_norm, team_abbr, home_team_abbr,
    away_team_abbr, market, stat, side, line, book, locked_price, current_price,
    minimum_acceptable_price, model_prob, market_prob, edge, ev, current_ev,
    pred_value, model_family, link, blockers, reasons, source_prediction_id,
    prediction_key, prop_offer_id, prop_offer_source_row_id, minimum_stake_usd,
    model_meta
) VALUES (
    %(pick_key)s, %(generated_at_utc)s, %(game_date_et)s, %(sport)s, %(source)s,
    %(market_type)s, %(recommendation_tier)s, %(pick_status)s, %(qualifies_now)s,
    %(ranking_score)s, %(game_slug)s, %(player_id)s, %(player_name)s,
    %(player_name_norm)s, %(team_abbr)s, %(home_team_abbr)s, %(away_team_abbr)s,
    %(market)s, %(stat)s, %(side)s, %(line)s, %(book)s, %(locked_price)s,
    %(current_price)s, %(minimum_acceptable_price)s, %(model_prob)s,
    %(market_prob)s, %(edge)s, %(ev)s, %(current_ev)s, %(pred_value)s,
    %(model_family)s, %(link)s, %(blockers)s, %(reasons)s,
    %(source_prediction_id)s, %(prediction_key)s, %(prop_offer_id)s,
    %(prop_offer_source_row_id)s, %(minimum_stake_usd)s, %(model_meta)s
) ON CONFLICT (pick_key) DO UPDATE SET
    generated_at_utc = EXCLUDED.generated_at_utc,
    source = EXCLUDED.source,
    market_type = EXCLUDED.market_type,
    recommendation_tier = EXCLUDED.recommendation_tier,
    pick_status = EXCLUDED.pick_status,
    qualifies_now = EXCLUDED.qualifies_now,
    ranking_score = EXCLUDED.ranking_score,
    current_price = EXCLUDED.current_price,
    minimum_acceptable_price = EXCLUDED.minimum_acceptable_price,
    model_prob = EXCLUDED.model_prob,
    market_prob = EXCLUDED.market_prob,
    edge = EXCLUDED.edge,
    ev = EXCLUDED.ev,
    current_ev = EXCLUDED.current_ev,
    model_family = EXCLUDED.model_family,
    link = EXCLUDED.link,
    blockers = EXCLUDED.blockers,
    reasons = EXCLUDED.reasons,
    minimum_stake_usd = EXCLUDED.minimum_stake_usd,
    model_meta = EXCLUDED.model_meta,
    updated_at_utc = NOW()
"""


def save_ai_pick_rows(conn, rows: Iterable[Mapping[str, Any]], *, setup_schema: bool = True) -> int:
    payload = [dict(row) for row in rows]
    if not payload:
        return 0
    if setup_schema:
        ensure_ai_pick_schema(conn)
    game_dates = sorted({row.get("game_date_et") for row in payload if row.get("game_date_et") is not None})
    saved = 0
    with conn.cursor() as cur:
        if game_dates:
            cur.execute(
                """
                DELETE FROM bets.mlb_ai_pick_engine
                WHERE sport = 'mlb'
                  AND source = 'supernovabets_ai_pick_engine'
                  AND game_date_et = ANY(%s)
                """,
                (game_dates,),
            )
        for row in payload:
            row["blockers"] = ", ".join(row.get("blockers") or [])
            row["reasons"] = ", ".join(row.get("reasons") or [])
            row["model_meta"] = psycopg2.extras.Json(_json_safe(row.get("model_meta") or {}))
            cur.execute(_UPSERT_SQL, row)
            saved += 1
    conn.commit()
    return saved


def _split_csvish(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def load_ai_pick_rows(
    conn,
    game_date: date,
    *,
    market_type: str | None = None,
    only_active: bool = False,
) -> list[dict[str, Any]]:
    """Load saved AI pick rows for a slate from bets.mlb_ai_pick_engine."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('bets.mlb_ai_pick_engine')")
        if cur.fetchone()[0] is None:
            return []
    where = ["game_date_et = %(game_date)s"]
    params: dict[str, Any] = {"game_date": game_date}
    if market_type:
        where.append("market_type = %(market_type)s")
        params["market_type"] = market_type
    if only_active:
        where.append("qualifies_now IS TRUE")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT *
            FROM bets.mlb_ai_pick_engine
            WHERE {' AND '.join(where)}
            ORDER BY
                qualifies_now DESC,
                CASE recommendation_tier
                    WHEN 'bankroll' THEN 100
                    WHEN 'starter' THEN 90
                    WHEN 'micro' THEN 80
                    WHEN 'watch' THEN 55
                    WHEN 'paper_common' THEN 35
                    WHEN 'paper_game' THEN 32
                    WHEN 'lottery' THEN 15
                    WHEN 'one_sided_fanduel' THEN 10
                    ELSE 0
                END DESC,
                ranking_score DESC NULLS LAST
            """,
            params,
        )
        rows = [dict(row) for row in cur.fetchall()]
    for row in rows:
        row["blockers"] = _split_csvish(row.get("blockers"))
        row["reasons"] = _split_csvish(row.get("reasons"))
        if not isinstance(row.get("model_meta"), dict):
            try:
                row["model_meta"] = json.loads(row.get("model_meta") or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                row["model_meta"] = {}
    return rows


def refresh_ai_pick_engine(
    conn,
    game_date: date,
    *,
    csv_dir: Path = _DEFAULT_CSV_DIR,
    setup_schema: bool = True,
) -> dict[str, Any]:
    """Rebuild, save, and write canonical AI pick outputs for one slate."""
    payload = build_ai_pick_payload(conn, game_date)
    saved = save_ai_pick_rows(conn, payload.get("rows") or [], setup_schema=setup_schema)
    json_path, report_path, csv_path = write_ai_pick_outputs(payload, csv_dir=csv_dir)
    payload["saved_rows"] = saved
    payload["json_path"] = str(json_path)
    payload["report_path"] = str(report_path)
    payload["csv_path"] = str(csv_path)
    return payload


def ai_pick_to_prop_prediction_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one AI-pick prop row into model_pick_ledger prop-row shape."""
    meta = row.get("model_meta") if isinstance(row.get("model_meta"), Mapping) else {}
    side = _text(row.get("side")).lower()
    model_prob = _float(row.get("model_prob"))
    pred_prob_over = model_prob if side == "over" else (1.0 - model_prob if model_prob is not None else None)
    pred_count = meta.get("pred_count") if meta.get("pred_count") is not None else row.get("pred_value")
    warning_parts = [
        *(row.get("blockers") or []),
        *(row.get("reasons") or []),
        "ai_pick_engine_source",
    ]
    return {
        "game_date_et": row.get("game_date_et"),
        "game_slug": row.get("game_slug"),
        "player_id": row.get("player_id"),
        "player_name": row.get("player_name"),
        "team_abbr": row.get("team_abbr"),
        "stat": row.get("stat") or row.get("market"),
        "pred_value": row.get("pred_value"),
        "pred_count": pred_count,
        "pred_prob_over": pred_prob_over,
        "book_line": row.get("line"),
        "edge": row.get("edge"),
        "edge_type": meta.get("edge_type") or "probability",
        "model_family": row.get("model_family"),
        "bet_side": side,
        "line_bucket": meta.get("line_bucket"),
        "over_price": meta.get("over_price"),
        "under_price": meta.get("under_price"),
        "bet_price": row.get("current_price") if row.get("current_price") is not None else row.get("locked_price"),
        "breakeven_prob": row.get("market_prob"),
        "ev": row.get("current_ev") if row.get("current_ev") is not None else row.get("ev"),
        "bookmaker_key": row.get("book"),
        "bankroll_tier": "micro_projection" if row.get("recommendation_tier") == "micro" else row.get("recommendation_tier"),
        "bankroll_candidate": bool(row.get("recommendation_tier") in {"bankroll", "starter"} and row.get("qualifies_now")),
        "bankroll_reasons": "; ".join(warning_parts),
        "stake_pct": None,
        "prediction_key": row.get("prediction_key") or row.get("source_prediction_id"),
        "prop_offer_id": row.get("prop_offer_id"),
        "prop_offer_source_row_id": row.get("prop_offer_source_row_id"),
        "bet_link": row.get("link"),
        "minimum_acceptable_price": row.get("minimum_acceptable_price"),
        "stake_usd": row.get("minimum_stake_usd") if row.get("recommendation_tier") == "micro" else None,
        "selector_tier": "micro_projection" if row.get("recommendation_tier") == "micro" else row.get("recommendation_tier"),
        "selector_score": row.get("ranking_score"),
        "selector_ev": row.get("current_ev") if row.get("current_ev") is not None else row.get("ev"),
        "selector_reasons": row.get("reasons"),
        "micro_projection_candidate": bool(row.get("recommendation_tier") == "micro" and row.get("qualifies_now")),
        "micro_projection_prob_side": row.get("model_prob"),
        "micro_projection_raw_prob_side": meta.get("micro_projection_raw_prob_side") or row.get("model_prob"),
        "micro_projection_prob_source": meta.get("micro_projection_prob_source") or row.get("model_family") or "ai_pick_engine",
        "micro_projection_edge": row.get("edge"),
        "micro_projection_ev": row.get("current_ev") if row.get("current_ev") is not None else row.get("ev"),
        "micro_projection_required_ev": 0.0,
        "micro_external_agreement": meta.get("external_agreement"),
        "external_agreement_count": meta.get("external_agreement_count"),
        "external_disagreement_count": meta.get("external_disagreement_count"),
        "external_agreement_strength": meta.get("external_agreement_strength"),
        "external_match_level": meta.get("external_match_level"),
        "external_platforms": meta.get("external_platforms"),
        "external_best_grade": meta.get("external_best_grade"),
        "external_max_ev": meta.get("external_max_ev"),
        "external_max_probability": meta.get("external_max_probability"),
        "clv_beat_prob": meta.get("clv_beat_prob"),
        "bookable_prob": meta.get("bookable_prob"),
        "close_capture_prob": meta.get("close_capture_prob"),
        "pair_quality": meta.get("pair_quality"),
        "market_prob_source": meta.get("market_prob_source"),
        "locked_at_utc": row.get("generated_at_utc"),
        "lock_snapshot_id": meta.get("lock_snapshot_id"),
        "ai_pick_key": row.get("pick_key"),
        "ai_pick_tier": row.get("recommendation_tier"),
        "ai_pick_status": row.get("pick_status"),
        "ai_pick_blockers": row.get("blockers"),
        "ai_pick_reasons": row.get("reasons"),
    }


def _row_for_csv(row: Mapping[str, Any]) -> dict[str, Any]:
    meta = row.get("model_meta") if isinstance(row.get("model_meta"), Mapping) else {}
    return {
        column: (
            ", ".join(str(item) for item in (row.get(column) or []))
            if column in {"blockers", "reasons"} and isinstance(row.get(column), list)
            else (
                ", ".join(str(item) for item in (meta.get("external_platforms") or []))
                if column == "external_platforms"
                else (
                    meta.get("external_match_level")
                    if column == "external_match_level"
                    else row.get(column)
                )
            )
        )
        for column in _CSV_COLUMNS
    }


def write_ai_pick_csv(rows: Iterable[Mapping[str, Any]], *, csv_dir: Path, game_date: date) -> Path:
    csv_dir.mkdir(parents=True, exist_ok=True)
    path = csv_dir / f"{game_date.isoformat()}.csv"
    serializable = [_row_for_csv(row) for row in rows]
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(_CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(serializable)
    atomic_write_text(path, handle.getvalue())
    return path


def _display(row: Mapping[str, Any]) -> str:
    meta = row.get("model_meta") if isinstance(row.get("model_meta"), Mapping) else {}
    market = _text(row.get("market"))
    side = _text(row.get("side")).upper()
    line = _fmt_num(row.get("line"), 1)
    book = _text(row.get("book")) or "-"
    price = _fmt_price(row.get("current_price") if row.get("current_price") is not None else row.get("locked_price"))
    prob = _fmt_pct(row.get("model_prob"))
    ev = _fmt_pct(row.get("current_ev") if row.get("current_ev") is not None else row.get("ev"), signed=True)
    name = _text(row.get("player_name")) or f"{row.get('away_team_abbr')} @ {row.get('home_team_abbr')}"
    ai_ml_score = _float(meta.get("ai_ml_score"))
    ai_ml = ""
    if ai_ml_score is not None:
        evidence = _text(meta.get("ai_ml_fanduel_evidence_tier") or meta.get("ai_ml_clean_evidence_tier"))
        evidence_s = f" {evidence}" if evidence and evidence not in {"not_fanduel", "true_same_book_pair"} else ""
        ai_ml = (
            f" | AI-ML {_fmt_num(ai_ml_score, 1)}"
            f" G {_fmt_pct(meta.get('ai_ml_good_bet_prob'))}"
            f" Pwin {_fmt_pct(meta.get('ai_ml_win_prob'))}"
            f" Pclv {_fmt_pct(meta.get('ai_ml_clv_beat_prob'))}"
            f"{evidence_s}"
        )
    return f"{name} | {market} {side} {line} | {book} {price} | P {prob} | EV {ev}{ai_ml}"


def _section(lines: list[str], title: str, rows: list[Mapping[str, Any]], *, limit: int = 15) -> None:
    lines.extend([f"## {title}", ""])
    if not rows:
        lines.extend(["None.", ""])
        return
    lines.extend([
        "| Pick | Tier | Status | Score | Blockers |",
        "|---|---|---|---:|---|",
    ])
    for row in rows[:limit]:
        blockers = ", ".join(row.get("blockers") or []) if isinstance(row.get("blockers"), list) else _text(row.get("blockers"))
        lines.append(
            f"| {_safe(_display(row))} | {row.get('recommendation_tier')} | "
            f"{row.get('pick_status')} | {_fmt_num(row.get('ranking_score'), 2)} | "
            f"{_safe(blockers) or '-'} |"
        )
    lines.append("")


def _render_report(payload: Mapping[str, Any], *, csv_path: Path | None = None) -> str:
    rows = [dict(row) for row in payload.get("rows") or []]
    summary = payload.get("summary") or {}
    lines = [
        "# MLB AI Pick Engine",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Date: {payload.get('game_date')}",
        f"CSV: `{csv_path or '-'}`",
        "",
        "## Summary",
        "",
        f"- Total pick rows: {summary.get('rows', 0)}",
        f"- Bettable now: {summary.get('bettable_now', 0)}",
        f"- Tiers: `{json.dumps(summary.get('by_tier') or {}, sort_keys=True)}`",
        f"- Statuses: `{json.dumps(summary.get('by_status') or {}, sort_keys=True)}`",
        "",
    ]
    _section(
        lines,
        "Bankroll / Starter Bets",
        [
            row for row in rows
            if row.get("recommendation_tier") in {"bankroll", "starter"}
            and row.get("qualifies_now")
        ],
        limit=20,
    )
    _section(
        lines,
        "$1 Micro Props",
        [
            row for row in rows
            if row.get("recommendation_tier") == "micro"
            and row.get("qualifies_now")
        ],
        limit=20,
    )
    _section(
        lines,
        "Blocked Actionable Candidates",
        [
            row for row in rows
            if row.get("recommendation_tier") in _ACTIONABLE_TIERS
            and not row.get("qualifies_now")
        ],
        limit=25,
    )
    _section(
        lines,
        "Watch / Both-Agree Research",
        [
            row for row in rows
            if row.get("recommendation_tier") == "watch"
            or (row.get("model_meta") or {}).get("external_agreement")
        ],
        limit=25,
    )
    _section(
        lines,
        "Best Paper Common",
        [row for row in rows if row.get("recommendation_tier") in {"paper_common", "paper_game"}],
        limit=25,
    )
    _section(
        lines,
        "Lottery / One-Sided FanDuel",
        [row for row in rows if row.get("recommendation_tier") in {"lottery", "one_sided_fanduel"}],
        limit=15,
    )
    return "\n".join(lines) + "\n"


def _render_discord(payload: Mapping[str, Any], *, csv_path: Path | None = None) -> str:
    rows = [dict(row) for row in payload.get("rows") or []]
    summary = payload.get("summary") or {}
    lines = [
        f"Date: `{payload.get('game_date')}` | rows `{summary.get('rows', 0)}` | bettable now `{summary.get('bettable_now', 0)}`",
    ]
    if csv_path:
        lines.append(f"CSV: `{csv_path}`")

    def add(title: str, section_rows: list[Mapping[str, Any]], limit: int = 8) -> None:
        lines.extend(["", f"**{title}**"])
        if not section_rows:
            lines.append("_None_")
            return
        for row in section_rows[:limit]:
            blockers = row.get("blockers") or []
            blocker_text = ""
            if blockers:
                blocker_text = " | block: " + ", ".join(str(item) for item in blockers[:3])
            lines.append(
                f"- {_display(row)} | score {_fmt_num(row.get('ranking_score'), 1)}{blocker_text}"
            )

    add(
        "BANKROLL / STARTER",
        [
            row for row in rows
            if row.get("recommendation_tier") in {"bankroll", "starter"}
            and row.get("qualifies_now")
        ],
    )
    add(
        "$1 MICRO",
        [
            row for row in rows
            if row.get("recommendation_tier") == "micro"
            and row.get("qualifies_now")
        ],
    )
    add(
        "BLOCKED ACTIONABLE",
        [
            row for row in rows
            if row.get("recommendation_tier") in _ACTIONABLE_TIERS
            and not row.get("qualifies_now")
        ],
        limit=5,
    )
    add(
        "WATCH / BOTH AGREE",
        [
            row for row in rows
            if row.get("recommendation_tier") == "watch"
            or (row.get("model_meta") or {}).get("external_agreement")
        ],
        limit=10,
    )
    return "\n".join(lines)


def write_ai_pick_outputs(
    payload: dict[str, Any],
    *,
    csv_dir: Path = _DEFAULT_CSV_DIR,
    model_dir: Path = _MODEL_DIR,
    report_dir: Path = _REPORT_DIR,
) -> tuple[Path, Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    game_date = date.fromisoformat(str(payload["game_date"]))
    csv_path = write_ai_pick_csv(payload.get("rows") or [], csv_dir=csv_dir, game_date=game_date)
    json_path = model_dir / "ai_pick_engine.json"
    report_path = report_dir / "mlb_ai_pick_engine_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, _render_report(payload, csv_path=csv_path))
    return json_path, report_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical MLB AI pick engine output.")
    parser.add_argument("--date", default=os.getenv("MLB_ET_DATE"), help="Slate date in YYYY-MM-DD ET")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--csv-dir", type=Path, default=Path(os.getenv("MLB_AI_PICK_CSV_DIR", str(_DEFAULT_CSV_DIR))))
    parser.add_argument("--skip-save", action="store_true", help="Do not write bets.mlb_ai_pick_engine.")
    parser.add_argument("--skip-schema", action="store_true", help="Assume bets.mlb_ai_pick_engine already exists.")
    parser.add_argument("--discord-format", action="store_true", help="Print a compact Discord-ready summary instead of JSON.")
    args = parser.parse_args()

    game_date = date.fromisoformat(args.date) if args.date else datetime.now(_ET).date()
    with psycopg2.connect(args.pg_dsn) as conn:
        payload = build_ai_pick_payload(conn, game_date)
        saved = 0
        if not args.skip_save:
            saved = save_ai_pick_rows(conn, payload.get("rows") or [], setup_schema=not args.skip_schema)
        json_path, report_path, csv_path = write_ai_pick_outputs(payload, csv_dir=args.csv_dir)
    if args.discord_format:
        print(_render_discord(payload, csv_path=csv_path))
    else:
        print(json.dumps({
            "status": payload.get("status"),
            "game_date": game_date.isoformat(),
            "rows": (payload.get("summary") or {}).get("rows", 0),
            "bettable_now": (payload.get("summary") or {}).get("bettable_now", 0),
            "saved_rows": saved,
            "json": str(json_path),
            "report": str(report_path),
            "csv": str(csv_path),
        }, indent=2, default=str))


if __name__ == "__main__":
    main()
