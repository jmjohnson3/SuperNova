"""Locked ledger for every MLB model pick.

Prediction tables can be overwritten by reruns. This table locks the first
observed model side so displayed/ranked picks can be graded without hindsight.
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import psycopg2.extras

from .bankroll_ledger import (
    _STAT_COL,
    _cfg_thresholds,
    _clean_float,
    _clean_int,
    _json,
    _normalize_name,
    _pick_key,
    _price_clv,
    _profit_units,
    _run_line_side_label,
    _ET,
)
from .game_line_clv import game_line_clv, resolve_valid_game_close
from .external_pick_ledger import attach_external_agreement
from .prop_candidate_engine import _current_exact_offer
from .prop_offer_snapshots import (
    ensure_prop_offer_snapshot_schema,
    minimum_american_price,
    resolve_valid_prop_close,
)
from .side_recalibration import price_bucket

log = logging.getLogger("mlb_pipeline.modeling.model_pick_ledger")

_SCHEMA_READY = False


def _model_pick_ledger_has_required_columns(conn) -> bool:
    required = {
        "pick_key", "source", "game_date_et", "game_slug", "market", "stat",
        "prediction_key", "prop_offer_id", "prop_offer_source_row_id",
        "minimum_acceptable_price", "stake_usd",
        "closing_snapshot_id", "lock_snapshot_id", "locked_at_utc",
        "clv_valid", "clv_status", "clv_unknown_reason",
    }
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'bets'
              AND table_name = 'mlb_model_pick_ledger'
            """
        )
        existing = {str(row[0]) for row in cur.fetchall()}
    return required.issubset(existing)


def _ev_per_unit(p_win, price) -> float | None:
    p = _clean_float(p_win)
    px = _clean_float(price)
    if p is None or px is None or px == 0:
        return None
    mult = px / 100.0 if px > 0 else 100.0 / abs(px)
    return round(p * mult - (1.0 - p), 4)


def _prob_for_side(p_over, side: str | None) -> float | None:
    p = _clean_float(p_over)
    if p is None or side not in {"over", "under", "home", "away"}:
        return None
    if side in {"over", "home"}:
        return p
    return 1.0 - p


def _prob_from_edge_sigma(edge, sigma) -> float | None:
    edge_v = _clean_float(edge)
    sigma_v = _clean_float(sigma)
    if edge_v is None or sigma_v is None or sigma_v <= 0:
        return None
    p = 1.0 / (1.0 + math.exp(-abs(edge_v) / sigma_v))
    return max(0.0, min(0.99, p))


def _clean_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return False


def _row_stale_after_status(row: dict[str, Any]) -> tuple[bool, str]:
    stale_after = row.get("stale_after_utc") or row.get("start_ts_utc") or row.get("commence_time_utc")
    if stale_after is None:
        return False, "stale_after_missing"
    try:
        ts = pd.Timestamp(stale_after)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ts_utc = ts.tz_convert("UTC")
    except Exception:
        return False, "stale_after_invalid"
    if pd.Timestamp.now(tz="UTC") > ts_utc:
        return False, "stale_after_expired"
    return True, ""


def _prop_price_drift_status(row: dict[str, Any], prop_lines) -> tuple[bool, str, float | None, str | None, str | None]:
    """Return whether the exact prop offer is still bettable at the guard price."""
    stale_ok, stale_reason = _row_stale_after_status(row)
    if not stale_ok:
        return False, stale_reason, None, row.get("bet_link"), row.get("bookmaker_key")
    line_data = (prop_lines or {}).get((_normalize_name(row.get("player_name") or ""), row.get("stat")), {})
    side = str(row.get("bet_side") or "").lower()
    current_offer = _current_exact_offer(
        line_data,
        prop_offer_id=row.get("prop_offer_id"),
        side=side,
        line=row.get("book_line"),
        bookmaker_key=row.get("bookmaker_key"),
    )
    current_price = _clean_float((current_offer or {}).get("price"))
    minimum_price = _clean_float(row.get("minimum_acceptable_price"))
    link = (current_offer or {}).get("link") or row.get("bet_link")
    bookmaker = (current_offer or {}).get("bookmaker_key") or row.get("bookmaker_key")
    if current_offer is None:
        return False, "offer_no_longer_available", current_price, link, bookmaker
    if current_price is None:
        return False, "current_price_missing", current_price, link, bookmaker
    if minimum_price is None:
        return False, "minimum_acceptable_price_missing", current_price, link, bookmaker
    if current_price < minimum_price:
        return False, "price_drift_below_minimum", current_price, link, bookmaker
    return True, "", current_price, link, bookmaker


def _attach_prop_game_start_times(conn, rows: list[dict[str, Any]]) -> None:
    missing = [
        str(row.get("game_slug"))
        for row in rows
        if row.get("game_slug") and not (row.get("start_ts_utc") or row.get("commence_time_utc"))
    ]
    if not missing:
        return
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT game_slug, start_ts_utc
            FROM raw.mlb_games
            WHERE game_slug = ANY(%s)
            """,
            (sorted(set(missing)),),
        )
        starts = {str(row["game_slug"]): row["start_ts_utc"] for row in cur.fetchall()}
    for row in rows:
        if not (row.get("start_ts_utc") or row.get("commence_time_utc")):
            start = starts.get(str(row.get("game_slug")))
            if start is not None:
                row["start_ts_utc"] = start


def _apply_selector_for_micro_ledger(
    row: dict[str, Any],
    *,
    prop_lines,
    selector_ctx,
    selector_cfg,
) -> None:
    """Populate micro selector fields when the caller passed raw DB prediction rows."""
    if (
        str(row.get("selector_tier") or "").lower() == "micro_projection"
        and _clean_bool(row.get("micro_projection_candidate"))
        and row.get("micro_projection_prob_side") is not None
        and row.get("micro_projection_ev") is not None
    ):
        return
    if selector_ctx is None or selector_cfg is None:
        return
    side = str(row.get("bet_side") or "").lower()
    line_data = (prop_lines or {}).get((_normalize_name(row.get("player_name") or ""), row.get("stat")), {})
    current_offer = _current_exact_offer(
        line_data,
        prop_offer_id=row.get("prop_offer_id"),
        side=side,
        line=row.get("book_line"),
        bookmaker_key=row.get("bookmaker_key"),
    )
    if current_offer is None:
        return
    score_row = dict(row)
    current_price = _clean_float(current_offer.get("price"))
    if current_price is not None:
        score_row["bet_price"] = current_price
    score_row["price_drift_ok"] = True
    try:
        from .prop_shadow_selector import score_prediction_row

        selector = score_prediction_row(score_row, ctx=selector_ctx, cfg=selector_cfg)
    except Exception:
        log.exception("Could not score micro selector row for model-pick ledger")
        return
    for key in (
        "selector_score",
        "selector_tier",
        "selector_real_candidate",
        "selector_prob_side",
        "selector_ev",
        "pair_quality",
        "market_prob_source",
        "policy_variant",
        "bucket_trust_status",
        "bucket_key",
        "line_surface",
        "line_bucket",
        "price_bucket",
        "micro_projection_candidate",
        "micro_projection_prob_side",
        "micro_projection_raw_prob_side",
        "micro_projection_prob_source",
        "micro_probability_calibration_key",
        "micro_probability_calibration_status",
        "micro_probability_calibration_target",
        "micro_probability_calibration_cap",
        "micro_probability_calibration_shrink",
        "micro_tb15_high_pa_power_cap_applied",
        "micro_tb15_high_pa_power_cap",
        "micro_tb15_high_pa_power_cap_reason",
        "micro_tb15_high_pa_power_flags",
        "micro_projection_edge",
        "micro_projection_ev",
        "micro_projection_required_prob_edge",
        "micro_projection_required_ev",
        "micro_projection_clv_confirms",
        "micro_projection_bucket_history_confirms",
        "micro_approved_model",
        "micro_approved_model_key",
        "micro_approved_model_reason",
        "micro_external_agreement",
        "micro_external_agreement_reason",
        "micro_external_agreement_blockers",
        "external_agreement_count",
        "external_disagreement_count",
        "external_agreement_strength",
        "external_match_level",
        "external_platforms",
        "external_best_grade",
        "external_max_ev",
        "external_max_probability",
        "micro_relaxed_trial_lane",
        "micro_truth_filter_status",
        "micro_truth_filter_reason",
        "micro_truth_filter_key",
        "micro_truth_filter_graded",
        "micro_truth_filter_record",
        "micro_truth_filter_roi",
        "micro_truth_filter_clv_rows",
        "micro_truth_filter_clv_beat_rate",
        "micro_truth_filter_avg_clv",
        "clv_beat_prob",
        "expected_clv_price",
        "residual_clv_beat_prob",
        "event_side_line_clv_beat_prob",
        "bookable_prob",
        "line_available_prob",
        "close_capture_prob",
        "bucket_roi",
        "bucket_clv_beat_rate",
        "bucket_avg_clv",
        "residual_bucket_decision",
        "distribution_bucket_decision",
        "tb15_calibration_key",
        "micro_trial_ready",
        "micro_trial_blockers",
        "micro_clv_beat_prob",
        "exact_bucket_clv_beat_prob",
        "exact_bucket_avg_clv",
        "exact_bucket_clv_micro_confirmed",
        "k_under_repair_gate_key",
        "k_under_repair_micro_allowed",
        "k_under_repair_blockers",
    ):
        row[key] = selector.get(key)
    row["selector_reasons"] = "; ".join(selector.get("selector_reasons") or [])
    if selector.get("selector_tier") == "micro_projection" and _clean_bool(selector.get("micro_projection_candidate")):
        row["bankroll_tier"] = "micro_projection"
        selector_micro_required_ev = _clean_float(selector.get("micro_projection_required_ev"))
        micro_min = minimum_american_price(
            selector.get("micro_projection_prob_side"),
            selector_cfg.micro_projection_min_ev if selector_micro_required_ev is None else selector_micro_required_ev,
        )
        if micro_min is not None:
            row["minimum_acceptable_price"] = micro_min
    elif str(row.get("bankroll_tier") or "").lower() == "micro_projection":
        row["bankroll_tier"] = None


def _link_bookmaker(link: str | None, fallback: str | None = None) -> str | None:
    if link and "fanduel.com" in link:
        return "fanduel"
    if link and "draftkings.com" in link:
        return "draftkings"
    return fallback


def ensure_model_pick_ledger_schema(conn) -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('bets.mlb_model_pick_ledger')")
        exists = cur.fetchone()[0] is not None
    if exists and _model_pick_ledger_has_required_columns(conn):
        _SCHEMA_READY = True
        ensure_prop_offer_snapshot_schema(conn)
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE SCHEMA IF NOT EXISTS bets;
            CREATE TABLE IF NOT EXISTS bets.mlb_model_pick_ledger (
                id BIGSERIAL PRIMARY KEY,
                pick_key TEXT NOT NULL UNIQUE,
                inserted_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                sport TEXT NOT NULL DEFAULT 'mlb',
                source TEXT NOT NULL,
                game_date_et DATE NOT NULL,
                game_slug TEXT NOT NULL,
                market TEXT NOT NULL,
                stat TEXT,
                prediction_key TEXT,
                prop_offer_id BIGINT,
                prop_offer_source_row_id INTEGER,
                side TEXT NOT NULL,
                label TEXT,
                team_abbr TEXT,
                opponent_abbr TEXT,
                home_team_abbr TEXT,
                away_team_abbr TEXT,
                player_id BIGINT,
                player_name TEXT,
                player_name_norm TEXT,
                bookmaker_key TEXT,
                market_line NUMERIC,
                bet_line NUMERIC,
                market_price NUMERIC,
                minimum_acceptable_price NUMERIC,
                link TEXT,
                pred_value NUMERIC,
                pred_count NUMERIC,
                model_prob NUMERIC,
                edge NUMERIC,
                edge_type TEXT,
                ev NUMERIC,
                kelly_fraction NUMERIC,
                model_tier TEXT,
                warning_reasons TEXT,
                stake_pct NUMERIC,
                stake_usd NUMERIC,
                model_meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                thresholds JSONB NOT NULL DEFAULT '{}'::jsonb,
                result_status TEXT NOT NULL DEFAULT 'pending',
                won BOOLEAN,
                push BOOLEAN NOT NULL DEFAULT FALSE,
                profit_units NUMERIC,
                actual_value NUMERIC,
                actual_home_score NUMERIC,
                actual_away_score NUMERIC,
                actual_run_diff NUMERIC,
                actual_total NUMERIC,
                over_hit BOOLEAN,
                closing_line NUMERIC,
                closing_price NUMERIC,
                clv_line NUMERIC,
                clv_price NUMERIC,
                closing_snapshot_id BIGINT,
                lock_snapshot_id BIGINT,
                locked_at_utc TIMESTAMPTZ,
                clv_valid BOOLEAN,
                clv_status TEXT,
                clv_unknown_reason TEXT,
                graded_at_utc TIMESTAMPTZ,
                grade_source TEXT
            );
            ALTER TABLE bets.mlb_model_pick_ledger
                ADD COLUMN IF NOT EXISTS prediction_key TEXT,
                ADD COLUMN IF NOT EXISTS prop_offer_id BIGINT,
                ADD COLUMN IF NOT EXISTS prop_offer_source_row_id INTEGER,
                ADD COLUMN IF NOT EXISTS minimum_acceptable_price NUMERIC,
                ADD COLUMN IF NOT EXISTS stake_usd NUMERIC,
                ADD COLUMN IF NOT EXISTS closing_snapshot_id BIGINT,
                ADD COLUMN IF NOT EXISTS lock_snapshot_id BIGINT,
                ADD COLUMN IF NOT EXISTS locked_at_utc TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS clv_valid BOOLEAN,
                ADD COLUMN IF NOT EXISTS clv_status TEXT,
                ADD COLUMN IF NOT EXISTS clv_unknown_reason TEXT;
            CREATE INDEX IF NOT EXISTS idx_mlb_model_pick_ledger_date
                ON bets.mlb_model_pick_ledger (game_date_et);
            CREATE INDEX IF NOT EXISTS idx_mlb_model_pick_ledger_pending
                ON bets.mlb_model_pick_ledger (result_status, game_date_et);
            CREATE INDEX IF NOT EXISTS idx_mlb_model_pick_ledger_market
                ON bets.mlb_model_pick_ledger (source, market, side);
            CREATE INDEX IF NOT EXISTS idx_mlb_model_pick_ledger_prop_offer
                ON bets.mlb_model_pick_ledger (prop_offer_id);
            """
        )
    ensure_prop_offer_snapshot_schema(conn)
    _SCHEMA_READY = True


_INSERT_SQL = """
INSERT INTO bets.mlb_model_pick_ledger (
    pick_key, source, game_date_et, game_slug, market, stat,
    prediction_key, prop_offer_id, prop_offer_source_row_id, side, label,
    team_abbr, opponent_abbr, home_team_abbr, away_team_abbr,
    player_id, player_name, player_name_norm, bookmaker_key,
    market_line, bet_line, market_price, link, pred_value, pred_count, model_prob,
    edge, edge_type, ev, kelly_fraction, model_tier, warning_reasons, stake_pct,
    stake_usd, minimum_acceptable_price, locked_at_utc, model_meta, thresholds
) VALUES (
    %(pick_key)s, %(source)s, %(game_date_et)s, %(game_slug)s, %(market)s, %(stat)s,
    %(prediction_key)s, %(prop_offer_id)s, %(prop_offer_source_row_id)s,
    %(side)s, %(label)s, %(team_abbr)s, %(opponent_abbr)s, %(home_team_abbr)s,
    %(away_team_abbr)s, %(player_id)s, %(player_name)s, %(player_name_norm)s,
    %(bookmaker_key)s, %(market_line)s, %(bet_line)s, %(market_price)s, %(link)s,
    %(pred_value)s, %(pred_count)s, %(model_prob)s, %(edge)s, %(edge_type)s, %(ev)s,
    %(kelly_fraction)s, %(model_tier)s, %(warning_reasons)s, %(stake_pct)s,
    %(stake_usd)s, %(minimum_acceptable_price)s, %(locked_at_utc)s, %(model_meta)s, %(thresholds)s
) ON CONFLICT (pick_key) DO NOTHING
"""


def insert_model_pick_rows(
    conn,
    rows: Iterable[dict[str, Any]],
    *,
    setup_schema: bool = True,
) -> int:
    payload = list(rows)
    if not payload:
        return 0
    if setup_schema:
        ensure_model_pick_ledger_schema(conn)
    elif not _model_pick_ledger_has_required_columns(conn):
        raise RuntimeError(
            "bets.mlb_model_pick_ledger schema is not ready; run the ledger setup/grade step before prediction"
        )
    for row in payload:
        row.setdefault("prediction_key", None)
        row.setdefault("prop_offer_id", None)
        row.setdefault("prop_offer_source_row_id", None)
        row.setdefault("stake_usd", None)
        row.setdefault("minimum_acceptable_price", None)
        row.setdefault("locked_at_utc", datetime.now(timezone.utc))
        row["model_meta"] = _json(row.get("model_meta"))
        row["thresholds"] = _json(row.get("thresholds"))
    inserted = 0
    with conn.cursor() as cur:
        for row in payload:
            cur.execute(_INSERT_SQL + " RETURNING id", row)
            if cur.fetchone() is not None:
                inserted += 1
    conn.commit()
    return inserted


def backfill_missing_game_model_pick_probabilities(conn, *, setup_schema: bool = True) -> int:
    """Fill game model-pick probabilities from locked edge/sigma metadata."""
    if setup_schema:
        ensure_model_pick_ledger_schema(conn)
    elif not _model_pick_ledger_has_required_columns(conn):
        raise RuntimeError("bets.mlb_model_pick_ledger schema is not ready")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, market, edge, market_price, model_meta
            FROM bets.mlb_model_pick_ledger
            WHERE source = 'game'
              AND model_prob IS NULL
              AND edge IS NOT NULL
            """
        )
        rows = cur.fetchall()
        updates = []
        for row in rows:
            meta = row.get("model_meta") or {}
            sigma_key = "sigma_q_rl" if row.get("market") == "run_line" else "sigma_q_total"
            p_win = _prob_from_edge_sigma(row.get("edge"), meta.get(sigma_key))
            if p_win is None:
                continue
            ev = _ev_per_unit(p_win, row.get("market_price"))
            updates.append((p_win, ev, row["id"]))
        for p_win, ev, row_id in updates:
            cur.execute(
                """
                UPDATE bets.mlb_model_pick_ledger
                SET model_prob = %s,
                    ev = COALESCE(ev, %s),
                    model_meta = model_meta || '{"model_prob_source":"edge_sigma"}'::jsonb
                WHERE id = %s
                  AND model_prob IS NULL
                """,
                (p_win, ev, row_id),
            )
    conn.commit()
    return len(updates)


def insert_game_model_pick_ledger(conn, rows: list[dict[str, Any]], *, fd_links=None, cfg=None) -> int:
    thresholds = _cfg_thresholds(cfg)
    out: list[dict[str, Any]] = []
    for row in rows:
        game_date = row.get("game_date_et")
        slug = row.get("game_slug")
        home = row.get("home_team_abbr")
        away = row.get("away_team_abbr")
        fd = (fd_links or {}).get(slug) or (fd_links or {}).get((home, away))

        edge_rl = _clean_float(row.get("edge_run_line"))
        home_line = _clean_float(row.get("market_run_line"))
        if edge_rl is not None and abs(edge_rl) > 1e-9 and home_line is not None:
            side = "home" if edge_rl > 0 else "away"
            team = home if side == "home" else away
            opp = away if side == "home" else home
            link = None
            price = None
            if fd is not None:
                link = getattr(fd, "spread_home_link", None) if side == "home" else getattr(fd, "spread_away_link", None)
                price = getattr(fd, "spread_home_price", None) if side == "home" else getattr(fd, "spread_away_price", None)
            p_home_cover = row.get("p_home_cover_clf")
            p_side = _prob_for_side(p_home_cover, side)
            if p_side is None and row.get("run_line_bet_side") == side:
                p_side = _clean_float(row.get("win_prob_rl"))
            if p_side is None:
                p_side = _prob_from_edge_sigma(edge_rl, row.get("sigma_q_rl"))
            line_label = _run_line_side_label(home_line, side)
            out.append({
                "pick_key": _pick_key("mlb", "model", "game", game_date, slug, "run_line", side),
                "source": "game",
                "game_date_et": game_date,
                "game_slug": slug,
                "market": "run_line",
                "stat": None,
                "side": side,
                "label": f"{team} {line_label}" if line_label else f"{team} run line",
                "team_abbr": team,
                "opponent_abbr": opp,
                "home_team_abbr": home,
                "away_team_abbr": away,
                "player_id": None,
                "player_name": None,
                "player_name_norm": None,
                "bookmaker_key": _link_bookmaker(link, "fanduel" if link else None),
                "market_line": home_line,
                "bet_line": home_line if side == "home" else -home_line,
                "market_price": _clean_float(price),
                "link": link,
                "pred_value": _clean_float(row.get("pred_run_diff")),
                "pred_count": None,
                "model_prob": p_side,
                "edge": edge_rl,
                "edge_type": "run_line",
                "ev": _ev_per_unit(p_side, price),
                "kelly_fraction": _clean_float(row.get("kelly_fraction_rl")),
                "model_tier": row.get("bankroll_tier_rl"),
                "warning_reasons": row.get("bankroll_reasons_rl"),
                "stake_pct": _clean_float(row.get("stake_pct_rl")),
                "model_meta": {
                    "season": row.get("season"),
                    "used_residual_model": row.get("used_residual_model"),
                    "sigma_q_rl": row.get("sigma_q_rl"),
                    "p_home_cover_clf": row.get("p_home_cover_clf"),
                    "source_prediction_column": "edge_run_line",
                },
                "thresholds": thresholds,
            })

        edge_total = _clean_float(row.get("edge_total"))
        total_line = _clean_float(row.get("market_total"))
        if edge_total is not None and abs(edge_total) > 1e-9 and total_line is not None:
            side = "over" if edge_total > 0 else "under"
            link = None
            price = None
            if fd is not None:
                link = getattr(fd, "total_over_link", None) if side == "over" else getattr(fd, "total_under_link", None)
                price = getattr(fd, "total_over_price", None) if side == "over" else getattr(fd, "total_under_price", None)
            p_total_over = row.get("p_total_over_clf")
            p_side = _prob_for_side(p_total_over, side)
            if p_side is None and row.get("total_bet_side") == side:
                p_side = _clean_float(row.get("win_prob_total"))
            if p_side is None:
                p_side = _prob_from_edge_sigma(edge_total, row.get("sigma_q_total"))
            out.append({
                "pick_key": _pick_key("mlb", "model", "game", game_date, slug, "total", side),
                "source": "game",
                "game_date_et": game_date,
                "game_slug": slug,
                "market": "total",
                "stat": None,
                "side": side,
                "label": f"{side.upper()} {total_line}",
                "team_abbr": None,
                "opponent_abbr": None,
                "home_team_abbr": home,
                "away_team_abbr": away,
                "player_id": None,
                "player_name": None,
                "player_name_norm": None,
                "bookmaker_key": _link_bookmaker(link, "fanduel" if link else None),
                "market_line": total_line,
                "bet_line": total_line,
                "market_price": _clean_float(price),
                "link": link,
                "pred_value": _clean_float(row.get("pred_total")),
                "pred_count": None,
                "model_prob": p_side,
                "edge": edge_total,
                "edge_type": "total",
                "ev": _ev_per_unit(p_side, price),
                "kelly_fraction": _clean_float(row.get("kelly_fraction_total")),
                "model_tier": row.get("bankroll_tier_total"),
                "warning_reasons": row.get("bankroll_reasons_total"),
                "stake_pct": _clean_float(row.get("stake_pct_total")),
                "model_meta": {
                    "season": row.get("season"),
                    "used_residual_model": row.get("used_residual_model"),
                    "sigma_q_total": row.get("sigma_q_total"),
                    "p_total_over_clf": row.get("p_total_over_clf"),
                    "source_prediction_column": "edge_total",
                },
                "thresholds": thresholds,
            })
    inserted = insert_model_pick_rows(conn, out, setup_schema=False)
    filled = backfill_missing_game_model_pick_probabilities(conn, setup_schema=False)
    if filled:
        log.info("Filled %d MLB game model-pick probability gaps from locked edge/sigma", filled)
    return inserted


def insert_prop_model_pick_ledger(conn, rows: list[dict[str, Any]], *, prop_lines=None, cfg=None) -> int:
    thresholds = _cfg_thresholds(cfg)
    min_ev = _clean_float(getattr(cfg, "min_ev", None)) if cfg is not None else 0.02
    min_ev = 0.02 if min_ev is None else min_ev
    micro_cap = int(getattr(cfg, "projection_micro_max_props", 5) or 0) if cfg is not None else 5
    micro_stake_usd = _clean_float(getattr(cfg, "projection_micro_stake_usd", None)) if cfg is not None else 1.0
    micro_stake_usd = 1.0 if micro_stake_usd is None else micro_stake_usd
    micro_min_ev = _clean_float(getattr(cfg, "projection_micro_min_ev", None)) if cfg is not None else 0.03
    micro_min_ev = 0.03 if micro_min_ev is None else micro_min_ev
    canonical_ai_micro = os.getenv("MLB_AI_PICK_ENGINE_CANONICAL_MICRO", "1").strip().lower() not in {"0", "false", "no", "off"}
    out: list[dict[str, Any]] = []
    micro_candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    _attach_prop_game_start_times(conn, rows)
    try:
        attach_external_agreement(conn, rows, game_date=getattr(cfg, "et_date", None))
    except Exception:
        log.exception("Could not attach external prop agreement metadata")
    selector_ctx = selector_cfg = None
    try:
        from .prop_shadow_selector import SelectorContext, ShadowSelectorConfig

        selector_ctx = SelectorContext(getattr(cfg, "model_dir", None) or Path(__file__).resolve().parent / "models" / "player_props")
        selector_cfg = ShadowSelectorConfig(
            pg_dsn=getattr(cfg, "pg_dsn", None) or "",
            report_date=getattr(cfg, "et_date", None),
            model_dir=getattr(cfg, "model_dir", None) or selector_ctx.model_dir,
            min_ev=min_ev,
        )
    except Exception:
        log.exception("Could not load prop selector context for automatic micro ledger scoring")

    def _with_reason(existing: Any, reason: str) -> str:
        parts = [part.strip() for part in str(existing or "").split(";") if part.strip()]
        if reason not in parts:
            parts.append(reason)
        return "; ".join(parts)

    def _ledger_row(
        row: dict[str, Any],
        *,
        ev: float,
        model_prob: float | None,
        model_tier: str | None,
        warning_reasons: str | None,
        market_price: float | None,
        link: str | None,
        bookmaker: str | None,
        stake_usd: float | None,
        stake_pct: float | None,
        micro_price_drift_ok: bool | None = None,
        micro_price_drift_reason: str | None = None,
    ) -> dict[str, Any] | None:
        side = (row.get("bet_side") or "").lower()
        if side not in {"over", "under"}:
            return None
        name = row.get("player_name") or ""
        stat = row.get("stat")
        norm = _normalize_name(name)
        line = _clean_float(row.get("book_line"))
        prop_offer_id = _clean_int(row.get("prop_offer_id"))
        prop_offer_source_row_id = _clean_int(row.get("prop_offer_source_row_id"))
        pick_identity = prop_offer_id or row.get("prediction_key") or link
        if model_tier == "micro_projection":
            pick_identity = f"micro_projection:{pick_identity}"
        return {
            "pick_key": _pick_key(
                "mlb", "model", "prop", row.get("game_date_et"), row.get("game_slug"),
                row.get("player_id"), stat, side, line, bookmaker,
                pick_identity,
            ),
            "source": "prop",
            "game_date_et": row.get("game_date_et"),
            "game_slug": row.get("game_slug"),
            "market": stat,
            "stat": stat,
            "prediction_key": row.get("prediction_key"),
            "prop_offer_id": prop_offer_id,
            "prop_offer_source_row_id": prop_offer_source_row_id,
            "side": side,
            "label": f"{name} {stat} {side} {line}",
            "team_abbr": row.get("team_abbr"),
            "opponent_abbr": None,
            "home_team_abbr": None,
            "away_team_abbr": None,
            "player_id": _clean_int(row.get("player_id")),
            "player_name": name,
            "player_name_norm": norm,
            "bookmaker_key": bookmaker,
            "market_line": line,
            "bet_line": line,
            "market_price": market_price,
            "link": link,
            "pred_value": _clean_float(row.get("pred_value")),
            "pred_count": _clean_float(row.get("pred_count")),
            "model_prob": model_prob,
            "edge": _clean_float(row.get("edge")),
            "edge_type": row.get("edge_type"),
            "ev": ev,
            "kelly_fraction": _clean_float(row.get("kelly_fraction")),
            "model_tier": model_tier,
            "warning_reasons": warning_reasons,
            "stake_pct": stake_pct,
            "stake_usd": stake_usd,
            "minimum_acceptable_price": _clean_float(row.get("minimum_acceptable_price")),
            "model_meta": {
                "model_family": row.get("model_family"),
                "line_bucket": row.get("line_bucket"),
                "price_bucket": row.get("price_bucket") or price_bucket(market_price),
                "bookmaker_key": row.get("bookmaker_key"),
                "prediction_key": row.get("prediction_key"),
                "prop_offer_id": prop_offer_id,
                "prop_offer_source_row_id": prop_offer_source_row_id,
                "selector_tier": row.get("selector_tier"),
                "selector_score": row.get("selector_score"),
                "selector_ev": row.get("selector_ev"),
                "selector_reasons": row.get("selector_reasons"),
                "micro_projection_candidate": row.get("micro_projection_candidate"),
                "micro_projection_prob_side": row.get("micro_projection_prob_side"),
                "micro_projection_raw_prob_side": row.get("micro_projection_raw_prob_side"),
                "micro_projection_prob_source": row.get("micro_projection_prob_source"),
                "micro_probability_calibration_key": row.get("micro_probability_calibration_key"),
                "micro_probability_calibration_status": row.get("micro_probability_calibration_status"),
                "micro_probability_calibration_target": row.get("micro_probability_calibration_target"),
                "micro_probability_calibration_cap": row.get("micro_probability_calibration_cap"),
                "micro_probability_calibration_shrink": row.get("micro_probability_calibration_shrink"),
                "micro_tb15_high_pa_power_cap_applied": row.get("micro_tb15_high_pa_power_cap_applied"),
                "micro_tb15_high_pa_power_cap": row.get("micro_tb15_high_pa_power_cap"),
                "micro_tb15_high_pa_power_cap_reason": row.get("micro_tb15_high_pa_power_cap_reason"),
                "micro_tb15_high_pa_power_flags": row.get("micro_tb15_high_pa_power_flags"),
                "micro_projection_edge": row.get("micro_projection_edge"),
                "micro_projection_ev": row.get("micro_projection_ev"),
                "micro_projection_required_prob_edge": row.get("micro_projection_required_prob_edge"),
                "micro_projection_required_ev": row.get("micro_projection_required_ev"),
                "micro_current_ev": ev if model_tier == "micro_projection" else None,
                "micro_projection_clv_confirms": row.get("micro_projection_clv_confirms"),
                "micro_projection_bucket_history_confirms": row.get("micro_projection_bucket_history_confirms"),
                "micro_approved_model": row.get("micro_approved_model"),
                "micro_approved_model_key": row.get("micro_approved_model_key"),
                "micro_approved_model_reason": row.get("micro_approved_model_reason"),
                "micro_external_agreement": row.get("micro_external_agreement"),
                "micro_external_agreement_reason": row.get("micro_external_agreement_reason"),
                "micro_external_agreement_blockers": row.get("micro_external_agreement_blockers"),
                "external_agreement_count": row.get("external_agreement_count"),
                "external_disagreement_count": row.get("external_disagreement_count"),
                "external_agreement_strength": row.get("external_agreement_strength"),
                "external_match_level": row.get("external_match_level"),
                "external_platforms": row.get("external_platforms"),
                "external_best_grade": row.get("external_best_grade"),
                "external_max_ev": row.get("external_max_ev"),
                "external_max_probability": row.get("external_max_probability"),
                "micro_relaxed_trial_lane": row.get("micro_relaxed_trial_lane"),
                "micro_truth_filter_status": row.get("micro_truth_filter_status"),
                "micro_truth_filter_reason": row.get("micro_truth_filter_reason"),
                "micro_truth_filter_key": row.get("micro_truth_filter_key"),
                "micro_truth_filter_graded": row.get("micro_truth_filter_graded"),
                "micro_truth_filter_record": row.get("micro_truth_filter_record"),
                "micro_truth_filter_roi": row.get("micro_truth_filter_roi"),
                "micro_truth_filter_clv_rows": row.get("micro_truth_filter_clv_rows"),
                "micro_truth_filter_clv_beat_rate": row.get("micro_truth_filter_clv_beat_rate"),
                "micro_truth_filter_avg_clv": row.get("micro_truth_filter_avg_clv"),
                "micro_projection_price_drift_ok": micro_price_drift_ok,
                "micro_projection_price_drift_reason": micro_price_drift_reason,
                "micro_projection_stake_usd": micro_stake_usd if model_tier == "micro_projection" else None,
                "clv_beat_prob": row.get("clv_beat_prob"),
                "expected_clv_price": row.get("expected_clv_price"),
                "residual_clv_beat_prob": row.get("residual_clv_beat_prob"),
                "event_side_line_clv_beat_prob": row.get("event_side_line_clv_beat_prob"),
                "bookable_prob": row.get("bookable_prob"),
                "line_available_prob": row.get("line_available_prob"),
                "close_capture_prob": row.get("close_capture_prob"),
                "pair_quality": row.get("pair_quality"),
                "market_prob_source": row.get("market_prob_source"),
                "policy_variant": row.get("policy_variant"),
                "residual_bucket_decision": row.get("residual_bucket_decision"),
                "distribution_bucket_decision": row.get("distribution_bucket_decision"),
                "bucket_trust_status": row.get("bucket_trust_status"),
                "bucket_roi": row.get("bucket_roi"),
                "bucket_clv_beat_rate": row.get("bucket_clv_beat_rate"),
                "bucket_avg_clv": row.get("bucket_avg_clv"),
                "tb15_calibration_key": row.get("tb15_calibration_key"),
                "micro_trial_ready": row.get("micro_trial_ready"),
                "micro_trial_blockers": row.get("micro_trial_blockers"),
                "micro_clv_beat_prob": row.get("micro_clv_beat_prob"),
                "exact_bucket_clv_beat_prob": row.get("exact_bucket_clv_beat_prob"),
                "exact_bucket_avg_clv": row.get("exact_bucket_avg_clv"),
                "exact_bucket_clv_micro_confirmed": row.get("exact_bucket_clv_micro_confirmed"),
                "k_under_repair_gate_key": row.get("k_under_repair_gate_key"),
                "k_under_repair_micro_allowed": row.get("k_under_repair_micro_allowed"),
                "k_under_repair_blockers": row.get("k_under_repair_blockers"),
            },
            "thresholds": thresholds,
        }

    for row in rows:
        side = (row.get("bet_side") or "").lower()
        if side not in {"over", "under"}:
            continue
        _apply_selector_for_micro_ledger(
            row,
            prop_lines=prop_lines,
            selector_ctx=selector_ctx,
            selector_cfg=selector_cfg,
        )
        is_micro_projection = (
            str(row.get("selector_tier") or "").lower() == "micro_projection"
            and _clean_bool(row.get("micro_projection_candidate"))
        )
        if is_micro_projection:
            if canonical_ai_micro:
                continue
            micro_ev = _clean_float(row.get("micro_projection_ev"))
            micro_prob = _clean_float(row.get("micro_projection_prob_side"))
            if micro_ev is None or micro_ev <= 0 or micro_prob is None:
                continue
            micro_required_ev = _clean_float(row.get("micro_projection_required_ev"))
            if micro_required_ev is None:
                micro_required_ev = micro_min_ev
            micro_min = minimum_american_price(micro_prob, micro_required_ev)
            drift_row = dict(row)
            if micro_min is not None:
                drift_row["minimum_acceptable_price"] = micro_min
            drift_ok, drift_reason, current_price, link, bookmaker = _prop_price_drift_status(drift_row, prop_lines)
            if not drift_ok or not link:
                continue
            current_ev = _ev_per_unit(micro_prob, current_price)
            if current_ev is None or current_ev <= 0.0:
                continue
            bookmaker = _link_bookmaker(link, bookmaker)
            ledger = _ledger_row(
                drift_row,
                ev=current_ev,
                model_prob=micro_prob,
                model_tier="micro_projection",
                warning_reasons=_with_reason(row.get("bankroll_reasons"), "micro_projection_not_bankroll_proven"),
                market_price=current_price,
                link=link,
                bookmaker=bookmaker,
                stake_usd=micro_stake_usd,
                stake_pct=0.0,
                micro_price_drift_ok=True,
                micro_price_drift_reason=drift_reason or None,
            )
            if ledger is not None:
                micro_candidates.append((
                    (
                        _clean_float(row.get("selector_score")) or -999.0,
                        _clean_float(row.get("clv_beat_prob")) or -999.0,
                        _clean_float(row.get("bucket_clv_beat_rate")) or -999.0,
                        _clean_float(row.get("bookable_prob")) or -999.0,
                        current_ev,
                        _clean_float(row.get("micro_projection_edge")) or -999.0,
                        micro_prob,
                    ),
                    ledger,
                ))
            continue

        ev = _clean_float(row.get("ev"))
        if ev is None or ev < min_ev:
            continue
        stat = row.get("stat")
        name = row.get("player_name") or ""
        norm = _normalize_name(name)
        ld = (prop_lines or {}).get((norm, stat), {})
        link = row.get("bet_link") or (ld.get("over_link") if side == "over" else ld.get("under_link"))
        bookmaker = row.get("bookmaker_key")
        if side == "under" and ld.get("under_link_book"):
            bookmaker = ld.get("under_link_book")
        bookmaker = _link_bookmaker(link, bookmaker)
        p_over = _clean_float(row.get("pred_prob_over"))
        model_prob = p_over if side == "over" else (1.0 - p_over if p_over is not None else None)
        ledger = _ledger_row(
            row,
            ev=ev,
            model_prob=model_prob,
            model_tier=row.get("bankroll_tier"),
            warning_reasons=row.get("bankroll_reasons"),
            market_price=_clean_float(row.get("bet_price")),
            link=link,
            bookmaker=bookmaker,
            stake_usd=_clean_float(row.get("stake_usd")),
            stake_pct=_clean_float(row.get("stake_pct")),
        )
        if ledger is not None:
            out.append(ledger)
    micro_candidates.sort(key=lambda item: item[0], reverse=True)
    if micro_cap > 0:
        out.extend(row for _score, row in micro_candidates[:micro_cap])
    return insert_model_pick_rows(conn, out, setup_schema=False)


def grade_model_pick_ledger(conn) -> int:
    ensure_model_pick_ledger_schema(conn)
    updated = _grade_game_model_pick_ledger(conn) + _grade_prop_model_pick_ledger(conn)
    conn.commit()
    return updated


def _grade_game_model_pick_ledger(conn) -> int:
    today = datetime.now(_ET).date()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT *
            FROM bets.mlb_model_pick_ledger
            WHERE source = 'game'
              AND result_status = 'pending'
              AND game_date_et <= %s
            """,
            (today,),
        )
        rows = cur.fetchall()
    if not rows:
        return 0

    slugs = list({r["game_slug"] for r in rows})
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT game_slug, home_score, away_score
            FROM raw.mlb_games
            WHERE game_slug = ANY(%s)
              AND status = 'final'
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
            """,
            (slugs,),
        )
        finals = {r["game_slug"]: r for r in cur.fetchall()}

    updated = 0
    with conn.cursor() as cur:
        for row in rows:
            final = finals.get(row["game_slug"])
            if not final:
                continue
            home_score = float(final["home_score"])
            away_score = float(final["away_score"])
            actual_run_diff = home_score - away_score
            actual_total = home_score + away_score
            market_line = _clean_float(row["market_line"])
            side = row["side"]
            won = None
            push = False
            over_hit = None

            if row["market"] == "run_line" and market_line is not None:
                threshold = -market_line
                push = abs(actual_run_diff - threshold) <= 1e-9
                home_covered = actual_run_diff > threshold
                won = None if push else (home_covered if side == "home" else not home_covered)
            elif row["market"] == "total" and market_line is not None:
                push = abs(actual_total - market_line) <= 1e-9
                over_hit = actual_total > market_line
                won = None if push else (over_hit if side == "over" else not over_hit)

            close = resolve_valid_game_close(conn, dict(row))
            closing_line = closing_price = clv_line = clv_price = None
            if close["valid"]:
                closing_line = _clean_float(close.get("closing_line"))
                closing_price = _clean_float(close.get("closing_price"))
                clv_line = game_line_clv(
                    row["market"],
                    side,
                    close.get("entry_line"),
                    closing_line,
                )
                if close.get("entry_line") == closing_line:
                    clv_price = _price_clv(close.get("entry_price"), closing_price)

            profit = _profit_units(won, push, row["market_price"])
            cur.execute(
                """
                UPDATE bets.mlb_model_pick_ledger
                SET result_status = 'graded',
                    won = %s,
                    push = %s,
                    profit_units = %s,
                    actual_home_score = %s,
                    actual_away_score = %s,
                    actual_run_diff = %s,
                    actual_total = %s,
                    over_hit = %s,
                    closing_line = %s,
                    closing_price = %s,
                    clv_line = %s,
                    clv_price = %s,
                    clv_valid = %s,
                    clv_status = %s,
                    clv_unknown_reason = %s,
                    graded_at_utc = NOW(),
                    grade_source = 'raw.mlb_games'
                WHERE id = %s
                """,
                (
                    won,
                    push,
                    profit,
                    home_score,
                    away_score,
                    actual_run_diff,
                    actual_total,
                    over_hit,
                    closing_line,
                    closing_price,
                    clv_line,
                    clv_price,
                    close["valid"],
                    close["status"],
                    close["unknown_reason"],
                    row["id"],
                ),
            )
            updated += 1
    log.info("Graded %d game model-pick ledger rows", updated)
    return updated


def _grade_prop_model_pick_ledger(conn) -> int:
    today = datetime.now(_ET).date()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT *
            FROM bets.mlb_model_pick_ledger
            WHERE source = 'prop'
              AND (result_status = 'pending' OR clv_status IS NULL)
              AND game_date_et <= %s
            """,
            (today,),
        )
        rows = cur.fetchall()
    if not rows:
        return 0

    slugs = list({r["game_slug"] for r in rows})
    player_ids = list({int(r["player_id"]) for r in rows if r["player_id"] is not None})
    if not slugs or not player_ids:
        return 0
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT gl.game_slug, gl.player_id,
                   gl.strikeouts_pitcher,
                   gl.hits,
                   gl.home_runs,
                   gl.total_bases,
                   gl.walks_batter
            FROM raw.mlb_player_gamelogs gl
            JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
            WHERE gl.game_slug = ANY(%s)
              AND gl.player_id = ANY(%s)
              AND g.status = 'final'
            """,
            (slugs, player_ids),
        )
        actuals = {(r["game_slug"], int(r["player_id"])): r for r in cur.fetchall()}

    updated = 0
    with conn.cursor() as cur:
        for row in rows:
            key = (row["game_slug"], int(row["player_id"])) if row["player_id"] is not None else None
            actual = actuals.get(key)
            if not actual:
                continue
            stat = row["stat"] or row["market"]
            gl_col = _STAT_COL.get(stat)
            if not gl_col:
                continue
            actual_value = _clean_float(actual.get(gl_col))
            market_line = _clean_float(row["market_line"])
            if actual_value is None or market_line is None:
                continue
            push = abs(actual_value - market_line) <= 1e-9
            over_hit = None if push else actual_value > market_line
            won = None if push else (over_hit if row["side"] == "over" else not over_hit)
            close = resolve_valid_prop_close(conn, dict(row))
            closing_line = closing_price = None
            if close["valid"]:
                closing_line = _clean_float(close.get("line"))
                closing_price = _clean_float(
                    close.get("over_price") if row["side"] == "over" else close.get("under_price")
                )
            clv_line = None
            if closing_line is not None:
                clv_line = round(closing_line - market_line, 2) if row["side"] == "over" else round(market_line - closing_line, 2)
            clv_price = _price_clv(row["market_price"], closing_price)
            profit = _profit_units(won, push, row["market_price"])
            cur.execute(
                """
                UPDATE bets.mlb_model_pick_ledger
                SET result_status = 'graded',
                    won = %s,
                    push = %s,
                    profit_units = %s,
                    actual_value = %s,
                    over_hit = %s,
                    closing_line = %s,
                    closing_price = %s,
                    clv_line = %s,
                    clv_price = %s,
                    closing_snapshot_id = %s,
                    clv_valid = %s,
                    clv_status = %s,
                    clv_unknown_reason = %s,
                    graded_at_utc = NOW(),
                    grade_source = 'raw.mlb_player_gamelogs'
                WHERE id = %s
                """,
                (
                    won,
                    push,
                    profit,
                    actual_value,
                    over_hit,
                    closing_line,
                    closing_price,
                    clv_line,
                    clv_price,
                    close.get("snapshot_id") if close["valid"] else None,
                    close["valid"],
                    close["status"],
                    close["unknown_reason"],
                    row["id"],
                ),
            )
            updated += 1
    log.info("Graded %d prop model-pick ledger rows", updated)
    return updated
