"""Locked bankroll ledger for MLB real-money candidates.

Prediction tables are allowed to be overwritten by reruns. This ledger is not:
the first observed bankroll candidate for a deterministic pick key is inserted
once and later graded in place.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2
import psycopg2.extras

from .game_line_clv import game_line_clv, resolve_valid_game_close
from .prop_offer_snapshots import (
    ensure_prop_offer_snapshot_schema,
    resolve_valid_prop_close,
)

log = logging.getLogger("mlb_pipeline.modeling.bankroll_ledger")

_ET = ZoneInfo("America/New_York")
_SCHEMA_READY = False
_SCHEMA_LOCK_KEY = "mlb_prop_schema_ddl"

_STAT_COL = {
    "pitcher_strikeouts": "strikeouts_pitcher",
    "batter_hits": "hits",
    "batter_home_runs": "home_runs",
    "batter_total_bases": "total_bases",
    "batter_walks": "walks_batter",
}


def _normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "")
    ascii_str = s.encode("ascii", "ignore").decode("ascii")
    ascii_str = re.sub(r"[^a-z0-9\s]", "", ascii_str.lower())
    return re.sub(r"\s+", " ", ascii_str).strip()


def _clean_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_int(value) -> int | None:
    val = _clean_float(value)
    return int(val) if val is not None else None


def _json(data: dict[str, Any] | None) -> psycopg2.extras.Json:
    return psycopg2.extras.Json(data or {}, dumps=lambda obj: json.dumps(obj, default=str))


def _pick_key(*parts: Any) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def game_bankroll_pick_key(row: dict[str, Any], market: str) -> str | None:
    if market == "run_line":
        side = row.get("run_line_bet_side")
    elif market == "total":
        side = row.get("total_bet_side")
    else:
        return None
    if not side:
        return None
    return _pick_key(
        "mlb",
        "game",
        row.get("game_date_et"),
        row.get("game_slug"),
        market,
        side,
    )


def game_bankroll_risk_slot(row: dict[str, Any], market: str) -> str | None:
    if market not in {"run_line", "total"}:
        return None
    return _pick_key(
        "mlb",
        "bankroll_risk",
        "game",
        row.get("game_date_et"),
        row.get("game_slug"),
        market,
    )


def _prop_link_bookmaker(
    row: dict[str, Any],
    prop_lines=None,
) -> tuple[str | None, str | None]:
    side = (row.get("bet_side") or "").lower()
    name = row.get("player_name") or ""
    stat = row.get("stat")
    norm = _normalize_name(name)
    ld = (prop_lines or {}).get((norm, stat), {})
    link = row.get("bet_link") or (
        ld.get("over_link") if side == "over" else ld.get("under_link")
    )
    bookmaker = row.get("bookmaker_key")
    if link and "fanduel.com" in link:
        bookmaker = "fanduel"
    elif link and "draftkings.com" in link:
        bookmaker = "draftkings"
    elif side == "under" and ld.get("under_link_book"):
        bookmaker = ld.get("under_link_book")
    return link, bookmaker


def prop_bankroll_pick_key(row: dict[str, Any], prop_lines=None) -> str | None:
    side = (row.get("bet_side") or "").lower()
    if side not in {"over", "under"}:
        return None
    link, bookmaker = _prop_link_bookmaker(row, prop_lines)
    line = _clean_float(row.get("book_line"))
    prop_offer_id = _clean_int(row.get("prop_offer_id"))
    return _pick_key(
        "mlb",
        "prop",
        row.get("game_date_et"),
        row.get("game_slug"),
        row.get("player_id"),
        row.get("stat"),
        side,
        line,
        bookmaker,
        prop_offer_id or row.get("prediction_key") or link,
    )


def prop_bankroll_risk_slot(row: dict[str, Any]) -> str | None:
    if not row.get("stat"):
        return None
    return _pick_key(
        "mlb",
        "bankroll_risk",
        "prop",
        row.get("game_date_et"),
        row.get("game_slug"),
        row.get("player_id"),
        row.get("stat"),
    )


def _ledger_risk_slot(row: dict[str, Any]) -> str | None:
    source = row.get("source")
    if source == "game":
        return game_bankroll_risk_slot(row, row.get("market"))
    if source == "prop":
        return prop_bankroll_risk_slot(row)
    return None


def _american_profit_mult(price) -> float | None:
    price = _clean_float(price)
    if price is None or price == 0:
        return None
    if price > 0:
        return price / 100.0
    return 100.0 / abs(price)


def _profit_units(won: bool | None, push: bool, price) -> float | None:
    if push:
        return 0.0
    if won is None:
        return None
    mult = _american_profit_mult(price)
    if mult is None:
        return None
    return mult if won else -1.0


def _ev_per_unit(p_win, price) -> float | None:
    p = _clean_float(p_win)
    mult = _american_profit_mult(price)
    if p is None or mult is None:
        return None
    return round(p * mult - (1.0 - p), 4)


def _american_to_prob(price: float) -> float:
    return 100.0 / (price + 100.0) if price >= 100 else abs(price) / (abs(price) + 100.0)


def _price_clv(entry_price, closing_price) -> float | None:
    entry = _clean_float(entry_price)
    close = _clean_float(closing_price)
    if entry is None or close is None:
        return None
    return round((_american_to_prob(close) - _american_to_prob(entry)) * 100.0, 2)


def _run_line_side_label(home_market_line, side: str | None) -> str | None:
    line = _clean_float(home_market_line)
    if line is None or side not in {"home", "away"}:
        return None
    side_line = line if side == "home" else -line
    return f"{side_line:+.1f}"


def _cfg_thresholds(cfg) -> dict[str, Any]:
    if cfg is None:
        return {}
    keys = [
        "min_edge_run_line",
        "min_edge_total",
        "max_run_line_lay_price",
        "max_away_dog_run_line_lay_price",
        "max_total_lay_price",
        "threshold_strikeouts",
        "threshold_strikeouts_over",
        "threshold_strikeouts_under",
        "threshold_hits",
        "threshold_total_bases",
        "threshold_total_bases_over",
        "threshold_total_bases_under",
        "threshold_home_runs_over",
        "threshold_home_runs_under",
        "threshold_clf",
        "min_ev",
        "bankroll_max_stake_pct",
        "bankroll_max_daily_exposure_pct",
        "bankroll_max_lay_price",
        "bankroll_reference_usd",
        "bankroll_micro_stake_usd",
        "bankroll_starter_stake_pct",
    ]
    return {key: getattr(cfg, key) for key in keys if hasattr(cfg, key)}


def ensure_bankroll_ledger_schema(conn) -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    if _bankroll_ledger_exists(conn) and _bankroll_ledger_has_required_columns(conn):
        _SCHEMA_READY = True
        return
    try:
        ensure_prop_offer_snapshot_schema(conn)
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '2s'")
            cur.execute("SET LOCAL statement_timeout = '15s'")
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (_SCHEMA_LOCK_KEY,))
            cur.execute(
                """
                CREATE SCHEMA IF NOT EXISTS bets;
                CREATE TABLE IF NOT EXISTS bets.mlb_bankroll_ledger (
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
                    bankroll_tier TEXT,
                    bankroll_reasons TEXT,
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
                ALTER TABLE bets.mlb_bankroll_ledger
                    ADD COLUMN IF NOT EXISTS model_meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                    ADD COLUMN IF NOT EXISTS thresholds JSONB NOT NULL DEFAULT '{}'::jsonb,
                    ADD COLUMN IF NOT EXISTS prediction_key TEXT,
                    ADD COLUMN IF NOT EXISTS prop_offer_id BIGINT,
                    ADD COLUMN IF NOT EXISTS prop_offer_source_row_id INTEGER,
                    ADD COLUMN IF NOT EXISTS bookmaker_key TEXT,
                    ADD COLUMN IF NOT EXISTS bet_line NUMERIC,
                    ADD COLUMN IF NOT EXISTS minimum_acceptable_price NUMERIC,
                    ADD COLUMN IF NOT EXISTS player_name_norm TEXT,
                    ADD COLUMN IF NOT EXISTS push BOOLEAN NOT NULL DEFAULT FALSE,
                    ADD COLUMN IF NOT EXISTS clv_price NUMERIC,
                    ADD COLUMN IF NOT EXISTS closing_snapshot_id BIGINT,
                    ADD COLUMN IF NOT EXISTS lock_snapshot_id BIGINT,
                    ADD COLUMN IF NOT EXISTS locked_at_utc TIMESTAMPTZ,
                    ADD COLUMN IF NOT EXISTS clv_valid BOOLEAN,
                    ADD COLUMN IF NOT EXISTS clv_status TEXT,
                    ADD COLUMN IF NOT EXISTS clv_unknown_reason TEXT;
                ALTER TABLE bets.mlb_bankroll_ledger
                    ADD COLUMN IF NOT EXISTS stake_usd NUMERIC;
                UPDATE bets.mlb_bankroll_ledger
                SET bet_line = CASE
                    WHEN market = 'run_line' AND side = 'away' THEN -market_line
                    ELSE market_line
                END
                WHERE bet_line IS NULL
                  AND market_line IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_mlb_bankroll_ledger_date
                    ON bets.mlb_bankroll_ledger (game_date_et);
                CREATE INDEX IF NOT EXISTS idx_mlb_bankroll_ledger_pending
                    ON bets.mlb_bankroll_ledger (result_status, game_date_et);
                CREATE INDEX IF NOT EXISTS idx_mlb_bankroll_ledger_prop_offer
                    ON bets.mlb_bankroll_ledger (prop_offer_id);
                """
            )
        conn.commit()
        _SCHEMA_READY = True
    except Exception:
        conn.rollback()
        raise


def _is_schema_lock_error(exc: Exception) -> bool:
    return getattr(exc, "pgcode", None) in {"40P01", "55P03"}


def _bankroll_ledger_exists(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('bets.mlb_bankroll_ledger')")
        return cur.fetchone()[0] is not None


def _bankroll_ledger_has_required_columns(conn) -> bool:
    required = {
        "pick_key", "source", "game_date_et", "game_slug", "market", "side",
        "prediction_key", "prop_offer_id", "prop_offer_source_row_id",
        "bookmaker_key", "market_line", "bet_line", "market_price",
        "minimum_acceptable_price", "stake_pct", "stake_usd", "model_meta",
        "thresholds", "result_status", "profit_units", "lock_snapshot_id",
        "locked_at_utc", "clv_valid", "clv_status", "clv_unknown_reason",
    }
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'bets'
              AND table_name = 'mlb_bankroll_ledger'
            """
        )
        existing = {str(row[0]) for row in cur.fetchall()}
    return required.issubset(existing)


_INSERT_SQL = """
INSERT INTO bets.mlb_bankroll_ledger (
    pick_key, source, game_date_et, game_slug, market, stat,
    prediction_key, prop_offer_id, prop_offer_source_row_id, side, label,
    team_abbr, opponent_abbr, home_team_abbr, away_team_abbr,
    player_id, player_name, player_name_norm, bookmaker_key,
    market_line, bet_line, market_price, link, pred_value, pred_count, model_prob,
    edge, edge_type, ev, kelly_fraction, bankroll_tier, bankroll_reasons,
    stake_pct, stake_usd, minimum_acceptable_price, locked_at_utc, model_meta, thresholds
) VALUES (
    %(pick_key)s, %(source)s, %(game_date_et)s, %(game_slug)s, %(market)s, %(stat)s,
    %(prediction_key)s, %(prop_offer_id)s, %(prop_offer_source_row_id)s,
    %(side)s, %(label)s, %(team_abbr)s, %(opponent_abbr)s, %(home_team_abbr)s,
    %(away_team_abbr)s, %(player_id)s, %(player_name)s, %(player_name_norm)s,
    %(bookmaker_key)s, %(market_line)s, %(bet_line)s, %(market_price)s, %(link)s,
    %(pred_value)s, %(pred_count)s, %(model_prob)s, %(edge)s, %(edge_type)s, %(ev)s,
    %(kelly_fraction)s, %(bankroll_tier)s, %(bankroll_reasons)s, %(stake_pct)s,
    %(stake_usd)s, %(minimum_acceptable_price)s, %(locked_at_utc)s, %(model_meta)s, %(thresholds)s
) ON CONFLICT (pick_key) DO NOTHING
"""


def _locked_bankroll_state_for_date(cur, game_date_et) -> tuple[float, set[str], set[str]]:
    cur.execute(
        """
        SELECT
            pick_key, source, game_date_et, game_slug, market, stat, player_id,
            COALESCE(stake_pct, 0) AS stake_pct
        FROM bets.mlb_bankroll_ledger
        WHERE game_date_et = %s
          AND result_status <> 'voided'
        """,
        (game_date_et,),
    )
    exposure = 0.0
    pick_keys: set[str] = set()
    risk_slots: set[str] = set()
    for row in cur.fetchall():
        if not isinstance(row, dict):
            row = {
                "pick_key": row[0],
                "source": row[1],
                "game_date_et": row[2],
                "game_slug": row[3],
                "market": row[4],
                "stat": row[5],
                "player_id": row[6],
                "stake_pct": row[7],
            }
        pick_keys.add(row["pick_key"])
        slot = _ledger_risk_slot(row)
        if slot:
            risk_slots.add(slot)
        exposure += max(_clean_float(row.get("stake_pct")) or 0.0, 0.0)
    return exposure, pick_keys, risk_slots


def locked_bankroll_state(conn, game_date_et) -> tuple[float, set[str], set[str]]:
    """Return non-voided daily exposure, exact pick keys, and occupied risk slots."""
    try:
        ensure_bankroll_ledger_schema(conn)
    except psycopg2.Error as exc:
        conn.rollback()
        if not _is_schema_lock_error(exc) or not _bankroll_ledger_exists(conn):
            raise
        log.warning(
            "Bankroll ledger schema ensure hit transient lock/deadlock; "
            "reading existing ledger state without running DDL",
            exc_info=True,
        )
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        return _locked_bankroll_state_for_date(cur, game_date_et)


def insert_ledger_rows(
    conn,
    rows: Iterable[dict[str, Any]],
    *,
    max_daily_exposure_pct: float | None = None,
) -> int:
    payload = list(rows)
    if not payload:
        return 0
    ensure_bankroll_ledger_schema(conn)
    cap = _clean_float(max_daily_exposure_pct)
    for row in payload:
        row.setdefault("prediction_key", None)
        row.setdefault("prop_offer_id", None)
        row.setdefault("prop_offer_source_row_id", None)
        row.setdefault("stake_usd", None)
        row.setdefault("minimum_acceptable_price", None)
        row.setdefault("locked_at_utc", datetime.now(timezone.utc))
        row["model_meta"] = _json(row.get("model_meta"))
        row["thresholds"] = _json(row.get("thresholds"))
    rows_by_date: dict[Any, list[dict[str, Any]]] = {}
    for row in payload:
        rows_by_date.setdefault(row.get("game_date_et"), []).append(row)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        inserted = 0
        for game_date_et, date_rows in rows_by_date.items():
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"mlb_bankroll_ledger:{game_date_et}",),
            )
            exposure, pick_keys, risk_slots = _locked_bankroll_state_for_date(
                cur,
                game_date_et,
            )
            for row in date_rows:
                pick_key = row.get("pick_key")
                if pick_key in pick_keys:
                    continue
                risk_slot = _ledger_risk_slot(row)
                if risk_slot and risk_slot in risk_slots:
                    log.warning(
                        "Skipped MLB bankroll lock for occupied risk slot: %s",
                        row.get("label") or pick_key,
                    )
                    continue
                stake = max(_clean_float(row.get("stake_pct")) or 0.0, 0.0)
                if cap is not None and exposure + stake > cap + 1e-12:
                    log.warning(
                        "Skipped MLB bankroll lock above daily cap: %s "
                        "(used=%.4f stake=%.4f cap=%.4f)",
                        row.get("label") or pick_key,
                        exposure,
                        stake,
                        cap,
                    )
                    continue
                cur.execute(_INSERT_SQL + " RETURNING id", row)
                if cur.fetchone() is not None:
                    inserted += 1
                    exposure += stake
                    pick_keys.add(pick_key)
                    if risk_slot:
                        risk_slots.add(risk_slot)
    conn.commit()
    return inserted


def void_negative_ev_game_bankroll_candidates(conn, cfg=None) -> int:
    """Void pending game bankroll rows that do not clear price-adjusted EV."""
    ensure_bankroll_ledger_schema(conn)
    min_ev = _clean_float(getattr(cfg, "min_game_ev", None)) if cfg is not None else 0.02
    min_ev = 0.02 if min_ev is None else min_ev
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, model_prob, market_price, bankroll_reasons
            FROM bets.mlb_bankroll_ledger
            WHERE source = 'game'
              AND result_status = 'pending'
              AND model_prob IS NOT NULL
              AND market_price IS NOT NULL
            """
        )
        fills = []
        voids = []
        for row in cur.fetchall():
            ev = _ev_per_unit(row.get("model_prob"), row.get("market_price"))
            if ev is None:
                continue
            fills.append((ev, row["id"]))
            if ev >= min_ev:
                continue
            reason = "negative_price_ev" if ev < 0 else "below_min_price_ev"
            existing = row.get("bankroll_reasons") or ""
            parts = [p.strip() for p in existing.split(";") if p.strip()]
            if reason not in parts:
                parts.append(reason)
            voids.append((ev, "; ".join(parts), row["id"]))
        for ev, row_id in fills:
            cur.execute(
                """
                UPDATE bets.mlb_bankroll_ledger
                SET ev = COALESCE(ev, %s)
                WHERE id = %s
                """,
                (ev, row_id),
            )
        for ev, reasons, row_id in voids:
            cur.execute(
                """
                UPDATE bets.mlb_bankroll_ledger
                SET result_status = 'voided',
                    ev = %s,
                    bankroll_reasons = %s,
                    profit_units = 0,
                    graded_at_utc = NOW(),
                    grade_source = 'price_ev_reconciliation'
                WHERE id = %s
                  AND result_status = 'pending'
                """,
                (ev, reasons, row_id),
            )
    conn.commit()
    return len(voids)


def insert_game_bankroll_ledger(conn, rows: list[dict[str, Any]], *, fd_links=None, cfg=None) -> int:
    thresholds = _cfg_thresholds(cfg)
    out: list[dict[str, Any]] = []
    for row in rows:
        game_date = row.get("game_date_et")
        slug = row.get("game_slug")
        home = row.get("home_team_abbr")
        away = row.get("away_team_abbr")
        fd = (fd_links or {}).get(slug) or (fd_links or {}).get((home, away))

        if row.get("bankroll_candidate_rl"):
            side = row.get("run_line_bet_side")
            team = home if side == "home" else away
            opp = away if side == "home" else home
            link = None
            if fd is not None:
                link = getattr(fd, "spread_home_link", None) if side == "home" else getattr(fd, "spread_away_link", None)
            line_label = _run_line_side_label(row.get("market_run_line"), side)
            home_line = _clean_float(row.get("market_run_line"))
            bet_line = home_line if side == "home" else (-home_line if home_line is not None else None)
            model_prob = _clean_float(row.get("win_prob_rl"))
            market_price = _clean_float(row.get("market_rl_price"))
            out.append({
                "pick_key": game_bankroll_pick_key(row, "run_line"),
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
                "bookmaker_key": "fanduel" if link else None,
                "market_line": home_line,
                "bet_line": bet_line,
                "market_price": market_price,
                "link": link,
                "pred_value": _clean_float(row.get("pred_run_diff")),
                "pred_count": None,
                "model_prob": model_prob,
                "edge": _clean_float(row.get("edge_run_line")),
                "edge_type": "run_line",
                "ev": _ev_per_unit(model_prob, market_price),
                "kelly_fraction": _clean_float(row.get("kelly_fraction_rl")),
                "bankroll_tier": row.get("bankroll_tier_rl"),
                "bankroll_reasons": row.get("bankroll_reasons_rl"),
                "stake_pct": _clean_float(row.get("stake_pct_rl")),
                "model_meta": {
                    "season": row.get("season"),
                    "used_residual_model": row.get("used_residual_model"),
                    "sigma_q_rl": row.get("sigma_q_rl"),
                    "p_home_cover_clf": row.get("p_home_cover_clf"),
                },
                "thresholds": thresholds,
            })

        if row.get("bankroll_candidate_total"):
            side = row.get("total_bet_side")
            link = None
            if fd is not None:
                link = getattr(fd, "total_over_link", None) if side == "over" else getattr(fd, "total_under_link", None)
            model_prob = _clean_float(row.get("win_prob_total"))
            market_price = _clean_float(row.get("market_total_price"))
            out.append({
                "pick_key": game_bankroll_pick_key(row, "total"),
                "source": "game",
                "game_date_et": game_date,
                "game_slug": slug,
                "market": "total",
                "stat": None,
                "side": side,
                "label": f"{side.upper()} {row.get('market_total')}",
                "team_abbr": None,
                "opponent_abbr": None,
                "home_team_abbr": home,
                "away_team_abbr": away,
                "player_id": None,
                "player_name": None,
                "player_name_norm": None,
                "bookmaker_key": "fanduel" if link else None,
                "market_line": _clean_float(row.get("market_total")),
                "bet_line": _clean_float(row.get("market_total")),
                "market_price": market_price,
                "link": link,
                "pred_value": _clean_float(row.get("pred_total")),
                "pred_count": None,
                "model_prob": model_prob,
                "edge": _clean_float(row.get("edge_total")),
                "edge_type": "total",
                "ev": _ev_per_unit(model_prob, market_price),
                "kelly_fraction": _clean_float(row.get("kelly_fraction_total")),
                "bankroll_tier": row.get("bankroll_tier_total"),
                "bankroll_reasons": row.get("bankroll_reasons_total"),
                "stake_pct": _clean_float(row.get("stake_pct_total")),
                "model_meta": {
                    "season": row.get("season"),
                    "used_residual_model": row.get("used_residual_model"),
                    "sigma_q_total": row.get("sigma_q_total"),
                    "p_total_over_clf": row.get("p_total_over_clf"),
                },
                "thresholds": thresholds,
            })
    cap = _clean_float(getattr(cfg, "bankroll_max_daily_exposure_pct", None)) if cfg else None
    inserted = insert_ledger_rows(conn, out, max_daily_exposure_pct=cap)
    voided = void_negative_ev_game_bankroll_candidates(conn, cfg=cfg)
    if voided:
        log.info("Voided %d pending MLB game bankroll rows below price-adjusted EV minimum", voided)
    return inserted


def insert_prop_bankroll_ledger(conn, rows: list[dict[str, Any]], *, prop_lines=None, cfg=None) -> int:
    thresholds = _cfg_thresholds(cfg)
    out: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("bankroll_candidate"):
            continue
        side = (row.get("bet_side") or "").lower()
        if side not in {"over", "under"}:
            continue
        name = row.get("player_name") or ""
        stat = row.get("stat")
        norm = _normalize_name(name)
        link, bookmaker = _prop_link_bookmaker(row, prop_lines)
        p_over = _clean_float(row.get("pred_prob_over"))
        model_prob = p_over if side == "over" else (1.0 - p_over if p_over is not None else None)
        line = _clean_float(row.get("book_line"))
        prop_offer_id = _clean_int(row.get("prop_offer_id"))
        prop_offer_source_row_id = _clean_int(row.get("prop_offer_source_row_id"))
        out.append({
            "pick_key": prop_bankroll_pick_key(row, prop_lines),
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
            "market_price": _clean_float(row.get("bet_price")),
            "link": link,
            "pred_value": _clean_float(row.get("pred_value")),
            "pred_count": _clean_float(row.get("pred_count")),
            "model_prob": model_prob,
            "edge": _clean_float(row.get("edge")),
            "edge_type": row.get("edge_type"),
            "ev": _clean_float(row.get("ev")),
            "kelly_fraction": _clean_float(row.get("kelly_fraction")),
            "bankroll_tier": row.get("bankroll_tier"),
            "bankroll_reasons": row.get("bankroll_reasons"),
            "stake_pct": _clean_float(row.get("stake_pct")),
            "stake_usd": _clean_float(row.get("stake_usd")),
            "minimum_acceptable_price": _clean_float(row.get("minimum_acceptable_price")),
            "model_meta": {
                "model_family": row.get("model_family"),
                "line_bucket": row.get("line_bucket"),
                "bookmaker_key": row.get("bookmaker_key"),
                "prediction_key": row.get("prediction_key"),
                "prop_offer_id": prop_offer_id,
                "prop_offer_source_row_id": prop_offer_source_row_id,
            },
            "thresholds": thresholds,
        })
    cap = _clean_float(getattr(cfg, "bankroll_max_daily_exposure_pct", None)) if cfg else None
    return insert_ledger_rows(conn, out, max_daily_exposure_pct=cap)


def grade_bankroll_ledger(conn) -> int:
    ensure_bankroll_ledger_schema(conn)
    updated = _grade_game_ledger(conn) + _grade_prop_ledger(conn)
    conn.commit()
    return updated


def _grade_game_ledger(conn) -> int:
    today = datetime.now(_ET).date()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT *
            FROM bets.mlb_bankroll_ledger
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
                UPDATE bets.mlb_bankroll_ledger
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
    log.info("Graded %d game bankroll ledger rows", updated)
    return updated


def _prop_close(conn, row: dict) -> tuple[float | None, float | None, float | None, float | None]:
    close = resolve_valid_prop_close(conn, dict(row))
    if not close["valid"]:
        return None, None, None, None
    closing_line = _clean_float(close.get("line"))
    closing_price = _clean_float(close.get("over_price") if row["side"] == "over" else close.get("under_price"))
    return closing_line, closing_price, None, None


def _grade_prop_ledger(conn) -> int:
    today = datetime.now(_ET).date()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT *
            FROM bets.mlb_bankroll_ledger
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
                UPDATE bets.mlb_bankroll_ledger
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
    log.info("Graded %d prop bankroll ledger rows", updated)
    return updated
