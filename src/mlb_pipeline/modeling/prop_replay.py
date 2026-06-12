"""Utilities for MLB prop prediction replay/backtest storage.

The replay table is intentionally separate from bets.mlb_prop_predictions:
daily predictions remain the live/latest surface, while replay rows are locked
snapshots used for calibration, betting-layer training, diagnostics, and CLV.
"""
from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Iterable

import psycopg2
import psycopg2.extras

from .prop_offer_snapshots import (
    ensure_prop_offer_snapshot_schema,
    insert_lock_snapshots_for_predictions,
    resolve_valid_prop_close,
)

_STAT_COL = {
    "pitcher_strikeouts": "strikeouts_pitcher",
    "batter_hits": "hits",
    "batter_home_runs": "home_runs",
    "batter_total_bases": "total_bases",
    "batter_walks": "walks_batter",
}


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_name = nfkd.encode("ascii", errors="ignore").decode("ascii")
    return re.sub(r"[^a-z ]", "", ascii_name.lower()).strip()


def american_to_prob(price: Any) -> float | None:
    if price is None:
        return None
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p) or p == 0:
        return None
    return 100.0 / (p + 100.0) if p > 0 else abs(p) / (abs(p) + 100.0)


def no_vig_probs(over_price: Any, under_price: Any) -> tuple[float | None, float | None]:
    po = american_to_prob(over_price)
    pu = american_to_prob(under_price)
    if po is None or pu is None:
        return None, None
    total = po + pu
    if total <= 0:
        return None, None
    return po / total, pu / total


def ev_per_unit(p_win: Any, price: Any) -> float | None:
    try:
        p = float(p_win)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p):
        return None
    try:
        pr = float(price)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(pr) or pr == 0:
        return None
    profit = pr / 100.0 if pr > 0 else 100.0 / abs(pr)
    return p * profit - (1.0 - p)


def price_clv(entry_price: Any, closing_price: Any) -> float | None:
    entry = american_to_prob(entry_price)
    close = american_to_prob(closing_price)
    if entry is None or close is None:
        return None
    return round((close - entry) * 100.0, 2)


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def ensure_prop_replay_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE SCHEMA IF NOT EXISTS bets;
            CREATE TABLE IF NOT EXISTS bets.mlb_prop_prediction_replay (
                id BIGSERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                run_started_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
                source_pred_id INTEGER,
                prediction_key TEXT,
                prop_offer_id BIGINT,
                prop_offer_source_row_id INTEGER,
                lock_snapshot_id BIGINT,
                locked_at_utc TIMESTAMPTZ,
                game_date_et DATE NOT NULL,
                game_slug TEXT NOT NULL,
                player_id BIGINT NOT NULL,
                player_name TEXT,
                player_name_norm TEXT,
                team_abbr TEXT,
                stat TEXT NOT NULL,
                model_family TEXT,
                edge_type TEXT,
                side TEXT,
                line_bucket TEXT,
                bookmaker_key TEXT,
                market_line NUMERIC,
                over_price NUMERIC,
                under_price NUMERIC,
                market_price NUMERIC,
                minimum_acceptable_price NUMERIC,
                bet_link TEXT,
                breakeven_prob NUMERIC,
                market_prob_over NUMERIC,
                market_prob_under NUMERIC,
                no_vig_prob_over NUMERIC,
                no_vig_prob_under NUMERIC,
                pred_value NUMERIC,
                pred_count NUMERIC,
                model_prob_over NUMERIC,
                model_prob_side NUMERIC,
                prob_edge_vs_market NUMERIC,
                count_edge_vs_line NUMERIC,
                edge NUMERIC,
                ev NUMERIC,
                kelly_fraction NUMERIC,
                bankroll_tier TEXT,
                bankroll_candidate BOOLEAN,
                bankroll_reasons TEXT,
                stake_pct NUMERIC,
                stake_usd NUMERIC,
                actual_value NUMERIC,
                over_hit BOOLEAN,
                won BOOLEAN,
                push BOOLEAN,
                profit_units NUMERIC,
                closing_line NUMERIC,
                closing_price NUMERIC,
                clv_line NUMERIC,
                clv_price NUMERIC,
                closing_source_row_id BIGINT,
                closing_snapshot_id BIGINT,
                closing_fetched_at_utc TIMESTAMPTZ,
                clv_match_method TEXT,
                clv_valid BOOLEAN,
                clv_status TEXT,
                clv_unknown_reason TEXT,
                result_status TEXT NOT NULL DEFAULT 'pending',
                graded_at_utc TIMESTAMPTZ,
                source_created_at TIMESTAMPTZ,
                UNIQUE (run_id, source_pred_id)
            );
            ALTER TABLE bets.mlb_prop_prediction_replay
                ADD COLUMN IF NOT EXISTS prediction_key TEXT,
                ADD COLUMN IF NOT EXISTS prop_offer_id BIGINT,
                ADD COLUMN IF NOT EXISTS prop_offer_source_row_id INTEGER,
                ADD COLUMN IF NOT EXISTS lock_snapshot_id BIGINT,
                ADD COLUMN IF NOT EXISTS locked_at_utc TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS bet_link TEXT,
                ADD COLUMN IF NOT EXISTS minimum_acceptable_price NUMERIC,
                ADD COLUMN IF NOT EXISTS stake_usd NUMERIC,
                ADD COLUMN IF NOT EXISTS closing_source_row_id BIGINT,
                ADD COLUMN IF NOT EXISTS closing_snapshot_id BIGINT,
                ADD COLUMN IF NOT EXISTS closing_fetched_at_utc TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS clv_match_method TEXT,
                ADD COLUMN IF NOT EXISTS clv_valid BOOLEAN,
                ADD COLUMN IF NOT EXISTS clv_status TEXT,
                ADD COLUMN IF NOT EXISTS clv_unknown_reason TEXT;
            ALTER TABLE IF EXISTS bets.mlb_prop_predictions
                ADD COLUMN IF NOT EXISTS lock_snapshot_id BIGINT,
                ADD COLUMN IF NOT EXISTS locked_at_utc TIMESTAMPTZ;
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_replay_date
                ON bets.mlb_prop_prediction_replay (game_date_et);
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_replay_status
                ON bets.mlb_prop_prediction_replay (result_status, game_date_et);
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_replay_bucket
                ON bets.mlb_prop_prediction_replay (stat, side, line_bucket, model_family);
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_replay_offer
                ON bets.mlb_prop_prediction_replay (prop_offer_id);
            """
        )
    conn.commit()
    ensure_prop_offer_snapshot_schema(conn)


def _row_from_prediction(row: dict[str, Any], run_id: str, run_started_at_utc: datetime) -> dict[str, Any]:
    side = (row.get("bet_side") or "").lower() or None
    p_over = _clean_float(row.get("pred_prob_over"))
    p_side = None
    if p_over is not None:
        if side == "over":
            p_side = p_over
        elif side == "under":
            p_side = 1.0 - p_over
    over_price = _clean_float(row.get("over_price"))
    under_price = _clean_float(row.get("under_price"))
    no_vig_over, no_vig_under = no_vig_probs(over_price, under_price)
    market_prob_over = american_to_prob(over_price)
    market_prob_under = american_to_prob(under_price)
    market_prob_side = no_vig_over if side == "over" else (no_vig_under if side == "under" else None)
    prob_edge_vs_market = None
    if p_side is not None and market_prob_side is not None:
        prob_edge_vs_market = p_side - market_prob_side
    pred_count = _clean_float(row.get("pred_count"))
    market_line = _clean_float(row.get("book_line"))
    count_edge = None
    if pred_count is not None and market_line is not None:
        count_edge = pred_count - market_line
    return {
        "run_id": run_id,
        "run_started_at_utc": run_started_at_utc,
        "source_pred_id": row.get("id"),
        "prediction_key": row.get("prediction_key"),
        "prop_offer_id": row.get("prop_offer_id"),
        "prop_offer_source_row_id": row.get("prop_offer_source_row_id"),
        "lock_snapshot_id": row.get("lock_snapshot_id"),
        "locked_at_utc": row.get("locked_at_utc"),
        "game_date_et": row.get("game_date_et"),
        "game_slug": row.get("game_slug"),
        "player_id": row.get("player_id"),
        "player_name": row.get("player_name"),
        "player_name_norm": normalize_name(row.get("player_name")),
        "team_abbr": row.get("team_abbr"),
        "stat": row.get("stat"),
        "model_family": row.get("model_family"),
        "edge_type": row.get("edge_type"),
        "side": side,
        "line_bucket": row.get("line_bucket"),
        "bookmaker_key": row.get("bookmaker_key"),
        "market_line": market_line,
        "over_price": over_price,
        "under_price": under_price,
        "market_price": _clean_float(row.get("bet_price")),
        "minimum_acceptable_price": _clean_float(row.get("minimum_acceptable_price")),
        "bet_link": row.get("bet_link"),
        "breakeven_prob": _clean_float(row.get("breakeven_prob")),
        "market_prob_over": market_prob_over,
        "market_prob_under": market_prob_under,
        "no_vig_prob_over": no_vig_over,
        "no_vig_prob_under": no_vig_under,
        "pred_value": _clean_float(row.get("pred_value")),
        "pred_count": pred_count,
        "model_prob_over": p_over,
        "model_prob_side": p_side,
        "prob_edge_vs_market": prob_edge_vs_market,
        "count_edge_vs_line": count_edge,
        "edge": _clean_float(row.get("edge")),
        "ev": _clean_float(row.get("ev")),
        "kelly_fraction": _clean_float(row.get("kelly_fraction")),
        "bankroll_tier": row.get("bankroll_tier"),
        "bankroll_candidate": row.get("bankroll_candidate"),
        "bankroll_reasons": row.get("bankroll_reasons"),
        "stake_pct": _clean_float(row.get("stake_pct")),
        "stake_usd": _clean_float(row.get("stake_usd")),
        "actual_value": _clean_float(row.get("actual_value")),
        "over_hit": row.get("over_hit"),
        "source_created_at": row.get("created_at"),
    }


_REPLAY_INSERT_SQL = """
INSERT INTO bets.mlb_prop_prediction_replay (
    run_id, run_started_at_utc, source_pred_id, prediction_key,
    prop_offer_id, prop_offer_source_row_id, lock_snapshot_id, locked_at_utc,
    game_date_et, game_slug,
    player_id, player_name, player_name_norm, team_abbr, stat, model_family,
    edge_type, side, line_bucket, bookmaker_key, market_line, over_price,
    under_price, market_price, minimum_acceptable_price, bet_link, breakeven_prob, market_prob_over,
    market_prob_under, no_vig_prob_over, no_vig_prob_under, pred_value,
    pred_count, model_prob_over, model_prob_side, prob_edge_vs_market,
    count_edge_vs_line, edge, ev, kelly_fraction, bankroll_tier,
    bankroll_candidate, bankroll_reasons, stake_pct, stake_usd, actual_value, over_hit,
    source_created_at
) VALUES (
    %(run_id)s, %(run_started_at_utc)s, %(source_pred_id)s, %(prediction_key)s,
    %(prop_offer_id)s, %(prop_offer_source_row_id)s, %(lock_snapshot_id)s,
    %(locked_at_utc)s, %(game_date_et)s, %(game_slug)s,
    %(player_id)s, %(player_name)s, %(player_name_norm)s, %(team_abbr)s, %(stat)s, %(model_family)s,
    %(edge_type)s, %(side)s, %(line_bucket)s, %(bookmaker_key)s, %(market_line)s, %(over_price)s,
    %(under_price)s, %(market_price)s, %(minimum_acceptable_price)s, %(bet_link)s, %(breakeven_prob)s, %(market_prob_over)s,
    %(market_prob_under)s, %(no_vig_prob_over)s, %(no_vig_prob_under)s, %(pred_value)s,
    %(pred_count)s, %(model_prob_over)s, %(model_prob_side)s, %(prob_edge_vs_market)s,
    %(count_edge_vs_line)s, %(edge)s, %(ev)s, %(kelly_fraction)s, %(bankroll_tier)s,
    %(bankroll_candidate)s, %(bankroll_reasons)s, %(stake_pct)s, %(stake_usd)s, %(actual_value)s, %(over_hit)s,
    %(source_created_at)s
)
ON CONFLICT (run_id, source_pred_id) DO UPDATE SET
    actual_value = COALESCE(EXCLUDED.actual_value, bets.mlb_prop_prediction_replay.actual_value),
    over_hit = COALESCE(EXCLUDED.over_hit, bets.mlb_prop_prediction_replay.over_hit),
    bankroll_tier = EXCLUDED.bankroll_tier,
    bankroll_candidate = EXCLUDED.bankroll_candidate,
    bankroll_reasons = EXCLUDED.bankroll_reasons,
    stake_pct = EXCLUDED.stake_pct,
    stake_usd = EXCLUDED.stake_usd,
    lock_snapshot_id = COALESCE(
        bets.mlb_prop_prediction_replay.lock_snapshot_id,
        EXCLUDED.lock_snapshot_id
    ),
    locked_at_utc = COALESCE(
        bets.mlb_prop_prediction_replay.locked_at_utc,
        EXCLUDED.locked_at_utc
    )
"""


def snapshot_prop_predictions(
    conn,
    *,
    run_id: str,
    date_from,
    date_to,
    include_no_side: bool = True,
    active_only: bool = True,
) -> int:
    ensure_prop_replay_schema(conn)
    run_started = datetime.now(timezone.utc)
    side_filter = "" if include_no_side else "AND bet_side IN ('over', 'under')"
    active_filter = "AND COALESCE(is_active, TRUE) IS TRUE" if active_only else ""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT *
            FROM bets.mlb_prop_predictions
            WHERE game_date_et BETWEEN %s AND %s
              {side_filter}
              {active_filter}
            ORDER BY game_date_et, game_slug, stat, player_id
            """,
            (date_from, date_to),
        )
        prediction_rows = [dict(r) for r in cur.fetchall()]
    if not prediction_rows:
        return 0
    lock_ids = insert_lock_snapshots_for_predictions(
        conn,
        prediction_rows,
        run_id=run_id,
        snapshot_at_utc=run_started,
    )
    for prediction in prediction_rows:
        prediction_id = prediction.get("id")
        if prediction_id in lock_ids:
            prediction["lock_snapshot_id"] = lock_ids[prediction_id]
            prediction["locked_at_utc"] = run_started
    with conn.cursor() as cur:
        for prediction_id, lock_snapshot_id in lock_ids.items():
            cur.execute(
                """
                UPDATE bets.mlb_prop_predictions
                SET lock_snapshot_id = %s,
                    locked_at_utc = %s
                WHERE id = %s
                """,
                (lock_snapshot_id, run_started, prediction_id),
            )
        for ledger_table in ("bets.mlb_model_pick_ledger", "bets.mlb_bankroll_ledger"):
            cur.execute("SELECT to_regclass(%s)", (ledger_table,))
            if cur.fetchone()[0] is None:
                continue
            cur.execute(
                f"""
                UPDATE {ledger_table} AS ledger
                SET lock_snapshot_id = snapshot.id,
                    locked_at_utc = snapshot.snapshot_at_utc
                FROM odds.mlb_player_prop_line_snapshots AS snapshot
                WHERE snapshot.snapshot_role = 'lock'
                  AND snapshot.run_id = %s
                  AND snapshot.prediction_key = ledger.prediction_key
                  AND ledger.lock_snapshot_id IS NULL
                """,
                (run_id,),
            )
    rows = [_row_from_prediction(row, run_id, run_started) for row in prediction_rows]
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, _REPLAY_INSERT_SQL, rows, page_size=500)
    conn.commit()
    return len(rows)


def _profit_units(won: bool | None, push: bool, price: Any) -> float | None:
    if push:
        return 0.0
    if won is None:
        return None
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if not won:
        return -1.0
    if p > 0:
        return round(p / 100.0, 4)
    if p < 0:
        return round(100.0 / abs(p), 4)
    return None


def _clv_fields_for_row(conn, row: dict[str, Any]) -> dict[str, Any]:
    side = (row.get("side") or row.get("bet_side") or "").lower()
    market_line = _clean_float(row.get("market_line") or row.get("book_line") or row.get("bet_line"))
    close = resolve_valid_prop_close(conn, dict(row))
    closing_line = closing_price = None
    closing_source_row_id = closing_snapshot_id = None
    closing_fetched_at_utc = clv_match_method = None
    if close["valid"]:
        closing_line = _clean_float(close.get("line"))
        closing_price = _clean_float(close.get("over_price") if side == "over" else close.get("under_price"))
        closing_source_row_id = close.get("source_row_id")
        closing_snapshot_id = close.get("snapshot_id")
        closing_fetched_at_utc = close.get("fetched_at_utc")
        clv_match_method = close.get("match_method")
    elif close.get("match_method"):
        clv_match_method = close.get("match_method")

    clv_line = None
    if closing_line is not None and market_line is not None and side in {"over", "under"}:
        clv_line = (
            round(closing_line - market_line, 2)
            if side == "over"
            else round(market_line - closing_line, 2)
        )
    clv_price = price_clv(row.get("market_price") or row.get("bet_price"), closing_price)
    return {
        "closing_line": closing_line,
        "closing_price": closing_price,
        "clv_line": clv_line,
        "clv_price": clv_price,
        "closing_source_row_id": closing_source_row_id,
        "closing_snapshot_id": closing_snapshot_id,
        "closing_fetched_at_utc": closing_fetched_at_utc,
        "clv_match_method": clv_match_method,
        "clv_valid": close["valid"],
        "clv_status": close["status"],
        "clv_unknown_reason": close["unknown_reason"],
    }


def grade_prop_replay(
    conn,
    *,
    run_ids: Iterable[str] | None = None,
    include_graded: bool = False,
) -> int:
    ensure_prop_replay_schema(conn)
    params: list[Any] = []
    run_filter = ""
    if run_ids:
        run_filter = "AND run_id = ANY(%s)"
        params.append(list(run_ids))
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        status_filter = (
            "result_status IN ('pending', 'graded')"
            if include_graded
            else "(result_status = 'pending' OR (result_status = 'graded' AND clv_status IS NULL))"
        )
        cur.execute(
            f"""
            SELECT *
            FROM bets.mlb_prop_prediction_replay
            WHERE {status_filter}
              {run_filter}
            """,
            params,
        )
        pending = cur.fetchall()
    if not pending:
        return 0
    slugs = list({r["game_slug"] for r in pending})
    player_ids = list({int(r["player_id"]) for r in pending})
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
        for row in pending:
            actual = actuals.get((row["game_slug"], int(row["player_id"])))
            if not actual:
                continue
            stat_col = _STAT_COL.get(row["stat"])
            if not stat_col:
                continue
            actual_value = _clean_float(actual.get(stat_col))
            market_line = _clean_float(row.get("market_line"))
            if actual_value is None or market_line is None:
                continue
            push = abs(actual_value - market_line) <= 1e-9
            over_hit = None if push else actual_value > market_line
            side = (row.get("side") or "").lower()
            won = None if push or side not in {"over", "under"} else (over_hit if side == "over" else not over_hit)
            clv = _clv_fields_for_row(conn, dict(row))
            profit = _profit_units(won, push, row.get("market_price"))
            cur.execute(
                """
                UPDATE bets.mlb_prop_prediction_replay
                SET actual_value = %s,
                    over_hit = %s,
                    won = %s,
                    push = %s,
                    profit_units = %s,
                    closing_line = %s,
                    closing_price = %s,
                    clv_line = %s,
                    clv_price = %s,
                    closing_source_row_id = %s,
                    closing_snapshot_id = %s,
                    closing_fetched_at_utc = %s,
                    clv_match_method = %s,
                    clv_valid = %s,
                    clv_status = %s,
                    clv_unknown_reason = %s,
                    result_status = 'graded',
                    graded_at_utc = now()
                WHERE id = %s
                """,
                (
                    actual_value,
                    over_hit,
                    won,
                    push,
                    profit,
                    clv["closing_line"],
                    clv["closing_price"],
                    clv["clv_line"],
                    clv["clv_price"],
                    clv["closing_source_row_id"],
                    clv["closing_snapshot_id"],
                    clv["closing_fetched_at_utc"],
                    clv["clv_match_method"],
                    clv["clv_valid"],
                    clv["clv_status"],
                    clv["clv_unknown_reason"],
                    row["id"],
                ),
            )
            updated += 1
    conn.commit()
    return updated


def refresh_prop_replay_clv(
    conn,
    *,
    run_ids: Iterable[str] | None = None,
    date_from: Any | None = None,
    date_to: Any | None = None,
    include_graded: bool = True,
    only_missing: bool = False,
) -> int:
    """Attach valid close snapshots to locked replay rows before final grading.

    Outcomes arrive after games finish, but CLV can be known as soon as a valid
    close snapshot exists.  This refresh keeps pending replay rows useful for
    CLV diagnostics, walk-forward reports, and CLV-target training.
    """
    ensure_prop_replay_schema(conn)
    filters = [
        "side IN ('over', 'under')",
        "market_line IS NOT NULL",
    ]
    params: list[Any] = []
    if run_ids:
        filters.append("run_id = ANY(%s)")
        params.append(list(run_ids))
    if date_from is not None:
        filters.append("game_date_et >= %s")
        params.append(date_from)
    if date_to is not None:
        filters.append("game_date_et <= %s")
        params.append(date_to)
    if include_graded:
        filters.append("result_status IN ('pending', 'graded')")
    else:
        filters.append("result_status = 'pending'")
    if only_missing:
        filters.append("(clv_status IS NULL OR clv_valid IS NULL OR clv_status = 'unknown')")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT *
            FROM bets.mlb_prop_prediction_replay
            WHERE {' AND '.join(filters)}
            ORDER BY game_date_et, id
            """,
            params,
        )
        rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        return 0

    updated = 0
    with conn.cursor() as cur:
        for row in rows:
            clv = _clv_fields_for_row(conn, row)
            cur.execute(
                """
                UPDATE bets.mlb_prop_prediction_replay
                SET closing_line = %s,
                    closing_price = %s,
                    clv_line = %s,
                    clv_price = %s,
                    closing_source_row_id = %s,
                    closing_snapshot_id = %s,
                    closing_fetched_at_utc = %s,
                    clv_match_method = %s,
                    clv_valid = %s,
                    clv_status = %s,
                    clv_unknown_reason = %s
                WHERE id = %s
                """,
                (
                    clv["closing_line"],
                    clv["closing_price"],
                    clv["clv_line"],
                    clv["clv_price"],
                    clv["closing_source_row_id"],
                    clv["closing_snapshot_id"],
                    clv["closing_fetched_at_utc"],
                    clv["clv_match_method"],
                    clv["clv_valid"],
                    clv["clv_status"],
                    clv["clv_unknown_reason"],
                    row["id"],
                ),
            )
            updated += 1
    conn.commit()
    return updated
