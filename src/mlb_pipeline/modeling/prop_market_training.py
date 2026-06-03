"""Build side-level MLB prop market training examples.

Each row represents one offered side for one replayed model prediction:
player/date/market/line/book/side/price.  This table is the common source for
direct side classifiers, betting-layer models, CLV models, recalibration, and
bucket reopen decisions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

import psycopg2
import psycopg2.extras

from .prop_replay import (
    american_to_prob,
    ensure_prop_replay_schema,
    ev_per_unit,
    no_vig_probs,
    price_clv,
)
from .side_recalibration import price_bucket, prop_line_bucket

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MARKETS = ("pitcher_strikeouts", "batter_hits", "batter_total_bases", "batter_home_runs")


@dataclass(frozen=True)
class PropMarketTrainingConfig:
    pg_dsn: str = _PG_DSN
    lookback_days: int = 365
    date_from: date | None = None
    date_to: date | None = None
    run_ids: tuple[str, ...] = ()
    include_pending: bool = False
    replace: bool = True


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _kelly_from_price(p_win: Any, price: Any) -> float | None:
    p = _clean_float(p_win)
    pr = _clean_float(price)
    if p is None or pr is None or pr == 0:
        return None
    b = pr / 100.0 if pr > 0 else 100.0 / abs(pr)
    if b <= 0:
        return None
    k = (b * p - (1.0 - p)) / b
    return max(0.0, k)


def _profit_units(won: bool | None, push: bool, price: Any) -> float | None:
    if push:
        return 0.0
    if won is None:
        return None
    pr = _clean_float(price)
    if pr is None or pr == 0:
        return None
    if not won:
        return -1.0
    return round(pr / 100.0 if pr > 0 else 100.0 / abs(pr), 4)


def ensure_prop_market_training_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE SCHEMA IF NOT EXISTS features;
            CREATE TABLE IF NOT EXISTS features.mlb_prop_market_training_examples (
                id BIGSERIAL PRIMARY KEY,
                source TEXT NOT NULL DEFAULT 'replay',
                run_id TEXT NOT NULL,
                replay_id BIGINT NOT NULL,
                source_pred_id INTEGER,
                game_date_et DATE NOT NULL,
                game_slug TEXT NOT NULL,
                player_id BIGINT NOT NULL,
                player_name TEXT,
                player_name_norm TEXT,
                team_abbr TEXT,
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                bookmaker_key TEXT,
                market_line NUMERIC,
                market_price NUMERIC,
                paired_price NUMERIC,
                raw_market_prob NUMERIC,
                no_vig_market_prob NUMERIC,
                market_prob_side NUMERIC,
                price_bucket TEXT,
                line_bucket TEXT,
                model_family TEXT,
                edge_type TEXT,
                pred_value NUMERIC,
                pred_count NUMERIC,
                model_prob_over NUMERIC,
                model_prob_side NUMERIC,
                count_edge_side NUMERIC,
                prob_edge_vs_market NUMERIC,
                ev NUMERIC,
                kelly_fraction NUMERIC,
                actual_value NUMERIC,
                over_hit BOOLEAN,
                won BOOLEAN,
                push BOOLEAN,
                profit_units NUMERIC,
                closing_line NUMERIC,
                closing_price NUMERIC,
                clv_line NUMERIC,
                clv_price NUMERIC,
                beat_clv_line BOOLEAN,
                beat_clv_price BOOLEAN,
                result_status TEXT NOT NULL DEFAULT 'pending',
                source_created_at TIMESTAMPTZ,
                example_updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (run_id, replay_id, side)
            );
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_market_training_date
                ON features.mlb_prop_market_training_examples (game_date_et);
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_market_training_bucket
                ON features.mlb_prop_market_training_examples
                (market, side, line_bucket, price_bucket, model_family);
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_market_training_status
                ON features.mlb_prop_market_training_examples (result_status, game_date_et);
            """
        )
    conn.commit()


def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1 FROM information_schema.tables
              WHERE table_schema = %s AND table_name = %s
            )
            """,
            (schema, table),
        )
        return bool(cur.fetchone()[0])


def _load_replay_rows(conn, cfg: PropMarketTrainingConfig) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)
    date_from = cfg.date_from or cutoff
    date_to = cfg.date_to
    filters = [
        "game_date_et >= %s",
        "stat = ANY(%s)",
        "model_prob_over IS NOT NULL",
        "market_line IS NOT NULL",
    ]
    params: list[Any] = [date_from, list(_MARKETS)]
    if date_to is not None:
        filters.append("game_date_et <= %s")
        params.append(date_to)
    if cfg.run_ids:
        filters.append("run_id = ANY(%s)")
        params.append(list(cfg.run_ids))
    if not cfg.include_pending:
        filters.append("actual_value IS NOT NULL")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT *
            FROM bets.mlb_prop_prediction_replay
            WHERE {' AND '.join(filters)}
            ORDER BY game_date_et, game_slug, stat, player_id, run_id, id
            """,
            params,
        )
        return [dict(r) for r in cur.fetchall()]


def _load_latest_closes(conn, dates: Iterable[date]) -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    dates = sorted({d for d in dates if d is not None})
    if not dates:
        return {}, {}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (as_of_date, player_name_norm, stat, bookmaker_key)
                as_of_date, player_name_norm, stat, bookmaker_key,
                line, over_price, under_price
            FROM odds.mlb_player_prop_lines
            WHERE as_of_date = ANY(%s::date[])
              AND stat = ANY(%s)
              AND line IS NOT NULL
            ORDER BY as_of_date, player_name_norm, stat, bookmaker_key, fetched_at_utc DESC
            """,
            (dates, list(_MARKETS)),
        )
        by_book = {
            (r["as_of_date"], r["player_name_norm"], r["stat"], r["bookmaker_key"]): dict(r)
            for r in cur.fetchall()
        }
        cur.execute(
            """
            SELECT DISTINCT ON (as_of_date, player_name_norm, stat)
                as_of_date, player_name_norm, stat, bookmaker_key,
                line, over_price, under_price
            FROM odds.mlb_player_prop_lines
            WHERE as_of_date = ANY(%s::date[])
              AND stat = ANY(%s)
              AND line IS NOT NULL
            ORDER BY
                as_of_date, player_name_norm, stat,
                CASE WHEN bookmaker_key = 'fanduel' THEN 0 ELSE 1 END,
                fetched_at_utc DESC
            """,
            (dates, list(_MARKETS)),
        )
        generic = {
            (r["as_of_date"], r["player_name_norm"], r["stat"]): dict(r)
            for r in cur.fetchall()
        }
    return by_book, generic


def _close_for(row: dict[str, Any], by_book: dict[tuple, dict], generic: dict[tuple, dict]) -> dict | None:
    key = (
        row.get("game_date_et"),
        row.get("player_name_norm"),
        row.get("stat"),
        row.get("bookmaker_key"),
    )
    close = by_book.get(key)
    if close:
        return close
    return generic.get((row.get("game_date_et"), row.get("player_name_norm"), row.get("stat")))


def _example_rows(row: dict[str, Any], close: dict | None) -> list[dict[str, Any]]:
    p_over = _clean_float(row.get("model_prob_over"))
    line = _clean_float(row.get("market_line"))
    if p_over is None or line is None:
        return []
    over_price = _clean_float(row.get("over_price"))
    under_price = _clean_float(row.get("under_price"))
    nv_over, nv_under = no_vig_probs(over_price, under_price)
    actual = _clean_float(row.get("actual_value"))
    push = bool(actual is not None and abs(actual - line) <= 1e-9)
    over_hit = row.get("over_hit")
    if over_hit is None and actual is not None:
        over_hit = actual > line
    market = str(row.get("stat") or "")
    lb = row.get("line_bucket") or prop_line_bucket(market, line)
    if lb == "unknown":
        lb = prop_line_bucket(market, line)
    pred_count = _clean_float(row.get("pred_count"))
    selected_side = (row.get("side") or "").lower()
    out: list[dict[str, Any]] = []
    for side in ("over", "under"):
        price = over_price if side == "over" else under_price
        paired = under_price if side == "over" else over_price
        raw_mkt = american_to_prob(price)
        no_vig = nv_over if side == "over" else nv_under
        market_prob = no_vig if no_vig is not None else raw_mkt
        p_side = p_over if side == "over" else 1.0 - p_over
        target_won = None
        if over_hit is not None and not push:
            target_won = bool(over_hit) if side == "over" else not bool(over_hit)
        count_edge = None
        if pred_count is not None:
            count_edge = pred_count - line if side == "over" else line - pred_count
        prob_edge = p_side - market_prob if market_prob is not None else None

        closing_line = closing_price = None
        if close:
            closing_line = _clean_float(close.get("line"))
            closing_price = _clean_float(close.get("over_price") if side == "over" else close.get("under_price"))
        if closing_line is None and selected_side == side:
            closing_line = _clean_float(row.get("closing_line"))
        if closing_price is None and selected_side == side:
            closing_price = _clean_float(row.get("closing_price"))

        clv_line = None
        if closing_line is not None:
            clv_line = round(closing_line - line, 2) if side == "over" else round(line - closing_line, 2)
        clv_p = price_clv(price, closing_price)
        ev = ev_per_unit(p_side, price)
        out.append({
            "source": "replay",
            "run_id": row.get("run_id"),
            "replay_id": row.get("id"),
            "source_pred_id": row.get("source_pred_id"),
            "game_date_et": row.get("game_date_et"),
            "game_slug": row.get("game_slug"),
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "player_name_norm": row.get("player_name_norm"),
            "team_abbr": row.get("team_abbr"),
            "market": market,
            "side": side,
            "bookmaker_key": row.get("bookmaker_key") or (close or {}).get("bookmaker_key"),
            "market_line": line,
            "market_price": price,
            "paired_price": paired,
            "raw_market_prob": raw_mkt,
            "no_vig_market_prob": no_vig,
            "market_prob_side": market_prob,
            "price_bucket": price_bucket(price),
            "line_bucket": lb,
            "model_family": row.get("model_family") or "unknown",
            "edge_type": row.get("edge_type"),
            "pred_value": _clean_float(row.get("pred_value")),
            "pred_count": pred_count,
            "model_prob_over": p_over,
            "model_prob_side": p_side,
            "count_edge_side": count_edge,
            "prob_edge_vs_market": prob_edge,
            "ev": ev,
            "kelly_fraction": _kelly_from_price(p_side, price),
            "actual_value": actual,
            "over_hit": over_hit,
            "won": target_won,
            "push": push if actual is not None else None,
            "profit_units": _profit_units(target_won, push, price),
            "closing_line": closing_line,
            "closing_price": closing_price,
            "clv_line": clv_line,
            "clv_price": clv_p,
            "beat_clv_line": None if clv_line is None else clv_line > 0,
            "beat_clv_price": None if clv_p is None else clv_p > 0,
            "result_status": "graded" if actual is not None else "pending",
            "source_created_at": row.get("source_created_at"),
        })
    return out


_INSERT_SQL = """
INSERT INTO features.mlb_prop_market_training_examples (
    source, run_id, replay_id, source_pred_id, game_date_et, game_slug,
    player_id, player_name, player_name_norm, team_abbr, market, side,
    bookmaker_key, market_line, market_price, paired_price, raw_market_prob,
    no_vig_market_prob, market_prob_side, price_bucket, line_bucket,
    model_family, edge_type, pred_value, pred_count, model_prob_over,
    model_prob_side, count_edge_side, prob_edge_vs_market, ev, kelly_fraction,
    actual_value, over_hit, won, push, profit_units, closing_line,
    closing_price, clv_line, clv_price, beat_clv_line, beat_clv_price,
    result_status, source_created_at, example_updated_at
) VALUES (
    %(source)s, %(run_id)s, %(replay_id)s, %(source_pred_id)s, %(game_date_et)s, %(game_slug)s,
    %(player_id)s, %(player_name)s, %(player_name_norm)s, %(team_abbr)s, %(market)s, %(side)s,
    %(bookmaker_key)s, %(market_line)s, %(market_price)s, %(paired_price)s, %(raw_market_prob)s,
    %(no_vig_market_prob)s, %(market_prob_side)s, %(price_bucket)s, %(line_bucket)s,
    %(model_family)s, %(edge_type)s, %(pred_value)s, %(pred_count)s, %(model_prob_over)s,
    %(model_prob_side)s, %(count_edge_side)s, %(prob_edge_vs_market)s, %(ev)s, %(kelly_fraction)s,
    %(actual_value)s, %(over_hit)s, %(won)s, %(push)s, %(profit_units)s, %(closing_line)s,
    %(closing_price)s, %(clv_line)s, %(clv_price)s, %(beat_clv_line)s, %(beat_clv_price)s,
    %(result_status)s, %(source_created_at)s, now()
)
ON CONFLICT (run_id, replay_id, side) DO UPDATE SET
    bookmaker_key = EXCLUDED.bookmaker_key,
    market_price = EXCLUDED.market_price,
    paired_price = EXCLUDED.paired_price,
    raw_market_prob = EXCLUDED.raw_market_prob,
    no_vig_market_prob = EXCLUDED.no_vig_market_prob,
    market_prob_side = EXCLUDED.market_prob_side,
    price_bucket = EXCLUDED.price_bucket,
    line_bucket = EXCLUDED.line_bucket,
    model_family = EXCLUDED.model_family,
    edge_type = EXCLUDED.edge_type,
    model_prob_side = EXCLUDED.model_prob_side,
    count_edge_side = EXCLUDED.count_edge_side,
    prob_edge_vs_market = EXCLUDED.prob_edge_vs_market,
    ev = EXCLUDED.ev,
    kelly_fraction = EXCLUDED.kelly_fraction,
    actual_value = EXCLUDED.actual_value,
    over_hit = EXCLUDED.over_hit,
    won = EXCLUDED.won,
    push = EXCLUDED.push,
    profit_units = EXCLUDED.profit_units,
    closing_line = EXCLUDED.closing_line,
    closing_price = EXCLUDED.closing_price,
    clv_line = EXCLUDED.clv_line,
    clv_price = EXCLUDED.clv_price,
    beat_clv_line = EXCLUDED.beat_clv_line,
    beat_clv_price = EXCLUDED.beat_clv_price,
    result_status = EXCLUDED.result_status,
    example_updated_at = now()
"""


def _delete_existing(conn, cfg: PropMarketTrainingConfig) -> int:
    filters = ["TRUE"]
    params: list[Any] = []
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)
    date_from = cfg.date_from or cutoff
    filters.append("game_date_et >= %s")
    params.append(date_from)
    if cfg.date_to is not None:
        filters.append("game_date_et <= %s")
        params.append(cfg.date_to)
    if cfg.run_ids:
        filters.append("run_id = ANY(%s)")
        params.append(list(cfg.run_ids))
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM features.mlb_prop_market_training_examples WHERE {' AND '.join(filters)}",
            params,
        )
        deleted = cur.rowcount
    conn.commit()
    return deleted


def refresh_prop_market_training_examples(cfg: PropMarketTrainingConfig) -> dict[str, int]:
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_prop_replay_schema(conn)
        ensure_prop_market_training_schema(conn)
        deleted = _delete_existing(conn, cfg) if cfg.replace else 0
        if not _table_exists(conn, "bets", "mlb_prop_prediction_replay"):
            return {"deleted": deleted, "replay_rows": 0, "examples": 0}
        replay_rows = _load_replay_rows(conn, cfg)
        closes_by_book, closes_generic = _load_latest_closes(
            conn,
            [r.get("game_date_et") for r in replay_rows],
        )
        examples: list[dict[str, Any]] = []
        for row in replay_rows:
            examples.extend(_example_rows(row, _close_for(row, closes_by_book, closes_generic)))
        if examples:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, _INSERT_SQL, examples, page_size=500)
            conn.commit()
    return {"deleted": deleted, "replay_rows": len(replay_rows), "examples": len(examples)}
