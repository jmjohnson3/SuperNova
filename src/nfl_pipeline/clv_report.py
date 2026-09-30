"""Attach NFL close-line value labels to locked predictions."""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.markets import normalize_name

log = logging.getLogger("nfl_pipeline.clv_report")

_CLOSE_WINDOW_BEFORE_MINUTES = 120.0
_CLOSE_WINDOW_AFTER_MINUTES = 0.0


@dataclass(frozen=True)
class ClvConfig:
    pg_dsn: str = PG_DSN
    game_date: date | None = None


def _clean_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _american_to_prob(price: Any) -> float | None:
    price_f = _clean_float(price)
    if price_f is None or price_f == 0:
        return None
    return 100.0 / (price_f + 100.0) if price_f > 0 else abs(price_f) / (abs(price_f) + 100.0)


def _prob_delta(lock_price: Any, close_price: Any) -> float | None:
    lock_p = _american_to_prob(lock_price)
    close_p = _american_to_prob(close_price)
    if lock_p is None or close_p is None:
        return None
    return float(close_p - lock_p)


def _minutes_to_start(close_time: Any, commence_time: Any) -> float | None:
    if not isinstance(close_time, datetime) or not isinstance(commence_time, datetime):
        return None
    try:
        return float((commence_time - close_time).total_seconds() / 60.0)
    except Exception:
        return None


def _close_quality(
    *,
    close_time: Any,
    lock_time: Any,
    commence_time: Any,
    close_price: Any,
    line_available_at_close: bool | None,
) -> tuple[str, float | None, bool]:
    minutes_to_start = _minutes_to_start(close_time, commence_time)
    if close_time is None:
        if line_available_at_close is False:
            return "line_unavailable_at_close", minutes_to_start, False
        return "missing_close", minutes_to_start, False
    if not isinstance(lock_time, datetime):
        return "missing_lock_time", minutes_to_start, False
    if isinstance(close_time, datetime) and close_time <= lock_time:
        return "stale_close_before_lock", minutes_to_start, False
    if line_available_at_close is False:
        return "line_unavailable_at_close", minutes_to_start, False
    if _clean_float(close_price) is None or abs(float(close_price)) < 100:
        return "same_line_missing_side_price", minutes_to_start, False
    if minutes_to_start is None:
        return "missing_commence_time", minutes_to_start, False
    if minutes_to_start > _CLOSE_WINDOW_BEFORE_MINUTES:
        return "close_outside_two_hour_window", minutes_to_start, False
    if minutes_to_start <= -_CLOSE_WINDOW_AFTER_MINUTES:
        return "close_after_kickoff", minutes_to_start, False
    return "valid_close", minutes_to_start, True


def _game_close_fields(row: dict[str, Any]) -> tuple[float | None, int | None]:
    market = str(row.get("market") or "").lower()
    side = str(row.get("side") or "").lower()
    if market == "spread":
        if side == "home":
            return _clean_float(row.get("spread_home_points")), row.get("spread_home_price")
        if side == "away":
            return _clean_float(row.get("spread_away_points")), row.get("spread_away_price")
    if market == "total":
        if side == "over":
            return _clean_float(row.get("total_points")), row.get("total_over_price")
        if side == "under":
            return _clean_float(row.get("total_points")), row.get("total_under_price")
    return None, None


def _prop_close_fields(row: dict[str, Any]) -> tuple[float | None, int | None]:
    side = str(row.get("side") or "").lower()
    if side == "over":
        return _clean_float(row.get("line")), row.get("over_price")
    if side == "under":
        return _clean_float(row.get("line")), row.get("under_price")
    return None, None


def attach_game_clv(conn, cfg: ClvConfig) -> int:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                p.id AS prediction_id, p.game_date_et, p.game_id, p.market, p.side, p.book,
                p.line::float AS locked_line, p.price AS locked_price,
                p.created_at_utc, p.integrity_version,
                COALESCE(g.start_ts_utc, c.commence_time_utc) AS commence_time_utc,
                c.id AS close_row_id,
                c.fetched_at_utc AS close_fetched_at_utc,
                c.spread_home_points::float AS spread_home_points,
                c.spread_home_price,
                c.spread_away_points::float AS spread_away_points,
                c.spread_away_price,
                c.total_points::float AS total_points,
                c.total_over_price,
                c.total_under_price
            FROM bets.nfl_game_predictions p
            LEFT JOIN raw.nfl_games g
              ON g.game_id = p.game_id
            LEFT JOIN LATERAL (
                SELECT *
                FROM odds.nfl_game_lines c
                WHERE c.snapshot_role = 'close'
                  AND c.as_of_date = p.game_date_et
                  AND c.bookmaker_key = p.book
                  AND (
                      c.event_id = p.game_id
                      OR (
                          c.home_team_abbr = p.home_team_abbr
                          AND c.away_team_abbr = p.away_team_abbr
                      )
                  )
                  AND c.fetched_at_utc > p.created_at_utc
                ORDER BY
                  CASE
                    WHEN c.fetched_at_utc >= COALESCE(g.start_ts_utc, c.commence_time_utc) - interval '120 minutes'
                     AND c.fetched_at_utc < COALESCE(g.start_ts_utc, c.commence_time_utc)
                    THEN 0 ELSE 1
                  END,
                  c.fetched_at_utc DESC
                LIMIT 1
            ) c ON TRUE
            WHERE (%(game_date)s IS NULL OR p.game_date_et = %(game_date)s)
            """,
            {"game_date": cfg.game_date},
        )
        rows = cur.fetchall()
    out_rows: list[tuple] = []
    for row in rows:
        close_line, close_price = _game_close_fields(dict(row))
        line_available = True if row["close_row_id"] is not None else None
        status, minutes_to_start, valid_close = _close_quality(
            close_time=row["close_fetched_at_utc"],
            lock_time=row["created_at_utc"],
            commence_time=row["commence_time_utc"],
            close_price=close_price,
            line_available_at_close=line_available,
        )
        if close_line is not None and _clean_float(row["locked_line"]) is not None and abs(close_line - float(row["locked_line"])) > 1e-9:
            status = "line_moved"
            valid_close = False
        if row['integrity_version'] != 'nfl-asof-v2':
            status, valid_close = 'legacy_lock_unverified', False
        out_rows.append((
            "game", row["prediction_id"], row["game_date_et"], row["book"], row["market"], None,
            row["side"], row["locked_line"], row["locked_price"], close_line, close_price,
            row["close_fetched_at_utc"], _prob_delta(row["locked_price"], close_price) if valid_close else None, status,
            row["commence_time_utc"], minutes_to_start, line_available, valid_close, status,
        ))
    return _upsert_clv_rows(conn, out_rows)


def prop_close_rows(conn, cfg: ClvConfig):
    # One indexed batch join. Identity comes from the immutable lock offer, not
    # a same-name player on another event or another provider's event ID.
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT p.id AS prediction_id, p.game_date_et, p.player_name, p.stat, p.side, p.book,
                   p.line::float AS locked_line, p.price AS locked_price,
                   p.created_at_utc, p.integrity_version, o.id AS lock_offer_id,
                   g.start_ts_utc, p.is_current,
                   EXISTS (SELECT 1 FROM bets.nfl_bet_ledger l
                           WHERE l.source_kind='prop' AND l.prediction_id=p.id
                             AND l.tier <> 'paper') AS selected_real_tier,
                   c.line::float AS line, c.over_price, c.under_price,
                   c.fetched_at_utc, c.commence_time_utc,
                   m.fresh_market_rows,m.observed_lines,b.fresh_book_rows
            FROM bets.nfl_player_prop_predictions p
            LEFT JOIN raw.nfl_games g ON g.game_id=p.game_id
            LEFT JOIN odds.nfl_player_prop_lines o ON o.id=p.offer_id
              AND o.fetched_at_utc<=p.created_at_utc
              AND o.bookmaker_key=p.book AND o.stat=p.stat AND o.line=p.line
              AND o.player_name_norm=p.offer_player_name_norm
            LEFT JOIN LATERAL (
                SELECT c.* FROM odds.nfl_player_prop_lines c
                WHERE c.snapshot_role='close' AND c.provider=o.provider
                  AND c.event_id=o.event_id AND c.bookmaker_key=o.bookmaker_key
                  AND c.player_name_norm=o.player_name_norm
                  AND c.stat=o.stat AND c.line=o.line
                  AND c.as_of_date=p.game_date_et
                ORDER BY CASE WHEN c.fetched_at_utc>p.created_at_utc
                    AND c.fetched_at_utc>=g.start_ts_utc-interval '120 minutes'
                    AND c.fetched_at_utc<g.start_ts_utc
                    AND abs(CASE WHEN p.side='over' THEN c.over_price ELSE c.under_price END)>=100
                    THEN 0 ELSE 1 END,
                    c.fetched_at_utc DESC
                LIMIT 1
            ) c ON TRUE
            LEFT JOIN LATERAL (
                SELECT count(*)::int AS fresh_market_rows,array_agg(DISTINCT q.line) AS observed_lines
                FROM odds.nfl_player_prop_lines q
                WHERE q.provider=o.provider AND q.event_id=o.event_id AND q.bookmaker_key=o.bookmaker_key
                  AND q.player_name_norm=o.player_name_norm AND q.stat=o.stat AND q.snapshot_role='close'
                  AND q.fetched_at_utc>p.created_at_utc
                  AND q.fetched_at_utc>=g.start_ts_utc-interval '120 minutes' AND q.fetched_at_utc<g.start_ts_utc
            ) m ON TRUE
            LEFT JOIN LATERAL (
                SELECT count(*)::int AS fresh_book_rows FROM odds.nfl_player_prop_lines q
                WHERE q.provider=o.provider AND q.event_id=o.event_id AND q.bookmaker_key=o.bookmaker_key
                  AND q.snapshot_role='close' AND q.fetched_at_utc>p.created_at_utc
                  AND q.fetched_at_utc>=g.start_ts_utc-interval '120 minutes' AND q.fetched_at_utc<g.start_ts_utc
            ) b ON TRUE
            WHERE (%(game_date)s IS NULL OR p.game_date_et=%(game_date)s)
              AND p.side IN ('over','under') AND p.line IS NOT NULL
        """, {"game_date": cfg.game_date})
        return cur.fetchall()


def classify_prop_close(p):
    close_line, close_price = _prop_close_fields(dict(p))
    start = p['start_ts_utc'] or p['commence_time_utc']
    status, minutes, valid = _close_quality(close_time=p['fetched_at_utc'], lock_time=p['created_at_utc'],
        commence_time=start, close_price=close_price, line_available_at_close=None)
    if p['integrity_version'] != 'nfl-asof-v2':
        status, valid = 'legacy_lock_unverified', False
    elif p['lock_offer_id'] is None:
        status, valid = 'missing_lock_offer', False
    elif not valid and status in ('missing_close', 'close_outside_two_hour_window', 'stale_close_before_lock', 'close_after_kickoff'):
        if p.get('fresh_market_rows') and p['locked_line'] not in (p.get('observed_lines') or []):
            status = 'exact_line_unavailable_in_captured_feed'
        elif p.get('fresh_book_rows') and not p.get('fresh_market_rows'):
            status = 'player_market_not_observed_in_capture'
    # A partial feed cannot prove that a sportsbook removed a line. Keep actual
    # bookability unknown unless the exact quote was observed at valid timing.
    return dict(close_line=close_line, close_price=close_price, status=status, minutes=minutes,
                valid=valid, start=start, available=True if valid else None)


def attach_prop_clv(conn, cfg: ClvConfig) -> int:
    preds = prop_close_rows(conn, cfg)
    rows = []
    for p in preds:
        quality = classify_prop_close(p)
        close_line, close_price = quality['close_line'], quality['close_price']
        valid, status = quality['valid'], quality['status']
        rows.append(("prop",p["prediction_id"],p["game_date_et"],p["book"],None,p["stat"],
                     p["side"],p["locked_line"],p["locked_price"],close_line,close_price,
                     p["fetched_at_utc"],_prob_delta(p["locked_price"],close_price) if valid else None,
                     status,quality['start'],quality['minutes'],quality['available'],valid,status))
    return _upsert_clv_rows(conn, rows)


def _upsert_clv_rows(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO bets.nfl_prediction_clv (
                source_kind, prediction_id, game_date_et, book, market, stat,
                side, locked_line, locked_price, close_line, close_price,
                close_fetched_at_utc, clv_prob_delta, clv_status,
                commence_time_utc, minutes_to_start, line_available_at_close,
                valid_close_snapshot_captured, close_quality_reason
            )
            VALUES %s
            ON CONFLICT (source_kind, prediction_id) DO UPDATE SET
                close_line = EXCLUDED.close_line,
                close_price = EXCLUDED.close_price,
                close_fetched_at_utc = EXCLUDED.close_fetched_at_utc,
                clv_prob_delta = EXCLUDED.clv_prob_delta,
                clv_status = EXCLUDED.clv_status,
                commence_time_utc = EXCLUDED.commence_time_utc,
                minutes_to_start = EXCLUDED.minutes_to_start,
                line_available_at_close = EXCLUDED.line_available_at_close,
                valid_close_snapshot_captured = EXCLUDED.valid_close_snapshot_captured,
                close_quality_reason = EXCLUDED.close_quality_reason,
                updated_at_utc = NOW()
            WHERE (bets.nfl_prediction_clv.close_line,bets.nfl_prediction_clv.close_price,
                   bets.nfl_prediction_clv.close_fetched_at_utc,bets.nfl_prediction_clv.clv_prob_delta,
                   bets.nfl_prediction_clv.clv_status,bets.nfl_prediction_clv.commence_time_utc,
                   bets.nfl_prediction_clv.line_available_at_close,bets.nfl_prediction_clv.valid_close_snapshot_captured)
              IS DISTINCT FROM (EXCLUDED.close_line,EXCLUDED.close_price,EXCLUDED.close_fetched_at_utc,
                                EXCLUDED.clv_prob_delta,EXCLUDED.clv_status,EXCLUDED.commence_time_utc,
                                EXCLUDED.line_available_at_close,EXCLUDED.valid_close_snapshot_captured)
            """,
            rows,
            page_size=1000,
        )
    conn.commit()
    return len(rows)


def attach_clv(cfg: ClvConfig) -> dict[str, Any]:
    with psycopg2.connect(cfg.pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout='120s'")
            cur.execute("SET lock_timeout='5s'")
        game_rows = attach_game_clv(conn, cfg)
        prop_rows = attach_prop_clv(conn, cfg)
    return {"status": "ok", "game_clv_rows": game_rows, "prop_clv_rows": prop_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach NFL CLV labels from close snapshots")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    result = attach_clv(ClvConfig(
        pg_dsn=args.pg_dsn,
        game_date=date.fromisoformat(args.date) if args.date else None,
    ))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
