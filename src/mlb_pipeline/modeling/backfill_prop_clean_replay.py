"""Backfill clean prop lock/close replay rows from saved predictions.

This script is for shadow-slate reconstruction. It does not mutate the live
bets.mlb_prop_predictions table. Instead it:
  1. matches old prediction rows to exact prop offer rows,
  2. creates immutable lock snapshots at the original prediction created_at,
  3. skips rows whose prediction was created after the close snapshot window,
  4. writes replay rows and optionally grades them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import psycopg2
import psycopg2.extras

from .prop_offer_snapshots import insert_lock_snapshots_for_predictions, normalize_name
from .prop_replay import (
    _REPLAY_INSERT_SQL,
    _row_from_prediction,
    ensure_prop_replay_schema,
    grade_prop_replay,
)

from mlb_pipeline.db import PG_DSN as _PG_DSN
@dataclass
class ReplaySummary:
    run_id: str
    rows_seen: int = 0
    matched_rows: int = 0
    locked_rows: int = 0
    replay_rows: int = 0
    graded_rows: int = 0
    skips: Counter[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rows_seen": self.rows_seen,
            "matched_rows": self.matched_rows,
            "locked_rows": self.locked_rows,
            "replay_rows": self.replay_rows,
            "graded_rows": self.graded_rows,
            "skips": dict(self.skips or {}),
        }


def _as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _line_key(value: Any) -> str | None:
    if value is None:
        return None
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not dec.is_finite():
        return None
    return format(dec.normalize(), "f")


def _clean_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return None
    return v


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _load_predictions(
    conn,
    *,
    date_from: date,
    date_to: date,
    include_inactive: bool,
) -> list[dict[str, Any]]:
    active_clause = "" if include_inactive else "AND COALESCE(is_active, TRUE) IS TRUE"
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT *
            FROM bets.mlb_prop_predictions
            WHERE game_date_et BETWEEN %s AND %s
              AND bet_side IN ('over', 'under')
              {active_clause}
            ORDER BY game_date_et, game_slug, stat, player_id, id
            """,
            (date_from, date_to),
        )
        return [dict(row) for row in cur.fetchall()]


def _load_games(conn, game_slugs: list[str]) -> dict[str, datetime]:
    if not game_slugs:
        return {}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT game_slug, start_ts_utc
            FROM raw.mlb_games
            WHERE game_slug = ANY(%s)
            """,
            (game_slugs,),
        )
        rows = cur.fetchall()
    result: dict[str, datetime] = {}
    for row in rows:
        start = _as_utc(row.get("start_ts_utc"))
        if start is not None:
            result[str(row["game_slug"])] = start
    return result


def _load_offer_index(
    conn,
    *,
    date_from: date,
    date_to: date,
) -> dict[tuple[Any, str, str, str, str], list[dict[str, Any]]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT *
            FROM odds.mlb_player_prop_lines
            WHERE as_of_date BETWEEN %s AND %s
            """,
            (date_from, date_to),
        )
        rows = [dict(row) for row in cur.fetchall()]
    index: dict[tuple[Any, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        line = _line_key(row.get("line"))
        book = str(row.get("bookmaker_key") or "").lower()
        player = str(row.get("player_name_norm") or normalize_name(row.get("player_name")))
        stat = str(row.get("stat") or "")
        if not row.get("as_of_date") or not line or not book or not player or not stat:
            continue
        index[(row["as_of_date"], player, stat, book, line)].append(row)
    return index


def _choose_offer(
    prediction: dict[str, Any],
    offers: list[dict[str, Any]],
    games: dict[str, datetime],
) -> dict[str, Any] | None:
    side = str(prediction.get("bet_side") or "").lower()
    side_price_col = "over_price" if side == "over" else "under_price"
    candidates = [offer for offer in offers if _clean_int(offer.get(side_price_col)) is not None]
    if not candidates:
        return None
    target_start = games.get(str(prediction.get("game_slug") or ""))
    if target_start is None:
        return sorted(candidates, key=lambda offer: _as_utc(offer.get("commence_time_utc")) or datetime.max.replace(tzinfo=timezone.utc))[0]

    def score(offer: dict[str, Any]) -> float:
        commence = _as_utc(offer.get("commence_time_utc"))
        if commence is None:
            return float("inf")
        return abs((commence - target_start).total_seconds())

    best = min(candidates, key=score)
    if score(best) > 6 * 60 * 60:
        return None
    return best


def _prediction_key(row: dict[str, Any]) -> tuple[Any, str, str, str, str] | None:
    line = _line_key(row.get("book_line"))
    book = str(row.get("bookmaker_key") or "").lower()
    player = normalize_name(row.get("player_name"))
    stat = str(row.get("stat") or "")
    game_date = row.get("game_date_et")
    if not game_date or not line or not book or not player or not stat:
        return None
    return (game_date, player, stat, book, line)


def _prepare_row(
    prediction: dict[str, Any],
    offer: dict[str, Any],
    *,
    lock_at: datetime,
) -> dict[str, Any]:
    row = dict(prediction)
    side = str(row.get("bet_side") or "").lower()
    over_price = _clean_int(row.get("over_price"))
    under_price = _clean_int(row.get("under_price"))
    if over_price is None:
        over_price = _clean_int(offer.get("over_price"))
    if under_price is None:
        under_price = _clean_int(offer.get("under_price"))
    bet_price = _clean_int(row.get("bet_price"))
    if bet_price is None:
        bet_price = over_price if side == "over" else under_price
    bet_link = row.get("bet_link")
    if not bet_link:
        bet_link = offer.get("over_link") if side == "over" else offer.get("under_link")

    row.update(
        {
            "prop_offer_source_row_id": offer.get("id"),
            "book_line": row.get("book_line") if row.get("book_line") is not None else offer.get("line"),
            "bookmaker_key": str(row.get("bookmaker_key") or offer.get("bookmaker_key") or "").lower(),
            "over_price": over_price,
            "under_price": under_price,
            "bet_price": bet_price,
            "bet_link": bet_link,
            "locked_at_utc": lock_at,
        }
    )
    return row


def backfill_clean_replay(
    conn,
    *,
    date_from: date,
    date_to: date,
    run_id: str,
    close_minutes_before_start: int = 30,
    include_inactive: bool = False,
    grade: bool = False,
) -> ReplaySummary:
    if close_minutes_before_start <= 0:
        raise ValueError("close_minutes_before_start must be positive")
    ensure_prop_replay_schema(conn)
    summary = ReplaySummary(run_id=run_id, skips=Counter())
    predictions = _load_predictions(
        conn,
        date_from=date_from,
        date_to=date_to,
        include_inactive=include_inactive,
    )
    summary.rows_seen = len(predictions)
    if not predictions:
        return summary

    games = _load_games(conn, sorted({str(row.get("game_slug")) for row in predictions if row.get("game_slug")}))
    offers = _load_offer_index(conn, date_from=date_from, date_to=date_to)
    prepared_by_lock: dict[datetime, list[dict[str, Any]]] = defaultdict(list)

    for prediction in predictions:
        if prediction.get("player_id") is None:
            summary.skips["missing_player_id"] += 1
            continue
        key = _prediction_key(prediction)
        if key is None:
            summary.skips["missing_offer_identity"] += 1
            continue
        offer = _choose_offer(prediction, offers.get(key, []), games)
        if offer is None:
            summary.skips["no_exact_offer_match"] += 1
            continue
        lock_at = _as_utc(prediction.get("created_at"))
        if lock_at is None:
            summary.skips["missing_prediction_created_at"] += 1
            continue
        commence = _as_utc(offer.get("commence_time_utc")) or games.get(str(prediction.get("game_slug") or ""))
        if commence is None:
            summary.skips["missing_commence_time"] += 1
            continue
        close_cutoff = commence - timedelta(minutes=close_minutes_before_start)
        if lock_at >= close_cutoff:
            summary.skips["created_after_close_window"] += 1
            continue
        summary.matched_rows += 1
        prepared_by_lock[lock_at].append(_prepare_row(prediction, offer, lock_at=lock_at))

    replay_rows: list[dict[str, Any]] = []
    for lock_at, rows in sorted(prepared_by_lock.items(), key=lambda item: item[0]):
        lock_ids = insert_lock_snapshots_for_predictions(
            conn,
            rows,
            run_id=run_id,
            snapshot_at_utc=lock_at,
        )
        for row in rows:
            prediction_id = int(row["id"])
            lock_snapshot_id = lock_ids.get(prediction_id)
            if lock_snapshot_id is None:
                summary.skips["lock_snapshot_not_created"] += 1
                continue
            row["lock_snapshot_id"] = lock_snapshot_id
            row["locked_at_utc"] = lock_at
            replay_rows.append(_row_from_prediction(row, run_id, lock_at))

    summary.locked_rows = len(replay_rows)
    if replay_rows:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, _REPLAY_INSERT_SQL, replay_rows, page_size=500)
        conn.commit()
    summary.replay_rows = len(replay_rows)
    if grade and replay_rows:
        summary.graded_rows = grade_prop_replay(conn, run_ids=[run_id], include_graded=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill clean MLB prop replay locks from saved predictions")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--date-from", required=True, type=date.fromisoformat)
    parser.add_argument("--date-to", required=True, type=date.fromisoformat)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--close-minutes-before-start", type=int, default=30)
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument("--grade", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or f"prop_clean_backfill_{args.date_from:%Y%m%d}_{args.date_to:%Y%m%d}"
    conn = psycopg2.connect(args.pg_dsn)
    try:
        summary = backfill_clean_replay(
            conn,
            date_from=args.date_from,
            date_to=args.date_to,
            run_id=run_id,
            close_minutes_before_start=args.close_minutes_before_start,
            include_inactive=args.include_inactive,
            grade=args.grade,
        )
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
