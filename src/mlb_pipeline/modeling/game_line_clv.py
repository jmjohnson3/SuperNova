"""Strict MLB game closing-line resolution from immutable odds snapshots."""
from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import psycopg2.extras


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _row_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _market_and_side(row: dict[str, Any]) -> tuple[str | None, str | None]:
    market = row.get("market")
    side = row.get("side")
    if market in {"run_line", "total"} and side:
        return str(market), str(side).lower()
    if row.get("run_line_bet_side"):
        return "run_line", str(row["run_line_bet_side"]).lower()
    if row.get("total_bet_side"):
        return "total", str(row["total_bet_side"]).lower()
    return None, None


def _bookmaker(row: dict[str, Any], market: str | None) -> str | None:
    book = _row_value(
        row,
        "bookmaker_key",
        "market_bookmaker_key",
        "market_rl_bookmaker_key" if market == "run_line" else "market_total_bookmaker_key",
    )
    if book:
        return str(book).lower()
    # Game entry prices and links are currently sourced from FanDuel.
    price = _row_value(
        row,
        "market_price",
        "market_rl_price" if market == "run_line" else "market_total_price",
    )
    link = row.get("link")
    if price is not None or (link and "fanduel.com" in str(link)):
        return "fanduel"
    return None


def _entry_line(row: dict[str, Any], market: str, side: str) -> float | None:
    line = _clean_float(row.get("bet_line"))
    if line is not None:
        return line
    if market == "run_line":
        home_line = _clean_float(_row_value(row, "market_line", "market_run_line"))
        if home_line is None:
            return None
        return home_line if side == "home" else -home_line
    return _clean_float(_row_value(row, "market_line", "market_total"))


def _entry_price(row: dict[str, Any], market: str) -> float | None:
    return _clean_float(
        _row_value(
            row,
            "market_price",
            "market_rl_price" if market == "run_line" else "market_total_price",
        )
    )


def _close_line_price(snapshot: dict[str, Any], market: str, side: str) -> tuple[float | None, float | None]:
    if market == "run_line":
        home_line = _clean_float(snapshot.get("spread_home_points"))
        away_line = _clean_float(snapshot.get("spread_away_points"))
        if side == "home":
            return home_line, _clean_float(snapshot.get("spread_home_price"))
        return away_line if away_line is not None else (-home_line if home_line is not None else None), _clean_float(
            snapshot.get("spread_away_price")
        )
    return _clean_float(snapshot.get("total_points")), _clean_float(
        snapshot.get("total_over_price") if side == "over" else snapshot.get("total_under_price")
    )


def resolve_valid_game_close(
    conn,
    row: dict[str, Any],
    *,
    max_hours_before_start: float = 2.0,
) -> dict[str, Any]:
    """Resolve a same-book game close captured after lock and near first pitch."""
    unknown = {
        "valid": False,
        "status": "unknown",
        "unknown_reason": "no_valid_close_snapshot",
        "match_method": None,
        "event_id": None,
        "fetched_at_utc": None,
        "closing_line": None,
        "closing_price": None,
        "entry_line": None,
        "entry_price": None,
    }
    market, side = _market_and_side(row)
    if market not in {"run_line", "total"} or side not in {"home", "away", "over", "under"}:
        return {**unknown, "unknown_reason": "missing_market_or_side"}
    game_date = row.get("game_date_et")
    home = row.get("home_team_abbr")
    away = row.get("away_team_abbr")
    game_slug = row.get("game_slug")
    book = _bookmaker(row, market)
    lock_at = _row_value(row, "locked_at_utc", "inserted_at_utc", "predicted_at_utc", "created_at_utc")
    if not game_date or not home or not away:
        return {**unknown, "unknown_reason": "missing_game_identity"}
    if not book:
        return {**unknown, "unknown_reason": "missing_bookmaker"}
    if lock_at is None:
        return {**unknown, "unknown_reason": "missing_lock_timestamp"}

    with conn.cursor() as cur:
        cur.execute(
            "SELECT start_ts_utc FROM raw.mlb_games WHERE game_slug = %s LIMIT 1",
            (game_slug,),
        )
        start_row = cur.fetchone()
    commence = start_row[0] if start_row else None
    if commence is None:
        return {**unknown, "unknown_reason": "missing_commence_time"}

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            WITH nearest_event AS (
                SELECT event_id
                FROM odds.mlb_game_lines
                WHERE as_of_date = %s
                  AND home_team = %s
                  AND away_team = %s
                  AND bookmaker_key = %s
                  AND NULLIF(commence_time_utc, '') IS NOT NULL
                  AND (NULLIF(commence_time_utc, '')::timestamptz AT TIME ZONE 'America/New_York')::date = as_of_date
                ORDER BY ABS(EXTRACT(EPOCH FROM (NULLIF(commence_time_utc, '')::timestamptz - %s)))
                LIMIT 1
            )
            SELECT line.*
            FROM odds.mlb_game_lines AS line
            JOIN nearest_event AS event USING (event_id)
            WHERE line.as_of_date = %s
              AND line.home_team = %s
              AND line.away_team = %s
              AND line.bookmaker_key = %s
            ORDER BY line.fetched_at_utc DESC
            """,
            (game_date, home, away, book, commence, game_date, home, away, book),
        )
        snapshots = [dict(candidate) for candidate in cur.fetchall()]

    entry_line = _entry_line(row, market, side)
    entry_price = _entry_price(row, market)
    earliest_valid = commence - timedelta(hours=float(max_hours_before_start))
    valid: list[dict[str, Any]] = []
    for snapshot in snapshots:
        fetched_at = snapshot.get("fetched_at_utc")
        close_line, close_price = _close_line_price(snapshot, market, side)
        if (
            fetched_at is not None
            and fetched_at > lock_at
            and earliest_valid <= fetched_at <= commence
            and close_line is not None
            and close_price is not None
        ):
            snapshot["_closing_line"] = close_line
            snapshot["_closing_price"] = close_price
            valid.append(snapshot)
    if valid:
        close = valid[0]
        close_line = close["_closing_line"]
        close_price = close["_closing_price"]
        no_move = entry_line == close_line and entry_price is not None and entry_price == close_price
        return {
            "valid": True,
            "status": "true_no_movement" if no_move else "valid_movement",
            "unknown_reason": None,
            "match_method": "same_book_nearest_event_snapshot",
            "event_id": close.get("event_id"),
            "fetched_at_utc": close.get("fetched_at_utc"),
            "closing_line": close_line,
            "closing_price": close_price,
            "entry_line": entry_line,
            "entry_price": entry_price,
        }

    if snapshots:
        after_lock = [s for s in snapshots if s.get("fetched_at_utc") and s["fetched_at_utc"] > lock_at]
        if after_lock:
            return {**unknown, "unknown_reason": "close_outside_two_hour_window"}
        return {**unknown, "unknown_reason": "stale_close_before_lock"}
    return unknown


def game_line_clv(market: str, side: str, entry_line: Any, closing_line: Any) -> float | None:
    """Positive line CLV means the bettor received a more favorable line."""
    entry = _clean_float(entry_line)
    close = _clean_float(closing_line)
    if entry is None or close is None:
        return None
    if market == "run_line":
        return round(entry - close, 2)
    if market == "total" and side == "over":
        return round(close - entry, 2)
    if market == "total" and side == "under":
        return round(entry - close, 2)
    return None
