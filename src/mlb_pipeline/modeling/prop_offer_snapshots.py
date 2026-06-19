"""Immutable MLB prop market snapshots and strict closing-line resolution."""
from __future__ import annotations

import hashlib
import logging
import math
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import psycopg2.extras

log = logging.getLogger("mlb_pipeline.modeling.prop_offer_snapshots")

SNAPSHOT_TABLE = "odds.mlb_player_prop_line_snapshots"
SNAPSHOT_ROLES = frozenset({"open", "lock", "close"})
_SCHEMA_READY = False
_SCHEMA_LOCK_KEY = "mlb_prop_schema_ddl"


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_name = nfkd.encode("ascii", errors="ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9\s]", "", ascii_name.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clean_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _snapshot_key(*parts: Any) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ensure_prop_offer_snapshot_schema(conn) -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '30s'")
            cur.execute("SET LOCAL statement_timeout = '60s'")
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (_SCHEMA_LOCK_KEY,))
            cur.execute(
                """
                CREATE SCHEMA IF NOT EXISTS odds;
                CREATE SCHEMA IF NOT EXISTS bets;
                CREATE TABLE IF NOT EXISTS odds.mlb_player_prop_line_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    snapshot_key TEXT NOT NULL UNIQUE,
                    snapshot_role TEXT NOT NULL
                        CHECK (snapshot_role IN ('open', 'lock', 'close')),
                    snapshot_at_utc TIMESTAMPTZ NOT NULL,
                    source_prop_line_id BIGINT,
                    prediction_id INTEGER,
                    prediction_key TEXT,
                    run_id TEXT,
                    as_of_date DATE NOT NULL,
                    event_id TEXT,
                    commence_time_utc TIMESTAMPTZ,
                    bookmaker_key TEXT NOT NULL,
                    home_team TEXT,
                    away_team TEXT,
                    player_name TEXT NOT NULL,
                    player_name_norm TEXT NOT NULL,
                    stat TEXT NOT NULL,
                    selected_side TEXT CHECK (selected_side IN ('over', 'under')),
                    line NUMERIC NOT NULL,
                    over_price INTEGER,
                    under_price INTEGER,
                    over_link TEXT,
                    under_link TEXT,
                    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_mlb_prop_snapshots_offer
                    ON odds.mlb_player_prop_line_snapshots
                    (as_of_date, player_name_norm, stat, bookmaker_key, line, snapshot_role);
                CREATE UNIQUE INDEX IF NOT EXISTS uq_mlb_prop_snapshots_open_offer
                    ON odds.mlb_player_prop_line_snapshots
                    (as_of_date, event_id, bookmaker_key, player_name_norm, stat, line)
                    WHERE snapshot_role = 'open';
                CREATE INDEX IF NOT EXISTS idx_mlb_prop_snapshots_event
                    ON odds.mlb_player_prop_line_snapshots
                    (event_id, bookmaker_key, stat, snapshot_role);
                CREATE INDEX IF NOT EXISTS idx_mlb_prop_snapshots_time
                    ON odds.mlb_player_prop_line_snapshots
                    (snapshot_role, snapshot_at_utc);
                CREATE INDEX IF NOT EXISTS idx_mlb_prop_snapshots_prediction
                    ON odds.mlb_player_prop_line_snapshots
                    (prediction_id, snapshot_role);
                CREATE INDEX IF NOT EXISTS idx_mlb_prop_snapshots_run_prediction_key
                    ON odds.mlb_player_prop_line_snapshots
                    (run_id, prediction_key, snapshot_role);
                DO $ddl$
                BEGIN
                    IF to_regprocedure('odds.reject_mlb_prop_snapshot_mutation()') IS NULL THEN
                        EXECUTE $fn$
                            CREATE FUNCTION odds.reject_mlb_prop_snapshot_mutation()
                            RETURNS trigger
                            LANGUAGE plpgsql
                            AS $body$
                            BEGIN
                                RAISE EXCEPTION 'odds.mlb_player_prop_line_snapshots is immutable';
                            END;
                            $body$;
                        $fn$;
                    END IF;
                END;
                $ddl$;
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_trigger
                        WHERE tgname = 'trg_reject_mlb_prop_snapshot_mutation'
                          AND tgrelid = 'odds.mlb_player_prop_line_snapshots'::regclass
                          AND NOT tgisinternal
                    ) THEN
                        CREATE TRIGGER trg_reject_mlb_prop_snapshot_mutation
                        BEFORE UPDATE OR DELETE OR TRUNCATE
                        ON odds.mlb_player_prop_line_snapshots
                        FOR EACH STATEMENT
                        EXECUTE FUNCTION odds.reject_mlb_prop_snapshot_mutation();
                    END IF;
                END;
                $$;
                ALTER TABLE IF EXISTS bets.mlb_prop_predictions
                    ADD COLUMN IF NOT EXISTS lock_snapshot_id BIGINT,
                    ADD COLUMN IF NOT EXISTS locked_at_utc TIMESTAMPTZ;
                ALTER TABLE IF EXISTS bets.mlb_model_pick_ledger
                    ADD COLUMN IF NOT EXISTS lock_snapshot_id BIGINT,
                    ADD COLUMN IF NOT EXISTS locked_at_utc TIMESTAMPTZ;
                ALTER TABLE IF EXISTS bets.mlb_bankroll_ledger
                    ADD COLUMN IF NOT EXISTS lock_snapshot_id BIGINT,
                    ADD COLUMN IF NOT EXISTS locked_at_utc TIMESTAMPTZ;
                """
            )
        conn.commit()
        _SCHEMA_READY = True
    except Exception:
        conn.rollback()
        raise


_SNAPSHOT_INSERT_SQL = """
INSERT INTO odds.mlb_player_prop_line_snapshots (
    snapshot_key, snapshot_role, snapshot_at_utc, source_prop_line_id,
    prediction_id, prediction_key, run_id, as_of_date, event_id,
    commence_time_utc, bookmaker_key, home_team, away_team, player_name,
    player_name_norm, stat, selected_side, line, over_price, under_price,
    over_link, under_link
) VALUES (
    %(snapshot_key)s, %(snapshot_role)s, %(snapshot_at_utc)s,
    %(source_prop_line_id)s, %(prediction_id)s, %(prediction_key)s,
    %(run_id)s, %(as_of_date)s, %(event_id)s, %(commence_time_utc)s,
    %(bookmaker_key)s, %(home_team)s, %(away_team)s, %(player_name)s,
    %(player_name_norm)s, %(stat)s, %(selected_side)s, %(line)s,
    %(over_price)s, %(under_price)s, %(over_link)s, %(under_link)s
)
ON CONFLICT DO NOTHING
"""


def insert_market_prop_snapshots(
    conn,
    rows: Iterable[tuple],
    *,
    snapshot_role: str,
) -> int:
    """Insert parser tuples as immutable open or close market observations."""
    role = str(snapshot_role or "").lower()
    if role not in {"open", "close"}:
        raise ValueError(f"Market snapshot role must be open or close, got {snapshot_role!r}")
    payload: list[dict[str, Any]] = []
    for row in rows:
        if len(row) < 17 or row[10] is None or not row[4] or not row[8] or not row[9]:
            continue
        fetched_at = row[1]
        payload.append({
            "snapshot_key": _snapshot_key(
                "market", role, fetched_at, row[2], str(row[4]).lower(),
                row[8], row[9], row[10],
            ),
            "snapshot_role": role,
            "snapshot_at_utc": fetched_at,
            "source_prop_line_id": None,
            "prediction_id": None,
            "prediction_key": None,
            "run_id": None,
            "as_of_date": row[0],
            "event_id": row[2],
            "commence_time_utc": row[3],
            "bookmaker_key": str(row[4]).lower(),
            "home_team": row[5],
            "away_team": row[6],
            "player_name": row[7],
            "player_name_norm": row[8],
            "stat": row[9],
            "selected_side": None,
            "line": row[10],
            "over_price": _clean_int(row[11]),
            "under_price": _clean_int(row[12]),
            "over_link": row[13],
            "under_link": row[14],
        })
    if not payload:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, _SNAPSHOT_INSERT_SQL, payload, page_size=500)
    return len(payload)


def insert_lock_snapshots_for_predictions(
    conn,
    predictions: Iterable[dict[str, Any]],
    *,
    run_id: str,
    snapshot_at_utc: datetime | None = None,
) -> dict[int, int]:
    """Lock the exact offered price carried by each persisted prediction row."""
    rows = [dict(row) for row in predictions]
    if not rows:
        return {}
    ensure_prop_offer_snapshot_schema(conn)
    locked_at = snapshot_at_utc or datetime.now(timezone.utc)
    source_ids = sorted({
        int(row["prop_offer_source_row_id"])
        for row in rows
        if row.get("prop_offer_source_row_id") is not None
    })
    source_by_id: dict[int, dict[str, Any]] = {}
    if source_ids:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM odds.mlb_player_prop_lines
                WHERE id = ANY(%s)
                """,
                (source_ids,),
            )
            source_by_id = {int(row["id"]): dict(row) for row in cur.fetchall()}

    game_slugs = sorted({str(row["game_slug"]) for row in rows if row.get("game_slug")})
    games: dict[str, dict[str, Any]] = {}
    if game_slugs:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT game_slug, start_ts_utc, home_team_abbr, away_team_abbr
                FROM raw.mlb_games
                WHERE game_slug = ANY(%s)
                """,
                (game_slugs,),
            )
            games = {str(row["game_slug"]): dict(row) for row in cur.fetchall()}

    payload: list[dict[str, Any]] = []
    keys_by_prediction: dict[int, str] = {}
    for row in rows:
        prediction_id = _clean_int(row.get("id"))
        line = _clean_float(row.get("book_line"))
        book = str(row.get("bookmaker_key") or "").lower()
        stat = str(row.get("stat") or "")
        player_name = str(row.get("player_name") or "")
        player_norm = normalize_name(player_name)
        if prediction_id is None or line is None or not book or not stat or not player_norm:
            continue
        source_id = _clean_int(row.get("prop_offer_source_row_id"))
        source = source_by_id.get(source_id or -1, {})
        game = games.get(str(row.get("game_slug") or ""), {})
        side = str(row.get("bet_side") or "").lower()
        side = side if side in {"over", "under"} else None
        over_price = _clean_int(row.get("over_price"))
        under_price = _clean_int(row.get("under_price"))
        if side == "over" and over_price is None:
            over_price = _clean_int(row.get("bet_price"))
        if side == "under" and under_price is None:
            under_price = _clean_int(row.get("bet_price"))
        over_link = source.get("over_link")
        under_link = source.get("under_link")
        if side == "over" and row.get("bet_link"):
            over_link = row.get("bet_link")
        if side == "under" and row.get("bet_link"):
            under_link = row.get("bet_link")
        key = _snapshot_key("lock", run_id, prediction_id, row.get("prediction_key"))
        keys_by_prediction[prediction_id] = key
        payload.append({
            "snapshot_key": key,
            "snapshot_role": "lock",
            "snapshot_at_utc": locked_at,
            "source_prop_line_id": source_id,
            "prediction_id": prediction_id,
            "prediction_key": row.get("prediction_key"),
            "run_id": run_id,
            "as_of_date": row.get("game_date_et"),
            "event_id": source.get("event_id"),
            "commence_time_utc": source.get("commence_time_utc") or game.get("start_ts_utc"),
            "bookmaker_key": book,
            "home_team": source.get("home_team") or game.get("home_team_abbr"),
            "away_team": source.get("away_team") or game.get("away_team_abbr"),
            "player_name": player_name,
            "player_name_norm": player_norm,
            "stat": stat,
            "selected_side": side,
            "line": line,
            "over_price": over_price,
            "under_price": under_price,
            "over_link": over_link,
            "under_link": under_link,
        })
    if not payload:
        return {}
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, _SNAPSHOT_INSERT_SQL, payload, page_size=500)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, prediction_id
            FROM odds.mlb_player_prop_line_snapshots
            WHERE snapshot_key = ANY(%s)
            """,
            (list(keys_by_prediction.values()),),
        )
        return {
            int(row["prediction_id"]): int(row["id"])
            for row in cur.fetchall()
            if row.get("prediction_id") is not None
        }


def _lock_snapshot_for_row(conn, row: dict[str, Any]) -> dict[str, Any] | None:
    lock_snapshot_id = _clean_int(row.get("lock_snapshot_id"))
    prediction_key = row.get("prediction_key")
    source_pred_id = _clean_int(row.get("source_pred_id") or row.get("prediction_id"))
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if lock_snapshot_id is not None:
            cur.execute(
                """
                SELECT *
                FROM odds.mlb_player_prop_line_snapshots
                WHERE id = %s AND snapshot_role = 'lock'
                LIMIT 1
                """,
                (lock_snapshot_id,),
            )
            lock = cur.fetchone()
            if lock:
                return dict(lock)
        if prediction_key or source_pred_id is not None:
            cur.execute(
                """
                SELECT s.*
                FROM bets.mlb_prop_predictions p
                JOIN odds.mlb_player_prop_line_snapshots s
                  ON s.id = p.lock_snapshot_id
                 AND s.snapshot_role = 'lock'
                WHERE (%s IS NOT NULL AND p.prediction_key = %s)
                   OR (%s IS NOT NULL AND p.id = %s)
                ORDER BY p.locked_at_utc ASC NULLS LAST, s.snapshot_at_utc ASC
                LIMIT 1
                """,
                (prediction_key, prediction_key, source_pred_id, source_pred_id),
            )
            lock = cur.fetchone()
            if lock:
                return dict(lock)
    return None


def _game_commence_time(conn, game_slug: Any) -> Any:
    if not game_slug:
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT start_ts_utc FROM raw.mlb_games WHERE game_slug = %s LIMIT 1",
            (game_slug,),
        )
        row = cur.fetchone()
    return row[0] if row else None


def _side_price(snapshot: dict[str, Any], side: str) -> float | None:
    return _clean_float(snapshot.get("over_price") if side == "over" else snapshot.get("under_price"))


def resolve_valid_prop_close(
    conn,
    row: dict[str, Any],
    *,
    max_hours_before_start: float = 2.0,
) -> dict[str, Any]:
    """Resolve a close only when it is a valid same-offer pregame observation."""
    unknown = {
        "valid": False,
        "status": "unknown",
        "unknown_reason": "no_valid_close_snapshot",
        "snapshot_id": None,
        "source_row_id": None,
        "fetched_at_utc": None,
        "line": None,
        "over_price": None,
        "under_price": None,
        "match_method": None,
    }
    lock = _lock_snapshot_for_row(conn, row)
    if not lock:
        return {**unknown, "unknown_reason": "missing_lock_snapshot"}

    side = str(row.get("side") or row.get("bet_side") or lock.get("selected_side") or "").lower()
    if side not in {"over", "under"}:
        return {**unknown, "unknown_reason": "missing_side"}
    game_date = row.get("game_date_et") or row.get("as_of_date") or lock.get("as_of_date")
    player_norm = (
        row.get("player_name_norm")
        or lock.get("player_name_norm")
        or normalize_name(row.get("player_name"))
    )
    stat = row.get("stat") or row.get("market") or lock.get("stat")
    book = str(row.get("bookmaker_key") or lock.get("bookmaker_key") or "").lower()
    line = _clean_float(row.get("market_line"))
    if line is None:
        line = _clean_float(row.get("book_line"))
    if line is None:
        line = _clean_float(row.get("bet_line"))
    if line is None:
        line = _clean_float(lock.get("line"))
    if not game_date or not player_norm or not stat or not book or line is None:
        return {**unknown, "unknown_reason": "missing_offer_identity"}

    lock_at = lock.get("snapshot_at_utc")
    commence = lock.get("commence_time_utc") or _game_commence_time(conn, row.get("game_slug"))
    if lock_at is None:
        return {**unknown, "unknown_reason": "missing_lock_timestamp"}
    if commence is None:
        return {**unknown, "unknown_reason": "missing_commence_time"}
    event_id = lock.get("event_id")
    if not event_id:
        return {**unknown, "unknown_reason": "missing_event_id"}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT *
            FROM odds.mlb_player_prop_line_snapshots
            WHERE snapshot_role = 'close'
              AND as_of_date = %s
              AND player_name_norm = %s
              AND stat = %s
              AND bookmaker_key = %s
              AND line = %s
              AND event_id = %s
            ORDER BY snapshot_at_utc DESC, id DESC
            """,
            (game_date, player_norm, stat, book, line, event_id),
        )
        exact = [dict(candidate) for candidate in cur.fetchall()]

    earliest_valid = commence - timedelta(hours=float(max_hours_before_start))
    valid = [
        candidate for candidate in exact
        if candidate.get("snapshot_at_utc") is not None
        and candidate["snapshot_at_utc"] > lock_at
        and earliest_valid <= candidate["snapshot_at_utc"] <= commence
        and _side_price(candidate, side) is not None
    ]
    if valid:
        close = valid[0]
        entry_price = _clean_float(
            lock.get("over_price") if side == "over" else lock.get("under_price")
        )
        if entry_price is None:
            entry_price = _clean_float(row.get("market_price") or row.get("bet_price"))
        close_price = _side_price(close, side)
        no_move = (
            _clean_float(close.get("line")) == _clean_float(lock.get("line"))
            and entry_price is not None
            and close_price == entry_price
        )
        return {
            "valid": True,
            "status": "true_no_movement" if no_move else "valid_movement",
            "unknown_reason": None,
            "snapshot_id": close.get("id"),
            "source_row_id": close.get("source_prop_line_id"),
            "fetched_at_utc": close.get("snapshot_at_utc"),
            "line": close.get("line"),
            "over_price": close.get("over_price"),
            "under_price": close.get("under_price"),
            "match_method": "same_book_exact_line_snapshot",
            "lock_line": lock.get("line"),
            "lock_over_price": lock.get("over_price"),
            "lock_under_price": lock.get("under_price"),
        }

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT *
            FROM odds.mlb_player_prop_line_snapshots
            WHERE snapshot_role = 'close'
              AND as_of_date = %s
              AND player_name_norm = %s
              AND stat = %s
              AND bookmaker_key = %s
              AND event_id = %s
              AND snapshot_at_utc > %s
              AND snapshot_at_utc BETWEEN %s AND %s
              AND line <> %s
            ORDER BY snapshot_at_utc DESC, id DESC
            """,
            (
                game_date, player_norm, stat, book, event_id,
                lock_at, earliest_valid, commence, line,
            ),
        )
        same_book_other_line = next(
            (dict(candidate) for candidate in cur.fetchall() if _side_price(dict(candidate), side) is not None),
            None,
        )
    if same_book_other_line:
        return {
            **unknown,
            "unknown_reason": "line_disappeared_at_close",
            "match_method": "same_book_other_line_at_close",
        }

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT *
            FROM odds.mlb_player_prop_line_snapshots
            WHERE snapshot_role = 'close'
              AND as_of_date = %s
              AND player_name_norm = %s
              AND stat = %s
              AND bookmaker_key <> %s
              AND line = %s
              AND event_id = %s
              AND snapshot_at_utc > %s
              AND snapshot_at_utc BETWEEN %s AND %s
            ORDER BY snapshot_at_utc DESC, id DESC
            """,
            (
                game_date, player_norm, stat, book, line, event_id,
                lock_at, earliest_valid, commence,
            ),
        )
        fallback = next(
            (dict(candidate) for candidate in cur.fetchall() if _side_price(dict(candidate), side) is not None),
            None,
        )
    if fallback:
        return {
            **unknown,
            "unknown_reason": "fallback_other_book_only",
            "match_method": "fallback_other_book_only",
        }
    if exact:
        timed = [candidate for candidate in exact if candidate.get("snapshot_at_utc") is not None]
        after_lock = [candidate for candidate in timed if candidate["snapshot_at_utc"] > lock_at]
        in_window = [
            candidate for candidate in after_lock
            if earliest_valid <= candidate["snapshot_at_utc"] <= commence
        ]
        if in_window and all(_side_price(candidate, side) is None for candidate in in_window):
            return {**unknown, "unknown_reason": "missing_close_side_price"}
        if after_lock:
            return {**unknown, "unknown_reason": "close_outside_two_hour_window"}
        return {**unknown, "unknown_reason": "stale_close_before_lock"}
    return unknown


def minimum_american_price(p_win: Any, min_ev: Any = 0.0) -> int | None:
    """Return the worst American price that still clears the requested EV."""
    p = _clean_float(p_win)
    target = _clean_float(min_ev)
    if p is None or target is None or p <= 0.0 or p >= 1.0:
        return None
    required_profit = (1.0 + target - p) / p
    if required_profit <= 0:
        return None
    american = 100.0 * required_profit if required_profit >= 1.0 else -100.0 / required_profit
    return int(math.ceil(american))
