"""Clean-slate coverage helpers for MLB prop CLV collection.

A clean shadow slate is a date where model picks were locked first and enough
same-book exact-line close snapshots were captured near first pitch. Promotion
code uses these helpers so dirty CLV collection days cannot accidentally become
real-money evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import psycopg2.extras


@dataclass(frozen=True)
class CleanSlateThresholds:
    min_side_locks: int = 100
    min_valid_side_locks: int = 100
    min_valid_coverage: float = 0.25
    max_missing_lock_rate: float = 0.02
    max_stale_close_rate: float = 0.05
    require_close_times: bool = True


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rate(numerator: Any, denominator: Any) -> float | None:
    den = _safe_int(denominator)
    if den <= 0:
        return None
    return _safe_int(numerator) / den


def clean_slate_reasons(row: dict[str, Any], thresholds: CleanSlateThresholds) -> list[str]:
    side_locks = _safe_int(row.get("side_lock_rows"))
    valid_side_locks = _safe_int(row.get("valid_side_locks"))
    training_rows = _safe_int(row.get("training_rows"))
    close_times = _safe_int(row.get("close_times"))
    missing_lock_rate = _rate(row.get("missing_lock_examples"), training_rows)
    stale_close_rate = _rate(row.get("stale_close_examples"), training_rows)
    valid_coverage = _rate(valid_side_locks, side_locks)

    reasons: list[str] = []
    if side_locks < thresholds.min_side_locks:
        reasons.append(f"side_locks<{thresholds.min_side_locks}")
    if valid_side_locks < thresholds.min_valid_side_locks:
        reasons.append(f"valid_side_locks<{thresholds.min_valid_side_locks}")
    if valid_coverage is None or valid_coverage < thresholds.min_valid_coverage:
        reasons.append(f"valid_clv_coverage<{thresholds.min_valid_coverage:.2f}")
    if training_rows <= 0:
        reasons.append("no_training_rows")
    if missing_lock_rate is None or missing_lock_rate > thresholds.max_missing_lock_rate:
        reasons.append(f"missing_lock_rate>{thresholds.max_missing_lock_rate:.2f}")
    if stale_close_rate is None or stale_close_rate > thresholds.max_stale_close_rate:
        reasons.append(f"stale_close_rate>{thresholds.max_stale_close_rate:.2f}")
    if thresholds.require_close_times and close_times <= 0:
        reasons.append("no_close_snapshot_time")
    return reasons


def clean_slate_qualifies(row: dict[str, Any], thresholds: CleanSlateThresholds) -> bool:
    return not clean_slate_reasons(row, thresholds)


def decorate_clean_slate_row(row: dict[str, Any], thresholds: CleanSlateThresholds) -> dict[str, Any]:
    out = dict(row)
    training_rows = _safe_int(out.get("training_rows"))
    side_locks = _safe_int(out.get("side_lock_rows"))
    out["valid_clv_coverage"] = _rate(out.get("valid_side_locks"), side_locks)
    out["missing_lock_rate"] = _rate(out.get("missing_lock_examples"), training_rows)
    out["stale_close_rate"] = _rate(out.get("stale_close_examples"), training_rows)
    reasons = clean_slate_reasons(out, thresholds)
    out["clean_slate"] = not reasons
    out["clean_slate_reasons"] = reasons
    return out


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


def load_clean_slate_rows(
    conn,
    *,
    date_from: date,
    date_to: date,
    thresholds: CleanSlateThresholds,
) -> list[dict[str, Any]]:
    has_offers = _table_exists(conn, "features", "mlb_prop_offer_links")
    has_training = _table_exists(conn, "features", "mlb_prop_market_training_examples")
    has_snapshots = _table_exists(conn, "odds", "mlb_player_prop_line_snapshots")
    offers_cte = """
    offers AS (
        SELECT as_of_date AS slate_date, COUNT(*) AS offer_rows
        FROM features.mlb_prop_offer_links
        WHERE as_of_date BETWEEN %(date_from)s AND %(date_to)s
        GROUP BY as_of_date
    )
    """ if has_offers else """
    offers AS (
        SELECT NULL::date AS slate_date, 0::bigint AS offer_rows
        WHERE FALSE
    )
    """
    snapshots_cte = """
    snapshots AS (
        SELECT *
        FROM odds.mlb_player_prop_line_snapshots
        WHERE as_of_date BETWEEN %(date_from)s AND %(date_to)s
    )
    """ if has_snapshots else """
    snapshots AS (
        SELECT
            NULL::bigint AS id,
            NULL::date AS as_of_date,
            NULL::text AS snapshot_role,
            NULL::text AS selected_side,
            NULL::text AS run_id,
            NULL::timestamptz AS snapshot_at_utc,
            NULL::text AS event_id,
            NULL::text AS player_name_norm,
            NULL::text AS stat,
            NULL::text AS bookmaker_key,
            NULL::numeric AS line,
            NULL::timestamptz AS commence_time_utc,
            NULL::integer AS over_price,
            NULL::integer AS under_price
        WHERE FALSE
    )
    """
    training_cte = """
    training_counts AS (
        SELECT
            game_date_et AS slate_date,
            COUNT(*) AS training_rows,
            COUNT(*) FILTER (
                WHERE lock_snapshot_id IS NULL
                   OR clv_unknown_reason = 'missing_lock_snapshot'
            ) AS missing_lock_examples,
            COUNT(*) FILTER (WHERE clv_unknown_reason = 'stale_close_before_lock') AS stale_close_examples,
            COUNT(*) FILTER (WHERE clv_unknown_reason = 'close_outside_two_hour_window') AS outside_window_examples,
            COUNT(*) FILTER (WHERE clv_unknown_reason = 'no_valid_close_snapshot') AS no_valid_close_examples,
            COUNT(*) FILTER (WHERE clv_valid IS TRUE) AS valid_training_clv_rows
        FROM features.mlb_prop_market_training_examples
        WHERE game_date_et BETWEEN %(date_from)s AND %(date_to)s
        GROUP BY game_date_et
    )
    """ if has_training else """
    training_counts AS (
        SELECT
            NULL::date AS slate_date,
            0::bigint AS training_rows,
            0::bigint AS missing_lock_examples,
            0::bigint AS stale_close_examples,
            0::bigint AS outside_window_examples,
            0::bigint AS no_valid_close_examples,
            0::bigint AS valid_training_clv_rows
        WHERE FALSE
    )
    """
    sql = f"""
    WITH dates AS (
        SELECT generate_series(%(date_from)s::date, %(date_to)s::date, interval '1 day')::date AS slate_date
    ),
    {offers_cte},
    {snapshots_cte},
    snapshot_counts AS (
        SELECT
            as_of_date AS slate_date,
            COUNT(*) FILTER (WHERE snapshot_role = 'open') AS open_rows,
            COUNT(*) FILTER (WHERE snapshot_role = 'lock') AS lock_rows,
            COUNT(*) FILTER (
                WHERE snapshot_role = 'lock'
                  AND selected_side IN ('over', 'under')
            ) AS side_lock_rows,
            COUNT(*) FILTER (WHERE snapshot_role = 'close') AS close_rows,
            COUNT(DISTINCT run_id) FILTER (WHERE snapshot_role = 'lock') AS lock_phases,
            COUNT(DISTINCT snapshot_at_utc) FILTER (WHERE snapshot_role = 'close') AS close_times,
            COUNT(*) FILTER (
                WHERE snapshot_role = 'lock'
                  AND (event_id IS NULL OR commence_time_utc IS NULL)
            ) AS lock_identity_gaps
        FROM snapshots
        GROUP BY as_of_date
    ),
    lock_coverage AS (
        SELECT
            l.as_of_date AS slate_date,
            l.id AS lock_id,
            EXISTS (
                SELECT 1
                FROM odds.mlb_player_prop_line_snapshots c
                WHERE c.snapshot_role = 'close'
                  AND c.as_of_date = l.as_of_date
                  AND c.event_id = l.event_id
                  AND c.player_name_norm = l.player_name_norm
                  AND c.stat = l.stat
                  AND c.bookmaker_key = l.bookmaker_key
                  AND c.line = l.line
                  AND c.snapshot_at_utc > l.snapshot_at_utc
                  AND c.snapshot_at_utc BETWEEN
                        l.commence_time_utc - interval '2 hours'
                        AND l.commence_time_utc
                  AND (
                        (l.selected_side = 'over' AND c.over_price IS NOT NULL)
                     OR (l.selected_side = 'under' AND c.under_price IS NOT NULL)
                  )
            ) AS has_valid_close
        FROM snapshots l
        WHERE l.snapshot_role = 'lock'
          AND l.selected_side IN ('over', 'under')
    ),
    valid_counts AS (
        SELECT
            slate_date,
            COUNT(*) FILTER (WHERE has_valid_close) AS valid_side_locks
        FROM lock_coverage
        GROUP BY slate_date
    ),
    {training_cte}
    SELECT
        d.slate_date,
        COALESCE(o.offer_rows, 0) AS offer_rows,
        COALESCE(s.open_rows, 0) AS open_rows,
        COALESCE(s.lock_rows, 0) AS lock_rows,
        COALESCE(s.side_lock_rows, 0) AS side_lock_rows,
        COALESCE(s.close_rows, 0) AS close_rows,
        COALESCE(v.valid_side_locks, 0) AS valid_side_locks,
        COALESCE(s.lock_phases, 0) AS lock_phases,
        COALESCE(s.close_times, 0) AS close_times,
        COALESCE(s.lock_identity_gaps, 0) AS lock_identity_gaps,
        COALESCE(t.training_rows, 0) AS training_rows,
        COALESCE(t.valid_training_clv_rows, 0) AS valid_training_clv_rows,
        COALESCE(t.missing_lock_examples, 0) AS missing_lock_examples,
        COALESCE(t.stale_close_examples, 0) AS stale_close_examples,
        COALESCE(t.outside_window_examples, 0) AS outside_window_examples,
        COALESCE(t.no_valid_close_examples, 0) AS no_valid_close_examples
    FROM dates d
    LEFT JOIN offers o USING (slate_date)
    LEFT JOIN snapshot_counts s USING (slate_date)
    LEFT JOIN valid_counts v USING (slate_date)
    LEFT JOIN training_counts t USING (slate_date)
    ORDER BY d.slate_date DESC
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, {"date_from": date_from, "date_to": date_to})
        rows = [dict(row) for row in cur.fetchall()]
    return [decorate_clean_slate_row(row, thresholds) for row in rows]


def clean_date_set(rows: list[dict[str, Any]]) -> set[date]:
    return {
        row["slate_date"]
        for row in rows
        if bool(row.get("clean_slate")) and row.get("slate_date") is not None
    }
