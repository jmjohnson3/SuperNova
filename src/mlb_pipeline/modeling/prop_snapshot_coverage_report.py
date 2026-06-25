"""Report MLB prop shadow-lock and valid close-snapshot coverage by slate."""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from .prop_clean_slate import CleanSlateThresholds, load_clean_slate_rows
from .prop_offer_snapshots import ensure_prop_offer_snapshot_schema

_ET = ZoneInfo("America/New_York")
from mlb_pipeline.db import PG_DSN as _PG_DSN
_COVERAGE_SQL = """
WITH dates AS (
    SELECT generate_series(%s::date, %s::date, interval '1 day')::date AS slate_date
),
offers AS (
    SELECT as_of_date AS slate_date, COUNT(*) AS offer_rows
    FROM features.mlb_prop_offer_links
    WHERE as_of_date BETWEEN %s AND %s
    GROUP BY as_of_date
),
snapshots AS (
    SELECT *
    FROM odds.mlb_player_prop_line_snapshots
    WHERE as_of_date BETWEEN %s AND %s
),
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
)
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
    COALESCE(s.lock_identity_gaps, 0) AS lock_identity_gaps
FROM dates d
LEFT JOIN offers o USING (slate_date)
LEFT JOIN snapshot_counts s USING (slate_date)
LEFT JOIN valid_counts v USING (slate_date)
ORDER BY d.slate_date DESC
"""


def _pct(numerator: int, denominator: int) -> str:
    if not denominator:
        return "-"
    return f"{numerator / denominator:.1%}"


def slate_qualifies(
    *,
    side_locks: int,
    valid_side_locks: int,
    min_valid_locks: int,
    min_valid_coverage: float,
) -> bool:
    if side_locks <= 0 or valid_side_locks < min_valid_locks:
        return False
    return valid_side_locks / side_locks >= min_valid_coverage


def build_report(
    conn,
    *,
    date_from: date,
    date_to: date,
    target_slates: int,
    min_valid_locks_per_slate: int,
    min_valid_coverage: float,
    max_missing_lock_rate: float,
    max_stale_close_rate: float,
) -> str:
    ensure_prop_offer_snapshot_schema(conn)
    thresholds = CleanSlateThresholds(
        min_side_locks=min_valid_locks_per_slate,
        min_valid_side_locks=min_valid_locks_per_slate,
        min_valid_coverage=min_valid_coverage,
        max_missing_lock_rate=max_missing_lock_rate,
        max_stale_close_rate=max_stale_close_rate,
    )
    rows = load_clean_slate_rows(
        conn,
        date_from=date_from,
        date_to=date_to,
        thresholds=thresholds,
    )

    valid_slates = sum(1 for row in rows if row.get("clean_slate"))
    remaining = max(target_slates - valid_slates, 0)
    status = "COLLECTING" if remaining else "TARGET MET"
    lines = [
        "# MLB Prop Snapshot Coverage",
        "",
        f"Generated: {datetime.now(_ET).isoformat(timespec='seconds')}",
        f"Range: {date_from} to {date_to}",
        "",
        "## Collection Status",
        "",
        f"**{status}**",
        "",
        f"- Clean shadow slates: {valid_slates} / {target_slates}",
        f"- Additional clean slates needed: {remaining}",
        f"- A clean slate needs at least {min_valid_locks_per_slate} side locks, {min_valid_locks_per_slate} valid exact side closes, and {min_valid_coverage:.1%} valid-close coverage.",
        f"- Missing-lock rate must be <= {max_missing_lock_rate:.1%}; stale-close-before-lock rate must be <= {max_stale_close_rate:.1%}.",
        "- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.",
        "",
        "## Slate Coverage",
        "",
        "| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        side_locks = int(row["side_lock_rows"] or 0)
        valid_locks = int(row["valid_side_locks"] or 0)
        training_rows = int(row.get("training_rows") or 0)
        lines.append(
            "| {slate_date} | {clean} | {offer_rows} | {open_rows} | {lock_rows} | "
            "{side_lock_rows} | {close_rows} | {valid_side_locks} | {coverage} | "
            "{missing_lock} | {stale_close} | {lock_phases} | {close_times} | {reasons} |".format(
                **row,
                clean="yes" if row.get("clean_slate") else "no",
                coverage=_pct(valid_locks, side_locks),
                missing_lock=_pct(int(row.get("missing_lock_examples") or 0), training_rows),
                stale_close=_pct(int(row.get("stale_close_examples") or 0), training_rows),
                reasons=", ".join(row.get("clean_slate_reasons") or []),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report MLB prop lock and valid close snapshot coverage")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--target-slates", type=int, default=10)
    parser.add_argument("--min-valid-locks-per-slate", type=int, default=100)
    parser.add_argument("--min-valid-coverage", type=float, default=0.25)
    parser.add_argument("--max-missing-lock-rate", type=float, default=0.02)
    parser.add_argument("--max-stale-close-rate", type=float, default=0.05)
    parser.add_argument("--date", default=None, help="End date in YYYY-MM-DD format. Defaults to today ET.")
    parser.add_argument(
        "--output",
        default="reports/mlb_prop_snapshot_coverage_latest.md",
        help="Markdown report path.",
    )
    args = parser.parse_args()

    date_to = date.fromisoformat(args.date) if args.date else datetime.now(_ET).date()
    date_from = date_to - timedelta(days=max(args.lookback_days, 1) - 1)
    with psycopg2.connect(args.pg_dsn) as conn:
        report = build_report(
            conn,
            date_from=date_from,
            date_to=date_to,
            target_slates=max(args.target_slates, 1),
            min_valid_locks_per_slate=max(args.min_valid_locks_per_slate, 1),
            min_valid_coverage=min(max(args.min_valid_coverage, 0.0), 1.0),
            max_missing_lock_rate=min(max(args.max_missing_lock_rate, 0.0), 1.0),
            max_stale_close_rate=min(max(args.max_stale_close_rate, 0.0), 1.0),
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
