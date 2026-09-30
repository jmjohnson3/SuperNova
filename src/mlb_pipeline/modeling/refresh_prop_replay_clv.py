"""Refresh CLV fields on locked MLB prop replay rows."""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg2

from .prop_replay import refresh_prop_replay_clv

_ET = ZoneInfo("America/New_York")
from mlb_pipeline.db import PG_DSN as _PG_DSN
def _default_date() -> date:
    env_date = os.getenv("MLB_ET_DATE")
    if env_date:
        return date.fromisoformat(env_date)
    return datetime.now(_ET).date()


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach valid close snapshots to MLB prop replay rows")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--date", default=None, help="ET date to refresh. Defaults to MLB_ET_DATE or today ET.")
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--run-id", action="append", default=None, help="Replay run id to refresh. Can be repeated.")
    parser.add_argument("--market", action="append", default=None, help="Limit to a prop market/stat. Can be repeated.")
    parser.add_argument("--side", action="append", default=None, choices=("over", "under"), help="Limit to a side. Can be repeated.")
    parser.add_argument("--bookmaker", action="append", default=None, help="Limit to a bookmaker key. Can be repeated.")
    parser.add_argument("--line-bucket", action="append", default=None, help="Limit to a line bucket. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum replay rows to refresh.")
    parser.add_argument("--batch-size", type=int, default=100, help="Replay rows to process and commit per batch.")
    parser.add_argument("--statement-timeout-ms", type=int, default=30_000, help="Postgres statement timeout for refresh queries.")
    parser.add_argument("--lock-timeout-ms", type=int, default=2_000, help="Postgres lock timeout for refresh updates.")
    parser.add_argument("--progress-every", type=int, default=100, help="Log progress after this many scanned rows.")
    parser.add_argument(
        "--ensure-schema",
        action="store_true",
        help="Create/upgrade replay schema before refreshing. Use only in maintenance jobs, not live repair.",
    )
    parser.add_argument("--pending-only", action="store_true", help="Do not refresh already-graded replay rows.")
    parser.add_argument("--only-missing", action="store_true", help="Only rows without a known CLV status.")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    date_from = _parse_date(args.date_from)
    date_to = _parse_date(args.date_to)
    if args.date:
        date_from = date_to = date.fromisoformat(args.date)
    elif args.lookback_days is not None:
        date_to = _default_date()
        date_from = date_to - timedelta(days=max(1, args.lookback_days) - 1)
    elif date_from is None and date_to is None and not args.run_id:
        date_from = date_to = _default_date()

    with psycopg2.connect(args.pg_dsn) as conn:
        refreshed = refresh_prop_replay_clv(
            conn,
            run_ids=args.run_id,
            date_from=date_from,
            date_to=date_to,
            include_graded=not args.pending_only,
            only_missing=args.only_missing,
            markets=args.market,
            sides=args.side,
            bookmakers=args.bookmaker,
            line_buckets=args.line_bucket,
            limit=args.limit,
            batch_size=args.batch_size,
            statement_timeout_ms=args.statement_timeout_ms,
            lock_timeout_ms=args.lock_timeout_ms,
            progress_every=args.progress_every,
            ensure_schema=args.ensure_schema,
        )
    print(json.dumps({
        "refreshed_rows": refreshed,
        "run_ids": args.run_id or "all",
        "markets": args.market or "all",
        "sides": args.side or "all",
        "bookmakers": args.bookmaker or "all",
        "line_buckets": args.line_bucket or "all",
        "limit": args.limit,
        "batch_size": args.batch_size,
        "statement_timeout_ms": args.statement_timeout_ms,
        "lock_timeout_ms": args.lock_timeout_ms,
        "ensure_schema": bool(args.ensure_schema),
        "date_from": date_from.isoformat() if date_from else None,
        "date_to": date_to.isoformat() if date_to else None,
        "include_graded": not args.pending_only,
        "only_missing": bool(args.only_missing),
    }, indent=2))


if __name__ == "__main__":
    main()
