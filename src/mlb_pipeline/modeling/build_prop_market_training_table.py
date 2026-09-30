"""CLI for rebuilding side-level MLB prop market training examples."""
from __future__ import annotations

import argparse
import json
from datetime import date

from .prop_market_training import PropMarketTrainingConfig, refresh_prop_market_training_examples

from mlb_pipeline.db import PG_DSN as _PG_DSN
def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop side-level market training table")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--run-id", action="append", default=[], help="Limit to one replay run_id; repeatable.")
    parser.add_argument("--market", action="append", default=[], help="Limit to one market/stat; repeatable.")
    parser.add_argument("--side", action="append", default=[], choices=("over", "under"), help="Limit to one side; repeatable.")
    parser.add_argument("--bookmaker", action="append", default=[], help="Limit to one bookmaker key; repeatable.")
    parser.add_argument("--line-bucket", action="append", default=[], help="Limit to one prop line bucket; repeatable.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum replay rows to rebuild.")
    parser.add_argument("--statement-timeout-ms", type=int, default=300_000, help="Postgres statement timeout for the targeted rebuild.")
    parser.add_argument("--lock-timeout-ms", type=int, default=5_000, help="Postgres lock timeout for delete/upsert operations.")
    parser.add_argument("--include-pending", action="store_true")
    parser.add_argument(
        "--allow-unlocked",
        action="store_true",
        help="Include replay rows without lock snapshots. Research only; real-money reports should omit this.",
    )
    parser.add_argument("--no-replace", action="store_true", help="Upsert without deleting matching rows first.")
    parser.add_argument(
        "--ensure-schema",
        action="store_true",
        help="Create/upgrade required tables and compatibility views before refreshing. Use in maintenance/nightly jobs.",
    )
    args = parser.parse_args()

    result = refresh_prop_market_training_examples(PropMarketTrainingConfig(
        pg_dsn=args.pg_dsn,
        lookback_days=args.lookback_days,
        date_from=_parse_date(args.date_from),
        date_to=_parse_date(args.date_to),
        run_ids=tuple(args.run_id or ()),
        include_pending=args.include_pending,
        require_lock=not args.allow_unlocked,
        replace=not args.no_replace,
        ensure_schema=args.ensure_schema,
        markets=tuple(args.market or ()),
        sides=tuple(args.side or ()),
        bookmakers=tuple(args.bookmaker or ()),
        line_buckets=tuple(args.line_bucket or ()),
        limit=args.limit,
        statement_timeout_ms=args.statement_timeout_ms,
        lock_timeout_ms=args.lock_timeout_ms,
    ))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
