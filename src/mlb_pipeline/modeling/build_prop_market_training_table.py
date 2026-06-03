"""CLI for rebuilding side-level MLB prop market training examples."""
from __future__ import annotations

import argparse
import json
from datetime import date

from .prop_market_training import PropMarketTrainingConfig, refresh_prop_market_training_examples

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop side-level market training table")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--run-id", action="append", default=[], help="Limit to one replay run_id; repeatable.")
    parser.add_argument("--include-pending", action="store_true")
    parser.add_argument("--no-replace", action="store_true", help="Upsert without deleting matching rows first.")
    args = parser.parse_args()

    result = refresh_prop_market_training_examples(PropMarketTrainingConfig(
        pg_dsn=args.pg_dsn,
        lookback_days=args.lookback_days,
        date_from=_parse_date(args.date_from),
        date_to=_parse_date(args.date_to),
        run_ids=tuple(args.run_id or ()),
        include_pending=args.include_pending,
        replace=not args.no_replace,
    ))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
