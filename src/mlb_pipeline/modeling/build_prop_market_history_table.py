"""CLI for rebuilding historical side-level MLB prop market examples."""
from __future__ import annotations

import argparse
import json
from datetime import date

from .prop_market_history import PropMarketHistoryConfig, refresh_prop_market_history_examples

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical MLB prop side-level market table")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=540)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--market", action="append", default=[], help="Limit to one market; repeatable.")
    parser.add_argument("--bookmaker", action="append", default=[], help="Limit to one bookmaker; repeatable.")
    parser.add_argument("--no-replace", action="store_true", help="Upsert without deleting matching rows first.")
    args = parser.parse_args()

    result = refresh_prop_market_history_examples(PropMarketHistoryConfig(
        pg_dsn=args.pg_dsn,
        lookback_days=args.lookback_days,
        date_from=_parse_date(args.date_from),
        date_to=_parse_date(args.date_to),
        markets=tuple(args.market or ("pitcher_strikeouts", "batter_hits", "batter_total_bases", "batter_home_runs")),
        bookmakers=tuple(args.bookmaker or ("fanduel", "draftkings")),
        replace=not args.no_replace,
    ))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
