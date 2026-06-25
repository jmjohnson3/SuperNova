"""Build the normalized MLB prop offer/link table."""
from __future__ import annotations

import argparse
import json
from datetime import date

import psycopg2

from .prop_offer_links import DEFAULT_BOOKMAKERS, refresh_prop_offer_links

from mlb_pipeline.db import PG_DSN as _PG_DSN
def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize MLB prop offer links into side-level rows")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--date", default=None, help="Build one game date, YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--all", action="store_true", help="Build every date in odds.mlb_player_prop_lines")
    parser.add_argument("--bookmakers", nargs="*", default=list(DEFAULT_BOOKMAKERS))
    args = parser.parse_args()

    game_date = date.fromisoformat(args.date) if args.date else None
    lookback_days = None if args.all or game_date else args.lookback_days
    with psycopg2.connect(args.pg_dsn) as conn:
        result = refresh_prop_offer_links(
            conn,
            game_date=game_date,
            lookback_days=lookback_days,
            bookmakers=args.bookmakers,
        )
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
