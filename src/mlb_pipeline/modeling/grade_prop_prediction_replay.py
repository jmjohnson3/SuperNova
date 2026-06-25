"""Grade pending MLB prop shadow replay rows."""
from __future__ import annotations

import argparse
import json

import psycopg2

from .prop_replay import grade_prop_replay

from mlb_pipeline.db import PG_DSN as _PG_DSN
def main() -> None:
    parser = argparse.ArgumentParser(description="Grade pending MLB prop prediction replay rows")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--run-id", action="append", default=None, help="Replay run id to grade. Can be repeated.")
    parser.add_argument("--regrade", action="store_true", help="Recompute already-graded replay rows too.")
    args = parser.parse_args()

    with psycopg2.connect(args.pg_dsn) as conn:
        graded = grade_prop_replay(conn, run_ids=args.run_id, include_graded=args.regrade)
    print(json.dumps({
        "graded_rows": graded,
        "run_ids": args.run_id or "all_pending",
        "regrade": bool(args.regrade),
    }, indent=2))


if __name__ == "__main__":
    main()
