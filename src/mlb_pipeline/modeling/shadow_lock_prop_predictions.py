"""Lock active MLB prop predictions into shadow replay storage.

This is the daily shadow-mode capture step.  It snapshots the current active
offer-level rows from bets.mlb_prop_predictions without changing bankroll
eligibility.  The replay table is then used for grading, CLV coverage, direct
side models, bucket promotion, and post-mortems.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

import psycopg2

from .prop_replay import grade_prop_replay, snapshot_prop_predictions

_ET = ZoneInfo("America/New_York")
from mlb_pipeline.db import PG_DSN as _PG_DSN
def _default_date() -> date:
    env_date = os.getenv("MLB_ET_DATE")
    if env_date:
        return date.fromisoformat(env_date)
    return datetime.now(_ET).date()


def _default_run_id(day: date, phase: str | None = None) -> str:
    phase_label = "".join(ch for ch in str(phase or "").lower() if ch.isalnum() or ch in {"_", "-"})
    suffix = f"_{phase_label}" if phase_label else ""
    return f"prop_shadow{suffix}_{day.strftime('%Y%m%d')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock active MLB prop prediction rows into shadow replay storage")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--date", default=None, help="ET date to lock. Defaults to MLB_ET_DATE or today ET.")
    parser.add_argument("--run-id", default=None, help="Stable replay run id. Defaults to prop_shadow_YYYYMMDD.")
    parser.add_argument("--phase", default=None, help="Optional stable lock phase, such as pregame.")
    parser.add_argument("--side-only", action="store_true", help="Only lock rows with over/under model sides.")
    parser.add_argument("--include-inactive", action="store_true", help="Also lock inactive/stale prediction rows.")
    parser.add_argument("--grade", action="store_true", help="Grade locked rows after snapshotting.")
    parser.add_argument("--regrade", action="store_true", help="Recompute already-graded rows for this run.")
    args = parser.parse_args()

    day = date.fromisoformat(args.date) if args.date else _default_date()
    run_id = args.run_id or _default_run_id(day, args.phase)
    with psycopg2.connect(args.pg_dsn) as conn:
        locked = snapshot_prop_predictions(
            conn,
            run_id=run_id,
            date_from=day,
            date_to=day,
            include_no_side=not args.side_only,
            active_only=not args.include_inactive,
        )
        graded = 0
        if args.grade:
            graded = grade_prop_replay(conn, run_ids=[run_id], include_graded=args.regrade)
    print(json.dumps({
        "date": day.isoformat(),
        "run_id": run_id,
        "phase": args.phase,
        "locked_rows": locked,
        "graded_rows": graded,
        "active_only": not args.include_inactive,
        "side_only": bool(args.side_only),
    }, indent=2))


if __name__ == "__main__":
    main()
