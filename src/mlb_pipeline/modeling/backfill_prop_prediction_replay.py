"""Backfill MLB prop prediction replay rows.

Two modes:
  1. Snapshot existing bets.mlb_prop_predictions rows into replay storage.
  2. Optionally run the current predictor for each date first, then snapshot.

The replay table is used by side calibrators, betting-layer training, CLV
diagnostics, and failure reports.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

import psycopg2

from .prop_replay import ensure_prop_replay_schema, grade_prop_replay, snapshot_prop_predictions

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _run_predictor_for_date(day: date, *, timeout_s: int) -> None:
    env = os.environ.copy()
    env["DISCORD_FORMAT"] = "0"
    cmd = [
        sys.executable,
        "-m",
        "src.mlb_pipeline.modeling.predict_player_props",
        "--date",
        day.isoformat(),
    ]
    proc = subprocess.run(
        cmd,
        env=env,
        timeout=timeout_s,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode:
        if proc.stdout:
            print(proc.stdout[-4000:], file=sys.stdout)
        if proc.stderr:
            print(proc.stderr[-4000:], file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill MLB prop prediction replay rows")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--run-id", default=None, help="Replay run identifier. Defaults to replay_<UTC timestamp>.")
    parser.add_argument("--run-predictor", action="store_true", help="Run current prop predictor for each date before snapshotting.")
    parser.add_argument("--predict-timeout-s", type=int, default=900)
    parser.add_argument("--side-only", action="store_true", help="Only snapshot rows with over/under model sides.")
    parser.add_argument("--include-inactive", action="store_true", help="Snapshot inactive/stale prediction rows too.")
    parser.add_argument("--grade", action="store_true", help="Grade replay rows after snapshotting.")
    parser.add_argument("--regrade", action="store_true", help="Recompute already-graded replay rows for this run.")
    args = parser.parse_args()

    date_from = date.fromisoformat(args.date_from)
    date_to = date.fromisoformat(args.date_to)
    if date_to < date_from:
        raise SystemExit("--date-to must be >= --date-from")
    run_id = args.run_id or f"replay_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    total_snapshotted = 0
    with psycopg2.connect(args.pg_dsn) as conn:
        ensure_prop_replay_schema(conn)

    for day in _date_range(date_from, date_to):
        if args.run_predictor:
            print(f"[{day}] running current prop predictor...")
            _run_predictor_for_date(day, timeout_s=args.predict_timeout_s)
        with psycopg2.connect(args.pg_dsn) as conn:
            n = snapshot_prop_predictions(
                conn,
                run_id=run_id,
                date_from=day,
                date_to=day,
                include_no_side=not args.side_only,
                active_only=not args.include_inactive,
            )
            total_snapshotted += n
        print(f"[{day}] snapshotted {n} rows into run_id={run_id}")

    graded = 0
    if args.grade:
        with psycopg2.connect(args.pg_dsn) as conn:
            graded = grade_prop_replay(conn, run_ids=[run_id], include_graded=args.regrade)
        print(f"graded {graded} replay rows for run_id={run_id}")

    print(f"done run_id={run_id} rows={total_snapshotted} graded={graded}")


if __name__ == "__main__":
    main()
