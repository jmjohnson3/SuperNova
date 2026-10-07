"""Compact raw.nfl_context_observations without changing any as-of answer.

The capture trigger used to treat `created_at_utc` and `raw_json` as content, so re-importing an
unchanged roster row recorded a fresh "observation" that said nothing new. 92% of roster rows, 83% of
injuries and 52% of depth-chart rows were duplicates of their immediate predecessor, and the log grew
~162k rows a week. Every as-of rebuild reads all of it, which on a spinning disk with no spare RAM is
the difference between a 0.2s and a 38s lookup - and eventually a timed-out prediction run.

This removes an observation only when it is identical to the one immediately before it for the same
entity, so a value that changes and later changes back keeps both of its transitions. It also strips
`raw_json` from the payloads it keeps; nothing reads that column back through the *_at() builders.

  python -m nfl_pipeline.compact_context_log --dry-run
  python -m nfl_pipeline.compact_context_log --apply
"""
from __future__ import annotations

import argparse
import json

import psycopg2

from nfl_pipeline.db import PG_DSN

KINDS = ("nfl_rosters", "nfl_depth_charts", "nfl_injuries")
# What identifies "the same thing over time" in each log.
ENTITY = {
    "nfl_rosters": ("season", "player_id", "team_abbr"),
    "nfl_depth_charts": ("season", "player_id", "team_abbr", "pos_abb"),
    "nfl_injuries": ("season", "week", "player_id", "team_abbr"),
}

# An observation is redundant when its content equals the previous observation's content for the same
# entity. Content excludes the bookkeeping columns the trigger now also ignores.
REDUNDANT_SQL = """
WITH ordered AS (
  SELECT ctid, observed_at,
         md5((payload - 'updated_at_utc' - 'created_at_utc' - 'raw_json')::text) AS body,
         LAG(md5((payload - 'updated_at_utc' - 'created_at_utc' - 'raw_json')::text))
           OVER (PARTITION BY {entity} ORDER BY observed_at, ctid) AS prev_body
  FROM raw.nfl_context_observations WHERE kind = %(kind)s
)
SELECT ctid FROM ordered WHERE prev_body IS NOT NULL AND body = prev_body
"""


def _entity_sql(kind: str) -> str:
    return ", ".join(f"payload->>'{field}'" for field in ENTITY[kind])


def compact(conn, kind: str, apply: bool) -> dict:
    with conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout='30s'")
        cur.execute(f"SELECT count(*) FROM raw.nfl_context_observations WHERE kind=%s", (kind,))
        before = cur.fetchone()[0]
        select_redundant = REDUNDANT_SQL.format(entity=_entity_sql(kind))
        if not apply:
            cur.execute(f"SELECT count(*) FROM ({select_redundant}) r", {"kind": kind})
            return dict(kind=kind, rows=before, redundant=cur.fetchone()[0], applied=False)
        cur.execute(f"DELETE FROM raw.nfl_context_observations WHERE ctid IN ({select_redundant})", {"kind": kind})
        removed = cur.rowcount
        # Keepers no longer need raw_json; the as-of builders never read it back.
        cur.execute("""UPDATE raw.nfl_context_observations SET payload = payload - 'raw_json'
                       WHERE kind = %(kind)s AND payload ? 'raw_json'""", {"kind": kind})
        stripped = cur.rowcount
    conn.commit()
    return dict(kind=kind, rows=before, removed=removed, raw_json_stripped=stripped,
                remaining=before - removed, applied=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="without this, only report what would be removed")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    apply = args.apply and not args.dry_run
    out = []
    with psycopg2.connect(PG_DSN) as conn:
        for kind in KINDS:
            out.append(compact(conn, kind, apply))
    if apply:
        # VACUUM cannot run inside a transaction block, and psycopg2's connection context manager
        # keeps one open, so this needs its own autocommit connection.
        vacuum = psycopg2.connect(PG_DSN)
        vacuum.autocommit = True
        with vacuum, vacuum.cursor() as cur:
            cur.execute("VACUUM (FULL, ANALYZE) raw.nfl_context_observations")
            cur.execute("SELECT pg_size_pretty(pg_total_relation_size('raw.nfl_context_observations'))")
            out.append({"table_size_after": cur.fetchone()[0]})
        vacuum.close()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
