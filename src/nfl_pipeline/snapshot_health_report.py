"""Report NFL lock/close odds snapshot health."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "reports" / "nfl_snapshot_health_latest.md"
_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SnapshotHealthConfig:
    pg_dsn: str = PG_DSN
    game_date: date | None = None
    out_file: Path = DEFAULT_OUT


def _rows(conn, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def _one(conn, sql: str, params: dict[str, Any]) -> dict[str, Any]:
    rows = _rows(conn, sql, params)
    return rows[0] if rows else {}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def build_report(cfg: SnapshotHealthConfig) -> str:
    et_day = cfg.game_date or datetime.now(_ET).date()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        raw = _rows(
            conn,
            """
            SELECT endpoint, snapshot_role, COUNT(*) AS payloads, MAX(fetched_at_utc) AS latest_fetch
            FROM raw.nfl_api_responses
            WHERE as_of_date = %(game_date)s
            GROUP BY endpoint, snapshot_role
            ORDER BY endpoint, snapshot_role
            """,
            {"game_date": et_day},
        )
        game = _rows(
            conn,
            """
            SELECT snapshot_role, COUNT(*) AS rows, COUNT(DISTINCT event_id) AS events, MAX(fetched_at_utc) AS latest_fetch
            FROM odds.nfl_game_lines
            WHERE as_of_date = %(game_date)s
            GROUP BY snapshot_role
            ORDER BY snapshot_role
            """,
            {"game_date": et_day},
        )
        props = _rows(
            conn,
            """
            SELECT snapshot_role, COUNT(*) AS rows, COUNT(DISTINCT player_name_norm) AS players, MAX(fetched_at_utc) AS latest_fetch
            FROM odds.nfl_player_prop_lines
            WHERE as_of_date = %(game_date)s
            GROUP BY snapshot_role
            ORDER BY snapshot_role
            """,
            {"game_date": et_day},
        )
        coverage = _one(
            conn,
            """
            WITH locked_games AS (
                SELECT event_id, bookmaker_key, COUNT(*) AS lock_rows
                FROM odds.nfl_game_lines
                WHERE as_of_date = %(game_date)s AND snapshot_role = 'lock'
                GROUP BY event_id, bookmaker_key
            ),
            closed_games AS (
                SELECT event_id, bookmaker_key
                FROM odds.nfl_game_lines
                WHERE as_of_date = %(game_date)s AND snapshot_role = 'close'
                GROUP BY event_id, bookmaker_key
            ),
            locked_props AS (
                SELECT player_name_norm, stat, line, bookmaker_key, COUNT(*) AS lock_rows
                FROM odds.nfl_player_prop_lines
                WHERE as_of_date = %(game_date)s AND snapshot_role = 'lock'
                GROUP BY player_name_norm, stat, line, bookmaker_key
            ),
            closed_props AS (
                SELECT player_name_norm, stat, line, bookmaker_key
                FROM odds.nfl_player_prop_lines
                WHERE as_of_date = %(game_date)s AND snapshot_role = 'close'
                GROUP BY player_name_norm, stat, line, bookmaker_key
            )
            SELECT
                (SELECT COUNT(*) FROM locked_games) AS locked_game_markets,
                (SELECT COUNT(*) FROM locked_games lg JOIN closed_games cg USING (event_id, bookmaker_key)) AS closed_game_markets,
                (SELECT COUNT(*) FROM locked_props) AS locked_prop_markets,
                (SELECT COUNT(*) FROM locked_props lp JOIN closed_props cp USING (player_name_norm, stat, line, bookmaker_key)) AS closed_prop_markets
            """,
            {"game_date": et_day},
        )

    lines = [
        f"# NFL Snapshot Health - {et_day}",
        "",
        "## Raw Payloads",
        "",
        "| Endpoint | Role | Payloads | Latest Fetch |",
        "|---|---|---:|---|",
    ]
    if raw:
        for row in raw:
            lines.append(f"| {_fmt(row['endpoint'])} | {_fmt(row['snapshot_role'])} | {_fmt(row['payloads'])} | {_fmt(row['latest_fetch'])} |")
    else:
        lines.append("| none | - | 0 | - |")
    lines.extend(["", "## Parsed Game Lines", "", "| Role | Rows | Events | Latest Fetch |", "|---|---:|---:|---|"])
    if game:
        for row in game:
            lines.append(f"| {_fmt(row['snapshot_role'])} | {_fmt(row['rows'])} | {_fmt(row['events'])} | {_fmt(row['latest_fetch'])} |")
    else:
        lines.append("| none | 0 | 0 | - |")
    lines.extend(["", "## Parsed Player Props", "", "| Role | Rows | Players | Latest Fetch |", "|---|---:|---:|---|"])
    if props:
        for row in props:
            lines.append(f"| {_fmt(row['snapshot_role'])} | {_fmt(row['rows'])} | {_fmt(row['players'])} | {_fmt(row['latest_fetch'])} |")
    else:
        lines.append("| none | 0 | 0 | - |")
    locked_game = int(coverage.get("locked_game_markets") or 0)
    closed_game = int(coverage.get("closed_game_markets") or 0)
    locked_prop = int(coverage.get("locked_prop_markets") or 0)
    closed_prop = int(coverage.get("closed_prop_markets") or 0)
    lines.extend([
        "",
        "## Lock To Close Coverage",
        "",
        "| Type | Locked | Closed Same Market | Coverage |",
        "|---|---:|---:|---:|",
        f"| game | {locked_game} | {closed_game} | {(closed_game / locked_game if locked_game else 0):.1%} |",
        f"| prop | {locked_prop} | {closed_prop} | {(closed_prop / locked_prop if locked_prop else 0):.1%} |",
        "",
    ])
    cfg.out_file.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    cfg.out_file.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Write NFL lock/close snapshot health report")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    print(build_report(SnapshotHealthConfig(
        pg_dsn=args.pg_dsn,
        game_date=date.fromisoformat(args.date) if args.date else None,
        out_file=Path(args.out_file),
    )))


if __name__ == "__main__":
    main()
