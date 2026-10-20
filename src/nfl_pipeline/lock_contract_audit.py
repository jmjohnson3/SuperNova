"""Audit the exact FanDuel locks produced by the live prediction run."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import os
from pathlib import Path
import tempfile
from typing import Any

import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.game_scope import game_ids as scoped_game_ids
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.pregame_checkpoints import lock_verification_passed, verify_locks


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
CONTRACT = "nfl-fanduel-lock-contract-v1"


def _games(day: date, scope: list[str] | None) -> list[dict[str, Any]]:
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute(
            """
            SELECT game_id, home_team_abbr, away_team_abbr, start_ts_utc, status
            FROM raw.nfl_games
            WHERE game_date_et=%s AND (%s IS NULL OR game_id=ANY(%s))
            ORDER BY start_ts_utc, game_id
            """,
            (day, scope, scope),
        )
        return [dict(row) for row in cur.fetchall()]


def build_report(day: date, required: bool = False) -> dict[str, Any]:
    scope = scoped_game_ids()
    games = _games(day, scope)
    verification = verify_locks(day, [str(game["game_id"]) for game in games])
    rows: list[dict[str, Any]] = []
    for game in games:
        game_id = str(game["game_id"])
        check = verification.get(game_id) or {"game_id": game_id}
        passed = lock_verification_passed(check)
        if passed:
            status = "passed"
        elif int(check.get("current_fanduel_locks") or 0) == 0 and not required:
            status = "waiting_for_fanduel_locks"
        else:
            status = "failed"
        rows.append(
            {
                **game,
                "game_id": game_id,
                "status": status,
                "verification": check,
            }
        )
    failures = sum(row["status"] == "failed" for row in rows)
    waiting = sum(row["status"] == "waiting_for_fanduel_locks" for row in rows)
    passed = sum(row["status"] == "passed" for row in rows)
    if not rows:
        overall = "failed" if required else "no_games"
    elif failures:
        overall = "failed"
    elif waiting:
        overall = "waiting"
    else:
        overall = "passed"
    return {
        "contract": CONTRACT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date": day.isoformat(),
        "required": required,
        "game_scope": scope,
        "status": overall,
        "summary": {
            "games": len(rows),
            "passed": passed,
            "waiting": waiting,
            "failed": failures,
        },
        "games": rows,
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_report(report: dict[str, Any]) -> None:
    day = report["date"]
    for name in (f"nfl_lock_contract_{day}.json", "nfl_lock_contract_latest.json"):
        atomic_json(REPORTS / name, report)
    lines = [
        "# NFL FanDuel Lock Contract",
        "",
        f"- Date: `{day}`",
        f"- Status: **{report['status']}**",
        f"- Games: {report['summary']['games']}",
        f"- Passed / waiting / failed: {report['summary']['passed']} / "
        f"{report['summary']['waiting']} / {report['summary']['failed']}",
        "",
        "A pass requires current immutable FanDuel locks, working FanDuel links, "
        "explicit injury context, and receiving baseline/production/challenger "
        "probabilities captured on the same book, side, line, and price.",
        "",
    ]
    for row in report["games"]:
        check = row["verification"]
        matchup = f"{row.get('away_team_abbr')} @ {row.get('home_team_abbr')}"
        lines.extend(
            [
                f"## {matchup}",
                "",
                f"- Status: `{row['status']}`",
                f"- Kickoff: `{row.get('start_ts_utc')}`",
                f"- FanDuel locks: {check.get('current_fanduel_locks', 0)}",
                f"- Receiving same-line triplets: "
                f"{check.get('receiving_same_line_triplets', 0)} / "
                f"{check.get('receiving_locks', 0)}",
                f"- Explicit injury contexts: "
                f"{check.get('explicit_injury_context_locks', 0)} / "
                f"{check.get('current_fanduel_locks', 0)}",
                f"- Missing links: {check.get('all_locks_missing_links', 0)}",
                f"- Immutable lock violations: {check.get('immutable_lock_violations', 0)}",
                "",
            ]
        )
    rendered = "\n".join(lines).rstrip() + "\n"
    for name in (f"nfl_lock_contract_{day}.md", "nfl_lock_contract_latest.md"):
        _atomic_text(REPORTS / name, rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit current exact FanDuel forecast locks")
    parser.add_argument("--date", required=True, help="ET game date YYYY-MM-DD")
    parser.add_argument(
        "--required",
        action="store_true",
        help="Fail when scoped games have no FanDuel locks",
    )
    args = parser.parse_args()
    report = build_report(date.fromisoformat(args.date), required=args.required)
    write_report(report)
    print(report)
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
