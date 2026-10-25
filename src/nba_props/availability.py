"""Who is playing: nba.com daily lineups (starters, inactives) and ESPN's injury report.

Daily lineups (stats.nba.com/js/data/leaders/00_daily_lineups_YYYYMMDD.json, history from 2023-24):
every rostered player per game with rosterStatus Active/Inactive; the five with a position are the
starters, lineupStatus 'Expected' until the team confirms them (~30 minutes before tip). ESPN's
injury feed is current only (Out / Day-To-Day / ...), so it informs live runs, not training.

Each capture is appended with its time (nbaprops.lineup_snapshots, nbaprops.injury_snapshots); training
reads the last lineup capture per game, live runs read the latest capture before the projection.

    python -m nba_props.availability --backfill 2023-10-24     # history, one call per day
    python -m nba_props.availability                           # capture today's lineups and injuries
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2
import psycopg2.extras
import requests

from nfl_pipeline.db import PG_DSN
from nba_props.model import match_name

ET = ZoneInfo("America/New_York")
LINEUPS_URL = "https://stats.nba.com/js/data/leaders/00_daily_lineups_{day}.json"
INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/", "Accept": "application/json"}
ESPN_ABBR = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX"}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nbaprops.lineup_snapshots (
  captured_at_utc TIMESTAMPTZ NOT NULL, game_date DATE NOT NULL, game_id TEXT NOT NULL, team TEXT NOT NULL,
  player_id BIGINT NOT NULL, player_name TEXT NOT NULL, name_key TEXT NOT NULL, starter BOOLEAN NOT NULL,
  position TEXT, roster_status TEXT, lineup_status TEXT, source_timestamp TEXT,
  PRIMARY KEY (captured_at_utc, game_id, player_id));
CREATE INDEX IF NOT EXISTS lineup_snapshots_game ON nbaprops.lineup_snapshots (game_date, game_id, captured_at_utc);
CREATE TABLE IF NOT EXISTS nbaprops.injury_snapshots (
  captured_at_utc TIMESTAMPTZ NOT NULL, team TEXT NOT NULL, player_name TEXT NOT NULL, name_key TEXT NOT NULL,
  status TEXT NOT NULL, injury_type TEXT, return_date TEXT, reported_at TEXT,
  PRIMARY KEY (captured_at_utc, team, name_key));
CREATE INDEX IF NOT EXISTS injury_snapshots_time ON nbaprops.injury_snapshots (captured_at_utc);
"""


def fetch_lineups(day: date, session=requests) -> list[dict] | None:
    """Rows for every rostered player in the day's games; None when nba.com has no file for the day."""
    r = session.get(LINEUPS_URL.format(day=day.strftime("%Y%m%d")), headers=HEADERS, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    rows = []
    for g in r.json().get("games", []):
        for side in ("homeTeam", "awayTeam"):
            team = g[side]
            for p in team.get("players", []):
                rows.append(dict(game_date=day, game_id=str(g["gameId"]), team=team["teamAbbreviation"],
                                 player_id=int(p["personId"]), player_name=p["playerName"],
                                 name_key=match_name(p["playerName"]), starter=bool(p.get("position")),
                                 position=p.get("position") or None, roster_status=p.get("rosterStatus"),
                                 lineup_status=p.get("lineupStatus"), source_timestamp=p.get("timestamp")))
    return rows


def fetch_injuries(session=requests) -> list[dict]:
    d = session.get(INJURIES_URL, timeout=30).json()
    rows = []
    for team in d.get("injuries", []):
        for i in team.get("injuries", []):
            athlete = i.get("athlete") or {}
            abbr = (athlete.get("team") or {}).get("abbreviation") or ""
            rows.append(dict(team=ESPN_ABBR.get(abbr, abbr), player_name=athlete.get("displayName") or "",
                             name_key=match_name(athlete.get("displayName") or ""), status=i.get("status") or "",
                             injury_type=((i.get("details") or {}).get("type")),
                             return_date=((i.get("details") or {}).get("returnDate")), reported_at=i.get("date")))
    return [r for r in rows if r["name_key"]]


def save(conn, lineups: list[dict] | None, injuries: list[dict] | None, captured: datetime) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        if lineups:
            cols = ["game_date", "game_id", "team", "player_id", "player_name", "name_key", "starter", "position",
                    "roster_status", "lineup_status", "source_timestamp"]
            psycopg2.extras.execute_values(cur, f"""INSERT INTO nbaprops.lineup_snapshots (captured_at_utc, {', '.join(cols)})
                VALUES %s ON CONFLICT DO NOTHING""", [(captured, *[r[c] for c in cols]) for r in lineups], page_size=2000)
        if injuries:
            cols = ["team", "player_name", "name_key", "status", "injury_type", "return_date", "reported_at"]
            seen, rows = set(), []
            for r in injuries:
                if (r["team"], r["name_key"]) not in seen:
                    seen.add((r["team"], r["name_key"]))
                    rows.append((captured, *[r[c] for c in cols]))
            psycopg2.extras.execute_values(cur, f"""INSERT INTO nbaprops.injury_snapshots (captured_at_utc, {', '.join(cols)})
                VALUES %s ON CONFLICT DO NOTHING""", rows)
    conn.commit()


def news_digest(lineups: list[dict] | None, injuries: list[dict] | None, teams: set[str]) -> str:
    """A fingerprint of what matters for today's teams: starters, inactives and injury statuses."""
    parts = sorted((r["team"], r["name_key"], r["starter"], r["roster_status"], r["lineup_status"])
                   for r in (lineups or []) if r["team"] in teams)
    parts += sorted((r["team"], r["name_key"], r["status"]) for r in (injuries or []) if r["team"] in teams)
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def backfill(start: date, end: date) -> int:
    total = 0
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            cur.execute("SELECT DISTINCT game_date FROM nbaprops.lineup_snapshots")
            done = {r[0] for r in cur.fetchall()}
            cur.execute("SELECT DISTINCT game_date FROM nbaprops.player_games WHERE game_date BETWEEN %s AND %s", (start, end))
            days = sorted(r[0] for r in cur.fetchall() if r[0] not in done)
        conn.commit()
        for day in days:
            for attempt in range(3):
                try:
                    rows = fetch_lineups(day)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(10)
            if rows:
                # a backfilled file is the final pre-game state; stamp it at the day's end so live reads stay honest
                save(conn, rows, None, datetime.combine(day, datetime.min.time(), tzinfo=ET) + timedelta(hours=23, minutes=59))
                total += len(rows)
            print(day, len(rows or []), flush=True)
            time.sleep(0.8)
    return total


def capture(now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    day = now.astimezone(ET).date()
    lineups, injuries = fetch_lineups(day), fetch_injuries()
    with psycopg2.connect(PG_DSN) as conn:
        save(conn, lineups, injuries, now)
    return dict(day=str(day), lineup_rows=len(lineups or []), injury_rows=len(injuries))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backfill", type=date.fromisoformat, default=None, help="first day to backfill")
    args = ap.parse_args()
    if args.backfill:
        print(dict(rows=backfill(args.backfill, date.today() - timedelta(days=1))))
    else:
        print(capture())


if __name__ == "__main__":
    main()
