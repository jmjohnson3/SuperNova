"""Same-day NFL injury/inactive statuses from ESPN's public game summary, as timestamped context.

nflverse injury files lag days (Thursday 10/1 forecasts saw reports from 9/27 and still projected a
back ESPN had listed Out since 9/30). ESPN's game summary carries each team's current report, updated
through game day when inactives are posted (~90 minutes before kickoff). Rows go into raw.nfl_injuries
for the game's season/week (source 'espn_live'), so the existing as-of observation trigger records
when we learned them and live context picks the newest status for that week. A player ESPN previously
listed but no longer lists gets a 'Cleared' row so a stale 'Out' cannot linger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

import psycopg2
import requests

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.game_scope import game_ids
from nfl_pipeline.import_context import _upsert_injuries
from nfl_pipeline.markets import normalize_name, normalize_team

SOURCE = "espn_live"
CLEARED = "Cleared"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


def _row_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join("" if p is None else str(p) for p in parts).encode()).hexdigest()


def parse_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for team in summary.get("injuries") or []:
        abbr = normalize_team((team.get("team") or {}).get("abbreviation"))
        for item in team.get("injuries") or []:
            athlete = item.get("athlete") or {}
            if not athlete.get("displayName") or not abbr:
                continue
            entries.append(dict(team=abbr, espn_id=str(athlete.get("id") or "") or None, name=athlete["displayName"],
                                position=((athlete.get("position") or {}).get("abbreviation") or "").upper() or None,
                                status=item.get("status"), injury=(item.get("details") or {}).get("type"),
                                espn_updated_at=item.get("date")))
    return entries


def _rows(game: dict[str, Any], entries: list[dict[str, Any]], ids: dict[str, str], previous: dict[tuple, dict],
          url: str) -> list[tuple]:
    rows, listed = [], set()
    for e in entries:
        player_id = ids.get(e["espn_id"] or "")
        key = (e["team"], player_id or normalize_name(e["name"]))
        listed.add(key)
        rows.append(_tuple(game, e["team"], player_id, e["name"], e["position"], e["injury"], e["status"], url, e))
    for key, prev in previous.items():  # listed earlier by ESPN this week, absent now -> cleared
        if key not in listed and prev["report_status"] != CLEARED and key[0] in (game["home"], game["away"]):
            rows.append(_tuple(game, key[0], prev["player_id"], prev["player_name"], prev["position"], None, CLEARED, url,
                               {"cleared_after": prev["report_status"]}))
    return rows


def _tuple(game, team, player_id, name, position, injury, status, url, raw) -> tuple:
    return (_row_hash(SOURCE, game["season"], game["week"], team, player_id, name, status, injury),
            game["season"], "REG", "REG", game["week"], team, player_id, name, normalize_name(name), position,
            injury, None, status, None, None, None, SOURCE, url, json.dumps(raw, default=str))


def capture(day: date, *, now: datetime | None = None, session=requests) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    scope = game_ids()
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='60s'")
            cur.execute("""SELECT game_id, season, week, home_team_abbr, away_team_abbr FROM raw.nfl_games
                WHERE game_date_et = %s AND start_ts_utc > %s""", (day, now - timedelta(minutes=30)))
            games = [dict(zip(("game_id", "season", "week", "home", "away"), r)) for r in cur.fetchall()]
            games = [g for g in games if scope is None or g["game_id"] in scope]
            if not games:
                return dict(status="nothing_due", day=str(day))
            cur.execute("""SELECT DISTINCT ON (espn_id) espn_id, player_id FROM raw.nfl_rosters
                WHERE season = %s AND espn_id IS NOT NULL AND player_id IS NOT NULL ORDER BY espn_id, week DESC""",
                        (games[0]["season"],))
            ids = {str(e).split(".")[0]: p for e, p in cur.fetchall()}
        board = session.get(f"{ESPN}/scoreboard", params={"dates": day.strftime("%Y%m%d")}, timeout=20)
        board.raise_for_status()
        events = {}
        for event in board.json().get("events", []):
            teams = {(c.get("homeAway"), normalize_team((c.get("team") or {}).get("abbreviation")))
                     for comp in event.get("competitions", []) for c in comp.get("competitors", [])}
            home = next((t for side, t in teams if side == "home"), None)
            away = next((t for side, t in teams if side == "away"), None)
            events[(home, away)] = event["id"]
        result = dict(status="ok", day=str(day), games=[], unmatched=[])
        for game in games:
            event_id = events.get((normalize_team(game["home"]), normalize_team(game["away"])))
            if not event_id:
                result["unmatched"].append(game["game_id"])
                continue
            url = f"{ESPN}/summary?event={event_id}"
            summary = session.get(f"{ESPN}/summary", params={"event": event_id}, timeout=20)
            summary.raise_for_status()
            entries = parse_summary(summary.json())
            with conn.cursor() as cur:
                cur.execute("""SELECT DISTINCT ON (team_abbr, COALESCE(player_id, player_name_norm))
                        team_abbr, COALESCE(player_id, player_name_norm), player_id, player_name, position, report_status
                    FROM raw.nfl_injuries WHERE source = %s AND season = %s AND week = %s AND team_abbr IN (%s, %s)
                    ORDER BY team_abbr, COALESCE(player_id, player_name_norm), updated_at_utc DESC""",
                            (SOURCE, game["season"], game["week"], game["home"], game["away"]))
                previous = {(t, k): dict(player_id=p, player_name=n, position=pos, report_status=s)
                            for t, k, p, n, pos, s in cur.fetchall()}
            rows = list({r[0]: r for r in _rows(game, entries, ids, previous, url)}.values())  # one row per hash
            written = _upsert_injuries(conn, rows) if rows else 0
            result["games"].append(dict(game_id=game["game_id"], entries=len(entries), rows=written,
                                        mapped=sum(1 for e in entries if ids.get(e["espn_id"] or "")),
                                        out=sorted(e["name"] for e in entries if str(e["status"]).lower() in ("out", "doubtful"))))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    print(json.dumps(capture(parser.parse_args().date), indent=2, default=str))


if __name__ == "__main__":
    main()
