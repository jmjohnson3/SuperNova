"""Capture sharp-book (EU region: Pinnacle, exchanges) NFL prop prices as a market reference.

FanDuel is only worth betting where its price is off-market. SportsGameOdds' tier here does not
include sharp books, but The Odds API's EU region does (Pinnacle props are available). One event-level
call costs (markets x regions) credits and returns every EU book, so this captures each game at most
once per role: 'lock' in the pregame window and 'close' shortly before kickoff, for exact-line
comparison at lock and sharp-close CLV. Payloads go through the normal raw store and prop parser,
landing in odds.nfl_player_prop_lines with their own bookmaker_key.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg2
import requests

from nfl_pipeline.crawler_oddsapi import OddsCrawlerConfig, _full_url, _save_payload
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.game_scope import game_ids
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.markets import normalize_team

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "reports" / "nfl_sharp_lines_state.json"
PROVIDER = "oddsapi_eu"
SHARP_BOOKS = ("pinnacle", "betfair_ex_eu", "matchbook", "smarkets")  # preference order
MARKETS = ("player_reception_yds",)  # the bet market; each extra market multiplies credits
WINDOWS_MINUTES = {"lock": 150, "close": 25}  # capture when kickoff is this close (and still ahead)
MIN_CREDITS_REMAINING = 150  # never spend the monthly budget below this floor
MAX_CALLS_PER_RUN = 20
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"


def _state() -> dict[str, Any]:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {"captured": {}}


def due_games(day: date, role: str, now: datetime, captured: dict[str, Any]) -> list[dict[str, Any]]:
    scope = game_ids()
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute("""SELECT game_id, home_team_abbr, away_team_abbr, start_ts_utc FROM raw.nfl_games
            WHERE game_date_et = %s AND start_ts_utc > %s AND start_ts_utc <= %s""",
                    (day, now, now + timedelta(minutes=WINDOWS_MINUTES[role])))
        games = [dict(zip(("game_id", "home", "away", "start"), r)) for r in cur.fetchall()]
    return [g for g in games if (scope is None or g["game_id"] in scope)
            and f"{g['game_id']}|{role}" not in captured]


def capture(day: date, role: str, *, now: datetime | None = None, session=requests) -> dict[str, Any]:
    if role not in WINDOWS_MINUTES:
        raise ValueError("role must be 'lock' or 'close'")
    now = now or datetime.now(timezone.utc)
    cfg = OddsCrawlerConfig()
    state = _state()
    games = due_games(day, role, now, state["captured"])
    result = dict(provider=PROVIDER, role=role, day=str(day), due_games=[g["game_id"] for g in games],
                  captured=[], skipped={}, credits_remaining=None, markets=list(MARKETS))
    if not games:
        return dict(result, status="nothing_due")
    if not cfg.oddsapi_key:
        return dict(result, status="skipped", reason="ODDS_API_KEY is not set")
    events = session.get(f"{BASE}/events", params={"apiKey": cfg.oddsapi_key}, timeout=cfg.timeout_s)
    events.raise_for_status()  # the events list does not cost credits
    by_matchup = {(normalize_team(e["home_team"]), normalize_team(e["away_team"])): e for e in events.json()}
    remaining = int(events.headers.get("x-requests-remaining") or 0)
    result["credits_remaining"] = remaining
    with psycopg2.connect(PG_DSN) as conn:
        for game in games[:MAX_CALLS_PER_RUN]:
            event = by_matchup.get((normalize_team(game["home"]), normalize_team(game["away"])))
            if event is None:
                result["skipped"][game["game_id"]] = "no_matching_event"
                continue
            if remaining - len(MARKETS) < MIN_CREDITS_REMAINING:
                result["skipped"][game["game_id"]] = "credit_floor"
                continue
            params = {"apiKey": cfg.oddsapi_key, "regions": "eu", "markets": ",".join(MARKETS),
                      "oddsFormat": "american", "dateFormat": "iso"}
            url = f"{BASE}/events/{event['id']}/odds"
            response = session.get(url, params=params, timeout=cfg.timeout_s)
            if not response.ok:
                result["skipped"][game["game_id"]] = f"http_{response.status_code}"
                continue
            remaining = int(response.headers.get("x-requests-remaining") or remaining - len(MARKETS))
            payload = response.json()
            _save_payload(conn, endpoint="nfl_player_props", snapshot_role=role, as_of_date=day,
                          url=_full_url(url, params), payload=payload, provider=PROVIDER)
            books = sorted(b["key"] for b in payload.get("bookmakers", []))
            state["captured"][f"{game['game_id']}|{role}"] = dict(at=now.isoformat(), event_id=event["id"], books=books)
            result["captured"].append(dict(game_id=game["game_id"], books=books))
    result["credits_remaining"] = remaining
    if result["captured"]:
        from nfl_pipeline.parse_oddsapi import ParseConfig, parse_props
        parse_props(ParseConfig(as_of_date=day))
        atomic_json(STATE, state)
    return dict(result, status="ok" if result["captured"] else "nothing_captured")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--role", choices=sorted(WINDOWS_MINUTES), required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.date, args.role), indent=2, default=str))


if __name__ == "__main__":
    main()
