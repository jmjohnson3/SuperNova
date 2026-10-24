"""Load FBS history from collegefootballdata.com into the database.

Per season (a few calls each; the free tier allows 1,000 a month):
    /games                 every game: teams, conferences, neutral site, final score
    /lines                 every book's spread and total, with the opener where the book reports one
    /stats/game/advanced   per team-game efficiency (plays, PPA, success rate, explosiveness, ...)

Raw responses are kept in raw.cfbd_responses (one row per endpoint and season, replaced on refresh),
then parsed into ncaaf.games, ncaaf.lines and ncaaf.team_game_stats.

    python -m ncaaf_pipeline.ingest --seasons 2014-2026
    python -m ncaaf_pipeline.ingest --seasons 2026          # refresh the current season
"""
from __future__ import annotations

import argparse
import json
import os
import time

import psycopg2
import psycopg2.extras
import requests

from nfl_pipeline.db import PG_DSN

BASE = "https://api.collegefootballdata.com"
ENDPOINTS = {"games": "/games", "lines": "/lines", "advanced": "/stats/game/advanced"}

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS ncaaf;
CREATE TABLE IF NOT EXISTS raw.cfbd_responses (
  endpoint TEXT NOT NULL, season INTEGER NOT NULL, season_type TEXT NOT NULL,
  fetched_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(), payload JSONB NOT NULL,
  PRIMARY KEY (endpoint, season, season_type));
CREATE TABLE IF NOT EXISTS ncaaf.games (
  game_id BIGINT PRIMARY KEY, season INTEGER NOT NULL, week INTEGER, season_type TEXT,
  start_date TIMESTAMPTZ, neutral_site BOOLEAN, conference_game BOOLEAN, completed BOOLEAN,
  home_id INTEGER, home_team TEXT, home_conference TEXT, home_division TEXT, home_points INTEGER,
  away_id INTEGER, away_team TEXT, away_conference TEXT, away_division TEXT, away_points INTEGER,
  venue_id INTEGER);
CREATE TABLE IF NOT EXISTS ncaaf.lines (
  game_id BIGINT NOT NULL, provider TEXT NOT NULL, spread NUMERIC, spread_open NUMERIC,
  total NUMERIC, total_open NUMERIC, home_moneyline INTEGER, away_moneyline INTEGER,
  PRIMARY KEY (game_id, provider));
CREATE TABLE IF NOT EXISTS ncaaf.team_game_stats (
  game_id BIGINT NOT NULL, season INTEGER NOT NULL, week INTEGER, team TEXT NOT NULL, opponent TEXT,
  offense JSONB, defense JSONB, PRIMARY KEY (game_id, team));
"""


def api_key() -> str:
    key = os.getenv("CFBD_API_KEY")
    if not key:
        from supernovabets_config import _saved_windows_env
        key = _saved_windows_env("CFBD_API_KEY")
    if not key:
        raise SystemExit("CFBD_API_KEY is not set")
    return key


def fetch(endpoint: str, season: int, season_type: str, session=requests) -> tuple[list, str | None]:
    params = {"year": season, "seasonType": season_type}
    if endpoint == "games":
        params["classification"] = "fbs"
    for attempt in range(4):  # the API returns occasional 502/503s under load
        response = session.get(BASE + ENDPOINTS[endpoint], params=params,
                               headers={"Authorization": f"Bearer {api_key()}"}, timeout=120)
        if response.status_code not in (502, 503, 504) or attempt == 3:
            break
        time.sleep(5 * 2 ** attempt)
    response.raise_for_status()
    return response.json(), response.headers.get("X-CallLimit-Remaining")


def parse_games(payload: list) -> list[tuple]:
    rows = []
    for g in payload:
        rows.append((g["id"], g["season"], g.get("week"), g.get("seasonType"), g.get("startDate"),
                     g.get("neutralSite"), g.get("conferenceGame"), g.get("completed"),
                     g.get("homeId"), g.get("homeTeam"), g.get("homeConference"), g.get("homeClassification"),
                     g.get("homePoints"), g.get("awayId"), g.get("awayTeam"), g.get("awayConference"),
                     g.get("awayClassification"), g.get("awayPoints"), g.get("venueId")))
    return rows


def parse_lines(payload: list) -> list[tuple]:
    rows = {}
    for g in payload:
        for line in g.get("lines") or []:
            key = (g["id"], line.get("provider"))
            rows[key] = (g["id"], line.get("provider"), line.get("spread"), line.get("spreadOpen"),
                         line.get("overUnder"), line.get("overUnderOpen"),
                         line.get("homeMoneyline"), line.get("awayMoneyline"))
    return list(rows.values())


def parse_advanced(payload: list) -> list[tuple]:
    return [(g["gameId"], g["season"], g.get("week"), g["team"], g.get("opponent"),
             psycopg2.extras.Json(g.get("offense")), psycopg2.extras.Json(g.get("defense"))) for g in payload]


UPSERTS = {
    "games": ("""INSERT INTO ncaaf.games VALUES %s ON CONFLICT (game_id) DO UPDATE SET
                 week=EXCLUDED.week, season_type=EXCLUDED.season_type, start_date=EXCLUDED.start_date,
                 neutral_site=EXCLUDED.neutral_site, conference_game=EXCLUDED.conference_game,
                 completed=EXCLUDED.completed, home_points=EXCLUDED.home_points, away_points=EXCLUDED.away_points""",
              parse_games),
    "lines": ("""INSERT INTO ncaaf.lines VALUES %s ON CONFLICT (game_id, provider) DO UPDATE SET
                 spread=EXCLUDED.spread, spread_open=EXCLUDED.spread_open, total=EXCLUDED.total,
                 total_open=EXCLUDED.total_open, home_moneyline=EXCLUDED.home_moneyline,
                 away_moneyline=EXCLUDED.away_moneyline""", parse_lines),
    "advanced": ("""INSERT INTO ncaaf.team_game_stats VALUES %s ON CONFLICT (game_id, team) DO UPDATE SET
                    offense=EXCLUDED.offense, defense=EXCLUDED.defense""", parse_advanced),
}


def ingest(seasons: list[int], session=requests) -> dict:
    summary = {}
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
        for season in seasons:
            for season_type in ("regular", "postseason"):
                for endpoint, (sql, parse) in UPSERTS.items():
                    payload, remaining = fetch(endpoint, season, season_type, session)
                    rows = parse(payload)
                    with conn.cursor() as cur:
                        cur.execute("""INSERT INTO raw.cfbd_responses (endpoint, season, season_type, payload)
                                       VALUES (%s,%s,%s,%s) ON CONFLICT (endpoint, season, season_type) DO UPDATE
                                       SET payload=EXCLUDED.payload, fetched_at_utc=now()""",
                                    (endpoint, season, season_type, json.dumps(payload)))
                        if rows:
                            psycopg2.extras.execute_values(cur, sql, rows, page_size=2000)
                    conn.commit()
                    summary[f"{season}-{season_type}-{endpoint}"] = len(rows)
                    print(f"{season} {season_type:10s} {endpoint:9s} {len(rows):6d} rows (calls left {remaining})", flush=True)
                    time.sleep(0.3)
    return summary


def parse_seasons(text: str) -> list[int]:
    if "-" in text:
        start, end = map(int, text.split("-"))
        return list(range(start, end + 1))
    return [int(s) for s in text.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2026")
    args = ap.parse_args()
    ingest(parse_seasons(args.seasons))


if __name__ == "__main__":
    main()
