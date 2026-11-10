"""Adapter: the price snapshots the existing NBA and NFL scanners already save in Postgres.

Those scheduled jobs poll The Odds API shortly before each game and keep every response. Reading them costs no
API credits, so this is the default source of real data. Read-only: nothing here writes to those tables.
"""
from __future__ import annotations

from datetime import datetime

from sports_sniper.normalize import parse_time
from sports_sniper.providers import Snapshot

QUERIES = {
    "nba": """SELECT observed_at_utc, payload FROM bets.sharp_multi_quotes
              WHERE sport = 'nba' AND observed_at_utc >= %s ORDER BY observed_at_utc""",
    "nfl": """SELECT fetched_at_utc, payload FROM raw.nfl_api_responses
              WHERE provider = 'oddsapi_watch' AND endpoint = 'nfl_player_props' AND fetched_at_utc >= %s
              ORDER BY fetched_at_utc""",
}


def default_dsn() -> str | None:
    try:
        from nfl_pipeline.db import PG_DSN
        return PG_DSN
    except Exception:
        return None


def load(dsn: str | None, sports: list[str], since: datetime, connect=None) -> list[Snapshot]:
    """Saved snapshots observed at or after `since`, oldest first. Empty when the database is not there."""
    dsn = dsn or default_dsn()
    if dsn is None:
        return []
    if connect is None:
        import psycopg2
        connect = psycopg2.connect
    out: list[Snapshot] = []
    conn = connect(dsn)
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            for sport in sports:
                if sport not in QUERIES:
                    continue
                cur.execute("SELECT to_regclass(%s)", ("bets.sharp_multi_quotes" if sport == "nba" else "raw.nfl_api_responses",))
                if cur.fetchone()[0] is None:
                    continue
                cur.execute(QUERIES[sport], (since,))
                for observed, payload in cur.fetchall():
                    event = payload.get("data") if isinstance(payload, dict) and "bookmakers" not in payload else payload
                    observed = parse_time(observed)
                    if isinstance(event, dict) and event.get("bookmakers") and observed is not None:
                        out.append(Snapshot(sport=sport, observed_at=observed, event=event))
    finally:
        conn.close()
    return sorted(out, key=lambda s: s.observed_at)
