"""Adapter: games still to start, from the existing pipelines' Postgres (read-only), so the dashboard can say
when prices will next appear. The scheduled scanners look at a game's props in its last 40 minutes (NBA) or its
last hour (NFL); nothing is stored for a game before that."""
from __future__ import annotations

from datetime import datetime, timedelta

from sports_sniper.providers import saved_quotes

LEAD_MINUTES = {"nba": 40, "nfl": 60}       # how long before the start the scanners first look at a game's props
QUERIES = {
    "nba": ("nbaprops.game_projections",
            """SELECT DISTINCT ON (game_id) away_team, home_team, tip_utc FROM nbaprops.game_projections
               WHERE tip_utc > %s AND tip_utc <= %s ORDER BY game_id, run_at_utc DESC"""),
    "nfl": ("raw.nfl_games",
            """SELECT away_team_abbr, home_team_abbr, start_ts_utc FROM raw.nfl_games
               WHERE start_ts_utc > %s AND start_ts_utc <= %s"""),
}


def upcoming(dsn: str | None, now: datetime, hours: float = 18.0, connect=None) -> list[dict]:
    """Games starting in the next `hours`, soonest first: sport, name, start, and when their prices should appear.
    Empty when the database is not there."""
    dsn = dsn or saved_quotes.default_dsn()
    if dsn is None:
        return []
    if connect is None:
        import psycopg2
        connect = lambda d: psycopg2.connect(d, options="-c timezone=UTC")      # noqa: E731
    out = []
    conn = connect(dsn)
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            for sport, (table, sql) in QUERIES.items():
                cur.execute("SELECT to_regclass(%s)", (table,))
                if cur.fetchone()[0] is None:
                    continue
                cur.execute(sql, (now, now + timedelta(hours=hours)))
                for away, home, start in cur.fetchall():
                    if start is not None:
                        out.append(dict(sport=sport, name=f"{away} @ {home}", start=start,
                                        prices_from=start - timedelta(minutes=LEAD_MINUTES[sport])))
    finally:
        conn.close()
    return sorted(out, key=lambda g: g["start"])
