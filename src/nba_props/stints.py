"""Rotation stints from stats.nba.com: who was on the floor, and when, in every game. Research only.

One call per game (GameRotation) gives each player's stints: the time he came on and went off, in tenths of a
second from the tip. Stored once per game in nbaprops.rotation_stints; a game already stored is not asked for
again, so the download can be stopped and restarted.

    python -m nba_props.stints                 every finished game not stored yet (about 2 seconds a game)
    python -m nba_props.stints --limit 200     the next 200 games
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN

FIRST = "2022-10-01"           # the seasons from 2022-23 come first, oldest game first; 2021-22 after them
SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS nbaprops;
CREATE TABLE IF NOT EXISTS nbaprops.rotation_stints (
  game_id TEXT NOT NULL, player_id BIGINT NOT NULL, team_id BIGINT NOT NULL, in_time INTEGER NOT NULL,
  out_time INTEGER NOT NULL, pts INTEGER, pt_diff NUMERIC, usg_pct NUMERIC,
  PRIMARY KEY (game_id, player_id, in_time));
CREATE TABLE IF NOT EXISTS nbaprops.rotation_games (
  game_id TEXT PRIMARY KEY, stints INTEGER NOT NULL, fetched_at TIMESTAMPTZ NOT NULL DEFAULT now());
"""
COLUMNS = ("game_id", "player_id", "team_id", "in_time", "out_time", "pts", "pt_diff", "usg_pct")
RETRY_WAITS = (5, 20, 60)      # seconds between attempts at one game
MAX_FAILED_IN_A_ROW = 8        # then a long pause; stats.nba.com shuts the door for a while when it is asked too fast
LONG_PAUSE = 600
MAX_LONG_PAUSES = 4


def pending(conn, limit: int | None = None) -> list[str]:
    """Finished games without stored stints: 2022-23 onward oldest first, then the 2021-22 season."""
    with conn.cursor() as cur:
        cur.execute("""SELECT g.game_id FROM (SELECT game_id, min(game_date) AS game_date FROM nbaprops.player_games
                                              WHERE minutes > 0 GROUP BY game_id) g
                       LEFT JOIN nbaprops.rotation_games r USING (game_id)
                       WHERE r.game_id IS NULL AND g.game_date < current_date
                       ORDER BY g.game_date < %s, g.game_date, g.game_id""", (FIRST,))
        games = [r[0] for r in cur.fetchall()]
    return games[:limit] if limit else games


def to_rows(frames: list[pd.DataFrame]) -> list[tuple]:
    """The endpoint's two tables (away, home) as rows; a stint without both times is dropped, and when the feed
    repeats a player's entry time the longer stint is kept."""
    rows = {}
    for df in frames:
        for r in df.to_dict("records"):
            if pd.isna(r.get("IN_TIME_REAL")) or pd.isna(r.get("OUT_TIME_REAL")):
                continue
            number = lambda v: None if pd.isna(v) else float(v)
            row = (str(r["GAME_ID"]), int(r["PERSON_ID"]), int(r["TEAM_ID"]), int(round(r["IN_TIME_REAL"])),
                   int(round(r["OUT_TIME_REAL"])), None if pd.isna(r.get("PLAYER_PTS")) else int(r["PLAYER_PTS"]),
                   number(r.get("PT_DIFF")), number(r.get("USG_PCT")))
            if row[:4] not in rows or row[4] > rows[row[:4]][4]:
                rows[row[:4]] = row
    return [rows[key] for key in sorted(rows)]


def fetch(game_id: str) -> list[pd.DataFrame]:
    from nba_api.stats.endpoints import gamerotation
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            return gamerotation.GameRotation(game_id=game_id, timeout=30).get_data_frames()
        except Exception:
            if attempt == len(RETRY_WAITS):
                raise
            time.sleep(RETRY_WAITS[attempt])


def store(conn, game_id: str, rows: list[tuple]) -> None:
    with conn.cursor() as cur:
        if rows:
            psycopg2.extras.execute_values(cur, f"""INSERT INTO nbaprops.rotation_stints ({", ".join(COLUMNS)}) VALUES %s
                ON CONFLICT (game_id, player_id, in_time) DO UPDATE SET
                {", ".join(f"{c}=EXCLUDED.{c}" for c in COLUMNS[2:] if c != "in_time")}""", rows)
        cur.execute("""INSERT INTO nbaprops.rotation_games (game_id, stints) VALUES (%s, %s)
                       ON CONFLICT (game_id) DO UPDATE SET stints=EXCLUDED.stints, fetched_at=now()""", (game_id, len(rows)))
    conn.commit()


def download(limit: int | None = None, pause: float = 0.6, fetcher=fetch, log=print, sleep=time.sleep) -> dict:
    done = failed = failed_in_a_row = long_pauses = 0
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
        games = pending(conn, limit)
        log(f"{len(games)} games to download")
        for n, game_id in enumerate(games, 1):
            try:
                rows = to_rows(fetcher(game_id))
                store(conn, game_id, rows)
            except Exception as error:
                conn.rollback()
                failed, failed_in_a_row = failed + 1, failed_in_a_row + 1
                log(f"  {game_id}: failed ({type(error).__name__})")
                if failed_in_a_row >= MAX_FAILED_IN_A_ROW:
                    long_pauses += 1
                    if long_pauses > MAX_LONG_PAUSES:
                        log("stats.nba.com keeps refusing; stopping. Run again later to carry on.")
                        break
                    log(f"  {failed_in_a_row} failures in a row: waiting {LONG_PAUSE // 60} minutes")
                    sleep(LONG_PAUSE)
                    failed_in_a_row = 0
                continue
            failed_in_a_row = 0
            done += 1
            if n % 50 == 0:
                log(f"  {n}/{len(games)} games ({failed} failed)")
            sleep(pause)
    log(f"stored {done} games, {failed} failed")
    return dict(games=len(games), stored=done, failed=failed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pause", type=float, default=0.6, help="seconds between games")
    args = ap.parse_args()
    download(args.limit, args.pause, log=lambda text: print(text, flush=True))


if __name__ == "__main__":
    main()
