"""Load NBA player game logs from stats.nba.com (nba_api) into nbaprops.player_games.

One call per season and season type returns every player's games, so a full refresh is a few calls.

    python -m nba_props.ingest --seasons 2022-2026     # 2022-23 through 2026-27
    python -m nba_props.ingest                         # current season only
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date

import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN

SEASON_TYPES = ("Regular Season", "Playoffs")
SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS nbaprops;
CREATE TABLE IF NOT EXISTS nbaprops.player_games (
  game_id TEXT NOT NULL, player_id BIGINT NOT NULL, season TEXT NOT NULL, season_type TEXT NOT NULL,
  game_date DATE NOT NULL, player_name TEXT NOT NULL, team TEXT NOT NULL, opponent TEXT NOT NULL,
  home BOOLEAN NOT NULL, minutes NUMERIC, pts INTEGER, reb INTEGER, ast INTEGER, fg3m INTEGER,
  fga INTEGER, fta INTEGER, tov INTEGER, oreb INTEGER, stl INTEGER, blk INTEGER, plus_minus NUMERIC,
  PRIMARY KEY (game_id, player_id));
CREATE INDEX IF NOT EXISTS player_games_date ON nbaprops.player_games (game_date);
CREATE TABLE IF NOT EXISTS nbaprops.ingest_conflicts (
  game_id TEXT NOT NULL, player_id BIGINT NOT NULL, candidates JSONB NOT NULL,
  observed_at TIMESTAMPTZ NOT NULL DEFAULT now(), resolved BOOLEAN NOT NULL DEFAULT FALSE,
  PRIMARY KEY (game_id, player_id));
"""
COLUMNS = ("game_id", "player_id", "season", "season_type", "game_date", "player_name", "team", "opponent",
           "home", "minutes", "pts", "reb", "ast", "fg3m", "fga", "fta", "tov", "oreb", "stl", "blk", "plus_minus")


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def current_start_year(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 9 else today.year - 1


def fetch(season: str, season_type: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import playergamelogs
    for attempt in range(4):  # stats.nba.com times out now and then
        try:
            return playergamelogs.PlayerGameLogs(season_nullable=season, season_type_nullable=season_type,
                                                 timeout=90).get_data_frames()[0]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))


def prepare_rows(df: pd.DataFrame, season: str, season_type: str) -> tuple[list[tuple], list[dict]]:
    if df.empty:
        return [], []
    matchup = df.MATCHUP.str.split(r" (?:vs\.|@) ", regex=True)
    out = pd.DataFrame(dict(
        game_id=df.GAME_ID.astype(str), player_id=df.PLAYER_ID.astype(int), season=season, season_type=season_type,
        game_date=pd.to_datetime(df.GAME_DATE).dt.date, player_name=df.PLAYER_NAME, team=df.TEAM_ABBREVIATION,
        opponent=matchup.str[1], home=df.MATCHUP.str.contains(" vs. ", regex=False),
        minutes=df.MIN.astype(float).round(2), pts=df.PTS, reb=df.REB, ast=df.AST, fg3m=df.FG3M, fga=df.FGA,
        fta=df.FTA, tov=df.TOV, oreb=df.OREB, stl=df.STL, blk=df.BLK, plus_minus=df.PLUS_MINUS))
    out = out.astype(object).where(out.notna(), None)
    out = out.drop_duplicates()
    duplicate = out.duplicated(["game_id", "player_id"], keep=False)
    rows = list(out.loc[~duplicate, list(COLUMNS)].itertuples(index=False, name=None))
    conflicts = []
    for (game_id, player_id), group in out[duplicate].groupby(["game_id", "player_id"], sort=False):
        # Merge complementary missing fields only when every supplied value agrees.
        values = [group[c].dropna().unique().tolist() for c in COLUMNS]
        if any(len(v) > 1 for v in values):
            conflicts.append(dict(game_id=game_id, player_id=int(player_id),
                                  candidates=group.to_dict("records")))
        else:
            rows.append(tuple(v[0] if v else None for v in values))
    return rows, conflicts


def to_rows(df: pd.DataFrame, season: str, season_type: str) -> list[tuple]:
    rows, conflicts = prepare_rows(df, season, season_type)
    if conflicts:
        raise ValueError(f"Conflicting player-game records: {[(c['game_id'], c['player_id']) for c in conflicts]}")
    return rows


def conflicted_games(conn) -> list[str]:
    """Exclude an entire game while any of its player records is ambiguous."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('nbaprops.ingest_conflicts')")
        if cur.fetchone()[0] is None:
            return []
        cur.execute("SELECT DISTINCT game_id FROM nbaprops.ingest_conflicts WHERE NOT resolved")
        return [r[0] for r in cur.fetchall()]


def ingest(start_years: list[int]) -> dict:
    summary = {}
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
        for year in start_years:
            season = season_label(year)
            for season_type in SEASON_TYPES:
                rows, conflicts = prepare_rows(fetch(season, season_type), season, season_type)
                with conn.cursor() as cur:
                    if rows:
                        psycopg2.extras.execute_values(cur, """UPDATE nbaprops.ingest_conflicts c SET resolved=TRUE
                            FROM (VALUES %s) AS good(game_id, player_id)
                            WHERE c.game_id=good.game_id AND c.player_id=good.player_id""",
                            [(r[0], r[1]) for r in rows])
                    if conflicts:
                        psycopg2.extras.execute_values(cur, """INSERT INTO nbaprops.ingest_conflicts
                            (game_id, player_id, candidates) VALUES %s ON CONFLICT (game_id, player_id)
                            DO UPDATE SET candidates=EXCLUDED.candidates, observed_at=now(), resolved=FALSE""",
                            [(c["game_id"], c["player_id"], psycopg2.extras.Json(c["candidates"],
                              dumps=lambda v: json.dumps(v, default=str))) for c in conflicts])
                    excluded = set(conflicted_games(conn))
                    rows = [r for r in rows if r[0] not in excluded]
                    if rows:
                        psycopg2.extras.execute_values(cur, f"""
                            INSERT INTO nbaprops.player_games ({", ".join(COLUMNS)}) VALUES %s
                            ON CONFLICT (game_id, player_id) DO UPDATE SET
                            {", ".join(f"{c}=EXCLUDED.{c}" for c in COLUMNS[2:])}""", rows, page_size=5000)
                conn.commit()
                summary[f"{season} {season_type}"] = len(rows)
                if conflicts:
                    print(f"Quarantined {len(conflicts)} conflicting player-game keys; excluded {len(excluded)} games", flush=True)
                print(f"{season} {season_type:15s} {len(rows):6d} rows", flush=True)
                time.sleep(1.5)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default=None, help="start years, e.g. 2022-2026 (2022-23 .. 2026-27)")
    args = ap.parse_args()
    if args.seasons:
        a, _, b = args.seasons.partition("-")
        years = list(range(int(a), int(b or a) + 1))
    else:
        years = [current_start_year()]
    ingest(years)


if __name__ == "__main__":
    main()
