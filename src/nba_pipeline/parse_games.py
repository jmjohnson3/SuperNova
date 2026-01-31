import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Optional, Any, Iterable

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

log = logging.getLogger("nba_pipeline.parse_games")

# Requires `pip install tzdata` on Windows to support IANA tz names
try:
    ET = ZoneInfo("America/New_York")
except ZoneInfoNotFoundError as e:
    raise RuntimeError(
        "Missing IANA timezone data. On Windows, run: pip install tzdata"
    ) from e


@dataclass(frozen=True)
class GameRow:
    game_slug: str
    season: str
    game_date_et: date
    start_ts_utc: Optional[datetime]
    home_team_abbr: str
    away_team_abbr: str
    status: str
    home_score: Optional[int]
    away_score: Optional[int]
    source_fetched_at_utc: datetime


def _parse_msf_iso(ts: str) -> datetime:
    """
    MySportsFeeds times look like:
      2025-10-22T23:30:00.000Z
    """
    # fromisoformat can't parse trailing 'Z' directly
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _status_from_game_obj(game_obj: dict, start_ts_utc: Optional[datetime]) -> str:
    """
    Deterministic status:
      - final if both team totals exist (common for completed games in season index)
      - scheduled if start time is in the future
      - in_progress otherwise
    """
    home_score, away_score = _extract_scores(game_obj)

    # If both totals exist, treat as final (covers historical seasons cleanly)
    if home_score is not None and away_score is not None:
        return "final"

    if start_ts_utc is None:
        return "scheduled"

    now_utc = datetime.now(timezone.utc)
    if start_ts_utc > now_utc:
        return "scheduled"

    return "in_progress"



def _extract_scores(game_obj: dict) -> tuple[Optional[int], Optional[int]]:
    score = game_obj.get("score") or {}
    # Many MSF payloads use score.homeScoreTotal / awayScoreTotal with nested dicts
    # We'll handle both common shapes:
    # 1) {"homeScoreTotal": 110, "awayScoreTotal": 99}
    # 2) {"homeScoreTotal": {"total": 110}, "awayScoreTotal": {"total": 99}}
    def unwrap(v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, dict):
            # MySportsFeeds often uses {"total": <int>}
            for key in ("total", "score", "points"):
                val = v.get(key)
                if isinstance(val, int):
                    return val
        return None

    home = unwrap(score.get("homeScoreTotal"))
    away = unwrap(score.get("awayScoreTotal"))
    return home, away


def _build_game_slug(game_obj: dict) -> GameRow:
    raise NotImplementedError("Use parse_games_payload()")  # kept for clarity


def parse_games_payload(season: str, payload: dict, fetched_at_utc: datetime) -> list[GameRow]:
    """
    Parse one games_season payload into normalized rows.
    """
    rows: list[GameRow] = []
    games = payload.get("games") or []
    for g in games:
        sched = g.get("schedule") or {}

        start_time = sched.get("startTime")
        away_abbr = ((sched.get("awayTeam") or {}).get("abbreviation") or "").upper()
        home_abbr = ((sched.get("homeTeam") or {}).get("abbreviation") or "").upper()

        if not (start_time and away_abbr and home_abbr):
            continue

        dt_utc = _parse_msf_iso(start_time)
        dt_et = dt_utc.astimezone(ET)
        game_date_et = dt_et.date()
        game_slug = f"{dt_et.strftime('%Y%m%d')}-{away_abbr}-{home_abbr}"

        home_score, away_score = _extract_scores(g)
        status = _status_from_game_obj(g, dt_utc)

        rows.append(
            GameRow(
                game_slug=game_slug,
                season=season,
                game_date_et=game_date_et,
                start_ts_utc=dt_utc,
                home_team_abbr=home_abbr,
                away_team_abbr=away_abbr,
                status=status,
                home_score=home_score,
                away_score=away_score,
                source_fetched_at_utc=fetched_at_utc,
            )
        )

    return rows


def load_games_season_payloads(conn) -> Iterable[tuple[str, datetime, dict]]:
    """
    Yield (season, fetched_at_utc, payload_dict) for each games_season row in raw.api_responses.
    """
    sql = """
    SELECT season, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider = 'mysportsfeeds'
      AND endpoint = 'games_season'
    ORDER BY fetched_at_utc DESC
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        for r in cur.fetchall():
            # psycopg2 returns jsonb as python objects if configured; but safer to handle both
            payload = r["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            yield r["season"], r["fetched_at_utc"], payload


def upsert_nba_games(conn, game_rows: list[GameRow]) -> int:
    """
    Upsert by game_slug. We update scores/status as new data comes in.
    """
    if not game_rows:
        return 0

    sql = """
    INSERT INTO raw.nba_games (
      game_slug, season, game_date_et, start_ts_utc,
      home_team_abbr, away_team_abbr,
      status, home_score, away_score,
      source_fetched_at_utc, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (game_slug) DO UPDATE SET
      season = EXCLUDED.season,
      game_date_et = EXCLUDED.game_date_et,
      start_ts_utc = EXCLUDED.start_ts_utc,
      home_team_abbr = EXCLUDED.home_team_abbr,
      away_team_abbr = EXCLUDED.away_team_abbr,
      status = EXCLUDED.status,
      home_score = EXCLUDED.home_score,
      away_score = EXCLUDED.away_score,
      source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
      updated_at_utc = now()
    ;
    """

    values = [
        {
            "game_slug": gr.game_slug,
            "season": gr.season,
            "game_date_et": gr.game_date_et,
            "start_ts_utc": gr.start_ts_utc,
            "home_team_abbr": gr.home_team_abbr,
            "away_team_abbr": gr.away_team_abbr,
            "status": gr.status,
            "home_score": gr.home_score,
            "away_score": gr.away_score,
            "source_fetched_at_utc": gr.source_fetched_at_utc,
        }
        for gr in game_rows
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            values,
            template="""
            (
              %(game_slug)s, %(season)s, %(game_date_et)s, %(start_ts_utc)s,
              %(home_team_abbr)s, %(away_team_abbr)s,
              %(status)s, %(home_score)s, %(away_score)s,
              %(source_fetched_at_utc)s, now()
            )
            """,
            page_size=1000,
        )

    return len(game_rows)


def build_raw_nba_games(conn) -> int:
    """
    Reads raw.api_responses (games_season) and upserts into raw.nba_games.
    Returns number of rows processed (not necessarily changed).
    """
    total = 0
    for season, fetched_at_utc, payload in load_games_season_payloads(conn):
        parsed = parse_games_payload(season=season, payload=payload, fetched_at_utc=fetched_at_utc)
        total += upsert_nba_games(conn, parsed)
        log.info("Parsed season=%s fetched_at=%s games=%d", season, fetched_at_utc, len(parsed))
    return total


def main() -> None:
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    dsn = "postgresql://josh:password@localhost:5432/nba"
    if not dsn:
        raise RuntimeError("Set PG_DSN, e.g. postgresql://josh:password@localhost:5432/nba")

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        n = build_raw_nba_games(conn)
        conn.commit()
        log.info("Done. Total processed=%d", n)
    except Exception:
        conn.rollback()
        log.exception("Failed; rolled back")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
