from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Any, Iterable, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

log = logging.getLogger("mlb_pipeline.parse_games")

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
    venue_id: Optional[int]
    status: str
    home_score: Optional[int]
    away_score: Optional[int]
    source_fetched_at_utc: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_msf_iso(ts: str) -> datetime:
    """
    MySportsFeeds timestamps look like: 2025-04-01T17:10:00.000Z
    fromisoformat() cannot parse trailing 'Z' directly in Python < 3.11.
    """
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _as_int(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    try:
        return int(float(x))
    except Exception:
        return None


def _extract_scores(game_obj: dict) -> tuple[Optional[int], Optional[int]]:
    """
    MLB MSF score object: game.score.awayScoreTotal / homeScoreTotal (integers).
    Handles both plain int and nested {"total": int} shapes defensively.
    """
    score = game_obj.get("score") or {}

    def unwrap(v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, dict):
            for key in ("total", "score", "runs"):
                val = v.get(key)
                if isinstance(val, (int, float)):
                    return int(val)
        return None

    home = unwrap(score.get("homeScoreTotal"))
    away = unwrap(score.get("awayScoreTotal"))
    return home, away


def _status_from_game_obj(game_obj: dict, start_ts_utc: Optional[datetime]) -> str:
    """
    Derive a simple status string:
      - 'final'      if both team scores are present
      - 'scheduled'  if start time is in the future (or unknown)
      - 'in_progress' otherwise
    """
    home_score, away_score = _extract_scores(game_obj)

    if home_score is not None and away_score is not None:
        return "final"

    if start_ts_utc is None:
        return "scheduled"

    now_utc = datetime.now(timezone.utc)
    if start_ts_utc > now_utc:
        return "scheduled"

    return "in_progress"


# ---------------------------------------------------------------------------
# Parse payload
# ---------------------------------------------------------------------------

def parse_games_payload(
    season: str, payload: dict, fetched_at_utc: datetime
) -> list[GameRow]:
    """
    Parse one games_by_date (or games_season) MSF payload into a list of GameRow.

    MSF MLB games payload shape:
      payload["games"] -> list of game objects
      game.schedule.startTime          -> ISO 8601 UTC timestamp
      game.schedule.awayTeam.abbreviation
      game.schedule.homeTeam.abbreviation
      game.schedule.venue.id           -> int (may be absent)
      game.score.awayScoreTotal        -> int or null
      game.score.homeScoreTotal        -> int or null

    Slug format: YYYYMMDD-AWAY-HOME (uppercase abbreviations, ET date).
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

        try:
            dt_utc = _parse_msf_iso(start_time)
        except Exception:
            log.warning("Could not parse startTime %r; skipping game.", start_time)
            continue

        dt_et = dt_utc.astimezone(ET)
        game_date_et = dt_et.date()
        game_slug = f"{dt_et.strftime('%Y%m%d')}-{away_abbr}-{home_abbr}"

        venue = sched.get("venue") or {}
        venue_id = _as_int(venue.get("id")) if isinstance(venue, dict) else None

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
                venue_id=venue_id,
                status=status,
                home_score=home_score,
                away_score=away_score,
                source_fetched_at_utc=fetched_at_utc,
            )
        )

    return rows


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def upsert_mlb_games(conn, rows: list[GameRow]) -> int:
    """
    Upsert a list of GameRow into raw.mlb_games, keyed on game_slug.
    Scores and status are updated if newer data arrives.
    """
    if not rows:
        return 0

    sql = """
    INSERT INTO raw.mlb_games (
      game_slug, season, game_date_et, start_ts_utc,
      home_team_abbr, away_team_abbr, venue_id,
      status, home_score, away_score,
      source_fetched_at_utc, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (game_slug) DO UPDATE SET
      season               = EXCLUDED.season,
      game_date_et         = EXCLUDED.game_date_et,
      start_ts_utc         = EXCLUDED.start_ts_utc,
      home_team_abbr       = EXCLUDED.home_team_abbr,
      away_team_abbr       = EXCLUDED.away_team_abbr,
      venue_id             = EXCLUDED.venue_id,
      status               = EXCLUDED.status,
      home_score           = EXCLUDED.home_score,
      away_score           = EXCLUDED.away_score,
      source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
      updated_at_utc       = now()
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
            "venue_id": gr.venue_id,
            "status": gr.status,
            "home_score": gr.home_score,
            "away_score": gr.away_score,
            "source_fetched_at_utc": gr.source_fetched_at_utc,
        }
        for gr in rows
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            values,
            template="""
            (
              %(game_slug)s, %(season)s, %(game_date_et)s, %(start_ts_utc)s,
              %(home_team_abbr)s, %(away_team_abbr)s, %(venue_id)s,
              %(status)s, %(home_score)s, %(away_score)s,
              %(source_fetched_at_utc)s, now()
            )
            """,
            page_size=1000,
        )

    return len(rows)


def load_games_by_date_payloads(conn) -> Iterable[tuple[str, datetime, dict]]:
    """
    Yield (season, fetched_at_utc, payload_dict) for every games_by_date row
    stored in raw.api_responses for the MLB pipeline.
    Ordered oldest-first so the most-recent fetch wins on upsert conflicts.
    """
    sql = """
    SELECT season, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider = 'mysportsfeeds'
      AND endpoint = 'games_by_date'
      AND (league = 'mlb' OR league IS NULL)
    ORDER BY fetched_at_utc ASC
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        for r in cur.fetchall():
            payload = r["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            yield r["season"], r["fetched_at_utc"], payload


def sync_scores_from_boxscores(conn) -> int:
    """
    UPDATE raw.mlb_games scores from raw.mlb_boxscore_games for completed games.

    raw.mlb_boxscore_games is authoritative for final scores; raw.mlb_games may
    lag because games_by_date payloads are fetched pre-game (null scores).

    Only syncs rows where played_status='COMPLETED' to avoid caching partial
    in-game scores.  Also re-syncs games already present but not yet marked
    'final' (handles partial writes from earlier in the same day).

    Sync query mirrors the NBA equivalent but uses MLB run-based score columns:
      raw.mlb_boxscore_games: home_runs, away_runs, played_status
    """
    sql = """
    UPDATE raw.mlb_games g
    SET
        home_score     = b.home_runs,
        away_score     = b.away_runs,
        status         = 'final',
        updated_at_utc = now()
    FROM raw.mlb_boxscore_games b
    WHERE g.game_slug = b.game_slug
      AND b.home_runs IS NOT NULL
      AND b.away_runs IS NOT NULL
      AND b.played_status = 'COMPLETED'
      AND (g.home_score IS NULL OR g.away_score IS NULL OR g.status != 'final')
    ;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.rowcount


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_raw_mlb_games(conn) -> int:
    """
    Reads raw.api_responses (games_by_date) and upserts into raw.mlb_games.
    Then syncs final scores from raw.mlb_boxscore_games.

    Returns total number of game rows processed (upserted, not necessarily changed).
    """
    total = 0
    by_date_total = 0

    for season, fetched_at_utc, payload in load_games_by_date_payloads(conn):
        parsed = parse_games_payload(
            season=season, payload=payload, fetched_at_utc=fetched_at_utc
        )
        if parsed:
            n = upsert_mlb_games(conn, parsed)
            by_date_total += n
            log.info(
                "games_by_date: season=%s fetched_at=%s games=%d",
                season, fetched_at_utc, len(parsed),
            )

    if by_date_total:
        log.info("Upserted %d game rows from games_by_date payloads", by_date_total)
    total += by_date_total

    # Sync final scores from boxscore table (authoritative for completed games)
    synced = sync_scores_from_boxscores(conn)
    if synced:
        log.info("Synced scores for %d games from mlb_boxscore_games", synced)

    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    dsn = "postgresql://josh:password@localhost:5432/nba"

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        n = build_raw_mlb_games(conn)
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
