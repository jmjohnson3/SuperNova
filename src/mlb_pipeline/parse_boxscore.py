"""
mlb_pipeline.parse_boxscore
============================
Parse MLB boxscore payloads from raw.api_responses into:
  - raw.mlb_boxscore_games       — game-level runs/hits/errors
  - raw.mlb_boxscore_team_stats  — team stats JSONB blob
  - raw.mlb_boxscore_player_stats — per-player batting/pitching JSONB

Source rows:
  provider='mysportsfeeds', endpoint='boxscore', game_slug IS NOT NULL

Run:
  python -m mlb_pipeline.parse_boxscore
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

log = logging.getLogger("mlb_pipeline.parse_boxscore")

DSN = "postgresql://josh:password@localhost:5432/nba"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_int(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    try:
        return int(str(x))
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        return float(str(x))
    except Exception:
        return None


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def _dump_jsonb(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _dedup_by_key(rows: list[dict], key_fn) -> list[dict]:
    out: dict = {}
    for r in rows:
        out[key_fn(r)] = r
    return list(out.values())


def _ensure_obj(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    raise TypeError(f"Unexpected payload type: {type(payload)}")


# ---------------------------------------------------------------------------
# DDL helpers — create tables if they don't exist yet
# ---------------------------------------------------------------------------

_DDL_BOXSCORE_GAMES = """
CREATE TABLE IF NOT EXISTS raw.mlb_boxscore_games (
    game_slug           TEXT PRIMARY KEY,
    season              TEXT,
    start_ts_utc        TIMESTAMPTZ,
    home_team_abbr      TEXT,
    away_team_abbr      TEXT,
    home_team_id        INTEGER,
    away_team_id        INTEGER,
    home_runs           INTEGER,
    away_runs           INTEGER,
    home_hits           INTEGER,
    away_hits           INTEGER,
    home_errors         INTEGER,
    away_errors         INTEGER,
    innings_played      INTEGER,
    played_status       TEXT,
    attendance          INTEGER,
    source_fetched_at_utc TIMESTAMPTZ,
    updated_at_utc      TIMESTAMPTZ DEFAULT now()
);
"""

_DDL_TEAM_STATS = """
CREATE TABLE IF NOT EXISTS raw.mlb_boxscore_team_stats (
    game_slug           TEXT NOT NULL,
    team_abbr           TEXT NOT NULL,
    season              TEXT,
    team_id             INTEGER,
    is_home             BOOLEAN,
    stats               JSONB,
    source_fetched_at_utc TIMESTAMPTZ,
    updated_at_utc      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (game_slug, team_abbr)
);
"""

_DDL_PLAYER_STATS = """
CREATE TABLE IF NOT EXISTS raw.mlb_boxscore_player_stats (
    game_slug           TEXT NOT NULL,
    player_id           INTEGER NOT NULL,
    season              TEXT,
    team_abbr           TEXT,
    team_id             INTEGER,
    is_home             BOOLEAN,
    first_name          TEXT,
    last_name           TEXT,
    primary_position    TEXT,
    batting_order       INTEGER,
    stats               JSONB,
    source_fetched_at_utc TIMESTAMPTZ,
    updated_at_utc      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (game_slug, player_id)
);
"""


def _ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_DDL_BOXSCORE_GAMES)
        cur.execute(_DDL_TEAM_STATS)
        cur.execute(_DDL_PLAYER_STATS)
    conn.commit()


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_boxscore_payload(game_slug: str, payload: dict) -> tuple[dict, list[dict], list[dict]]:
    """
    Parse a single MSF MLB boxscore payload.

    Returns:
        (game_row_dict, list_of_team_stat_rows, list_of_player_stat_rows)

    MSF MLB boxscore payload shape (simplified):
        {
          "game": {
            "id": ...,
            "startTime": "2025-04-01T18:10:00Z",
            "scheduleStatus": "NORMAL",
            "playedStatus": "COMPLETED",
            "awayTeam": {"id": ..., "abbreviation": "NYY"},
            "homeTeam": {"id": ..., "abbreviation": "BOS"},
            "attendance": 37000
          },
          "scoring": {
            "currentInning": 9,
            "teamTotal": {
              "away": {"runs": 3, "hits": 7, "errors": 1},
              "home": {"runs": 5, "hits": 10, "errors": 0}
            }
          },
          "stats": {
            "away": {
              "teamStats": [{"batting": {...}, "pitching": {...}, "fielding": {...}}],
              "players": [
                {
                  "player": {"id": 123, "firstName": "John", "lastName": "Doe",
                             "primaryPosition": "SP"},
                  "battingOrder": 0,
                  "batting": {...},
                  "pitching": {...}
                },
                ...
              ]
            },
            "home": { ... }
          }
        }
    """
    g = payload.get("game") or {}
    scoring = payload.get("scoring") or {}
    stats_block = payload.get("stats") or {}

    away_team = g.get("awayTeam") or {}
    home_team = g.get("homeTeam") or {}
    home_team_abbr = (home_team.get("abbreviation") or "").upper()
    away_team_abbr = (away_team.get("abbreviation") or "").upper()
    home_team_id = _as_int(home_team.get("id"))
    away_team_id = _as_int(away_team.get("id"))

    # Scoring totals
    team_total = scoring.get("teamTotal") or {}
    away_totals = team_total.get("away") or {}
    home_totals = team_total.get("home") or {}

    home_runs = _as_int(home_totals.get("runs"))
    away_runs = _as_int(away_totals.get("runs"))
    home_hits = _as_int(home_totals.get("hits"))
    away_hits = _as_int(away_totals.get("hits"))
    home_errors = _as_int(home_totals.get("errors"))
    away_errors = _as_int(away_totals.get("errors"))

    innings_played = _as_int(scoring.get("currentInning"))

    # Infer played_status
    if (
        home_runs is not None
        and away_runs is not None
        and innings_played is not None
        and innings_played >= 9
    ):
        played_status = "COMPLETED"
    elif home_runs is not None or away_runs is not None:
        played_status = "IN_PROGRESS"
    else:
        played_status = g.get("playedStatus") or "UNPLAYED"

    game_row = {
        "game_slug": game_slug,
        "season": None,  # caller may inject season from api_responses row
        "start_ts_utc": _parse_iso(g.get("startTime")),
        "home_team_abbr": home_team_abbr,
        "away_team_abbr": away_team_abbr,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_runs": home_runs,
        "away_runs": away_runs,
        "home_hits": home_hits,
        "away_hits": away_hits,
        "home_errors": home_errors,
        "away_errors": away_errors,
        "innings_played": innings_played,
        "played_status": played_status,
        "attendance": _as_int(g.get("attendance")),
        "source_fetched_at_utc": None,  # caller sets this
    }

    team_rows: list[dict] = []
    player_rows: list[dict] = []

    for side_key, is_home in (("away", False), ("home", True)):
        side = stats_block.get(side_key) or {}
        team_abbr = home_team_abbr if is_home else away_team_abbr
        team_id = home_team_id if is_home else away_team_id

        # teamStats is typically a list of one dict
        team_stats_list = side.get("teamStats") or []
        team_stats = team_stats_list[0] if team_stats_list else {}

        team_rows.append(
            {
                "game_slug": game_slug,
                "season": None,
                "team_abbr": team_abbr,
                "team_id": team_id,
                "is_home": is_home,
                "stats": _dump_jsonb(team_stats),
                "source_fetched_at_utc": None,
            }
        )

        for p in (side.get("players") or []):
            player = p.get("player") or {}
            pid = _as_int(player.get("id"))
            if pid is None:
                continue

            # Merge batting + pitching dicts into a single stats blob
            batting = p.get("batting") or {}
            pitching = p.get("pitching") or {}
            merged_stats: dict = {}
            if batting:
                merged_stats["batting"] = batting
            if pitching:
                merged_stats["pitching"] = pitching

            batting_order_raw = p.get("battingOrder")
            batting_order = _as_int(batting_order_raw)

            player_rows.append(
                {
                    "game_slug": game_slug,
                    "season": None,
                    "player_id": pid,
                    "team_abbr": team_abbr,
                    "team_id": team_id,
                    "is_home": is_home,
                    "first_name": player.get("firstName"),
                    "last_name": player.get("lastName"),
                    "primary_position": player.get("primaryPosition"),
                    "batting_order": batting_order,
                    "stats": _dump_jsonb(merged_stats),
                    "source_fetched_at_utc": None,
                }
            )

    return game_row, team_rows, player_rows


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------

def upsert_mlb_boxscore_games(conn, rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
    INSERT INTO raw.mlb_boxscore_games (
        game_slug, season,
        start_ts_utc,
        home_team_abbr, away_team_abbr,
        home_team_id, away_team_id,
        home_runs, away_runs,
        home_hits, away_hits,
        home_errors, away_errors,
        innings_played, played_status,
        attendance,
        source_fetched_at_utc, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (game_slug) DO UPDATE SET
        season               = EXCLUDED.season,
        start_ts_utc         = EXCLUDED.start_ts_utc,
        home_team_abbr       = EXCLUDED.home_team_abbr,
        away_team_abbr       = EXCLUDED.away_team_abbr,
        home_team_id         = EXCLUDED.home_team_id,
        away_team_id         = EXCLUDED.away_team_id,
        home_runs            = EXCLUDED.home_runs,
        away_runs            = EXCLUDED.away_runs,
        home_hits            = EXCLUDED.home_hits,
        away_hits            = EXCLUDED.away_hits,
        home_errors          = EXCLUDED.home_errors,
        away_errors          = EXCLUDED.away_errors,
        innings_played       = EXCLUDED.innings_played,
        played_status        = EXCLUDED.played_status,
        attendance           = EXCLUDED.attendance,
        source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
        updated_at_utc       = now()
    ;
    """
    execute_values(
        conn.cursor(),
        sql,
        rows,
        template="""(
            %(game_slug)s, %(season)s,
            %(start_ts_utc)s,
            %(home_team_abbr)s, %(away_team_abbr)s,
            %(home_team_id)s, %(away_team_id)s,
            %(home_runs)s, %(away_runs)s,
            %(home_hits)s, %(away_hits)s,
            %(home_errors)s, %(away_errors)s,
            %(innings_played)s, %(played_status)s,
            %(attendance)s,
            %(source_fetched_at_utc)s, now()
        )""",
        page_size=500,
    )


def upsert_mlb_boxscore_team_stats(conn, rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
    INSERT INTO raw.mlb_boxscore_team_stats (
        game_slug, team_abbr,
        season, team_id, is_home,
        stats,
        source_fetched_at_utc, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (game_slug, team_abbr) DO UPDATE SET
        season               = EXCLUDED.season,
        team_id              = EXCLUDED.team_id,
        is_home              = EXCLUDED.is_home,
        stats                = EXCLUDED.stats,
        source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
        updated_at_utc       = now()
    ;
    """
    execute_values(
        conn.cursor(),
        sql,
        rows,
        template="""(
            %(game_slug)s, %(team_abbr)s,
            %(season)s, %(team_id)s, %(is_home)s,
            %(stats)s::jsonb,
            %(source_fetched_at_utc)s, now()
        )""",
        page_size=500,
    )


def upsert_mlb_boxscore_player_stats(conn, rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
    INSERT INTO raw.mlb_boxscore_player_stats (
        game_slug, player_id,
        season, team_abbr, team_id, is_home,
        first_name, last_name, primary_position, batting_order,
        stats,
        source_fetched_at_utc, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (game_slug, player_id) DO UPDATE SET
        season               = EXCLUDED.season,
        team_abbr            = EXCLUDED.team_abbr,
        team_id              = EXCLUDED.team_id,
        is_home              = EXCLUDED.is_home,
        first_name           = EXCLUDED.first_name,
        last_name            = EXCLUDED.last_name,
        primary_position     = EXCLUDED.primary_position,
        batting_order        = EXCLUDED.batting_order,
        stats                = EXCLUDED.stats,
        source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
        updated_at_utc       = now()
    ;
    """
    execute_values(
        conn.cursor(),
        sql,
        rows,
        template="""(
            %(game_slug)s, %(player_id)s,
            %(season)s, %(team_abbr)s, %(team_id)s, %(is_home)s,
            %(first_name)s, %(last_name)s, %(primary_position)s, %(batting_order)s,
            %(stats)s::jsonb,
            %(source_fetched_at_utc)s, now()
        )""",
        page_size=1000,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_mlb_boxscores(conn, *, commit_every: int = 250) -> None:
    """
    Load all MLB boxscore payloads from raw.api_responses, parse them, and
    upsert into raw.mlb_boxscore_games / _team_stats / _player_stats.

    Uses DISTINCT ON (game_slug) to take the most-recently-fetched payload for
    each game so the tables always reflect the final box score.
    """
    _ensure_tables(conn)

    q = """
    SELECT DISTINCT ON (game_slug)
           season, game_slug, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider = 'mysportsfeeds'
      AND endpoint = 'boxscore'
      AND game_slug IS NOT NULL
    ORDER BY game_slug, fetched_at_utc DESC
    """

    games_batch: list[dict] = []
    teams_batch: list[dict] = []
    players_batch: list[dict] = []
    processed = 0

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q)
        for r in cur:
            game_slug = r["game_slug"]
            season = r.get("season")
            fetched_at = r["fetched_at_utc"]
            payload = _ensure_obj(r["payload"])

            try:
                game_row, team_rows, player_rows = parse_boxscore_payload(
                    game_slug=game_slug,
                    payload=payload,
                )
            except Exception:
                log.exception("Error parsing boxscore game_slug=%s — skipping", game_slug)
                continue

            # Inject season and fetched_at from the api_responses row
            game_row["season"] = season
            game_row["source_fetched_at_utc"] = fetched_at
            for row in team_rows:
                row["season"] = season
                row["source_fetched_at_utc"] = fetched_at
            for row in player_rows:
                row["season"] = season
                row["source_fetched_at_utc"] = fetched_at

            games_batch.append(game_row)
            teams_batch.extend(team_rows)
            players_batch.extend(player_rows)
            processed += 1

            # Deduplicate within the batch before upsert
            games_batch = _dedup_by_key(games_batch, lambda r: r["game_slug"])
            teams_batch = _dedup_by_key(teams_batch, lambda r: (r["game_slug"], r["team_abbr"]))
            players_batch = _dedup_by_key(players_batch, lambda r: (r["game_slug"], r["player_id"]))

            if processed % commit_every == 0:
                log.info(
                    "Upserting batch: processed=%d games=%d teams=%d players=%d",
                    processed, len(games_batch), len(teams_batch), len(players_batch),
                )
                upsert_mlb_boxscore_games(conn, games_batch)
                upsert_mlb_boxscore_team_stats(conn, teams_batch)
                upsert_mlb_boxscore_player_stats(conn, players_batch)
                conn.commit()
                games_batch.clear()
                teams_batch.clear()
                players_batch.clear()

    # Final flush
    if games_batch or teams_batch or players_batch:
        log.info(
            "Final upsert: processed=%d games=%d teams=%d players=%d",
            processed, len(games_batch), len(teams_batch), len(players_batch),
        )
        upsert_mlb_boxscore_games(conn, games_batch)
        upsert_mlb_boxscore_team_stats(conn, teams_batch)
        upsert_mlb_boxscore_player_stats(conn, players_batch)
        conn.commit()

    log.info("Done. MLB boxscores processed=%d", processed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    try:
        build_mlb_boxscores(conn)
    except Exception:
        conn.rollback()
        log.exception("Failed; rolled back")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
