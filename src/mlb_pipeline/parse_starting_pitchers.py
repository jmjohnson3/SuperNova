"""
mlb_pipeline.parse_starting_pitchers
=====================================
Parse starting pitcher info from lineup and boxscore payloads into:
  - raw.mlb_starting_pitchers  — one row per (game_slug, team_abbr)
  - Updates raw.mlb_games.home_sp_id / away_sp_id after processing

Data sources:
  1. lineup endpoint payloads (provider='mysportsfeeds', endpoint='lineup')
     — announced SPs before the game; source='announced'
  2. raw.mlb_boxscore_player_stats
     — actual starter (pitcher with gamesStarted=1 or most IP); source='actual'

Run:
  python -m mlb_pipeline.parse_starting_pitchers
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

log = logging.getLogger("mlb_pipeline.parse_starting_pitchers")

DSN = "postgresql://josh:password@localhost:5432/nba"

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_STARTING_PITCHERS = """
CREATE TABLE IF NOT EXISTS raw.mlb_starting_pitchers (
    game_slug    TEXT        NOT NULL,
    team_abbr    TEXT        NOT NULL,
    player_id    INTEGER,
    player_name  TEXT,
    source       TEXT        NOT NULL DEFAULT 'announced',
    updated_at_utc TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (game_slug, team_abbr)
);
CREATE INDEX IF NOT EXISTS idx_mlb_sp_game_slug
    ON raw.mlb_starting_pitchers (game_slug);
"""

_DDL_MLB_GAMES_SP_COLS = """
ALTER TABLE IF EXISTS raw.mlb_games
    ADD COLUMN IF NOT EXISTS home_sp_id INTEGER,
    ADD COLUMN IF NOT EXISTS away_sp_id INTEGER;
"""

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
        return int(float(str(x)))
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        return float(str(x))
    except Exception:
        return None


def _ensure_obj(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    raise TypeError(f"Unexpected payload type: {type(payload)}")


# ---------------------------------------------------------------------------
# Parse lineup payload
# ---------------------------------------------------------------------------

def parse_lineup_payload(
    game_slug: str, payload: dict, source: str = "announced"
) -> list[dict]:
    """Parse a single MSF MLB lineup payload.

    MSF MLB lineup payload shape:
      payload["teamLineups"] -> list of up to 2 items (home and away)
      Each item:
        teamLineup.team.abbreviation  -> str
        teamLineup.actual or teamLineup.expected
          -> lineup.lineupPositions -> list of player positions
          -> find player where lineupPosition.position == "SP"
             or lineupPosition.lineupSlot == "SP"
          -> player: lineupPosition.player.id, .firstName, .lastName

    Returns a list of dicts with keys:
      game_slug, team_abbr, player_id, player_name, source
    """
    rows: list[dict] = []
    team_lineups = (payload.get("teamLineups") or [])

    for team_entry in team_lineups:
        team = team_entry.get("team") or {}
        team_abbr = (team.get("abbreviation") or "").upper()
        if not team_abbr:
            continue

        # Prefer 'actual' lineup over 'expected'
        lineup_container = (
            team_entry.get("actual")
            or team_entry.get("expected")
            or {}
        )
        lineup_positions = (lineup_container.get("lineupPositions") or [])

        sp_player_id: Optional[int] = None
        sp_player_name: Optional[str] = None

        for lp in lineup_positions:
            position = (lp.get("position") or "").upper()
            lineup_slot = (lp.get("lineupSlot") or "").upper()
            if position == "SP" or lineup_slot == "SP":
                player = lp.get("player") or {}
                pid = _as_int(player.get("id"))
                first = player.get("firstName") or ""
                last = player.get("lastName") or ""
                full_name = f"{first} {last}".strip() or None
                sp_player_id = pid
                sp_player_name = full_name
                break  # only one SP per team lineup

        rows.append(
            {
                "game_slug": game_slug,
                "team_abbr": team_abbr,
                "player_id": sp_player_id,
                "player_name": sp_player_name,
                "source": source,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Extract actual starters from boxscores
# ---------------------------------------------------------------------------

def extract_actual_starters_from_boxscores(conn) -> list[dict]:
    """Extract actual starting pitchers from raw.mlb_boxscore_player_stats.

    Strategy (in order of preference per game/team):
      1. Player where (stats->'pitching'->>'gamesStarted')::int = 1
      2. Pitcher with highest (stats->'pitching'->>'inningsPitched')::float
         among players who have any pitching stats

    Returns a list of dicts:
      game_slug, team_abbr, player_id, player_name, source='actual'
    """
    sql = """
    WITH pitchers AS (
        SELECT
            p.game_slug,
            p.team_abbr,
            p.player_id,
            TRIM(COALESCE(p.first_name, '') || ' ' || COALESCE(p.last_name, '')) AS player_name,
            -- gamesStarted flag (1 = game starter)
            CASE
                WHEN (p.stats->'pitching'->>'gamesStarted') IS NOT NULL
                THEN (p.stats->'pitching'->>'gamesStarted')::numeric::int
                ELSE 0
            END AS games_started,
            -- innings pitched (used as tiebreaker)
            CASE
                WHEN (p.stats->'pitching'->>'inningsPitched') IS NOT NULL
                THEN (p.stats->'pitching'->>'inningsPitched')::numeric
                ELSE 0
            END AS innings_pitched
        FROM raw.mlb_boxscore_player_stats p
        WHERE p.stats ? 'pitching'
          AND p.stats->'pitching' IS NOT NULL
          AND p.stats->'pitching' <> 'null'::jsonb
    ),
    ranked AS (
        SELECT
            game_slug,
            team_abbr,
            player_id,
            player_name,
            games_started,
            innings_pitched,
            -- prefer games_started=1; among ties, highest IP wins
            ROW_NUMBER() OVER (
                PARTITION BY game_slug, team_abbr
                ORDER BY games_started DESC, innings_pitched DESC, player_id
            ) AS rn
        FROM pitchers
        WHERE innings_pitched > 0 OR games_started = 1
    )
    SELECT
        game_slug,
        team_abbr,
        player_id,
        player_name
    FROM ranked
    WHERE rn = 1
    ORDER BY game_slug, team_abbr
    ;
    """
    rows: list[dict] = []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        for r in cur.fetchall():
            rows.append(
                {
                    "game_slug": r["game_slug"],
                    "team_abbr": r["team_abbr"],
                    "player_id": _as_int(r["player_id"]),
                    "player_name": r["player_name"] or None,
                    "source": "actual",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_starting_pitchers(conn, rows: list[dict]) -> int:
    """Upsert rows into raw.mlb_starting_pitchers.

    On conflict (game_slug, team_abbr), always update to the latest values
    so that 'actual' source overwrites 'announced' as boxscores arrive.
    """
    if not rows:
        return 0

    sql = """
    INSERT INTO raw.mlb_starting_pitchers (
        game_slug, team_abbr, player_id, player_name, source, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (game_slug, team_abbr) DO UPDATE SET
        player_id      = EXCLUDED.player_id,
        player_name    = EXCLUDED.player_name,
        source         = EXCLUDED.source,
        updated_at_utc = now()
    ;
    """
    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            rows,
            template="""(
                %(game_slug)s, %(team_abbr)s,
                %(player_id)s, %(player_name)s,
                %(source)s, now()
            )""",
            page_size=500,
        )
    return len(rows)


# ---------------------------------------------------------------------------
# Update raw.mlb_games SP IDs
# ---------------------------------------------------------------------------

def update_game_sp_ids(conn) -> int:
    """UPDATE raw.mlb_games.home_sp_id / away_sp_id from raw.mlb_starting_pitchers.

    Prefers source='actual' over 'announced' when both exist.
    Uses DISTINCT ON to select the best (most-authoritative) row per game/team.
    """
    sql = """
    WITH best_sp AS (
        SELECT DISTINCT ON (game_slug, team_abbr)
            game_slug,
            team_abbr,
            player_id
        FROM raw.mlb_starting_pitchers
        WHERE player_id IS NOT NULL
        ORDER BY
            game_slug,
            team_abbr,
            -- actual beats announced; within same source, highest player_id as tiebreak
            CASE WHEN source = 'actual' THEN 0 ELSE 1 END,
            player_id
    )
    UPDATE raw.mlb_games g
    SET
        home_sp_id = home_sp.player_id,
        away_sp_id = away_sp.player_id,
        updated_at_utc = now()
    FROM best_sp home_sp
    JOIN best_sp away_sp
        ON away_sp.game_slug = home_sp.game_slug
        AND away_sp.team_abbr = g.away_team_abbr
    WHERE home_sp.game_slug = g.game_slug
      AND home_sp.team_abbr = g.home_team_abbr
      AND (
          g.home_sp_id IS DISTINCT FROM home_sp.player_id
          OR g.away_sp_id IS DISTINCT FROM away_sp.player_id
      )
    ;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.rowcount


# ---------------------------------------------------------------------------
# Ensure tables
# ---------------------------------------------------------------------------

def _ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_DDL_STARTING_PITCHERS)
        cur.execute(_DDL_MLB_GAMES_SP_COLS)
    conn.commit()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_mlb_starting_pitchers(conn) -> None:
    """Orchestrate all starting pitcher parsing steps.

    1. Load + parse lineup payloads (announced SPs)
    2. Extract actual SPs from mlb_boxscore_player_stats
    3. Upsert both sources into raw.mlb_starting_pitchers
    4. Update raw.mlb_games.home_sp_id / away_sp_id
    """
    _ensure_tables(conn)

    # ---- Step 1: Parse announced SPs from lineup payloads ----
    sql_lineups = """
    SELECT game_slug, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider = 'mysportsfeeds'
      AND endpoint = 'lineup'
      AND game_slug IS NOT NULL
    ORDER BY fetched_at_utc ASC
    ;
    """
    announced_rows: list[dict] = []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql_lineups)
        lineup_snaps = cur.fetchall()

    log.info("Found %d lineup payloads to parse.", len(lineup_snaps))

    for r in lineup_snaps:
        game_slug = r["game_slug"]
        try:
            payload = _ensure_obj(r["payload"])
        except Exception:
            log.warning("Could not decode lineup payload for game_slug=%s; skipping.", game_slug)
            continue

        try:
            rows = parse_lineup_payload(game_slug, payload, source="announced")
            announced_rows.extend(rows)
        except Exception:
            log.exception("Error parsing lineup payload for game_slug=%s; skipping.", game_slug)

    # Deduplicate by (game_slug, team_abbr) — keep last (most recent fetch)
    announced_dedup: dict[tuple, dict] = {}
    for row in announced_rows:
        announced_dedup[(row["game_slug"], row["team_abbr"])] = row
    announced_rows = list(announced_dedup.values())

    if announced_rows:
        n = upsert_starting_pitchers(conn, announced_rows)
        conn.commit()
        log.info("Upserted %d announced SP rows into raw.mlb_starting_pitchers.", n)
    else:
        log.info("No announced SP rows found from lineup payloads.")

    # ---- Step 2: Extract actual SPs from boxscore player stats ----
    try:
        actual_rows = extract_actual_starters_from_boxscores(conn)
    except Exception:
        log.exception("Failed to extract actual starters from boxscores; skipping.")
        actual_rows = []

    if actual_rows:
        n = upsert_starting_pitchers(conn, actual_rows)
        conn.commit()
        log.info("Upserted %d actual SP rows into raw.mlb_starting_pitchers.", n)
    else:
        log.info("No actual SP rows extracted from boxscores.")

    # ---- Step 3: Update raw.mlb_games SP ID columns ----
    try:
        updated = update_game_sp_ids(conn)
        conn.commit()
        log.info("Updated home_sp_id/away_sp_id for %d games in raw.mlb_games.", updated)
    except Exception:
        conn.rollback()
        log.exception("Failed to update game SP IDs; rolled back.")


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
        build_mlb_starting_pitchers(conn)
        log.info("Done.")
    except Exception:
        conn.rollback()
        log.exception("Failed; rolled back.")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
