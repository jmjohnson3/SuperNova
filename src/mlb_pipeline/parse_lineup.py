"""
mlb_pipeline.parse_lineup
==========================
Parse pre-game / post-game batting order lineups from MSF MLB lineup payloads
into raw.mlb_lineups.

Source priority per team entry: 'actual' > 'expected'.

Each batter appears twice in the payload:
  - once with position 'BO1'–'BO9'  → batting_order = 1–9
  - once with field position (RF, 3B, etc.)
Pitcher appears once with position 'P' (no batting slot).

Run:
  python -m mlb_pipeline.parse_lineup
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from mlb_pipeline.crawler import _norm_abbr

log = logging.getLogger("mlb_pipeline.parse_lineup")

DSN = "postgresql://josh:password@localhost:5432/nba"

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_MLB_LINEUPS = """
CREATE TABLE IF NOT EXISTS raw.mlb_lineups (
    game_slug      TEXT     NOT NULL,
    team_abbr      TEXT     NOT NULL,
    player_id      INTEGER  NOT NULL,
    player_name    TEXT,
    player_name_norm TEXT,
    batting_order  SMALLINT,           -- 1-9; NULL for pitcher/bench
    field_position TEXT,               -- RF, 3B, P, DH, etc.
    lineup_source  TEXT,               -- 'actual' or 'expected'
    updated_at     TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (game_slug, team_abbr, player_id)
);
ALTER TABLE raw.mlb_lineups
    ADD COLUMN IF NOT EXISTS player_name TEXT,
    ADD COLUMN IF NOT EXISTS player_name_norm TEXT;
CREATE INDEX IF NOT EXISTS idx_mlb_lineups_game_slug
    ON raw.mlb_lineups (game_slug, team_abbr);
CREATE INDEX IF NOT EXISTS idx_mlb_lineups_name_norm
    ON raw.mlb_lineups (game_slug, team_abbr, player_name_norm);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GAME_SLUG_RE = re.compile(r"games/([^/]+)/lineup")


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


def _normalize_name(name: Any) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _player_name(player: dict[str, Any]) -> str | None:
    full = (player.get("fullName") or "").strip()
    if full:
        return full
    first = (player.get("firstName") or "").strip()
    last = (player.get("lastName") or "").strip()
    full = " ".join(part for part in (first, last) if part)
    return full or None


def _ensure_obj(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    raise TypeError(f"Unexpected payload type: {type(payload)}")


# ---------------------------------------------------------------------------
# Parse a single lineup payload
# ---------------------------------------------------------------------------

def parse_lineup_payload(game_slug: str, payload: dict) -> list[dict]:
    """Parse one MSF MLB lineup payload into batter rows.

    Returns a list of dicts with keys:
      game_slug, team_abbr, player_id, player_name, player_name_norm,
      batting_order, field_position, lineup_source
    """
    rows: list[dict] = []
    team_lineups = payload.get("teamLineups") or []

    for team_entry in team_lineups:
        team = team_entry.get("team") or {}
        team_abbr = _norm_abbr(team.get("abbreviation") or "")
        if not team_abbr:
            continue

        # Prefer 'actual' lineup; fall back to 'expected'
        actual_container = team_entry.get("actual") or {}
        expected_container = team_entry.get("expected") or {}

        actual_positions = actual_container.get("lineupPositions") or []
        expected_positions = expected_container.get("lineupPositions") or []

        if actual_positions:
            lineup_positions = actual_positions
            source = "actual"
        elif expected_positions:
            lineup_positions = expected_positions
            source = "expected"
        else:
            continue

        # Merge BO slot + field position by player_id
        player_data: dict[int, dict] = {}
        for lp in lineup_positions:
            position = (lp.get("position") or "").strip()
            player = lp.get("player") or {}
            pid = _as_int(player.get("id"))
            if pid is None:
                continue  # TBD / empty slot
            player_name = _player_name(player)
            player_name_norm = _normalize_name(player_name) if player_name else None

            if pid not in player_data:
                player_data[pid] = {
                    "game_slug": game_slug,
                    "team_abbr": team_abbr,
                    "player_id": pid,
                    "player_name": player_name,
                    "player_name_norm": player_name_norm,
                    "batting_order": None,
                    "field_position": None,
                    "lineup_source": source,
                }
            elif player_name and not player_data[pid].get("player_name"):
                player_data[pid]["player_name"] = player_name
                player_data[pid]["player_name_norm"] = player_name_norm

            if position.startswith("BO") and len(position) > 2:
                try:
                    order = int(position[2:])
                    if 1 <= order <= 9:
                        player_data[pid]["batting_order"] = order
                except ValueError:
                    pass
            elif position:
                player_data[pid]["field_position"] = position

        rows.extend(player_data.values())

    return rows


# ---------------------------------------------------------------------------
# Ensure tables
# ---------------------------------------------------------------------------

def _ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_DDL_MLB_LINEUPS)
    conn.commit()


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_lineups(conn, rows: list[dict]) -> int:
    if not rows:
        return 0

    # Deduplicate within the batch on PK (game_slug, team_abbr, player_id).
    # Multiple payloads for the same game (e.g. crawled twice) or same player
    # appearing in overlapping API responses would otherwise trigger the
    # "ON CONFLICT DO UPDATE cannot affect row a second time" error.
    # Keep last occurrence — caller sorts payloads oldest-first so latest wins.
    dedup: dict[tuple, dict] = {}
    for r in rows:
        key = (r["game_slug"], r["team_abbr"], int(r["player_id"]))
        dedup[key] = r
    rows = list(dedup.values())

    sql = """
    INSERT INTO raw.mlb_lineups (
        game_slug, team_abbr, player_id, player_name, player_name_norm,
        batting_order, field_position, lineup_source, updated_at
    )
    VALUES %s
    ON CONFLICT (game_slug, team_abbr, player_id) DO UPDATE SET
        player_name    = COALESCE(EXCLUDED.player_name, raw.mlb_lineups.player_name),
        player_name_norm = COALESCE(EXCLUDED.player_name_norm, raw.mlb_lineups.player_name_norm),
        batting_order  = EXCLUDED.batting_order,
        field_position = EXCLUDED.field_position,
        lineup_source  = EXCLUDED.lineup_source,
        updated_at     = now()
    ;
    """
    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            rows,
            template="""(
                %(game_slug)s, %(team_abbr)s, %(player_id)s,
                %(player_name)s, %(player_name_norm)s,
                %(batting_order)s, %(field_position)s, %(lineup_source)s, now()
            )""",
            page_size=500,
        )
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    try:
        _ensure_tables(conn)

        # Fetch all MLB lineup payloads
        sql_fetch = """
        SELECT url, payload
        FROM raw.api_responses
        WHERE provider = 'mysportsfeeds'
          AND endpoint = 'lineup'
          AND url LIKE '%%/pull/mlb/%%'
        ORDER BY fetched_at_utc ASC
        ;
        """
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_fetch)
            lineup_rows = cur.fetchall()

        log.info("Found %d MLB lineup payloads to parse.", len(lineup_rows))

        all_rows: list[dict] = []
        for r in lineup_rows:
            url = r["url"] or ""
            m = _GAME_SLUG_RE.search(url)
            if not m:
                log.warning("Could not extract game_slug from URL: %s", url)
                continue
            game_slug = m.group(1)

            try:
                payload = _ensure_obj(r["payload"])
            except Exception:
                log.warning("Could not decode payload for game_slug=%s; skipping.", game_slug)
                continue

            try:
                rows = parse_lineup_payload(game_slug, payload)
                all_rows.extend(rows)
            except Exception:
                log.exception("Error parsing lineup for game_slug=%s; skipping.", game_slug)

        if all_rows:
            n = upsert_lineups(conn, all_rows)
            conn.commit()
            log.info("Upserted %d lineup rows into raw.mlb_lineups.", n)
        else:
            log.info("No lineup rows parsed.")

        log.info("Done.")
    except Exception:
        conn.rollback()
        log.exception("Failed; rolled back.")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
