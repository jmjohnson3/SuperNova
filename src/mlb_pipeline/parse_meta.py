from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

log = logging.getLogger("mlb_pipeline.parse_meta")


def _as_int(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.strip().isdigit():
        return int(x.strip())
    try:
        return int(float(x))
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (float, int)):
        return float(x)
    try:
        return float(x)
    except Exception:
        return None


def _ensure_obj(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    raise TypeError(f"Unexpected payload type: {type(payload)}")


# ---------------------------------------------------------------------------
# Venues
# ---------------------------------------------------------------------------

def upsert_venues(conn) -> int:
    """
    Reads venues payloads from raw.api_responses and upserts raw.mlb_venues.
    MLB MSF venues payload: payload["venues"] -> list of venue dicts with
    id, name, city, state, roofType, turfType, capacity.
    """
    q = """
    SELECT season, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider = 'mysportsfeeds'
      AND endpoint = 'venues'
      AND url LIKE '%/mlb/%'
    ORDER BY fetched_at_utc DESC
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q)
        rows = cur.fetchall()

    if not rows:
        log.info("No MLB venues payloads found.")
        return 0

    # Use the single most-recent payload (venues are not season-specific)
    latest_by_season: dict[str, dict] = {}
    for r in rows:
        season = r["season"] or "_all"
        if season not in latest_by_season:
            latest_by_season[season] = r

    by_venue_id: dict[int, dict] = {}

    for season, r in latest_by_season.items():
        payload = _ensure_obj(r["payload"])
        fetched_at = r["fetched_at_utc"]

        for item in payload.get("venues", []) or []:
            # MSF may wrap as {"venue": {...}} or directly as the venue dict
            v = item.get("venue") if isinstance(item, dict) and "venue" in item else item
            if not isinstance(v, dict):
                continue

            venue_id = _as_int(v.get("id"))
            if venue_id is None:
                continue

            home_team = item.get("homeTeam") or {} if isinstance(item, dict) else {}

            by_venue_id[venue_id] = {
                "venue_id": venue_id,
                "name": v.get("name") or f"venue_{venue_id}",
                "city": v.get("city"),
                "state": v.get("state"),
                "country": v.get("country"),
                "roof_type": v.get("roofType"),
                "turf_type": v.get("turfType"),
                "capacity": _as_int(v.get("capacity")),
                "home_team_abbr": (
                    home_team.get("abbreviation") if isinstance(home_team, dict) else None
                ),
                "source_fetched_at_utc": fetched_at,
            }

    upsert_rows = list(by_venue_id.values())
    if not upsert_rows:
        log.info("No venue records extracted from payloads.")
        return 0

    sql = """
    INSERT INTO raw.mlb_venues (
      venue_id, name, city, state, country,
      roof_type, turf_type, capacity,
      home_team_abbr,
      source_fetched_at_utc
    )
    VALUES %s
    ON CONFLICT (venue_id) DO UPDATE SET
      name               = EXCLUDED.name,
      city               = EXCLUDED.city,
      state              = EXCLUDED.state,
      country            = EXCLUDED.country,
      roof_type          = EXCLUDED.roof_type,
      turf_type          = EXCLUDED.turf_type,
      capacity           = EXCLUDED.capacity,
      home_team_abbr     = EXCLUDED.home_team_abbr,
      source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
      updated_at_utc     = now()
    ;
    """

    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            upsert_rows,
            template="""
            (
              %(venue_id)s, %(name)s, %(city)s, %(state)s, %(country)s,
              %(roof_type)s, %(turf_type)s, %(capacity)s,
              %(home_team_abbr)s,
              %(source_fetched_at_utc)s
            )
            """,
            page_size=500,
        )

    log.info("upsert_venues: %d rows", len(upsert_rows))
    return len(upsert_rows)


# ---------------------------------------------------------------------------
# Teams (extracted from standings payloads)
# ---------------------------------------------------------------------------

def upsert_teams(conn) -> int:
    """
    Extracts team dimension rows from standings payloads and upserts raw.mlb_teams.
    MSF standings payload: payload["standings"]["teams"] -> list with
    team.id, team.abbreviation, team.city, team.name,
    team.division.name, team.conference.name (AL/NL).
    """
    q = """
    SELECT season, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider = 'mysportsfeeds'
      AND endpoint = 'standings'
      AND url LIKE '%/mlb/%'
      AND season IS NOT NULL
    ORDER BY fetched_at_utc DESC
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q)
        rows = cur.fetchall()

    if not rows:
        log.info("No standings payloads found; skipping team upsert.")
        return 0

    # Collect unique teams across all seasons (latest season wins per team_id)
    by_team_id: dict[int, dict] = {}

    # Process oldest first so newest season overwrites
    for r in reversed(rows):
        payload = _ensure_obj(r["payload"])
        fetched_at = r["fetched_at_utc"]
        season = r["season"]

        standings_root = payload.get("standings") or payload
        team_entries = standings_root.get("teams", []) or []

        for t in team_entries:
            team = t.get("team") or {}
            team_id = _as_int(team.get("id"))
            abbr = team.get("abbreviation")
            if team_id is None or not abbr:
                continue

            division = team.get("division") or {}
            conference = team.get("conference") or {}

            by_team_id[team_id] = {
                "team_id": team_id,
                "abbreviation": abbr.upper(),
                "city": team.get("city"),
                "name": team.get("name"),
                "division_name": division.get("name") if isinstance(division, dict) else None,
                "league_name": conference.get("name") if isinstance(conference, dict) else None,
                "source_fetched_at_utc": fetched_at,
            }

    upsert_rows = list(by_team_id.values())
    if not upsert_rows:
        log.info("No team records extracted from standings payloads.")
        return 0

    sql = """
    INSERT INTO raw.mlb_teams (
      team_id, abbreviation, city, name,
      division_name, league_name,
      source_fetched_at_utc
    )
    VALUES %s
    ON CONFLICT (team_id) DO UPDATE SET
      abbreviation       = EXCLUDED.abbreviation,
      city               = EXCLUDED.city,
      name               = EXCLUDED.name,
      division_name      = EXCLUDED.division_name,
      league_name        = EXCLUDED.league_name,
      source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
      updated_at_utc     = now()
    ;
    """

    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            upsert_rows,
            template="""
            (
              %(team_id)s, %(abbreviation)s, %(city)s, %(name)s,
              %(division_name)s, %(league_name)s,
              %(source_fetched_at_utc)s
            )
            """,
            page_size=500,
        )

    log.info("upsert_teams: %d rows", len(upsert_rows))
    return len(upsert_rows)


# ---------------------------------------------------------------------------
# Standings
# ---------------------------------------------------------------------------

def insert_standings_snapshot(conn) -> int:
    """
    Inserts standings snapshots (keeps full history) using the latest payload per season.
    MSF payload: payload["standings"]["teams"] -> list with standings sub-object.
    Fields: wins, losses, winPct, runDifferential, divisionRank, conferenceRank.
    """
    q = """
    SELECT season, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider = 'mysportsfeeds'
      AND endpoint = 'standings'
      AND url LIKE '%/mlb/%'
      AND season IS NOT NULL
    ORDER BY fetched_at_utc DESC
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q)
        rows = cur.fetchall()

    if not rows:
        log.info("No standings payloads found.")
        return 0

    # Use latest payload per season
    latest_by_season: dict[str, dict] = {}
    for r in rows:
        season = r["season"]
        if season not in latest_by_season:
            latest_by_season[season] = r

    insert_rows = []
    for season, r in latest_by_season.items():
        payload = _ensure_obj(r["payload"])
        fetched_at = r["fetched_at_utc"]

        standings_root = payload.get("standings") or payload
        team_entries = standings_root.get("teams", []) or []

        for t in team_entries:
            team = t.get("team") or {}
            abbr = team.get("abbreviation")
            if not abbr:
                continue

            standings = t.get("standings") or {}
            division = team.get("division") or {}
            conference = team.get("conference") or {}

            insert_rows.append(
                {
                    "season": season,
                    "team_abbr": abbr.upper(),
                    "team_id": _as_int(team.get("id")),
                    "wins": _as_int(standings.get("wins")),
                    "losses": _as_int(standings.get("losses")),
                    "win_pct": _as_float(standings.get("winPct")),
                    "run_differential": _as_int(standings.get("runDifferential")),
                    "games_back": _as_float(standings.get("gamesBack")),
                    "division_rank": _as_int(standings.get("divisionRank")),
                    "conference_rank": _as_int(standings.get("conferenceRank")),
                    "conference_name": conference.get("name") if isinstance(conference, dict) else None,
                    "division_name": division.get("name") if isinstance(division, dict) else None,
                    "stats": json.dumps(t, ensure_ascii=False),
                    "source_fetched_at_utc": fetched_at,
                }
            )

    if not insert_rows:
        log.info("No standings records extracted.")
        return 0

    sql = """
    INSERT INTO raw.mlb_standings (
      season, team_abbr, team_id,
      wins, losses, win_pct, run_differential, games_back,
      division_rank, conference_rank,
      conference_name, division_name,
      stats,
      source_fetched_at_utc
    )
    VALUES %s
    ON CONFLICT DO NOTHING
    ;
    """

    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            insert_rows,
            template="""
            (
              %(season)s, %(team_abbr)s, %(team_id)s,
              %(wins)s, %(losses)s, %(win_pct)s, %(run_differential)s, %(games_back)s,
              %(division_rank)s, %(conference_rank)s,
              %(conference_name)s, %(division_name)s,
              %(stats)s::jsonb,
              %(source_fetched_at_utc)s
            )
            """,
            page_size=500,
        )

    log.info("insert_standings_snapshot: %d rows attempted", len(insert_rows))
    return len(insert_rows)


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------

def upsert_injuries(conn) -> int:
    """
    Loads the most-recent injuries payload and upserts raw.mlb_injuries
    (latest snapshot per player_id).
    MSF payload: payload["injuries"] -> list with player.id, player.firstName,
    player.lastName, player.currentTeam.abbreviation,
    injury.description, injury.playingProbability.
    """
    q = """
    SELECT fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider = 'mysportsfeeds'
      AND endpoint = 'injuries'
      AND url LIKE '%/mlb/%'
    ORDER BY fetched_at_utc DESC
    LIMIT 1
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q)
        r = cur.fetchone()

    if not r:
        log.info("No injuries payload found.")
        return 0

    payload = _ensure_obj(r["payload"])
    fetched_at = r["fetched_at_utc"]

    rows = []
    for entry in payload.get("injuries", []) or []:
        player = entry.get("player") or {}
        pid = _as_int(player.get("id"))
        if pid is None:
            continue

        team = player.get("currentTeam") or {}
        inj = entry.get("injury") or {}

        # playingProbability maps to a status string (e.g. "OUT", "DOUBTFUL", etc.)
        rows.append(
            {
                "player_id": pid,
                "first_name": player.get("firstName"),
                "last_name": player.get("lastName"),
                "primary_position": player.get("primaryPosition"),
                "jersey_number": (
                    str(player.get("jerseyNumber"))
                    if player.get("jerseyNumber") is not None
                    else None
                ),
                "team_abbr": (
                    team.get("abbreviation").upper()
                    if isinstance(team, dict) and team.get("abbreviation")
                    else None
                ),
                "roster_status": player.get("currentRosterStatus"),
                "injury_description": inj.get("description"),
                "playing_probability": inj.get("playingProbability"),
                "source_fetched_at_utc": fetched_at,
            }
        )

    if not rows:
        log.info("No injury records extracted from payload.")
        return 0

    sql = """
    INSERT INTO raw.mlb_injuries (
      player_id, first_name, last_name, primary_position, jersey_number,
      team_abbr, roster_status, injury_description, playing_probability,
      source_fetched_at_utc
    )
    VALUES %s
    ON CONFLICT (player_id) DO UPDATE SET
      first_name          = EXCLUDED.first_name,
      last_name           = EXCLUDED.last_name,
      primary_position    = EXCLUDED.primary_position,
      jersey_number       = EXCLUDED.jersey_number,
      team_abbr           = EXCLUDED.team_abbr,
      roster_status       = EXCLUDED.roster_status,
      injury_description  = EXCLUDED.injury_description,
      playing_probability = EXCLUDED.playing_probability,
      source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
      updated_at_utc      = now()
    ;
    """

    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            rows,
            template="""
            (
              %(player_id)s, %(first_name)s, %(last_name)s, %(primary_position)s, %(jersey_number)s,
              %(team_abbr)s, %(roster_status)s, %(injury_description)s, %(playing_probability)s,
              %(source_fetched_at_utc)s
            )
            """,
            page_size=1000,
        )

    log.info("upsert_injuries: %d rows", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# Snapshot injuries history
# ---------------------------------------------------------------------------

def snapshot_injuries_history(conn) -> int:
    """
    Copies the current contents of raw.mlb_injuries into raw.mlb_injuries_history
    keyed by today's date (ET timezone).  Idempotent via ON CONFLICT … DO UPDATE.
    """
    today_et = datetime.now(ZoneInfo("America/New_York")).date()

    sql = """
    INSERT INTO raw.mlb_injuries_history (
        as_of_date,
        player_id,
        first_name,
        last_name,
        primary_position,
        jersey_number,
        team_abbr,
        roster_status,
        injury_description,
        playing_probability,
        source_fetched_at_utc
    )
    SELECT
        %s AS as_of_date,
        player_id,
        first_name,
        last_name,
        primary_position,
        jersey_number,
        team_abbr,
        roster_status,
        injury_description,
        playing_probability,
        source_fetched_at_utc
    FROM raw.mlb_injuries
    ON CONFLICT (as_of_date, player_id) DO UPDATE SET
        first_name          = EXCLUDED.first_name,
        last_name           = EXCLUDED.last_name,
        primary_position    = EXCLUDED.primary_position,
        jersey_number       = EXCLUDED.jersey_number,
        team_abbr           = EXCLUDED.team_abbr,
        roster_status       = EXCLUDED.roster_status,
        injury_description  = EXCLUDED.injury_description,
        playing_probability = EXCLUDED.playing_probability,
        source_fetched_at_utc = EXCLUDED.source_fetched_at_utc
    ;
    """

    with conn.cursor() as cur:
        cur.execute(sql, (today_et,))
        row_count = cur.rowcount

    log.info("snapshot_injuries_history: as_of_date=%s  rows=%d", today_et, row_count)
    return row_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    dsn = "postgresql://josh:password@localhost:5432/nba"

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        n_venues = upsert_venues(conn)
        n_teams = upsert_teams(conn)
        n_standings = insert_standings_snapshot(conn)
        n_injuries = upsert_injuries(conn)
        n_injuries_hist = snapshot_injuries_history(conn)
        conn.commit()
        log.info(
            "Done. venues=%d teams=%d standings_rows=%d injuries=%d injuries_history=%d",
            n_venues, n_teams, n_standings, n_injuries, n_injuries_hist,
        )
    except Exception:
        conn.rollback()
        log.exception("Failed; rolled back")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
