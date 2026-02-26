from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

log = logging.getLogger("nba_pipeline.parse_meta")


def _as_int(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.isdigit():
        return int(x)
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


def upsert_venues(conn) -> int:
    q = """
    SELECT season, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider='mysportsfeeds'
      AND endpoint='venues'
    ORDER BY fetched_at_utc DESC
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q)
        rows = cur.fetchall()

    if not rows:
        log.info("No venues payloads found.")
        return 0

    # Use latest payload per season
    latest_by_season: dict[str, dict] = {}
    for r in rows:
        season = r["season"]
        if not season or season in latest_by_season:
            continue
        latest_by_season[season] = r

    # 👇 DEDUPE BY venue_id ACROSS ALL SEASONS/PAYLOADS
    by_venue_id: dict[int, dict] = {}

    for season, r in latest_by_season.items():
        payload = _ensure_obj(r["payload"])
        fetched_at = r["fetched_at_utc"]
        last_updated = payload.get("lastUpdatedOn")

        for item in payload.get("venues", []) or []:
            v = item.get("venue") or {}
            venue_id = _as_int(v.get("id"))
            if venue_id is None:
                continue

            geo = v.get("geoCoordinates") or {}
            home_team = item.get("homeTeam") or {}

            by_venue_id[venue_id] = {
                "venue_id": venue_id,
                "name": v.get("name") or f"venue_{venue_id}",
                "city": v.get("city"),
                "country": v.get("country"),
                "latitude": _as_float(geo.get("latitude")),
                "longitude": _as_float(geo.get("longitude")),
                "has_roof": v.get("hasRoof"),
                "has_retractable_roof": v.get("hasRetractableRoof"),
                "playing_surface": v.get("playingSurface"),
                "capacities": json.dumps(v.get("capacitiesByEventType", []) or []),
                "home_team_abbr": (home_team.get("abbreviation") if isinstance(home_team, dict) else None),
                "source_last_updated_on": last_updated,
                "source_fetched_at_utc": fetched_at,
            }

    upsert_rows = list(by_venue_id.values())
    if not upsert_rows:
        return 0

    sql = """
    INSERT INTO raw.nba_venues (
      venue_id, name, city, country, latitude, longitude,
      has_roof, has_retractable_roof, playing_surface,
      capacities, home_team_abbr,
      source_last_updated_on, source_fetched_at_utc
    )
    VALUES %s
    ON CONFLICT (venue_id) DO UPDATE SET
      name = EXCLUDED.name,
      city = EXCLUDED.city,
      country = EXCLUDED.country,
      latitude = EXCLUDED.latitude,
      longitude = EXCLUDED.longitude,
      has_roof = EXCLUDED.has_roof,
      has_retractable_roof = EXCLUDED.has_retractable_roof,
      playing_surface = EXCLUDED.playing_surface,
      capacities = EXCLUDED.capacities,
      home_team_abbr = EXCLUDED.home_team_abbr,
      source_last_updated_on = EXCLUDED.source_last_updated_on,
      source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
      updated_at_utc = now()
    ;
    """

    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            upsert_rows,
            template="""
            (
              %(venue_id)s, %(name)s, %(city)s, %(country)s, %(latitude)s, %(longitude)s,
              %(has_roof)s, %(has_retractable_roof)s, %(playing_surface)s,
              %(capacities)s::jsonb, %(home_team_abbr)s,
              %(source_last_updated_on)s, %(source_fetched_at_utc)s
            )
            """,
            page_size=500,
        )

    return len(upsert_rows)


def upsert_injuries(conn) -> int:
    """
    Loads newest injuries payload and upserts raw.nba_injuries (latest snapshot per player_id).
    """
    q = """
    SELECT fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider='mysportsfeeds'
      AND endpoint='injuries'
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
    for p in payload.get("players", []) or []:
        pid = _as_int(p.get("id"))
        if pid is None:
            continue

        team = p.get("currentTeam") or {}
        inj = p.get("currentInjury") or {}

        rows.append(
            {
                "player_id": pid,
                "first_name": p.get("firstName"),
                "last_name": p.get("lastName"),
                "primary_position": p.get("primaryPosition"),
                "jersey_number": str(p.get("jerseyNumber")) if p.get("jerseyNumber") is not None else None,
                "team_abbr": (team.get("abbreviation") if isinstance(team, dict) else None),
                "roster_status": p.get("currentRosterStatus"),
                "injury_description": inj.get("description"),
                "playing_probability": inj.get("playingProbability"),
                "source_fetched_at_utc": fetched_at,
            }
        )

    if not rows:
        return 0

    sql = """
    INSERT INTO raw.nba_injuries (
      player_id, first_name, last_name, primary_position, jersey_number,
      team_abbr, roster_status, injury_description, playing_probability,
      source_fetched_at_utc
    )
    VALUES %s
    ON CONFLICT (player_id) DO UPDATE SET
      first_name = EXCLUDED.first_name,
      last_name = EXCLUDED.last_name,
      primary_position = EXCLUDED.primary_position,
      jersey_number = EXCLUDED.jersey_number,
      team_abbr = EXCLUDED.team_abbr,
      roster_status = EXCLUDED.roster_status,
      injury_description = EXCLUDED.injury_description,
      playing_probability = EXCLUDED.playing_probability,
      source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
      updated_at_utc = now()
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

    return len(rows)


def snapshot_injuries_history(conn) -> int:
    """
    Copies the current contents of raw.nba_injuries into raw.nba_injuries_history
    keyed by today's date (ET timezone).  Idempotent via ON CONFLICT … DO UPDATE.
    """
    today_et = datetime.now(ZoneInfo("America/New_York")).date()

    sql = """
    INSERT INTO raw.nba_injuries_history (
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
    FROM raw.nba_injuries
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


def insert_standings_snapshot(conn) -> int:
    """
    Inserts standings snapshots (keeps history) using latest payload per season.
    """
    q = """
    SELECT season, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider='mysportsfeeds'
      AND endpoint='standings'
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

        for t in payload.get("teams", []) or []:
            team = t.get("team") or {}
            abbr = team.get("abbreviation")
            if not abbr:
                continue

            stats = t.get("stats") or {}
            standings = (stats.get("standings") or {}) if isinstance(stats, dict) else {}

            overall = t.get("overallRank") or {}
            conf = t.get("conferenceRank") or {}
            div = t.get("divisionRank") or {}
            playoff = t.get("playoffRank") or {}

            insert_rows.append(
                {
                    "season": season,
                    "team_abbr": abbr,
                    "team_id": _as_int(team.get("id")),
                    "wins": _as_int(standings.get("wins")),
                    "losses": _as_int(standings.get("losses")),
                    "win_pct": _as_float(standings.get("winPct")),
                    "games_back": _as_float(standings.get("gamesBack")),
                    "overall_rank": _as_int(overall.get("rank")) if isinstance(overall, dict) else None,
                    "conference_name": conf.get("conferenceName") if isinstance(conf, dict) else None,
                    "conference_rank": _as_int(conf.get("rank")) if isinstance(conf, dict) else None,
                    "division_name": div.get("divisionName") if isinstance(div, dict) else None,
                    "division_rank": _as_int(div.get("rank")) if isinstance(div, dict) else None,
                    "playoff_rank": _as_int(playoff.get("rank")) if isinstance(playoff, dict) else None,
                    "stats": json.dumps(t, ensure_ascii=False),
                    "source_fetched_at_utc": fetched_at,
                }
            )

    if not insert_rows:
        return 0

    sql = """
    INSERT INTO raw.nba_standings (
      season, team_abbr, team_id,
      wins, losses, win_pct, games_back,
      overall_rank, conference_name, conference_rank,
      division_name, division_rank, playoff_rank,
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
              %(wins)s, %(losses)s, %(win_pct)s, %(games_back)s,
              %(overall_rank)s, %(conference_name)s, %(conference_rank)s,
              %(division_name)s, %(division_rank)s, %(playoff_rank)s,
              %(stats)s::jsonb,
              %(source_fetched_at_utc)s
            )
            """,
            page_size=500,
        )

    return len(insert_rows)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    dsn = "postgresql://josh:password@localhost:5432/nba"
    if not dsn:
        raise RuntimeError("Set PG_DSN (e.g. postgresql://user:pass@localhost:5432/nba)")

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        n1 = upsert_venues(conn)
        n2 = upsert_injuries(conn)
        n2h = snapshot_injuries_history(conn)
        n3 = insert_standings_snapshot(conn)
        conn.commit()
        log.info(
            "Done. venues=%d injuries=%d injuries_history=%d standings_rows=%d",
            n1, n2, n2h, n3,
        )
    except Exception:
        conn.rollback()
        log.exception("Failed; rolled back")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
