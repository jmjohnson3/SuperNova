"""
mlb_pipeline.crawler_weather
==============================
Fetches game-time weather from Open-Meteo (free, no API key required).

For completed games (game_date < today - 5 days): uses archive-api.open-meteo.com
For today / recent / future games:                uses api.open-meteo.com

Data stored in raw.mlb_weather (DDL: sql/MLB010_mlb_weather_ddl.sql).

Usage (standalone):
  python -m mlb_pipeline.crawler_weather
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import execute_values

log = logging.getLogger("mlb_pipeline.crawler_weather")

ET = ZoneInfo("America/New_York")
from mlb_pipeline.db import PG_DSN as DSN
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Games older than this threshold use the archive API; newer use forecast
ARCHIVE_CUTOFF_DAYS = 5

# Hourly weather variables to request
HOURLY_VARS = "temperature_2m,windspeed_10m,winddirection_10m,precipitation_probability"

# Seconds between API calls (Open-Meteo is generous but be polite)
REQUEST_SLEEP = 0.3


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SuperNovaBets/1.0 (sports analytics)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Open-Meteo fetch
# ---------------------------------------------------------------------------

def _fetch_hourly(
    latitude: float,
    longitude: float,
    date_str: str,
    use_archive: bool,
) -> dict:
    """
    Fetch hourly weather for a single calendar date from Open-Meteo.

    Returns the raw API response dict (keys: 'hourly', 'hourly_units', ...).
    """
    base = ARCHIVE_URL if use_archive else FORECAST_URL
    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": HOURLY_VARS,
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "timezone": "America/New_York",
    }

    url = base + "?" + urllib.parse.urlencode(params)
    return _get_json(url)


def _extract_game_weather(data: dict, start_ts_utc: datetime) -> dict | None:
    """
    Find the hourly slot in the Open-Meteo response closest to game start time.

    Open-Meteo returns times in the requested timezone ("America/New_York") as
    naive ISO strings like "2024-04-01T13:00".

    Returns a dict with: temperature_f, wind_speed_mph, wind_direction_deg,
    precip_prob_pct; or None if no usable data.
    """
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return None

    temps   = hourly.get("temperature_2m", [])
    winds   = hourly.get("windspeed_10m", [])
    dirs    = hourly.get("winddirection_10m", [])
    precips = hourly.get("precipitation_probability", [])

    # Convert game start time to ET for matching
    start_et = start_ts_utc.astimezone(ET)

    best_idx = 0
    best_diff: float | None = None
    for i, t_str in enumerate(times):
        try:
            t_naive = datetime.fromisoformat(t_str)
            t_et = t_naive.replace(tzinfo=ET)
            diff = abs((t_et - start_et).total_seconds())
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = i
        except Exception:
            continue

    def _safe(lst: list, idx: int) -> float | None:
        try:
            v = lst[idx]
            return None if v is None else float(v)
        except (IndexError, TypeError, ValueError):
            return None

    return {
        "temperature_f":      _safe(temps,   best_idx),
        "wind_speed_mph":     _safe(winds,   best_idx),
        "wind_direction_deg": _safe(dirs,    best_idx),
        "precip_prob_pct":    _safe(precips, best_idx),
    }


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------

def _upsert_weather(conn, rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
    INSERT INTO raw.mlb_weather (
        game_slug, temperature_f, wind_speed_mph, wind_direction_deg,
        precip_prob_pct, fetched_at_utc, updated_at_utc
    ) VALUES %s
    ON CONFLICT (game_slug) DO UPDATE SET
        temperature_f      = EXCLUDED.temperature_f,
        wind_speed_mph     = EXCLUDED.wind_speed_mph,
        wind_direction_deg = EXCLUDED.wind_direction_deg,
        precip_prob_pct    = EXCLUDED.precip_prob_pct,
        updated_at_utc     = now()
    ;
    """
    with conn.cursor() as cur:
        execute_values(
            cur, sql, rows,
            template="""(
                %(game_slug)s, %(temperature_f)s, %(wind_speed_mph)s,
                %(wind_direction_deg)s, %(precip_prob_pct)s,
                now(), now()
            )""",
            page_size=100,
        )


# ---------------------------------------------------------------------------
# Main fetch functions
# ---------------------------------------------------------------------------

def fetch_weather_for_games(conn, game_slugs: list[str]) -> int:
    """
    Fetch and upsert weather for the given list of game_slugs.

    Looks up venue lat/lon and start_ts_utc from the database.
    Returns count of rows upserted.
    """
    if not game_slugs:
        return 0

    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                g.game_slug,
                g.start_ts_utc,
                g.game_date_et,
                v.latitude::FLOAT,
                v.longitude::FLOAT,
                v.roof_type
            FROM raw.mlb_games g
            LEFT JOIN raw.mlb_venues v ON v.venue_id = g.venue_id
            WHERE g.game_slug = ANY(%s)
              AND g.start_ts_utc IS NOT NULL
              AND v.latitude IS NOT NULL
              AND v.longitude IS NOT NULL
            ORDER BY g.game_date_et
        """, (game_slugs,))
        rows = cur.fetchall()

    if not rows:
        log.info("No games with venue lat/lon found for weather fetch")
        return 0

    cutoff = date.today() - timedelta(days=ARCHIVE_CUTOFF_DAYS)
    upsert_rows: list[dict] = []
    errors = 0

    for game_slug, start_ts_utc, game_date_et, lat, lon, roof_type in rows:
        use_archive = game_date_et < cutoff
        date_str = game_date_et.isoformat()

        try:
            data = _fetch_hourly(lat, lon, date_str, use_archive=use_archive)
            time.sleep(REQUEST_SLEEP)

            wx = _extract_game_weather(data, start_ts_utc)
            if wx is None:
                log.warning("No weather data extracted for %s", game_slug)
                continue

            upsert_rows.append({"game_slug": game_slug, **wx})

        except Exception as exc:
            errors += 1
            log.warning("Weather fetch failed for %s: %s", game_slug, exc)
            if errors > 20:
                log.error("Too many weather errors; stopping fetch")
                break

    if upsert_rows:
        _upsert_weather(conn, upsert_rows)
        log.info("Upserted weather for %d games", len(upsert_rows))

    return len(upsert_rows)


def fetch_all_missing_weather(conn) -> int:
    """
    Fetch weather for all final + scheduled games that don't have weather yet.

    Called from parse_all.main() after parse_games().
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT g.game_slug
            FROM raw.mlb_games g
            WHERE g.status IN ('final', 'scheduled')
              AND NOT EXISTS (
                  SELECT 1 FROM raw.mlb_weather w WHERE w.game_slug = g.game_slug
              )
              AND g.start_ts_utc IS NOT NULL
              AND g.game_date_et <= CURRENT_DATE + 6
            ORDER BY g.game_date_et
        """)
        slugs = [r[0] for r in cur.fetchall()]

    if not slugs:
        log.info("No missing weather data to fetch")
        return 0

    log.info("Fetching weather for %d games without data", len(slugs))
    return fetch_weather_for_games(conn, slugs)


def refresh_todays_weather(conn) -> int:
    """
    Re-fetch weather for today's scheduled games (updates forecast as game time nears).
    """
    today = date.today().isoformat()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT g.game_slug
            FROM raw.mlb_games g
            WHERE g.game_date_et = %s
              AND g.status = 'scheduled'
            ORDER BY g.start_ts_utc
        """, (today,))
        slugs = [r[0] for r in cur.fetchall()]

    if not slugs:
        return 0

    log.info("Refreshing weather forecast for %d today's games", len(slugs))
    return fetch_weather_for_games(conn, slugs)


# ---------------------------------------------------------------------------
# Entry point (standalone)
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    try:
        n = fetch_all_missing_weather(conn)
        conn.commit()
        log.info("Weather crawl complete: %d games updated", n)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
