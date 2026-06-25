"""
crawler_statcast_velocity.py
----------------------------
Fetches per-SP-per-game fastball velocity from Baseball Savant statcast_search CSV
and stores the results in raw.mlb_sp_start_velocity.

Data source: Baseball Savant statcast_search (group_by=name-game), pitch types FF+SI.
One row per (player_id, game_date) — primary key — storing average release speed and
pitch count for 4-seam fastballs (FF) and sinkers (SI).

Usage:
    python -m mlb_pipeline.crawler_statcast_velocity --years 2024 2025
"""
import argparse
import io
import logging
import time
from datetime import date, timedelta

import psycopg2
import requests

log = logging.getLogger("mlb_pipeline.crawler_statcast_velocity")

from mlb_pipeline.db import PG_DSN as _PG_DSN
# Pitch types to fetch: 4-seam fastball + sinker
_PITCH_TYPES = ["FF", "SI"]

# Rate-limit between requests (seconds)
_REQUEST_DELAY = 1.5

# Number of days before we consider a player's data stale and re-fetch
_STALE_DAYS = 7

_SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"


def _ensure_schema(conn) -> None:
    """Create raw.mlb_sp_start_velocity if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw.mlb_sp_start_velocity (
                player_id      INTEGER  NOT NULL,
                player_name    TEXT,
                season_year    INTEGER  NOT NULL,
                game_date      DATE     NOT NULL,
                game_pk        INTEGER,
                ff_avg_speed   FLOAT,
                si_avg_speed   FLOAT,
                ff_n_pitches   INTEGER,
                si_n_pitches   INTEGER,
                fetched_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (player_id, game_date)
            )
        """)


def _fetch_savant_csv(player_id: int, year: int) -> bytes | None:
    """Fetch statcast CSV for a pitcher+year. Returns raw CSV bytes or None on failure."""
    params = {
        "all":               "true",
        "player_type":       "pitcher",
        "pitchers_lookup[]": str(player_id),
        "hfPT":              "FF|SI|",
        "type":              "details",
        "group_by":          "name-game",
        "year":              str(year),
    }
    try:
        resp = requests.get(_SAVANT_URL, params=params, timeout=30)
        if resp.status_code == 429:
            log.warning("Rate-limited by Baseball Savant (429) for player_id=%d year=%d", player_id, year)
            time.sleep(30)
            return None
        if resp.status_code != 200:
            log.debug("HTTP %d for player_id=%d year=%d", resp.status_code, player_id, year)
            return None
        return resp.content
    except Exception as exc:
        log.warning("Request error for player_id=%d year=%d: %s", player_id, year, exc)
        return None


def _parse_csv(csv_bytes: bytes, player_id: int) -> list[dict]:
    """
    Parse Baseball Savant CSV bytes.

    Expected columns include: game_date, game_pk, player_name, pitch_type, release_speed.
    Returns list of dicts ready for upsert into raw.mlb_sp_start_velocity.
    """
    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    except Exception as exc:
        log.warning("Could not parse CSV for player_id=%d: %s", player_id, exc)
        return []

    if df.empty or "game_date" not in df.columns:
        return []

    # Normalize columns
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    df = df.dropna(subset=["game_date"])

    if "release_speed" not in df.columns:
        log.debug("No release_speed column for player_id=%d", player_id)
        return []

    df["release_speed"] = pd.to_numeric(df["release_speed"], errors="coerce")

    # Pivot by pitch_type per game_date
    rows: list[dict] = []
    for game_date, grp in df.groupby("game_date"):
        row: dict = {
            "player_id":   player_id,
            "player_name": str(grp["player_name"].iloc[0]) if "player_name" in grp.columns else None,
            "season_year": game_date.year,
            "game_date":   game_date,
            "game_pk":     int(grp["game_pk"].iloc[0]) if "game_pk" in grp.columns and grp["game_pk"].notna().any() else None,
            "ff_avg_speed": None,
            "si_avg_speed": None,
            "ff_n_pitches": None,
            "si_n_pitches": None,
        }
        for pt_col, avg_col, n_col in [("FF", "ff_avg_speed", "ff_n_pitches"),
                                        ("SI", "si_avg_speed", "si_n_pitches")]:
            if "pitch_type" in grp.columns:
                sub = grp[grp["pitch_type"] == pt_col]["release_speed"].dropna()
                if len(sub) > 0:
                    row[avg_col] = float(sub.mean())
                    row[n_col]   = int(len(sub))
        rows.append(row)

    return rows


def _upsert_rows(conn, rows: list[dict]) -> int:
    """Upsert velocity rows. Returns number of rows inserted/updated."""
    if not rows:
        return 0
    with conn.cursor() as cur:
        for row in rows:
            cur.execute("""
                INSERT INTO raw.mlb_sp_start_velocity
                    (player_id, player_name, season_year, game_date, game_pk,
                     ff_avg_speed, si_avg_speed, ff_n_pitches, si_n_pitches)
                VALUES
                    (%(player_id)s, %(player_name)s, %(season_year)s, %(game_date)s, %(game_pk)s,
                     %(ff_avg_speed)s, %(si_avg_speed)s, %(ff_n_pitches)s, %(si_n_pitches)s)
                ON CONFLICT (player_id, game_date) DO UPDATE SET
                    player_name    = EXCLUDED.player_name,
                    game_pk        = EXCLUDED.game_pk,
                    ff_avg_speed   = EXCLUDED.ff_avg_speed,
                    si_avg_speed   = EXCLUDED.si_avg_speed,
                    ff_n_pitches   = EXCLUDED.ff_n_pitches,
                    si_n_pitches   = EXCLUDED.si_n_pitches,
                    fetched_at_utc = now()
            """, row)
    return len(rows)


def _get_stale_pitcher_years(conn, years: list[int]) -> list[tuple[int, int]]:
    """
    Return (player_id, year) pairs that need fetching:
      - player_id × year combos from mlb_starting_pitchers not yet in mlb_sp_start_velocity
      - OR where the most recent game_date in mlb_sp_start_velocity is more than _STALE_DAYS old
        relative to today (to catch newly played starts).
    """
    stale_cutoff = date.today() - timedelta(days=_STALE_DAYS)
    year_list = list(years)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT sp.player_id,
                   EXTRACT(YEAR FROM g.game_date_et)::INT AS season_year
            FROM raw.mlb_starting_pitchers sp
            JOIN raw.mlb_games g ON g.game_slug = sp.game_slug
            WHERE EXTRACT(YEAR FROM g.game_date_et)::INT = ANY(%s)
              AND sp.player_id IS NOT NULL
        """, (year_list,))
        all_pairs = set(cur.fetchall())

        cur.execute("""
            SELECT player_id, season_year, MAX(game_date) AS last_date
            FROM raw.mlb_sp_start_velocity
            WHERE season_year = ANY(%s)
            GROUP BY player_id, season_year
        """, (year_list,))
        fetched = {(r[0], r[1]): r[2] for r in cur.fetchall()}

    result = []
    for (player_id, season_year) in all_pairs:
        last_date = fetched.get((player_id, season_year))
        if last_date is None or last_date < stale_cutoff:
            result.append((int(player_id), int(season_year)))

    return result


def fetch_all_sp_velocity(conn, years: list[int] | None = None) -> int:
    """
    Fetch velocity data for all SPs in mlb_starting_pitchers.
    Returns total number of pitcher-game rows upserted.
    """
    if years is None:
        years = [date.today().year, date.today().year - 1]

    _ensure_schema(conn)
    conn.commit()   # commit the CREATE TABLE before querying

    to_fetch = _get_stale_pitcher_years(conn, years)
    log.info("SP velocity: %d pitcher×year combos to fetch", len(to_fetch))

    total_rows = 0
    for idx, (player_id, season_year) in enumerate(to_fetch):
        if idx > 0:
            time.sleep(_REQUEST_DELAY)

        log.debug("Fetching velocity: player_id=%d year=%d (%d/%d)",
                  player_id, season_year, idx + 1, len(to_fetch))

        csv_bytes = _fetch_savant_csv(player_id, season_year)
        if csv_bytes is None:
            continue

        rows = _parse_csv(csv_bytes, player_id)
        if not rows:
            continue

        n = _upsert_rows(conn, rows)
        total_rows += n

        if (idx + 1) % 50 == 0:
            conn.commit()
            log.info("SP velocity progress: %d/%d fetched, %d rows upserted so far",
                     idx + 1, len(to_fetch), total_rows)

    return total_rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fetch SP per-start velocity from Baseball Savant")
    parser.add_argument(
        "--years", type=int, nargs="+",
        default=[date.today().year, date.today().year - 1],
        help="Season years to fetch (e.g. --years 2024 2025)",
    )
    args = parser.parse_args()

    log.info("Fetching SP velocity for years: %s", args.years)

    with psycopg2.connect(_PG_DSN) as conn:
        n = fetch_all_sp_velocity(conn, years=args.years)
        conn.commit()

    log.info("Done — %d pitcher-game rows upserted", n)


if __name__ == "__main__":
    main()
