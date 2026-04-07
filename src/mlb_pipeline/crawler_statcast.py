# src/mlb_pipeline/crawler_statcast.py
"""
Fetch season-level Statcast leaderboard data from Baseball Savant.

Downloads two CSV endpoints per year (batter + pitcher):
  1. Exit velocity / barrel stats  — avg_exit_velo, barrel_rate, hard_hit_pct, etc.
  2. Expected stats                — xBA, xSLG, xwOBA, xISO, etc.

Data is stored in:
  - raw.mlb_statcast_batting  (keyed by player_id, season_year)
  - raw.mlb_statcast_pitching (keyed by player_id, season_year)

No API key required — Baseball Savant is a public MLB resource.

Usage:
  python -m mlb_pipeline.crawler_statcast
  python -m mlb_pipeline.crawler_statcast --year 2024
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import execute_values

log = logging.getLogger("mlb_pipeline.crawler_statcast")

_ET = ZoneInfo("America/New_York")
_PG_DSN = "postgresql://josh:password@localhost:5432/nba"

# Baseball Savant leaderboard CSV endpoints
_BASE = "https://baseballsavant.mlb.com/leaderboard"

# Exit velocity + barrel data
_EV_BARREL_URL = (
    "{base}/statcast?type={ptype}&year={year}&position=&team=&min={min_bbe}"
    "&sort=5&sortDir=desc&csv=true"
)

# Expected statistics
_EXPECTED_URL = (
    "{base}/expected_statistics?type={ptype}&year={year}&position=&team="
    "&filterByPosition=&min={min_pa}&csv=true"
)

# Request headers to look like a normal browser
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                  " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
}

# ───────────────────────────────────────────────────────────────────────────
# DDL
# ───────────────────────────────────────────────────────────────────────────

_DDL_BATTING = """
CREATE TABLE IF NOT EXISTS raw.mlb_statcast_batting (
    player_id           INTEGER     NOT NULL,
    player_name         TEXT,
    season_year         INTEGER     NOT NULL,
    -- Exit velocity / barrel metrics
    avg_exit_velocity   FLOAT,
    max_exit_velocity   FLOAT,
    avg_launch_angle    FLOAT,
    barrel_batted_rate  FLOAT,      -- barrel % (0-100 scale from Savant)
    hard_hit_percent    FLOAT,      -- 95+ mph hit rate
    groundballs_percent FLOAT,
    flyballs_percent    FLOAT,
    linedrives_percent  FLOAT,
    batted_ball_events  INTEGER,    -- total BBE (sample size indicator)
    sweet_spot_percent  FLOAT,      -- 8-32 deg launch angle %
    -- Expected stats
    xba                 FLOAT,      -- expected batting average
    xslg                FLOAT,      -- expected slugging
    xwoba               FLOAT,      -- expected wOBA
    xiso                FLOAT,      -- expected ISO (xSLG - xBA)
    xobp                FLOAT,      -- expected OBP
    -- Metadata
    fetched_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, season_year)
);
"""

_DDL_PITCHING = """
CREATE TABLE IF NOT EXISTS raw.mlb_statcast_pitching (
    player_id           INTEGER     NOT NULL,
    player_name         TEXT,
    season_year         INTEGER     NOT NULL,
    -- Exit velocity / barrel metrics (against)
    avg_exit_velocity   FLOAT,
    max_exit_velocity   FLOAT,
    avg_launch_angle    FLOAT,
    barrel_batted_rate  FLOAT,
    hard_hit_percent    FLOAT,
    groundballs_percent FLOAT,
    flyballs_percent    FLOAT,
    linedrives_percent  FLOAT,
    batted_ball_events  INTEGER,
    sweet_spot_percent  FLOAT,
    -- Expected stats (against)
    xba                 FLOAT,
    xslg                FLOAT,
    xwoba               FLOAT,
    xiso                FLOAT,
    xobp                FLOAT,
    -- Metadata
    fetched_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, season_year)
);
"""

# ───────────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StatcastConfig:
    pg_dsn: str = _PG_DSN
    min_bbe_batter: int = 10      # min batted ball events for batter leaderboard
    min_bbe_pitcher: int = 10     # min BBE for pitcher leaderboard
    min_pa: int = 10              # min plate appearances for expected stats
    request_delay: float = 2.0    # seconds between requests (be polite)


# ───────────────────────────────────────────────────────────────────────────
# HTTP fetch
# ───────────────────────────────────────────────────────────────────────────

def _fetch_csv(url: str) -> list[dict]:
    """Fetch a CSV URL from Baseball Savant and return list of dicts."""
    log.info("Fetching: %s", url[:120])
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8-sig")  # BOM-safe
    except Exception:
        log.exception("Failed to fetch %s", url[:120])
        return []

    # Parse CSV
    reader = csv.DictReader(io.StringIO(raw))
    rows = list(reader)
    log.info("  → %d rows", len(rows))
    return rows


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None for blanks/errors."""
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    if val is None or str(val).strip() == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


# ───────────────────────────────────────────────────────────────────────────
# Fetch + merge batter / pitcher data
# ───────────────────────────────────────────────────────────────────────────

def _fetch_batter_data(year: int, cfg: StatcastConfig) -> dict[int, dict]:
    """Fetch and merge batter EV/barrel + expected stats for a season."""
    players: dict[int, dict] = {}

    # 1. Exit velocity / barrels
    url = _EV_BARREL_URL.format(
        base=_BASE, ptype="batter", year=year, min_bbe=cfg.min_bbe_batter,
    )
    for row in _fetch_csv(url):
        pid = _safe_int(row.get("player_id"))
        if not pid:
            continue
        players[pid] = {
            "player_id": pid,
            "player_name": row.get("player_name") or row.get("last_name, first_name") or "",
            "season_year": year,
            "avg_exit_velocity": _safe_float(row.get("avg_hit_speed")),
            "max_exit_velocity": _safe_float(row.get("max_hit_speed")),
            "avg_launch_angle": _safe_float(row.get("avg_hit_angle")),
            "barrel_batted_rate": _safe_float(row.get("barrel_batted_rate")),
            "hard_hit_percent": _safe_float(row.get("hard_hit_percent") or row.get("ev95percent")),
            "groundballs_percent": _safe_float(row.get("groundballs_percent")),
            "flyballs_percent": _safe_float(row.get("flyballs_percent")),
            "linedrives_percent": _safe_float(row.get("linedrives_percent")),
            "batted_ball_events": _safe_int(row.get("attempts") or row.get("bip")),
            "sweet_spot_percent": _safe_float(row.get("sweet_spot_percent")),
            # Expected stats filled in next step
            "xba": None, "xslg": None, "xwoba": None, "xiso": None, "xobp": None,
        }

    time.sleep(cfg.request_delay)

    # 2. Expected statistics
    url = _EXPECTED_URL.format(
        base=_BASE, ptype="batter", year=year, min_pa=cfg.min_pa,
    )
    for row in _fetch_csv(url):
        pid = _safe_int(row.get("player_id"))
        if not pid:
            continue
        if pid not in players:
            players[pid] = {
                "player_id": pid,
                "player_name": row.get("player_name") or row.get("last_name, first_name") or "",
                "season_year": year,
                "avg_exit_velocity": None, "max_exit_velocity": None,
                "avg_launch_angle": None, "barrel_batted_rate": None,
                "hard_hit_percent": None, "groundballs_percent": None,
                "flyballs_percent": None, "linedrives_percent": None,
                "batted_ball_events": None, "sweet_spot_percent": None,
            }
        players[pid]["xba"] = _safe_float(row.get("est_ba") or row.get("xba"))
        players[pid]["xslg"] = _safe_float(row.get("est_slg") or row.get("xslg"))
        players[pid]["xwoba"] = _safe_float(row.get("est_woba") or row.get("xwoba"))
        xba = players[pid]["xba"]
        xslg = players[pid]["xslg"]
        players[pid]["xiso"] = (xslg - xba) if (xba is not None and xslg is not None) else None
        players[pid]["xobp"] = _safe_float(row.get("est_obp") or row.get("xobp"))

    return players


def _fetch_pitcher_data(year: int, cfg: StatcastConfig) -> dict[int, dict]:
    """Fetch and merge pitcher EV/barrel + expected stats for a season."""
    players: dict[int, dict] = {}

    # 1. Exit velocity / barrels (against)
    url = _EV_BARREL_URL.format(
        base=_BASE, ptype="pitcher", year=year, min_bbe=cfg.min_bbe_pitcher,
    )
    for row in _fetch_csv(url):
        pid = _safe_int(row.get("player_id"))
        if not pid:
            continue
        players[pid] = {
            "player_id": pid,
            "player_name": row.get("player_name") or row.get("last_name, first_name") or "",
            "season_year": year,
            "avg_exit_velocity": _safe_float(row.get("avg_hit_speed")),
            "max_exit_velocity": _safe_float(row.get("max_hit_speed")),
            "avg_launch_angle": _safe_float(row.get("avg_hit_angle")),
            "barrel_batted_rate": _safe_float(row.get("barrel_batted_rate")),
            "hard_hit_percent": _safe_float(row.get("hard_hit_percent") or row.get("ev95percent")),
            "groundballs_percent": _safe_float(row.get("groundballs_percent")),
            "flyballs_percent": _safe_float(row.get("flyballs_percent")),
            "linedrives_percent": _safe_float(row.get("linedrives_percent")),
            "batted_ball_events": _safe_int(row.get("attempts") or row.get("bip")),
            "sweet_spot_percent": _safe_float(row.get("sweet_spot_percent")),
            "xba": None, "xslg": None, "xwoba": None, "xiso": None, "xobp": None,
        }

    time.sleep(cfg.request_delay)

    # 2. Expected statistics (against)
    url = _EXPECTED_URL.format(
        base=_BASE, ptype="pitcher", year=year, min_pa=cfg.min_pa,
    )
    for row in _fetch_csv(url):
        pid = _safe_int(row.get("player_id"))
        if not pid:
            continue
        if pid not in players:
            players[pid] = {
                "player_id": pid,
                "player_name": row.get("player_name") or row.get("last_name, first_name") or "",
                "season_year": year,
                "avg_exit_velocity": None, "max_exit_velocity": None,
                "avg_launch_angle": None, "barrel_batted_rate": None,
                "hard_hit_percent": None, "groundballs_percent": None,
                "flyballs_percent": None, "linedrives_percent": None,
                "batted_ball_events": None, "sweet_spot_percent": None,
            }
        players[pid]["xba"] = _safe_float(row.get("est_ba") or row.get("xba"))
        players[pid]["xslg"] = _safe_float(row.get("est_slg") or row.get("xslg"))
        players[pid]["xwoba"] = _safe_float(row.get("est_woba") or row.get("xwoba"))
        xba = players[pid]["xba"]
        xslg = players[pid]["xslg"]
        players[pid]["xiso"] = (xslg - xba) if (xba is not None and xslg is not None) else None
        players[pid]["xobp"] = _safe_float(row.get("est_obp") or row.get("xobp"))

    return players


# ───────────────────────────────────────────────────────────────────────────
# DB upsert
# ───────────────────────────────────────────────────────────────────────────

_COLS = [
    "player_id", "player_name", "season_year",
    "avg_exit_velocity", "max_exit_velocity", "avg_launch_angle",
    "barrel_batted_rate", "hard_hit_percent",
    "groundballs_percent", "flyballs_percent", "linedrives_percent",
    "batted_ball_events", "sweet_spot_percent",
    "xba", "xslg", "xwoba", "xiso", "xobp",
]

_UPSERT_BATTING = f"""
INSERT INTO raw.mlb_statcast_batting ({', '.join(_COLS)})
VALUES %s
ON CONFLICT (player_id, season_year) DO UPDATE SET
    player_name         = EXCLUDED.player_name,
    avg_exit_velocity   = EXCLUDED.avg_exit_velocity,
    max_exit_velocity   = EXCLUDED.max_exit_velocity,
    avg_launch_angle    = EXCLUDED.avg_launch_angle,
    barrel_batted_rate  = EXCLUDED.barrel_batted_rate,
    hard_hit_percent    = EXCLUDED.hard_hit_percent,
    groundballs_percent = EXCLUDED.groundballs_percent,
    flyballs_percent    = EXCLUDED.flyballs_percent,
    linedrives_percent  = EXCLUDED.linedrives_percent,
    batted_ball_events  = EXCLUDED.batted_ball_events,
    sweet_spot_percent  = EXCLUDED.sweet_spot_percent,
    xba                 = EXCLUDED.xba,
    xslg                = EXCLUDED.xslg,
    xwoba               = EXCLUDED.xwoba,
    xiso                = EXCLUDED.xiso,
    xobp                = EXCLUDED.xobp,
    fetched_at_utc      = now()
"""

_UPSERT_PITCHING = _UPSERT_BATTING.replace("mlb_statcast_batting", "mlb_statcast_pitching")


def _upsert_rows(cur, sql: str, players: dict[int, dict]) -> int:
    if not players:
        return 0
    tuples = [
        tuple(p.get(c) for c in _COLS)
        for p in players.values()
    ]
    execute_values(cur, sql, tuples, page_size=200)
    return len(tuples)


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fetch Statcast season leaderboards from Baseball Savant")
    parser.add_argument("--year", type=int, default=None,
                        help="Season year to fetch (default: current year)")
    parser.add_argument("--backfill", action="store_true",
                        help="Fetch 2024 + current year")
    args = parser.parse_args()

    cfg = StatcastConfig()
    current_year = datetime.now(_ET).year
    years = [current_year]

    if args.year:
        years = [args.year]
    elif args.backfill:
        years = [2024, 2025, current_year] if current_year > 2025 else [2024, current_year]
        # Deduplicate
        years = sorted(set(years))

    log.info("Statcast crawler: years=%s, min_bbe=%d, min_pa=%d",
             years, cfg.min_bbe_batter, cfg.min_pa)

    with psycopg2.connect(cfg.pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL_BATTING)
            cur.execute(_DDL_PITCHING)
        conn.commit()
        log.info("Ensured Statcast tables exist.")

        for year in years:
            log.info("─── Fetching %d Statcast data ───", year)

            # Batters
            batter_data = _fetch_batter_data(year, cfg)
            time.sleep(cfg.request_delay)

            # Pitchers
            pitcher_data = _fetch_pitcher_data(year, cfg)

            # Upsert
            with conn.cursor() as cur:
                nb = _upsert_rows(cur, _UPSERT_BATTING, batter_data)
                np_ = _upsert_rows(cur, _UPSERT_PITCHING, pitcher_data)
            conn.commit()

            log.info("Year %d: upserted %d batters, %d pitchers", year, nb, np_)

            if year != years[-1]:
                time.sleep(cfg.request_delay)

    log.info("Statcast crawler complete.")


if __name__ == "__main__":
    main()
