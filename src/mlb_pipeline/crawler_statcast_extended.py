# src/mlb_pipeline/crawler_statcast_extended.py
"""
Extended Statcast crawler — HR-prediction feature set.

Adds three data sources not covered by crawler_statcast.py:

  1. Spray-angle profile (pull_percent, opposite_percent, popup_percent, brl_pa)
     - Sourced from the SAME Baseball Savant EV/barrel endpoint the base crawler uses,
       but those columns were previously not parsed.
     - Written as new columns on raw.mlb_statcast_batting via ALTER TABLE ADD COLUMN
       IF NOT EXISTS + a partial upsert that only updates the four new fields.

  2. Sprint speed  →  raw.mlb_statcast_sprint_speed
     - Source: baseballsavant.mlb.com/running_splits
     - Key column: sprint_speed (ft/sec, 90th-percentile burst speed)
     - Athleticism proxy with non-trivial positive correlation to HR rate.

  3. Pitcher fastball-arsenal profile  →  raw.mlb_statcast_pitcher_arsenal
     - Source: baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats?pitchType=FF
     - Captures 4-seam fastball dominance (usage %, velocity, hard-hit% against,
       xwOBA against, barrel rate against).
     - High FB velocity + high FB usage → hard for batters to time → fewer HRs.
     - Separate fetch for sinkers (SI) allows distinguishing FB-heavy pitchers.

Usage:
  python -m mlb_pipeline.crawler_statcast_extended
  python -m mlb_pipeline.crawler_statcast_extended --year 2024
  python -m mlb_pipeline.crawler_statcast_extended --backfill   # 2024 + 2025 + current
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

log = logging.getLogger("mlb_pipeline.crawler_statcast_extended")

_ET = ZoneInfo("America/New_York")
from mlb_pipeline.db import PG_DSN as _PG_DSN
_BASE = "https://baseballsavant.mlb.com"

# Same EV/barrel endpoint as the base crawler — spray cols are in the same CSV
_EV_BARREL_URL = (
    "{base}/leaderboard/statcast?type={ptype}&year={year}&position=&team="
    "&min={min_bbe}&sort=5&sortDir=desc&csv=true"
)

# Sprint speed leaderboard (single sprint_speed value per player)
_SPRINT_URL = (
    "{base}/sprint_speed_leaderboard?year={year}&type=runner&min={min_sprint}&csv=true"
)

# Pitch arsenal (per pitch type)
_ARSENAL_URL = (
    "{base}/leaderboard/pitch-arsenal-stats?pitchType={pitch_type}"
    "&type=pitcher&year={year}&min={min_pitches}&csv=true"
)

# Plate discipline — batter (chase rate, contact rates, whiff)
_BATTER_DISC_COLS_STR = (
    "oz_swing_percent,iz_contact_percent,oz_contact_percent,"
    "swing_percent,whiff_percent,out_zone_percent,k_percent,bb_percent,pa"
)
_BATTER_DISC_URL = (
    "{base}/leaderboard/custom?year={year}&type=batter&filter=&sort=2&sortDir=asc"
    "&min={min_pa}&selections={cols}&csv=true"
)

# Plate discipline — pitcher (induced whiff/chase; zone_percent/strike_percent always empty for pitcher type)
_PITCHER_DISC_COLS_STR = (
    "oz_swing_percent,iz_contact_percent,oz_contact_percent,"
    "swing_percent,whiff_percent,k_percent,bb_percent,pa"
)
_PITCHER_DISC_URL = (
    "{base}/leaderboard/custom?year={year}&type=pitcher&filter=&sort=2&sortDir=asc"
    "&min={min_pa}&selections={cols}&csv=true"
)

# Catcher framing leaderboard
_FRAMING_URL = (
    "{base}/leaderboard/catcher-framing?year={year}&type=catcher&min={min_pitches}&csv=true"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
}


# ─────────────────────────────────────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────────────────────────────────────

# Extend the existing batting table with 4 new columns (safe / idempotent)
_DDL_BATTING_ALTER = """
ALTER TABLE raw.mlb_statcast_batting
    ADD COLUMN IF NOT EXISTS pull_percent     FLOAT,
    ADD COLUMN IF NOT EXISTS opposite_percent  FLOAT,
    ADD COLUMN IF NOT EXISTS popup_percent     FLOAT,
    ADD COLUMN IF NOT EXISTS brl_pa            FLOAT;
"""

_DDL_SPRINT = """
CREATE TABLE IF NOT EXISTS raw.mlb_statcast_sprint_speed (
    player_id        INTEGER      NOT NULL,
    player_name      TEXT,
    season_year      INTEGER      NOT NULL,
    sprint_speed     FLOAT,           -- ft/sec, 90th-percentile burst speed
    hp_to_1b         FLOAT,           -- home-to-first average (sec)
    competitive_runs INTEGER,         -- sample-size indicator
    fetched_at_utc   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, season_year)
);
"""

_DDL_ARSENAL = """
CREATE TABLE IF NOT EXISTS raw.mlb_statcast_pitcher_arsenal (
    player_id             INTEGER     NOT NULL,
    player_name           TEXT,
    season_year           INTEGER     NOT NULL,
    -- 4-seam fastball (FF) profile
    fb_percent            FLOAT,   -- usage % (0-100)
    fb_hard_hit_pct       FLOAT,   -- hard-hit % allowed on 4-seam FBs
    fb_xwoba              FLOAT,   -- xwOBA against on 4-seam FBs
    fb_run_value_per_100  FLOAT,   -- run value per 100 pitches (+ = bad for pitcher)
    fb_whiff_pct          FLOAT,   -- swing-and-miss % on 4-seam FBs
    fb_k_pct              FLOAT,   -- K% when this pitch is thrown
    fb_put_away           FLOAT,   -- put-away % (2-strike whiff)
    -- Sinker (SI) profile — separates heavy-groundball pitchers
    si_percent            FLOAT,
    si_hard_hit_pct       FLOAT,
    si_xwoba              FLOAT,
    si_whiff_pct          FLOAT,
    si_k_pct              FLOAT,
    -- Slider (SL) profile — highest whiff pitch for most pitchers
    sl_percent            FLOAT,
    sl_hard_hit_pct       FLOAT,
    sl_xwoba              FLOAT,
    sl_whiff_pct          FLOAT,
    sl_k_pct              FLOAT,
    sl_run_value_per_100  FLOAT,
    -- Changeup (CH) profile — deception / swing-miss off-speed
    ch_percent            FLOAT,
    ch_hard_hit_pct       FLOAT,
    ch_xwoba              FLOAT,
    ch_whiff_pct          FLOAT,
    ch_k_pct              FLOAT,
    ch_run_value_per_100  FLOAT,
    -- Combined fastball family (FF + SI)
    fastball_family_pct   FLOAT,   -- FB + sinker combined usage
    -- Pitch diversity (higher = more varied arsenal)
    pitch_diversity       FLOAT,   -- 1 - max_single_pitch_pct / 100
    fetched_at_utc        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, season_year)
);
"""

_DDL_BATTER_DISCIPLINE = """
CREATE TABLE IF NOT EXISTS raw.mlb_statcast_batter_discipline (
    player_id        INTEGER     NOT NULL,
    player_name      TEXT,
    season_year      INTEGER     NOT NULL,
    oz_swing_pct     FLOAT,   -- out-of-zone swing % (chase rate) — primary K predictor
    iz_contact_pct   FLOAT,   -- in-zone contact %  (low = more called Ks)
    oz_contact_pct   FLOAT,   -- out-of-zone contact % (low = more chase Ks)
    swing_pct        FLOAT,   -- overall swing %
    whiff_pct        FLOAT,   -- overall swing-and-miss %
    out_zone_pct     FLOAT,   -- % of pitches seen outside zone
    k_pct            FLOAT,   -- K% (season)
    bb_pct           FLOAT,   -- BB% (season)
    pa               INTEGER, -- plate appearances (sample size)
    fetched_at_utc   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, season_year)
);
"""

_DDL_PITCHER_DISCIPLINE = """
CREATE TABLE IF NOT EXISTS raw.mlb_statcast_pitcher_discipline (
    player_id        INTEGER     NOT NULL,
    player_name      TEXT,
    season_year      INTEGER     NOT NULL,
    oz_swing_pct     FLOAT,   -- out-of-zone swing % induced (chase rate)
    iz_contact_pct   FLOAT,   -- in-zone contact % allowed
    oz_contact_pct   FLOAT,   -- out-of-zone contact % allowed
    swing_pct        FLOAT,   -- overall swing % induced
    whiff_pct        FLOAT,   -- overall whiff % induced
    zone_pct         FLOAT,   -- % of pitches thrown in strike zone
    strike_pct       FLOAT,   -- overall strike %
    k_pct            FLOAT,   -- K% (season)
    bb_pct           FLOAT,   -- BB% (season)
    pa               INTEGER, -- batters faced
    fetched_at_utc   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, season_year)
);
"""

_DDL_CATCHER_FRAMING = """
CREATE TABLE IF NOT EXISTS raw.mlb_statcast_catcher_framing (
    player_id             INTEGER     NOT NULL,
    player_name           TEXT,
    season_year           INTEGER     NOT NULL,
    framing_rv_per_100    FLOAT,   -- run value per 100 borderline pitches (+= better framer)
    framing_rate          FLOAT,   -- % of borderline pitches received as strikes
    framing_pitches       INTEGER, -- sample: # of borderline pitches
    fetched_at_utc        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, season_year)
);
"""

# ALTER to add new columns to existing tables (idempotent)
_DDL_ARSENAL_ALTER = """
ALTER TABLE raw.mlb_statcast_pitcher_arsenal
    ADD COLUMN IF NOT EXISTS fb_whiff_pct         FLOAT,
    ADD COLUMN IF NOT EXISTS fb_k_pct             FLOAT,
    ADD COLUMN IF NOT EXISTS fb_put_away          FLOAT,
    ADD COLUMN IF NOT EXISTS si_whiff_pct         FLOAT,
    ADD COLUMN IF NOT EXISTS si_k_pct             FLOAT,
    ADD COLUMN IF NOT EXISTS sl_percent           FLOAT,
    ADD COLUMN IF NOT EXISTS sl_hard_hit_pct      FLOAT,
    ADD COLUMN IF NOT EXISTS sl_xwoba             FLOAT,
    ADD COLUMN IF NOT EXISTS sl_whiff_pct         FLOAT,
    ADD COLUMN IF NOT EXISTS sl_k_pct             FLOAT,
    ADD COLUMN IF NOT EXISTS sl_run_value_per_100 FLOAT,
    ADD COLUMN IF NOT EXISTS ch_percent           FLOAT,
    ADD COLUMN IF NOT EXISTS ch_hard_hit_pct      FLOAT,
    ADD COLUMN IF NOT EXISTS ch_xwoba             FLOAT,
    ADD COLUMN IF NOT EXISTS ch_whiff_pct         FLOAT,
    ADD COLUMN IF NOT EXISTS ch_k_pct             FLOAT,
    ADD COLUMN IF NOT EXISTS ch_run_value_per_100 FLOAT;
"""


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExtendedStatcastConfig:
    pg_dsn: str = _PG_DSN
    min_bbe_batter: int = 10      # min BBE for spray angle leaderboard
    min_sprint: int = 10          # min competitive sprint attempts
    min_pitches: int = 20         # min pitches thrown for arsenal leaderboard (was 50; lowered for April)
    min_pa_disc: int = 20         # min PA for plate discipline leaderboard (was 50; lowered for April)
    min_framing: int = 20         # min borderline pitches for catcher framing (was 50; lowered for April)
    request_delay: float = 2.5   # seconds between requests


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_csv(url: str) -> list[dict]:
    log.info("Fetching: %s", url[:120])
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8-sig")
    except Exception:
        log.exception("Failed to fetch %s", url[:120])
        return []
    rows = list(csv.DictReader(io.StringIO(raw)))
    log.info("  → %d rows", len(rows))
    return rows


def _safe_float(val) -> float | None:
    if val is None or str(val).strip() in ("", "null", "NULL"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    if val is None or str(val).strip() in ("", "null", "NULL"):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spray-angle + brl_pa (patch onto existing batting rows)
# ─────────────────────────────────────────────────────────────────────────────

_SPRAY_COLS = ["player_id", "player_name", "season_year", "brl_pa"]

_UPSERT_SPRAY = """
INSERT INTO raw.mlb_statcast_batting (player_id, player_name, season_year, brl_pa)
VALUES %s
ON CONFLICT (player_id, season_year) DO UPDATE SET
    player_name    = EXCLUDED.player_name,
    brl_pa         = EXCLUDED.brl_pa,
    fetched_at_utc = now()
"""


def _fetch_spray_data(year: int, cfg: ExtendedStatcastConfig) -> dict[int, dict]:
    """Return {player_id: row} with brl_pa from EV leaderboard.

    Note: pull_percent / opposite_percent / popup_percent are not available
    from the Baseball Savant leaderboard CSV endpoints and are left NULL.
    """
    url = _EV_BARREL_URL.format(
        base=_BASE, ptype="batter", year=year, min_bbe=cfg.min_bbe_batter,
    )
    players: dict[int, dict] = {}
    for row in _fetch_csv(url):
        pid = _safe_int(row.get("player_id"))
        if not pid:
            continue
        players[pid] = {
            "player_id":   pid,
            "player_name": row.get("player_name") or row.get("last_name, first_name") or "",
            "season_year": year,
            "brl_pa":      _safe_float(row.get("brl_pa") or row.get("brl_per_pa")),
        }
    return players


def _upsert_spray(cur, players: dict[int, dict]) -> int:
    if not players:
        return 0
    tuples = [tuple(p.get(c) for c in _SPRAY_COLS) for p in players.values()]
    execute_values(cur, _UPSERT_SPRAY, tuples, page_size=200)
    return len(tuples)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sprint speed
# ─────────────────────────────────────────────────────────────────────────────

_SPRINT_COLS = ["player_id", "player_name", "season_year",
                "sprint_speed", "hp_to_1b", "competitive_runs"]

_UPSERT_SPRINT = """
INSERT INTO raw.mlb_statcast_sprint_speed (player_id, player_name, season_year,
    sprint_speed, hp_to_1b, competitive_runs)
VALUES %s
ON CONFLICT (player_id, season_year) DO UPDATE SET
    player_name      = EXCLUDED.player_name,
    sprint_speed     = EXCLUDED.sprint_speed,
    hp_to_1b         = EXCLUDED.hp_to_1b,
    competitive_runs = EXCLUDED.competitive_runs,
    fetched_at_utc   = now()
"""


def _fetch_sprint_data(year: int, cfg: ExtendedStatcastConfig) -> dict[int, dict]:
    url = _SPRINT_URL.format(base=_BASE, year=year, min_sprint=cfg.min_sprint)
    players: dict[int, dict] = {}
    for row in _fetch_csv(url):
        pid = _safe_int(row.get("player_id"))
        if not pid:
            continue
        players[pid] = {
            "player_id":        pid,
            "player_name":      row.get("player_name") or row.get("last_name, first_name") or "",
            "season_year":      year,
            "sprint_speed":     _safe_float(row.get("sprint_speed")),
            "hp_to_1b":         _safe_float(row.get("hp_to_1b")),
            "competitive_runs": _safe_int(row.get("competitive_runs") or row.get("n")),
        }
    return players


def _upsert_sprint(cur, players: dict[int, dict]) -> int:
    if not players:
        return 0
    tuples = [tuple(p.get(c) for c in _SPRINT_COLS) for p in players.values()]
    execute_values(cur, _UPSERT_SPRINT, tuples, page_size=200)
    return len(tuples)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pitcher fastball arsenal
# ─────────────────────────────────────────────────────────────────────────────

_ARSENAL_COLS = [
    "player_id", "player_name", "season_year",
    "fb_percent", "fb_hard_hit_pct", "fb_xwoba", "fb_run_value_per_100",
    "fb_whiff_pct", "fb_k_pct", "fb_put_away",
    "si_percent", "si_hard_hit_pct", "si_xwoba", "si_whiff_pct", "si_k_pct",
    "sl_percent", "sl_hard_hit_pct", "sl_xwoba", "sl_whiff_pct", "sl_k_pct", "sl_run_value_per_100",
    "ch_percent", "ch_hard_hit_pct", "ch_xwoba", "ch_whiff_pct", "ch_k_pct", "ch_run_value_per_100",
    "fastball_family_pct", "pitch_diversity",
]

_UPSERT_ARSENAL = f"""
INSERT INTO raw.mlb_statcast_pitcher_arsenal ({', '.join(_ARSENAL_COLS)})
VALUES %s
ON CONFLICT (player_id, season_year) DO UPDATE SET
    player_name           = EXCLUDED.player_name,
    fb_percent            = EXCLUDED.fb_percent,
    fb_hard_hit_pct       = EXCLUDED.fb_hard_hit_pct,
    fb_xwoba              = EXCLUDED.fb_xwoba,
    fb_run_value_per_100  = EXCLUDED.fb_run_value_per_100,
    fb_whiff_pct          = EXCLUDED.fb_whiff_pct,
    fb_k_pct              = EXCLUDED.fb_k_pct,
    fb_put_away           = EXCLUDED.fb_put_away,
    si_percent            = EXCLUDED.si_percent,
    si_hard_hit_pct       = EXCLUDED.si_hard_hit_pct,
    si_xwoba              = EXCLUDED.si_xwoba,
    si_whiff_pct          = EXCLUDED.si_whiff_pct,
    si_k_pct              = EXCLUDED.si_k_pct,
    sl_percent            = EXCLUDED.sl_percent,
    sl_hard_hit_pct       = EXCLUDED.sl_hard_hit_pct,
    sl_xwoba              = EXCLUDED.sl_xwoba,
    sl_whiff_pct          = EXCLUDED.sl_whiff_pct,
    sl_k_pct              = EXCLUDED.sl_k_pct,
    sl_run_value_per_100  = EXCLUDED.sl_run_value_per_100,
    ch_percent            = EXCLUDED.ch_percent,
    ch_hard_hit_pct       = EXCLUDED.ch_hard_hit_pct,
    ch_xwoba              = EXCLUDED.ch_xwoba,
    ch_whiff_pct          = EXCLUDED.ch_whiff_pct,
    ch_k_pct              = EXCLUDED.ch_k_pct,
    ch_run_value_per_100  = EXCLUDED.ch_run_value_per_100,
    fastball_family_pct   = EXCLUDED.fastball_family_pct,
    pitch_diversity       = EXCLUDED.pitch_diversity,
    fetched_at_utc        = now()
"""


def _parse_arsenal_row(row: dict) -> tuple[int | None, dict]:
    """Return (player_id, field_dict) for one arsenal CSV row."""
    pid = _safe_int(row.get("player_id"))
    if not pid:
        return None, {}
    return pid, {
        "player_name":       row.get("player_name") or row.get("last_name, first_name") or "",
        "pitch_percent":     _safe_float(
            row.get("pitch_usage") or row.get("pitch_percent") or row.get("pitches_percent")
        ),
        "hard_hit_pct":      _safe_float(row.get("hard_hit_percent")),
        "xwoba":             _safe_float(row.get("est_woba") or row.get("xwoba")),
        "run_value_per_100": _safe_float(row.get("run_value_per_100")),
        "whiff_pct":         _safe_float(row.get("whiff_percent")),
        "k_pct":             _safe_float(row.get("k_percent")),
        "put_away":          _safe_float(row.get("put_away")),
    }


# Pitch types to fetch: ordered FF → SI → SL → CH
_PITCH_TYPES = ("FF", "SI", "SL", "CH")


def _fetch_arsenal_data(year: int, cfg: ExtendedStatcastConfig) -> dict[int, dict]:
    """
    Fetch 4-seam (FF), sinker (SI), slider (SL), and changeup (CH) profiles,
    pivot into one row per pitcher.
    """
    # Collect per-pitch-type data keyed by (player_id, pitch_type)
    pitch_data: dict[tuple[int, str], dict] = {}

    for i, pitch_type in enumerate(_PITCH_TYPES):
        url = _ARSENAL_URL.format(
            base=_BASE, pitch_type=pitch_type, year=year, min_pitches=cfg.min_pitches,
        )
        for row in _fetch_csv(url):
            pid, fields = _parse_arsenal_row(row)
            if pid is None:
                continue
            pitch_data[(pid, pitch_type)] = fields

        if i < len(_PITCH_TYPES) - 1:  # delay between pitch types, not after last
            time.sleep(cfg.request_delay)

    # Pivot: one row per pitcher
    all_pids: set[int] = {pid for (pid, _) in pitch_data}
    players: dict[int, dict] = {}

    for pid in all_pids:
        ff = pitch_data.get((pid, "FF"), {})
        si = pitch_data.get((pid, "SI"), {})
        sl = pitch_data.get((pid, "SL"), {})
        ch = pitch_data.get((pid, "CH"), {})

        fb_pct = ff.get("pitch_percent")
        si_pct = si.get("pitch_percent")
        sl_pct = sl.get("pitch_percent")
        ch_pct = ch.get("pitch_percent")

        family_pct = None
        if fb_pct is not None or si_pct is not None:
            family_pct = (fb_pct or 0.0) + (si_pct or 0.0)

        # pitch_diversity: 1 - max single-pitch usage (lower max → more diverse arsenal)
        diversity = None
        all_pcts = [p for p in (fb_pct, si_pct, sl_pct, ch_pct) if p is not None]
        if all_pcts:
            diversity = round(1.0 - max(all_pcts) / 100.0, 4)

        name = ff.get("player_name") or si.get("player_name") or sl.get("player_name") or ch.get("player_name") or ""
        players[pid] = {
            "player_id":            pid,
            "player_name":          name,
            "season_year":          year,
            "fb_percent":           fb_pct,
            "fb_hard_hit_pct":      ff.get("hard_hit_pct"),
            "fb_xwoba":             ff.get("xwoba"),
            "fb_run_value_per_100": ff.get("run_value_per_100"),
            "fb_whiff_pct":         ff.get("whiff_pct"),
            "fb_k_pct":             ff.get("k_pct"),
            "fb_put_away":          ff.get("put_away"),
            "si_percent":           si_pct,
            "si_hard_hit_pct":      si.get("hard_hit_pct"),
            "si_xwoba":             si.get("xwoba"),
            "si_whiff_pct":         si.get("whiff_pct"),
            "si_k_pct":             si.get("k_pct"),
            "sl_percent":           sl_pct,
            "sl_hard_hit_pct":      sl.get("hard_hit_pct"),
            "sl_xwoba":             sl.get("xwoba"),
            "sl_whiff_pct":         sl.get("whiff_pct"),
            "sl_k_pct":             sl.get("k_pct"),
            "sl_run_value_per_100": sl.get("run_value_per_100"),
            "ch_percent":           ch_pct,
            "ch_hard_hit_pct":      ch.get("hard_hit_pct"),
            "ch_xwoba":             ch.get("xwoba"),
            "ch_whiff_pct":         ch.get("whiff_pct"),
            "ch_k_pct":             ch.get("k_pct"),
            "ch_run_value_per_100": ch.get("run_value_per_100"),
            "fastball_family_pct":  family_pct,
            "pitch_diversity":      diversity,
        }

    return players


def _upsert_arsenal(cur, players: dict[int, dict]) -> int:
    if not players:
        return 0
    tuples = [tuple(p.get(c) for c in _ARSENAL_COLS) for p in players.values()]
    execute_values(cur, _UPSERT_ARSENAL, tuples, page_size=200)
    return len(tuples)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Batter plate discipline
# ─────────────────────────────────────────────────────────────────────────────

_BATTER_DISC_COLS = [
    "player_id", "player_name", "season_year",
    "oz_swing_pct", "iz_contact_pct", "oz_contact_pct",
    "swing_pct", "whiff_pct", "out_zone_pct", "k_pct", "bb_pct", "pa",
]

_UPSERT_BATTER_DISC = f"""
INSERT INTO raw.mlb_statcast_batter_discipline ({', '.join(_BATTER_DISC_COLS)})
VALUES %s
ON CONFLICT (player_id, season_year) DO UPDATE SET
    player_name      = EXCLUDED.player_name,
    oz_swing_pct     = EXCLUDED.oz_swing_pct,
    iz_contact_pct   = EXCLUDED.iz_contact_pct,
    oz_contact_pct   = EXCLUDED.oz_contact_pct,
    swing_pct        = EXCLUDED.swing_pct,
    whiff_pct        = EXCLUDED.whiff_pct,
    out_zone_pct     = EXCLUDED.out_zone_pct,
    k_pct            = EXCLUDED.k_pct,
    bb_pct           = EXCLUDED.bb_pct,
    pa               = EXCLUDED.pa,
    fetched_at_utc   = now()
"""


def _fetch_batter_discipline(year: int, cfg: ExtendedStatcastConfig) -> dict[int, dict]:
    url = _BATTER_DISC_URL.format(
        base=_BASE, year=year, min_pa=cfg.min_pa_disc, cols=_BATTER_DISC_COLS_STR,
    )
    players: dict[int, dict] = {}
    for row in _fetch_csv(url):
        pid = _safe_int(row.get("player_id"))
        if not pid:
            continue
        players[pid] = {
            "player_id":     pid,
            "player_name":   row.get("last_name, first_name") or row.get("player_name") or "",
            "season_year":   year,
            "oz_swing_pct":  _safe_float(row.get("oz_swing_percent")),
            "iz_contact_pct":_safe_float(row.get("iz_contact_percent")),
            "oz_contact_pct":_safe_float(row.get("oz_contact_percent")),
            "swing_pct":     _safe_float(row.get("swing_percent")),
            "whiff_pct":     _safe_float(row.get("whiff_percent")),
            "out_zone_pct":  _safe_float(row.get("out_zone_percent")),
            "k_pct":         _safe_float(row.get("k_percent")),
            "bb_pct":        _safe_float(row.get("bb_percent")),
            "pa":            _safe_int(row.get("pa")),
        }
    return players


def _upsert_batter_discipline(cur, players: dict[int, dict]) -> int:
    if not players:
        return 0
    tuples = [tuple(p.get(c) for c in _BATTER_DISC_COLS) for p in players.values()]
    execute_values(cur, _UPSERT_BATTER_DISC, tuples, page_size=200)
    return len(tuples)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pitcher plate discipline
# ─────────────────────────────────────────────────────────────────────────────

_PITCHER_DISC_COLS = [
    "player_id", "player_name", "season_year",
    "oz_swing_pct", "iz_contact_pct", "oz_contact_pct",
    "swing_pct", "whiff_pct", "zone_pct", "strike_pct", "k_pct", "bb_pct", "pa",
]

_UPSERT_PITCHER_DISC = f"""
INSERT INTO raw.mlb_statcast_pitcher_discipline ({', '.join(_PITCHER_DISC_COLS)})
VALUES %s
ON CONFLICT (player_id, season_year) DO UPDATE SET
    player_name      = EXCLUDED.player_name,
    oz_swing_pct     = EXCLUDED.oz_swing_pct,
    iz_contact_pct   = EXCLUDED.iz_contact_pct,
    oz_contact_pct   = EXCLUDED.oz_contact_pct,
    swing_pct        = EXCLUDED.swing_pct,
    whiff_pct        = EXCLUDED.whiff_pct,
    zone_pct         = EXCLUDED.zone_pct,
    strike_pct       = EXCLUDED.strike_pct,
    k_pct            = EXCLUDED.k_pct,
    bb_pct           = EXCLUDED.bb_pct,
    pa               = EXCLUDED.pa,
    fetched_at_utc   = now()
"""


def _fetch_pitcher_discipline(year: int, cfg: ExtendedStatcastConfig) -> dict[int, dict]:
    url = _PITCHER_DISC_URL.format(
        base=_BASE, year=year, min_pa=cfg.min_pa_disc, cols=_PITCHER_DISC_COLS_STR,
    )
    players: dict[int, dict] = {}
    for row in _fetch_csv(url):
        pid = _safe_int(row.get("player_id"))
        if not pid:
            continue
        players[pid] = {
            "player_id":     pid,
            "player_name":   row.get("last_name, first_name") or row.get("player_name") or "",
            "season_year":   year,
            "oz_swing_pct":  _safe_float(row.get("oz_swing_percent")),
            "iz_contact_pct":_safe_float(row.get("iz_contact_percent")),
            "oz_contact_pct":_safe_float(row.get("oz_contact_percent")),
            "swing_pct":     _safe_float(row.get("swing_percent")),
            "whiff_pct":     _safe_float(row.get("whiff_percent")),
            "zone_pct":      _safe_float(row.get("zone_percent")),
            "strike_pct":    _safe_float(row.get("strike_percent")),
            "k_pct":         _safe_float(row.get("k_percent")),
            "bb_pct":        _safe_float(row.get("bb_percent")),
            "pa":            _safe_int(row.get("pa")),
        }
    return players


def _upsert_pitcher_discipline(cur, players: dict[int, dict]) -> int:
    if not players:
        return 0
    tuples = [tuple(p.get(c) for c in _PITCHER_DISC_COLS) for p in players.values()]
    execute_values(cur, _UPSERT_PITCHER_DISC, tuples, page_size=200)
    return len(tuples)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Catcher framing
# ─────────────────────────────────────────────────────────────────────────────

_FRAMING_COLS = [
    "player_id", "player_name", "season_year",
    "framing_rv_per_100", "framing_rate", "framing_pitches",
]

_UPSERT_FRAMING = f"""
INSERT INTO raw.mlb_statcast_catcher_framing ({', '.join(_FRAMING_COLS)})
VALUES %s
ON CONFLICT (player_id, season_year) DO UPDATE SET
    player_name         = EXCLUDED.player_name,
    framing_rv_per_100  = EXCLUDED.framing_rv_per_100,
    framing_rate        = EXCLUDED.framing_rate,
    framing_pitches     = EXCLUDED.framing_pitches,
    fetched_at_utc      = now()
"""


def _fetch_catcher_framing(year: int, cfg: ExtendedStatcastConfig) -> dict[int, dict]:
    url = _FRAMING_URL.format(base=_BASE, year=year, min_pitches=cfg.min_framing)
    players: dict[int, dict] = {}
    for row in _fetch_csv(url):
        pid = _safe_int(row.get("id"))
        if not pid:
            continue
        players[pid] = {
            "player_id":        pid,
            "player_name":      row.get("name") or "",
            "season_year":      year,
            "framing_rv_per_100": _safe_float(row.get("rv_tot")),
            "framing_rate":     _safe_float(row.get("pct_tot")),
            "framing_pitches":  _safe_int(row.get("pitches")),
        }
    return players


def _upsert_catcher_framing(cur, players: dict[int, dict]) -> int:
    if not players:
        return 0
    tuples = [tuple(p.get(c) for c in _FRAMING_COLS) for p in players.values()]
    execute_values(cur, _UPSERT_FRAMING, tuples, page_size=200)
    return len(tuples)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Fetch extended Statcast data (spray angle, sprint speed, pitcher arsenal)"
    )
    parser.add_argument("--year", type=int, default=None,
                        help="Season year to fetch (default: current year)")
    parser.add_argument("--backfill", action="store_true",
                        help="Fetch 2024 + 2025 + current year")
    args = parser.parse_args()

    cfg = ExtendedStatcastConfig()
    current_year = datetime.now(_ET).year
    years = [current_year]

    if args.year:
        years = [args.year]
    elif args.backfill:
        years = sorted(set([2024, 2025, current_year]))

    log.info("Extended Statcast crawler: years=%s", years)

    with psycopg2.connect(cfg.pg_dsn) as conn:
        with conn.cursor() as cur:
            # Ensure base table exists before ALTER (base crawler may not have run yet)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS raw.mlb_statcast_batting (
                    player_id    INTEGER  NOT NULL,
                    player_name  TEXT,
                    season_year  INTEGER  NOT NULL,
                    fetched_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (player_id, season_year)
                )
            """)
            cur.execute(_DDL_BATTING_ALTER)
            cur.execute(_DDL_SPRINT)
            cur.execute(_DDL_ARSENAL)
            cur.execute(_DDL_ARSENAL_ALTER)
            cur.execute(_DDL_BATTER_DISCIPLINE)
            cur.execute(_DDL_PITCHER_DISCIPLINE)
            cur.execute(_DDL_CATCHER_FRAMING)
        conn.commit()
        log.info("Ensured extended Statcast tables/columns exist.")

        for year in years:
            log.info("─── %d: spray angle ───", year)
            spray = _fetch_spray_data(year, cfg)
            time.sleep(cfg.request_delay)

            log.info("─── %d: sprint speed ───", year)
            sprint = _fetch_sprint_data(year, cfg)
            time.sleep(cfg.request_delay)

            log.info("─── %d: pitcher arsenal (FF + SI + SL + CH) ───", year)
            arsenal = _fetch_arsenal_data(year, cfg)
            time.sleep(cfg.request_delay)

            log.info("─── %d: batter plate discipline ───", year)
            b_disc = _fetch_batter_discipline(year, cfg)
            time.sleep(cfg.request_delay)

            log.info("─── %d: pitcher plate discipline ───", year)
            p_disc = _fetch_pitcher_discipline(year, cfg)
            time.sleep(cfg.request_delay)

            log.info("─── %d: catcher framing ───", year)
            framing = _fetch_catcher_framing(year, cfg)

            with conn.cursor() as cur:
                n_spray   = _upsert_spray(cur, spray)
                n_sprint  = _upsert_sprint(cur, sprint)
                n_arsenal = _upsert_arsenal(cur, arsenal)
                n_b_disc  = _upsert_batter_discipline(cur, b_disc)
                n_p_disc  = _upsert_pitcher_discipline(cur, p_disc)
                n_framing = _upsert_catcher_framing(cur, framing)
            conn.commit()

            log.info(
                "Year %d: spray=%d, sprint=%d, arsenal=%d, batter_disc=%d, pitcher_disc=%d, framing=%d",
                year, n_spray, n_sprint, n_arsenal, n_b_disc, n_p_disc, n_framing,
            )

            if year != years[-1]:
                time.sleep(cfg.request_delay)

    log.info("Extended Statcast crawler complete.")


if __name__ == "__main__":
    main()
