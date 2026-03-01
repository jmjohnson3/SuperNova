"""scan_alt_lines_grid.py
Scan historical hit-rates for alt player-prop lines and apply a configurable
rule-set to surface only the highest-confidence bets for today's slate.

Rules implemented
-----------------
 R1  Starters only          – confirmed lineup or high-median-minutes proxy
 R2  Median minutes ≥ 28    – computed over ALL last_n games (not just 28+ min games)
 R3  Game total ≥ 215       – market/consensus total for today's game
 R4  Spread ≤ 8             – abs(spread) for today's game
 R5  Role stability         – coefficient of variation of minutes < 30 %
 R6  Rotation fragility     – ≤ 1 game with <20 min in last_n
 R7  Foul risk              – avg personal fouls per game < 3.5
 R8  Assist hub             – for assist legs: FGA/48 proxy ≥ 8 (ball-handler)
 R9  Tax-line kill          – NOTE: no odds data in DB; annotated in output only
R10  Injury/rest kill       – player in today's injury report (OUT/DOUBTFUL/QUESTIONABLE)
R11  Game exposure cap      – max 2 legs per game_slug (post-processing)
R12  Moved-line usage       – if line > avg + 3, require FGA/48 ≥ 10
R13  Second-half minutes    – (skipped: no per-quarter player minutes in DB)
R14  Late-game usage        – (skipped: no 4Q data in DB)
R15  Correlation ban        – same player across multiple stats; >1 assist leg per game
R16  Defensive profile      – skip props vs opponents with elite defense (def_rtg < threshold)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Optional, Sequence

import pandas as pd
from sqlalchemy import create_engine, text
from zoneinfo import ZoneInfo

log = logging.getLogger("nba_pipeline.modeling.scan_alt_lines_grid")

_ET = ZoneInfo("America/New_York")

Stat = Literal["points", "rebounds", "assists"]
Side = Literal["over", "under"]

STAT_COL: dict[str, str] = {
    "points":   "p.points",
    "rebounds": "p.rebounds",
    "assists":  "p.assists",
}

SIDE_OP: dict[str, str] = {
    "over":  ">=",
    "under": "<=",
}

DEFAULT_LINES: dict[str, list[float]] = {
    "points":   [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
    "rebounds": [4,5,6,7,8,9,10,11,12,13,14],
    "assists":  [3,4,5,6,7,8,9,10,11,12,13],
}

# Injury report strings that trigger the Rule 10 kill
_INJURY_KILL_STRINGS = {"OUT", "DOUBTFUL", "QUESTIONABLE"}


@dataclass(frozen=True)
class GridScanConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"

    stats: Sequence[Stat] = ("points", "rebounds", "assists")
    side: Side = "over"
    lines: Optional[Sequence[float]] = None

    last_n: int = 20
    min_minutes: float = 28.0       # minutes floor for hit-rate sample games
    min_hit_count: int = 20
    require_full_sample: bool = True

    only_players_in_todays_games: bool = True
    et_date: Optional[date] = None
    max_rows: int = 150
    verbose: bool = False           # if True, print filtered-out rows with reasons

    # ── R1  Starters only ─────────────────────────────────────────────────────
    starters_only: bool = True

    # ── R2  Median minutes floor ──────────────────────────────────────────────
    min_median_minutes: float = 28.0

    # ── R3  Game total ────────────────────────────────────────────────────────
    min_game_total: float = 215.0

    # ── R4  Spread filter ─────────────────────────────────────────────────────
    max_spread_abs: float = 8.0

    # ── R5  Role stability ────────────────────────────────────────────────────
    max_minutes_cv: float = 0.30    # stddev / mean of minutes

    # ── R6  Rotation fragility ────────────────────────────────────────────────
    max_low_min_games: int = 1      # games with < 20 min allowed in last_n

    # ── R7  Foul risk ─────────────────────────────────────────────────────────
    max_avg_fouls: float = 3.5

    # ── R8  Assist hub requirement ────────────────────────────────────────────
    assist_min_usage_proxy: float = 8.0     # FGA / 48 min

    # ── R10 Injury/rest kill ──────────────────────────────────────────────────
    injury_kill: bool = True

    # ── R11 Game exposure cap ─────────────────────────────────────────────────
    max_legs_per_game: int = 2

    # ── R12 Moved-line usage requirement ─────────────────────────────────────
    moved_line_delta: float = 3.0       # line > avg_value + this triggers check
    moved_line_min_usage: float = 10.0  # FGA/48 required when line is "moved"

    # ── R15 Correlation ban ───────────────────────────────────────────────────
    correlation_ban: bool = True

    # ── R16 Defensive profile ─────────────────────────────────────────────────
    opp_def_rtg_min: float = 110.0   # cut if opponent def_rtg_avg_10 < this


# ---------------------------------------------------------------------------
# SQL: today's game context (one row per team playing today)
# ---------------------------------------------------------------------------
SQL_TODAYS_GAME_CONTEXT = """
SELECT
    UPPER(g.home_team_abbr)                                         AS team_abbr,
    g.game_slug,
    UPPER(g.away_team_abbr)                                         AS opponent_abbr,
    TRUE                                                            AS is_home,
    COALESCE(g.market_total, g.consensus_total)                     AS game_total,
    ABS(COALESCE(g.market_spread_home, g.consensus_spread_home, 0)) AS spread_abs,
    g.away_def_rtg_avg_10                                           AS opp_def_rtg
FROM features.game_prediction_features g
WHERE g.game_date_et = :game_date
  AND g.start_ts_utc IS NOT NULL

UNION ALL

SELECT
    UPPER(g.away_team_abbr)                                         AS team_abbr,
    g.game_slug,
    UPPER(g.home_team_abbr)                                         AS opponent_abbr,
    FALSE                                                           AS is_home,
    COALESCE(g.market_total, g.consensus_total)                     AS game_total,
    ABS(COALESCE(g.market_spread_home, g.consensus_spread_home, 0)) AS spread_abs,
    g.home_def_rtg_avg_10                                           AS opp_def_rtg
FROM features.game_prediction_features g
WHERE g.game_date_et = :game_date
  AND g.start_ts_utc IS NOT NULL
"""

# ---------------------------------------------------------------------------
# SQL: today's confirmed/expected starters from nba_game_lineups
# Player IDs live at raw_json -> 'team' -> 'actual'/'expected' ->
# 'lineupPositions' -> elem -> 'player' -> 'id'
# ---------------------------------------------------------------------------
SQL_TODAYS_STARTERS = """
SELECT DISTINCT
    (elem->'player'->>'id')::int    AS player_id,
    UPPER(gl.team_abbr)             AS team_abbr,
    gl.game_slug
FROM raw.nba_game_lineups gl
JOIN raw.nba_games g ON g.game_slug = gl.game_slug
CROSS JOIN LATERAL jsonb_array_elements(
    COALESCE(
        NULLIF(gl.raw_json->'team'->'actual'->'lineupPositions',   'null'::jsonb),
        NULLIF(gl.raw_json->'team'->'expected'->'lineupPositions', 'null'::jsonb),
        '[]'::jsonb
    )
) AS elem
WHERE g.game_date_et  = :game_date
  AND (elem->>'position') LIKE 'Starter%'
  AND elem->'player'->>'id' IS NOT NULL
"""

# ---------------------------------------------------------------------------
# SQL: today's injury report entries (any uncertainty kills the leg)
# ---------------------------------------------------------------------------
SQL_INJURIES = """
SELECT DISTINCT player_id
FROM raw.nba_injuries
WHERE UPPER(COALESCE(playing_probability, '')) = ANY(:kill_strings)
   OR UPPER(COALESCE(injury_description, ''))  ILIKE '%out%'
   OR UPPER(COALESCE(injury_description, ''))  ILIKE '%doubtful%'
   OR UPPER(COALESCE(injury_description, ''))  ILIKE '%questionable%'
"""


# ---------------------------------------------------------------------------
# SQL: core hit-rate scan (per stat, per side)
# Includes a parallel player_char CTE over ALL last_n games (no minutes floor)
# so that median_minutes and volatility reflect true availability.
# ---------------------------------------------------------------------------
def build_sql(stat: Stat, side: Side, require_full_sample: bool) -> str:
    stat_expr = STAT_COL[stat]
    op = SIDE_OP[side]
    sample_clause = "AND h.games_count = :last_n" if require_full_sample else ""

    return f"""
WITH
-- ── All recent games (any minutes) — player characterisation ─────────────────
base_all AS (
  SELECT
    p.player_id,
    UPPER(p.team_abbr)                                                  AS team_abbr,
    p.season,
    g.start_ts_utc,
    COALESCE(p.minutes, 0)                                              AS minutes,
    COALESCE(p.fga, 0)                                                  AS fga,
    COALESCE(
        (p.stats->'miscellaneous'->>'foulPers')::float,
        (p.stats->'miscellaneous'->>'fouls')::float,
        0
    )                                                                   AS game_fouls
  FROM raw.nba_player_gamelogs p
  JOIN raw.nba_games g
    ON g.season   = p.season
   AND g.game_slug = p.game_slug
  WHERE p.minutes IS NOT NULL
),
ranked_all AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY start_ts_utc DESC) AS rn
  FROM base_all
),
last_n_all AS (
  SELECT * FROM ranked_all WHERE rn <= :last_n
),
-- Per-player characterisation stats (minutes volatility, fouls, usage)
player_char AS (
  SELECT
    player_id,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY minutes)                AS median_minutes,
    COALESCE(STDDEV(minutes), 0)                                        AS minutes_stddev,
    AVG(minutes)                                                        AS avg_minutes_all,
    SUM(CASE WHEN minutes < 20 THEN 1 ELSE 0 END)::int                  AS low_min_games,
    AVG(game_fouls)                                                     AS avg_fouls,
    CASE
      WHEN SUM(minutes) > 0
      THEN (SUM(fga)::float / SUM(minutes)) * 48
      ELSE 0
    END                                                                 AS usage_proxy_48
  FROM last_n_all
  GROUP BY player_id
),

-- ── Filtered games (minutes >= threshold) — hit-rate calculation ─────────────
base AS (
  SELECT
    p.season,
    p.game_slug,
    g.start_ts_utc,
    p.player_id,
    UPPER(p.team_abbr)  AS team_abbr,
    p.minutes,
    {stat_expr}          AS stat_value
  FROM raw.nba_player_gamelogs p
  JOIN raw.nba_games g
    ON g.season   = p.season
   AND g.game_slug = p.game_slug
  WHERE p.minutes IS NOT NULL
    AND p.minutes >= :min_minutes
    AND {stat_expr} IS NOT NULL
),
ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY start_ts_utc DESC) AS rn
  FROM base
),
last_n AS (
  SELECT * FROM ranked WHERE rn <= :last_n
),
player_sample AS (
  SELECT
    player_id,
    MAX(season)     AS season,
    MAX(team_abbr)  AS team_abbr,
    COUNT(*)        AS games_count,
    AVG(stat_value) AS avg_value
  FROM last_n
  GROUP BY player_id
),
lines AS (
  SELECT line::numeric AS line
  FROM unnest((:lines)::float8[]) AS line
),
hits_by_line AS (
  SELECT
    ln.player_id,
    ln.season,
    ln.team_abbr,
    ps.games_count,
    l.line,
    SUM(CASE WHEN ln.stat_value {op} l.line THEN 1 ELSE 0 END) AS hits,
    AVG(ln.stat_value)                                          AS avg_value
  FROM last_n ln
  JOIN player_sample ps ON ps.player_id = ln.player_id
  CROSS JOIN lines l
  GROUP BY ln.player_id, ln.season, ln.team_abbr, ps.games_count, l.line
),
best_per_player AS (
  SELECT DISTINCT ON (h.player_id)
    h.player_id,
    h.season,
    h.team_abbr,
    h.games_count,
    h.line,
    h.hits,
    (h.hits::float / NULLIF(h.games_count, 0))  AS hit_rate,
    h.avg_value,
    -- player characterisation
    COALESCE(pc.median_minutes,   0)             AS median_minutes,
    COALESCE(pc.minutes_stddev,   0)             AS minutes_stddev,
    COALESCE(pc.avg_minutes_all,  0)             AS avg_minutes_all,
    COALESCE(pc.low_min_games,    0)             AS low_min_games,
    COALESCE(pc.avg_fouls,        0)             AS avg_fouls,
    COALESCE(pc.usage_proxy_48,   0)             AS usage_proxy_48
  FROM hits_by_line h
  LEFT JOIN player_char pc ON pc.player_id = h.player_id
  WHERE h.hits >= :min_hit_count
    {sample_clause}
  ORDER BY h.player_id, h.line DESC, h.hits DESC, h.avg_value DESC
)
SELECT
  b.player_id,
  COALESCE(pl.player_name, CONCAT('player_id=', b.player_id::text)) AS player_name,
  b.season,
  b.team_abbr,
  b.games_count,
  b.line,
  b.hits,
  b.hit_rate,
  b.avg_value,
  b.median_minutes,
  b.minutes_stddev,
  b.avg_minutes_all,
  b.low_min_games,
  b.avg_fouls,
  b.usage_proxy_48
FROM best_per_player b
LEFT JOIN raw.nba_players pl ON pl.player_id = b.player_id
ORDER BY b.line DESC, b.hits DESC, b.avg_value DESC
LIMIT :max_rows
"""


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------
def _apply_row_filters(
    row: "pd.Series",
    stat: Stat,
    game_ctx: Optional[dict],
    injury_set: set[int],
    starter_set: Optional[set[int]],
    cfg: GridScanConfig,
) -> list[str]:
    """Return list of failed rule labels. Empty list → row passes all rules."""
    reasons: list[str] = []
    pid = int(row["player_id"])

    # R1 – Starters only
    if cfg.starters_only:
        if starter_set is not None:
            if pid not in starter_set:
                # fall back to minutes heuristic: median ≥ 30 as proxy
                if float(row["median_minutes"]) < 30.0:
                    reasons.append("R1:not-starter")
        else:
            # No lineup data — use conservative heuristic
            if float(row["median_minutes"]) < 30.0:
                reasons.append("R1:not-starter(no-lineup-data)")

    # R2 – Median minutes floor
    if float(row["median_minutes"]) < cfg.min_median_minutes:
        reasons.append(f"R2:median-min={row['median_minutes']:.1f}<{cfg.min_median_minutes}")

    # R3 – Game total (only enforce when market/consensus data is available)
    if game_ctx is not None:
        gt = game_ctx.get("game_total")
        if gt is not None and float(gt) < cfg.min_game_total:
            reasons.append(f"R3:total={float(gt):.0f}<{cfg.min_game_total}")
    # (if gt is None or no game_ctx → no odds data, skip rule silently)

    # R4 – Spread (only enforce when data is available)
    if game_ctx is not None:
        sa = game_ctx.get("spread_abs")
        if sa is not None and float(sa) > 0 and float(sa) > cfg.max_spread_abs:
            reasons.append(f"R4:spread={float(sa):.1f}>{cfg.max_spread_abs}")

    # R5 – Role stability (coefficient of variation)
    avg_min = float(row["avg_minutes_all"])
    if avg_min > 0:
        cv = float(row["minutes_stddev"]) / avg_min
        if cv > cfg.max_minutes_cv:
            reasons.append(f"R5:min-cv={cv:.2f}>{cfg.max_minutes_cv}")

    # R6 – Rotation fragility
    if int(row["low_min_games"]) > cfg.max_low_min_games:
        reasons.append(f"R6:low-min-games={row['low_min_games']}>{cfg.max_low_min_games}")

    # R7 – Foul risk
    if float(row["avg_fouls"]) > cfg.max_avg_fouls:
        reasons.append(f"R7:avg-fouls={row['avg_fouls']:.1f}>{cfg.max_avg_fouls}")

    # R8 – Assist hub (assists stat only)
    if stat == "assists":
        if float(row["usage_proxy_48"]) < cfg.assist_min_usage_proxy:
            reasons.append(f"R8:usage={row['usage_proxy_48']:.1f}<{cfg.assist_min_usage_proxy}")

    # R10 – Injury kill
    if cfg.injury_kill and pid in injury_set:
        reasons.append("R10:injured/questionable")

    # R12 – Moved-line usage requirement (points only — most relevant)
    if stat == "points":
        avg_val = float(row["avg_value"])
        line = float(row["line"])
        if (line - avg_val) > cfg.moved_line_delta:
            if float(row["usage_proxy_48"]) < cfg.moved_line_min_usage:
                reasons.append(
                    f"R12:moved-line({line:.0f}>{avg_val:.1f}+{cfg.moved_line_delta})"
                    f" usage={row['usage_proxy_48']:.1f}<{cfg.moved_line_min_usage}"
                )

    # R16 – Defensive profile (only enforce when def_rtg data is available)
    if game_ctx is not None:
        odr = game_ctx.get("opp_def_rtg")
        if odr is not None and float(odr) < cfg.opp_def_rtg_min:
            reasons.append(f"R16:elite-def={float(odr):.1f}<{cfg.opp_def_rtg_min}")
        # (if odr is None → rolling def_rtg not available for this game, skip)

    return reasons


def _apply_global_filters(all_rows: list[dict], cfg: GridScanConfig) -> list[dict]:
    """Apply cross-row rules (R11 exposure cap, R15 correlation ban).
    Only examines rows that already pass per-row filters.
    Returns the full list with filter_reasons updated for newly-cut rows.
    """
    passing = [r for r in all_rows if not r["filter_reasons"]]

    # R15 – Correlation ban
    # • Same player appearing in multiple stat results → keep only best leg
    # • More than one assist prop per game → keep first only
    if cfg.correlation_ban:
        # Sort: highest line first so we keep the strongest leg
        passing.sort(key=lambda r: (r["line"], r["hit_rate"]), reverse=True)
        seen_players: set[int] = set()
        seen_assist_games: set[str] = set()
        kept: list[dict] = []
        for row in passing:
            pid = row["player_id"]
            slug = row.get("game_slug", "")
            stat = row["stat"]

            if pid in seen_players:
                row["filter_reasons"].append("R15:same-player-multi-stat")
                continue
            if stat == "assists" and slug in seen_assist_games:
                row["filter_reasons"].append("R15:assist-stack-same-game")
                continue

            seen_players.add(pid)
            if stat == "assists":
                seen_assist_games.add(slug)
            kept.append(row)
        passing = kept

    # R11 – Game exposure cap
    if cfg.max_legs_per_game > 0:
        game_counts: dict[str, int] = {}
        kept = []
        for row in passing:
            slug = row.get("game_slug", "")
            count = game_counts.get(slug, 0)
            if count >= cfg.max_legs_per_game:
                row["filter_reasons"].append(
                    f"R11:game-cap({cfg.max_legs_per_game})"
                )
                continue
            game_counts[slug] = count + 1
            kept.append(row)
        passing = kept

    return passing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_lines(lines: Sequence[float]) -> list[float]:
    return sorted({float(x) for x in lines})


def _fmt_label(stat: Stat, side: Side) -> str:
    return f"{stat.upper()} {'OVER' if side == 'over' else 'UNDER'}"


# ---------------------------------------------------------------------------
# Context loaders
# ---------------------------------------------------------------------------
def _load_game_context(conn, et_day: date) -> dict[str, dict]:
    """Returns {team_abbr_upper: {game_slug, game_total, spread_abs, opp_def_rtg}}."""
    try:
        df = pd.read_sql(
            text(SQL_TODAYS_GAME_CONTEXT),
            conn,
            params={"game_date": et_day},
        )
        if df.empty:
            return {}
        df["team_abbr"] = df["team_abbr"].astype(str).str.upper()
        result: dict[str, dict] = {}
        for _, r in df.iterrows():
            result[r["team_abbr"]] = {
                "game_slug":   r.get("game_slug"),
                "game_total":  r.get("game_total"),
                "spread_abs":  r.get("spread_abs"),
                "opp_def_rtg": r.get("opp_def_rtg"),
            }
        return result
    except Exception as e:
        log.warning("Failed to load game context: %s", e)
        return {}


def _load_starters(conn, et_day: date) -> Optional[set[int]]:
    """Returns set of confirmed-starter player_ids for today, or None on failure."""
    try:
        df = pd.read_sql(
            text(SQL_TODAYS_STARTERS),
            conn,
            params={"game_date": et_day},
        )
        if df.empty:
            log.warning("No lineup data found for %s — R1 will use minutes heuristic.", et_day)
            return None
        ids = set(int(x) for x in df["player_id"].dropna())
        log.info("Loaded %d confirmed starters for %s.", len(ids), et_day)
        return ids
    except Exception as e:
        log.warning("Could not load starter data (%s) — R1 will use minutes heuristic.", e)
        return None


def _load_injuries(conn) -> set[int]:
    """Returns set of player_ids with kill-worthy injury status."""
    try:
        kill_strings = list(_INJURY_KILL_STRINGS)
        df = pd.read_sql(
            text(SQL_INJURIES),
            conn,
            params={"kill_strings": kill_strings},
        )
        if df.empty:
            return set()
        return set(int(x) for x in df["player_id"].dropna())
    except Exception as e:
        log.warning("Could not load injury data (%s) — R10 skipped.", e)
        return set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = GridScanConfig()

    # validate
    for s in cfg.stats:
        if s not in STAT_COL:
            raise ValueError(f"Unsupported stat={s!r}. Use one of: {list(STAT_COL)}")
    if cfg.side not in SIDE_OP:
        raise ValueError(f"Unsupported side={cfg.side!r}. Use one of: {list(SIDE_OP)}")
    if cfg.last_n <= 0:
        raise ValueError("last_n must be > 0")
    if not (0 <= cfg.min_hit_count <= cfg.last_n):
        raise ValueError("min_hit_count must be in [0, last_n]")

    et_day = cfg.et_date or datetime.now(_ET).date()
    engine = create_engine(cfg.pg_dsn)

    with engine.connect() as conn:
        # ── Load context data ──────────────────────────────────────────────
        game_ctx_by_team = _load_game_context(conn, et_day)
        starter_set      = _load_starters(conn, et_day) if cfg.starters_only else None
        injury_set       = _load_injuries(conn) if cfg.injury_kill else set()

        # Filter to today's teams (existing behaviour)
        today_teams: Optional[pd.DataFrame] = None
        if cfg.only_players_in_todays_games:
            if game_ctx_by_team:
                today_teams = pd.DataFrame(
                    [
                        {"season": None, "team_abbr": t}
                        for t in game_ctx_by_team.keys()
                    ]
                )
            else:
                log.warning(
                    "No game context for %s — showing ALL players.", et_day
                )

        # ── Scan each stat ─────────────────────────────────────────────────
        all_rows: list[dict] = []

        for stat in cfg.stats:
            lines = _normalize_lines(
                cfg.lines if cfg.lines is not None else DEFAULT_LINES[stat]
            )
            sql = build_sql(stat, cfg.side, require_full_sample=cfg.require_full_sample)

            log.info(
                "Grid scan | stat=%s side=%s | lines=%s | last_n=%d "
                "min_hit_count=%d min_minutes=%.1f | date=%s",
                stat, cfg.side, lines, cfg.last_n, cfg.min_hit_count,
                cfg.min_minutes, et_day,
            )

            df = pd.read_sql(
                text(sql),
                conn,
                params={
                    "lines":         [float(x) for x in lines],
                    "last_n":        cfg.last_n,
                    "min_minutes":   cfg.min_minutes,
                    "min_hit_count": cfg.min_hit_count,
                    "max_rows":      cfg.max_rows,
                },
            )

            if today_teams is not None and not df.empty:
                df["team_abbr"] = df["team_abbr"].astype(str).str.upper()
                # season may be None in today_teams; merge on team only
                today_abbrs = set(today_teams["team_abbr"].str.upper())
                df = df[df["team_abbr"].isin(today_abbrs)]

            if df.empty:
                continue

            # Build rows with per-row filter results
            for _, row in df.iterrows():
                team = str(row["team_abbr"]).upper()
                gctx = game_ctx_by_team.get(team)

                reasons = _apply_row_filters(
                    row, stat, gctx, injury_set, starter_set, cfg
                )

                all_rows.append({
                    "player_id":     int(row["player_id"]),
                    "player_name":   str(row["player_name"]),
                    "team_abbr":     team,
                    "season":        str(row["season"]),
                    "games_count":   int(row["games_count"]),
                    "line":          float(row["line"]),
                    "hits":          int(row["hits"]),
                    "hit_rate":      float(row["hit_rate"]),
                    "avg_value":     float(row["avg_value"]),
                    "median_minutes": float(row["median_minutes"]),
                    "avg_fouls":     float(row["avg_fouls"]),
                    "usage_proxy_48": float(row["usage_proxy_48"]),
                    "game_slug":     gctx["game_slug"] if gctx else None,
                    "game_total":    gctx["game_total"] if gctx else None,
                    "stat":          stat,
                    "filter_reasons": reasons,
                })

        # ── Global post-processing (R11, R15) ──────────────────────────────
        all_rows = _apply_global_filters(all_rows, cfg)

        # ── Print results ──────────────────────────────────────────────────
        label = _fmt_label(cfg.stats[0] if len(cfg.stats) == 1 else "MULTI", cfg.side)
        passing = [r for r in all_rows if not r["filter_reasons"]]
        cut     = [r for r in all_rows if r["filter_reasons"]]

        STAT_LABEL = {"points": "Points", "rebounds": "Rebounds", "assists": "Assists"}
        discord = os.getenv("DISCORD_FORMAT") == "1"
        if passing:
            for r in sorted(passing, key=lambda x: (-x["hit_rate"], -x["line"])):
                side_word = "Over" if cfg.side == "over" else "Under"
                stat_word = STAT_LABEL.get(r["stat"], r["stat"].title())
                pct = f"{r['hit_rate']*100:.0f}%"
                if discord:
                    print(f"✅ **{r['player_name']}** · {side_word} {r['line']:g} {stat_word} · {pct}")
                else:
                    print(f"{r['player_name']} {side_word} {r['line']:g} {stat_word}  {pct}")
        else:
            print("*No plays for today's slate*" if discord else "No plays for today's slate")

        log.info(
            "Scan complete | %d candidates → %d pass, %d cut",
            len(all_rows), len(passing), len(cut),
        )

        if cfg.verbose and cut:
            print("\n── Filtered out ──────────────────────────────────────────────")
            for r in sorted(cut, key=lambda x: (x["stat"], -x["line"])):
                lbl = _fmt_label(r["stat"], cfg.side)
                print(
                    f"  CUT {r['player_name']} ({r['team_abbr']}) {lbl} {r['line']:g}"
                    f" | {'; '.join(r['filter_reasons'])}"
                )


if __name__ == "__main__":
    main()
