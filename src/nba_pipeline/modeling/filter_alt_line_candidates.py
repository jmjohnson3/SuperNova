from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

import pandas as pd
from sqlalchemy import create_engine, text
from zoneinfo import ZoneInfo

log = logging.getLogger("nba_pipeline.modeling.filter_alt_line_candidates")

_ET = ZoneInfo("America/New_York")


# --- Tune these thresholds however you want ---
@dataclass(frozen=True)
class FilterConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"
    et_date: date | None = None

    last_n: int = 10
    require_hits: int = 10

    min_minutes_proj: float = 28.0
    min_game_total: float = 215.0
    max_abs_spread: float = 8.0

    # "low foul risk" proxy
    # if you have a better foul-risk feature, swap it in.
    max_pf_avg_10: float = 3.5
    max_pf_per_min_10: float = 0.12  # ~3.4 fouls in 28 minutes

    # default alt lines scanned
    pts_lines: Sequence[float] = (10, 15, 20, 25, 30, 35)
    reb_lines: Sequence[float] = (4, 6, 8, 10, 12, 14)
    ast_lines: Sequence[float] = (3, 5, 7, 9, 11, 13)

    max_rows_per_stat: int = 200


SQL_TODAYS_GAMES = """
SELECT
  season,
  game_slug,
  game_date_et,
  start_ts_utc,
  UPPER(home_team_abbr) AS home_team_abbr,
  UPPER(away_team_abbr) AS away_team_abbr,
  market_total,
  market_spread_home
FROM features.game_prediction_features
WHERE game_date_et = :game_date
  AND start_ts_utc IS NOT NULL
ORDER BY start_ts_utc, game_slug
"""


# This is basically your "player snapshot" (pregame) builder, but we include:
# - player_name
# - min_avg_10 (minutes projection proxy)
# - starter flag (heuristic, unless you have a true starter column)
SQL_PLAYER_SNAPSHOTS_FOR_DATE = """
WITH games_today AS (
    SELECT
      season,
      game_slug,
      game_date_et,
      start_ts_utc,
      UPPER(home_team_abbr) AS home_team_abbr,
      UPPER(away_team_abbr) AS away_team_abbr,
      market_total,
      market_spread_home
    FROM features.game_prediction_features
    WHERE game_date_et = :game_date
      AND start_ts_utc IS NOT NULL
),
teams_today AS (
    SELECT season, home_team_abbr AS team_abbr, away_team_abbr AS opponent_abbr, TRUE AS is_home, game_slug
    FROM games_today
    UNION ALL
    SELECT season, away_team_abbr AS team_abbr, home_team_abbr AS opponent_abbr, FALSE AS is_home, game_slug
    FROM games_today
),
hist AS (
    SELECT
      p.season,
      p.game_slug,
      g.start_ts_utc,
      p.player_id,
      COALESCE(pl.player_name, CONCAT('player_id=', p.player_id::text)) AS player_name,
      UPPER(p.team_abbr) AS team_abbr,
      UPPER(p.opponent_abbr) AS opponent_abbr,
      p.minutes
      -- If you have a true starter column, uncomment and use it:
      -- , p.started
    FROM raw.nba_player_gamelogs p
    JOIN raw.nba_games g
      ON g.season = p.season
     AND g.game_slug = p.game_slug
    LEFT JOIN raw.nba_players pl
      ON pl.player_id = p.player_id
    WHERE g.start_ts_utc < (SELECT MIN(start_ts_utc) FROM games_today)
      AND p.minutes IS NOT NULL
      AND p.minutes > 0
),
feat AS (
    SELECT
      h.*,
      AVG(h.minutes) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS min_avg_10,
      COUNT(*)       OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS n_prev_10
    FROM hist h
),
latest AS (
    SELECT DISTINCT ON (player_id, team_abbr)
      player_id,
      player_name,
      team_abbr,
      opponent_abbr,
      min_avg_10,
      n_prev_10
    FROM feat
    ORDER BY player_id, team_abbr, start_ts_utc DESC
),
joined AS (
    SELECT
      t.season,
      t.game_slug,
      gt.game_date_et,
      gt.start_ts_utc,
      t.team_abbr,
      t.opponent_abbr,
      t.is_home,
      gt.home_team_abbr,
      gt.away_team_abbr,
      gt.market_total,
      gt.market_spread_home,
      l.player_id,
      l.player_name,
      l.min_avg_10,
      l.n_prev_10,
      -- starter heuristic: if you're averaging starter minutes recently, treat as starter
      (CASE WHEN l.min_avg_10 >= 24 THEN TRUE ELSE FALSE END) AS is_projected_starter
    FROM teams_today t
    JOIN games_today gt
      ON gt.season = t.season
     AND gt.game_slug = t.game_slug
    JOIN latest l
      ON l.team_abbr = t.team_abbr
)
SELECT *
FROM joined
WHERE n_prev_10 >= 3
"""


# Foul-risk proxy over last N games (minutes-filtered)
# NOTE: you MUST confirm your foul column name:
# - If raw.nba_player_gamelogs has "personal_fouls" use that.
# - If it's "fouls" then swap it in.
SQL_FOUL_RISK_LAST_N = """WITH base AS (
  SELECT
    bps.player_id,
    g.start_ts_utc,
    (bps.stats->'miscellaneous'->>'minSeconds')::numeric AS min_seconds,
    (bps.stats->'miscellaneous'->>'foulPers')::numeric AS pf
  FROM raw.nba_boxscore_player_stats bps
  JOIN raw.nba_games g
    ON g.season = bps.season
   AND g.game_slug = bps.game_slug
  WHERE bps.stats IS NOT NULL
    AND (bps.stats->'miscellaneous'->>'foulPers') IS NOT NULL
    AND (bps.stats->'miscellaneous'->>'minSeconds') IS NOT NULL
    AND (bps.stats->'miscellaneous'->>'minSeconds')::numeric > 0
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY start_ts_utc DESC) AS rn
  FROM base
),
last_n AS (
  SELECT *
  FROM ranked
  WHERE rn <= :last_n
)
SELECT
  player_id,
  COUNT(*) AS games_count,
  AVG(pf) AS pf_avg,
  AVG(pf / NULLIF(min_seconds/60.0, 0)) AS pf_per_min_avg
FROM last_n
GROUP BY player_id;
"""


def _values_sql(lines: Sequence[float]) -> str:
    # Build a VALUES list: (10::numeric),(15::numeric),...
    uniq = sorted({float(x) for x in lines})
    return ",".join(f"({x}::numeric)" for x in uniq)


def build_alt_grid_sql(stat: str, lines: Sequence[float], op: str = ">=") -> str:
    """
    Returns "best line per player" for the given stat over last_n games.
    """
    if stat not in {"points", "rebounds", "assists"}:
        raise ValueError(f"Unsupported stat: {stat}")

    stat_expr = {
        "points": "p.points",
        "rebounds": "p.rebounds",
        "assists": "p.assists",
    }[stat]

    values = _values_sql(lines)

    return f"""
WITH base AS (
  SELECT
    p.player_id,
    UPPER(p.team_abbr) AS team_abbr,
    g.start_ts_utc,
    p.minutes,
    {stat_expr} AS stat_value
  FROM raw.nba_player_gamelogs p
  JOIN raw.nba_games g
    ON g.season = p.season
   AND g.game_slug = p.game_slug
  WHERE p.minutes IS NOT NULL
    AND p.minutes > 0
    AND {stat_expr} IS NOT NULL
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY start_ts_utc DESC) AS rn
  FROM base
),
last_n AS (
  SELECT *
  FROM ranked
  WHERE rn <= :last_n
),
player_sample AS (
  SELECT
    player_id,
    MAX(team_abbr) AS team_abbr,
    COUNT(*) AS games_count,
    AVG(stat_value) AS avg_value
  FROM last_n
  GROUP BY player_id
),
lines AS (
  SELECT line
  FROM (VALUES {values}) v(line)
),
hits_by_line AS (
  SELECT
    ln.player_id,
    ps.team_abbr,
    ps.games_count,
    l.line,
    SUM(CASE WHEN ln.stat_value {op} l.line THEN 1 ELSE 0 END) AS hits,
    AVG(ln.stat_value) AS avg_value
  FROM last_n ln
  JOIN player_sample ps
    ON ps.player_id = ln.player_id
  CROSS JOIN lines l
  GROUP BY ln.player_id, ps.team_abbr, ps.games_count, l.line
),
best_per_player AS (
  SELECT DISTINCT ON (player_id)
    player_id,
    team_abbr,
    games_count,
    line,
    hits,
    (hits::float / NULLIF(games_count, 0)) AS hit_rate,
    avg_value
  FROM hits_by_line
  WHERE hits >= :min_hit_count
    AND games_count = :last_n
  ORDER BY player_id, line DESC, hits DESC, avg_value DESC
)
SELECT
  :stat_label AS stat,
  player_id,
  team_abbr,
  games_count,
  line,
  hits,
  hit_rate,
  avg_value
FROM best_per_player
ORDER BY line DESC, hits DESC, avg_value DESC
LIMIT :max_rows
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = FilterConfig()
    et_day = cfg.et_date or datetime.now(_ET).date()
    engine = create_engine(cfg.pg_dsn)

    log.info("Filtering alt-line candidates for ET date=%s", et_day)

    with engine.connect() as conn:
        games = pd.read_sql(text(SQL_TODAYS_GAMES), conn, params={"game_date": et_day})
        if games.empty:
            log.warning("No games found for %s", et_day)
            return

        snaps = pd.read_sql(text(SQL_PLAYER_SNAPSHOTS_FOR_DATE), conn, params={"game_date": et_day})
        if snaps.empty:
            log.warning("No player snapshots found for %s", et_day)
            return

        # Foul risk table (proxy)
        try:
            foul = pd.read_sql(
                text(SQL_FOUL_RISK_LAST_N),
                conn,
                params={"last_n": cfg.last_n},
            )
        except Exception:
            log.exception(
                "Foul-risk query failed. If your column isn't personal_fouls, change it in SQL_FOUL_RISK_LAST_N."
            )
            raise

        # Alt-line best-per-player for each stat
        pts = pd.read_sql(
            text(build_alt_grid_sql("points", cfg.pts_lines)),
            conn,
            params={
                "stat_label": "PTS",
                "last_n": cfg.last_n,
                "min_hit_count": cfg.require_hits,
                "max_rows": cfg.max_rows_per_stat,
            },
        )
        reb = pd.read_sql(
            text(build_alt_grid_sql("rebounds", cfg.reb_lines)),
            conn,
            params={
                "stat_label": "REB",
                "last_n": cfg.last_n,
                "min_hit_count": cfg.require_hits,
                "max_rows": cfg.max_rows_per_stat,
            },
        )
        ast = pd.read_sql(
            text(build_alt_grid_sql("assists", cfg.ast_lines)),
            conn,
            params={
                "stat_label": "AST",
                "last_n": cfg.last_n,
                "min_hit_count": cfg.require_hits,
                "max_rows": cfg.max_rows_per_stat,
            },
        )

    # Combine + attach names and today context
    grid = pd.concat([pts, reb, ast], ignore_index=True)
    if grid.empty:
        log.warning("No grid hits found for PTS/REB/AST with %d/%d.", cfg.require_hits, cfg.last_n)
        return

    # Join grid -> today snapshots (to restrict to today's games and get game totals/spreads)
    snaps_cols = [
        "season", "game_slug", "start_ts_utc", "team_abbr", "opponent_abbr", "is_home",
        "home_team_abbr", "away_team_abbr", "market_total", "market_spread_home",
        "player_id", "player_name", "min_avg_10", "is_projected_starter",
    ]
    s = snaps[snaps_cols].copy()

    # Normalize
    s["team_abbr"] = s["team_abbr"].astype(str).str.upper()
    grid["team_abbr"] = grid["team_abbr"].astype(str).str.upper()

    merged = grid.merge(s, on=["player_id", "team_abbr"], how="inner")

    # Join foul risk
    merged = merged.merge(foul[["player_id", "pf_avg", "pf_per_min_avg"]], on="player_id", how="left")
    merged["market_total"] = pd.to_numeric(merged["market_total"], errors="coerce")
    merged["market_spread_home"] = pd.to_numeric(merged["market_spread_home"], errors="coerce")
    merged["abs_spread"] = merged["market_spread_home"].abs()

    if merged["market_total"].isna().any() or merged["market_spread_home"].isna().any():
        log.warning(
            "Missing markets for some rows: total_na=%d spread_na=%d (these rows will be filtered out by thresholds)",
            int(merged["market_total"].isna().sum()),
            int(merged["market_spread_home"].isna().sum()),
        )
    # Coerce
    merged["market_total"] = pd.to_numeric(merged["market_total"], errors="coerce")
    merged["market_spread_home"] = pd.to_numeric(merged["market_spread_home"], errors="coerce")
    merged["abs_spread"] = merged["market_spread_home"].abs()

    # "skip if missing" masks
    total_ok = merged["market_total"].isna() | (merged["market_total"] >= cfg.min_game_total)
    spread_ok = merged["abs_spread"].isna() | (merged["abs_spread"] <= cfg.max_abs_spread)

    filtered = merged[
        (merged["is_projected_starter"] == True) &
        (merged["min_avg_10"] >= cfg.min_minutes_proj) &
        total_ok &
        spread_ok &
        (merged["pf_avg"].fillna(99) <= cfg.max_pf_avg_10) &
        (merged["pf_per_min_avg"].fillna(99) <= cfg.max_pf_per_min_10)
        ].copy()

    if filtered.empty:
        print("\nNo players matched all filters. Try relaxing minutes or foul thresholds.\n")
        return

    # Pretty print
    filtered["start_ts_utc"] = pd.to_datetime(filtered["start_ts_utc"], utc=True).dt.tz_convert(_ET)
    filtered = filtered.sort_values(["start_ts_utc", "game_slug", "stat", "line"], ascending=[True, True, True, False])

    print("\n=== FILTERED ALT-LINE CANDIDATES (TODAY) ===")
    print(
        f"Filters: starter=True | min_avg_10>={cfg.min_minutes_proj:g} | "
        f"total>={cfg.min_game_total:g} | abs_spread<={cfg.max_abs_spread:g} | "
        f"pf_avg_10<={cfg.max_pf_avg_10:g} | pf/min<={cfg.max_pf_per_min_10:g}\n"
    )

    for (ts, slug), g in filtered.groupby(["start_ts_utc", "game_slug"], sort=False):
        # Get matchup line
        row0 = g.iloc[0]
        home = row0["home_team_abbr"]
        away = row0["away_team_abbr"]
        total = float(row0["market_total"]) if pd.notna(row0["market_total"]) else None
        spread = float(row0["market_spread_home"]) if pd.notna(row0["market_spread_home"]) else None

        total_s = f"{total:.1f}" if total is not None else "NA"
        spread_s = f"{spread:+.1f}" if spread is not None else "NA"

        print(f"\n{ts:%Y-%m-%d %I:%M%p ET} | {away} @ {home} | Total {total_s} | Spread(home) {spread_s} | {slug}")

        # Print per stat
        for stat in ["PTS", "REB", "AST"]:
            sg = g[g["stat"] == stat]
            if sg.empty:
                continue
            print(f"  {stat}:")
            for _, r in sg.sort_values(["line", "avg_value"], ascending=[False, False]).iterrows():
                print(
                    f"    {r['player_name']} ({r['team_abbr']}) | best_line={float(r['line']):g} "
                    f"| {int(r['hits'])}/{int(r['games_count'])} "
                    f"| avg={float(r['avg_value']):.1f} "
                    f"| min_avg_10={float(r['min_avg_10']):.1f} "
                    f"| pf_avg_10={float(r['pf_avg']):.2f} pf/min={float(r['pf_per_min_avg']):.3f}"
                )

    print("\n")


if __name__ == "__main__":
    main()
