import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Sequence

import pandas as pd
from sqlalchemy import create_engine, text
from zoneinfo import ZoneInfo

log = logging.getLogger("nba_pipeline.modeling.scan_alt_lines_grid")

_ET = ZoneInfo("America/New_York")

Stat = Literal["points", "rebounds", "assists"]
Side = Literal["over", "under"]

STAT_COL = {
    "points": "p.points",
    "rebounds": "p.rebounds",
    "assists": "p.assists",
}

SIDE_OP = {
    "over": ">=",
    "under": "<=",
}

DEFAULT_LINES: dict[Stat, list[float]] = {
    "points": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    "rebounds": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "assists": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
}


@dataclass(frozen=True)
class GridScanConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"

    # NEW: scan multiple stats in one run
    stats: Sequence[Stat] = ("points", "rebounds", "assists")

    side: Side = "over"

    # Optional global override. If None, each stat uses DEFAULT_LINES[stat]
    lines: Sequence[float] | None = None

    last_n: int = 20
    min_minutes: float = 28.0

    min_hit_count: int = 20
    require_full_sample: bool = True  # if True, only players with exactly last_n games

    only_players_in_todays_games: bool = True
    et_date: date | None = None

    max_rows: int = 150


SQL_TODAYS_TEAMS = """
WITH games_today AS (
  SELECT
    season,
    game_slug,
    game_date_et,
    start_ts_utc,
    UPPER(home_team_abbr) AS home_team_abbr,
    UPPER(away_team_abbr) AS away_team_abbr
  FROM features.game_prediction_features
  WHERE game_date_et = :game_date
    AND start_ts_utc IS NOT NULL
)
SELECT DISTINCT season, home_team_abbr AS team_abbr FROM games_today
UNION
SELECT DISTINCT season, away_team_abbr AS team_abbr FROM games_today
"""


def build_sql(stat: Stat, side: Side, require_full_sample: bool) -> str:
    stat_expr = STAT_COL[stat]
    op = SIDE_OP[side]

    # Only difference: honor require_full_sample properly
    sample_clause = "AND games_count = :last_n" if require_full_sample else ""

    return f"""
WITH base AS (
  SELECT
    p.season,
    p.game_slug,
    g.start_ts_utc,
    p.player_id,
    UPPER(p.team_abbr) AS team_abbr,
    p.minutes,
    {stat_expr} AS stat_value
  FROM raw.nba_player_gamelogs p
  JOIN raw.nba_games g
    ON g.season = p.season
   AND g.game_slug = p.game_slug
  WHERE p.minutes IS NOT NULL
    AND p.minutes >= :min_minutes
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
    MAX(season) AS season,
    MAX(team_abbr) AS team_abbr,
    COUNT(*) AS games_count,
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
    AVG(ln.stat_value) AS avg_value
  FROM last_n ln
  JOIN player_sample ps
    ON ps.player_id = ln.player_id
  CROSS JOIN lines l
  GROUP BY ln.player_id, ln.season, ln.team_abbr, ps.games_count, l.line
),
best_per_player AS (
  SELECT DISTINCT ON (player_id)
    player_id,
    season,
    team_abbr,
    games_count,
    line,
    hits,
    (hits::float / NULLIF(games_count, 0)) AS hit_rate,
    avg_value
  FROM hits_by_line
  WHERE hits >= :min_hit_count
    {sample_clause}
  ORDER BY player_id, line DESC, hits DESC, avg_value DESC
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
  b.avg_value
FROM best_per_player b
LEFT JOIN raw.nba_players pl
  ON pl.player_id = b.player_id
ORDER BY b.line DESC, b.hits DESC, b.avg_value DESC
LIMIT :max_rows
"""


def _normalize_lines(lines: Sequence[float]) -> list[float]:
    # numeric + unique + sorted
    return sorted({float(x) for x in lines})


def _print_block(df: pd.DataFrame, stat: Stat, side: Side, cfg: GridScanConfig, lines: list[float]) -> None:
    label = f"{stat.upper()} {'OVER' if side == 'over' else 'UNDER'}"
    sample = f"{cfg.min_hit_count}/{cfg.last_n}" if cfg.require_full_sample else f">={cfg.min_hit_count} hits"

    #print(f"\n=== BEST ALT LINE PER PLAYER | {label} | sample={sample} | min_minutes={cfg.min_minutes:g} ===")
    #print(f"Lines scanned: {lines}\n")

    if df.empty:
        print("(no matches)\n")
        return

    df = df.sort_values(["line", "hits", "avg_value"], ascending=[False, False, False]).head(cfg.max_rows)

    for _, r in df.iterrows():
        print(
            f"{r['player_name']} ({str(r['team_abbr']).upper()}) {label} {float(r['line']):g}"
            #f"best_line={float(r['line']):g} | {int(r['hits'])}/{int(r['games_count'])} "
            #f"({float(r['hit_rate'])*100:.0f}%) | avg={float(r['avg_value']):.1f}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    cfg = GridScanConfig()

    # validate
    for s in cfg.stats:
        if s not in STAT_COL:
            raise ValueError(f"Unsupported stat={s}. Use one of: {list(STAT_COL)}")
    if cfg.side not in SIDE_OP:
        raise ValueError(f"Unsupported side={cfg.side}. Use one of: {list(SIDE_OP)}")
    if cfg.last_n <= 0:
        raise ValueError("last_n must be > 0")
    if cfg.min_hit_count < 0 or cfg.min_hit_count > cfg.last_n:
        raise ValueError("min_hit_count must be in [0, last_n]")

    et_day = cfg.et_date or datetime.now(_ET).date()
    engine = create_engine(cfg.pg_dsn)

    with engine.connect() as conn:
        teams = None
        if cfg.only_players_in_todays_games:
            teams = pd.read_sql(text(SQL_TODAYS_TEAMS), conn, params={"game_date": et_day})
            if teams.empty:
                log.warning("No teams found for today in features.game_prediction_features. Showing ALL players instead.")
                teams = None
            else:
                teams["team_abbr"] = teams["team_abbr"].astype(str).str.upper()

        # Run each stat
        for stat in cfg.stats:
            lines = _normalize_lines(cfg.lines if cfg.lines is not None else DEFAULT_LINES[stat])
            sql = build_sql(stat, cfg.side, require_full_sample=cfg.require_full_sample)

            log.info(
                "Grid scan | stat=%s side=%s | lines=%s | last_n=%d min_hit_count=%d min_minutes=%.1f | today_only=%s date=%s",
                stat, cfg.side, lines, cfg.last_n, cfg.min_hit_count, cfg.min_minutes,
                cfg.only_players_in_todays_games, et_day,
            )

            df = pd.read_sql(
                text(sql),
                conn,
                params={
                    "lines": [float(x) for x in lines],
                    "last_n": cfg.last_n,
                    "min_minutes": cfg.min_minutes,
                    "min_hit_count": cfg.min_hit_count,
                    "max_rows": cfg.max_rows,
                },
            )

            if teams is not None and not df.empty:
                df["team_abbr"] = df["team_abbr"].astype(str).str.upper()
                df = df.merge(teams, on=["season", "team_abbr"], how="inner")

            _print_block(df, stat, cfg.side, cfg, lines)


if __name__ == "__main__":
    main()
