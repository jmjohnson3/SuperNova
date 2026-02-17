import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger("nba_pipeline.modeling.scan_alt_hit_rate")

Stat = Literal["points", "rebounds", "assists"]
Side = Literal["over", "under"]

# NOTE: We keep the stat column selection safe by mapping allowed values.
STAT_COL = {
    "points": "p.points",
    "rebounds": "p.rebounds",
    "assists": "p.assists",
}

SIDE_OP = {
    "over": ">=",
    "under": "<=",
}


@dataclass(frozen=True)
class ScanConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"

    stat: Stat = "points"
    side: Side = "over"
    line: float = 15.0

    last_n: int = 10
    min_games_required: int = 10  # require full sample to call it "100%"
    min_minutes: float = 10.0     # ignore super low-minute games (optional)
    only_100: bool = True         # if False, prints top hit rates

    max_rows: int = 100


def build_sql(stat: Stat, side: Side) -> str:
    stat_expr = STAT_COL[stat]
    op = SIDE_OP[side]

    # We compute last N games per player; then aggregate hit counts.
    return f"""
WITH base AS (
  SELECT
    p.season,
    p.game_slug,
    g.start_ts_utc,
    p.player_id,
    p.team_abbr,
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
agg AS (
  SELECT
    player_id,
    MAX(team_abbr) AS team_abbr,
    COUNT(*) AS games_count,
    SUM(CASE WHEN stat_value {op} :line THEN 1 ELSE 0 END) AS hits,
    AVG(stat_value) AS avg_value,
    MIN(stat_value) AS min_value,
    MAX(stat_value) AS max_value
  FROM last_n
  GROUP BY player_id
)
SELECT
  a.player_id,
  COALESCE(pl.player_name, CONCAT('player_id=', a.player_id::text)) AS player_name,
  a.team_abbr,
  a.games_count,
  a.hits,
  (a.hits::float / NULLIF(a.games_count, 0)) AS hit_rate,
  a.avg_value,
  a.min_value,
  a.max_value
FROM agg a
LEFT JOIN raw.nba_players pl
  ON pl.player_id = a.player_id
WHERE a.games_count >= :min_games_required
ORDER BY hit_rate DESC, hits DESC, avg_value DESC
LIMIT :max_rows
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    cfg = ScanConfig()

    if cfg.stat not in STAT_COL:
        raise ValueError(f"Unsupported stat={cfg.stat}. Use one of: {list(STAT_COL)}")
    if cfg.side not in SIDE_OP:
        raise ValueError(f"Unsupported side={cfg.side}. Use one of: {list(SIDE_OP)}")
    if cfg.last_n <= 0:
        raise ValueError("last_n must be > 0")
    if cfg.min_games_required > cfg.last_n:
        raise ValueError("min_games_required cannot exceed last_n")

    sql = build_sql(cfg.stat, cfg.side)
    engine = create_engine(cfg.pg_dsn)

    log.info(
        "Scanning alt hit rate | stat=%s side=%s line=%.2f | last_n=%d min_games=%d min_minutes=%.1f",
        cfg.stat, cfg.side, cfg.line, cfg.last_n, cfg.min_games_required, cfg.min_minutes
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            text(sql),
            conn,
            params={
                "line": cfg.line,
                "last_n": cfg.last_n,
                "min_games_required": cfg.min_games_required,
                "min_minutes": cfg.min_minutes,
                "max_rows": cfg.max_rows,
            },
        )

    if df.empty:
        log.warning("No players matched the criteria.")
        return

    if cfg.only_100:
        df = df[df["hits"] == df["games_count"]]

    if df.empty:
        log.warning("No 100%% players for this line. Try lowering the line or set only_100=False.")
        return

    label = f"{cfg.stat.upper()} {'OVER' if cfg.side == 'over' else 'UNDER'} {cfg.line:g}"
    print(f"\n=== LAST {cfg.last_n} GAMES | {label} | min_minutes={cfg.min_minutes:g} ===\n")

    for _, r in df.iterrows():
        print(
            f"{r['player_name']} ({str(r['team_abbr']).upper()}) | "
            f"{int(r['hits'])}/{int(r['games_count'])} ({float(r['hit_rate'])*100:.0f}%) | "
            f"avg={float(r['avg_value']):.1f} min={float(r['min_value']):.1f} max={float(r['max_value']):.1f}"
        )


if __name__ == "__main__":
    main()
