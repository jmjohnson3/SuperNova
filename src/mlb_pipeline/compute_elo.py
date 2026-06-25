# src/mlb_pipeline/compute_elo.py
"""Compute FiveThirtyEight-style Elo ratings for all MLB teams.

Reads completed games from raw.mlb_games (chronological order) and replaces
the derived per-game pre/post Elo values in raw.mlb_elo. Safe to re-run.

Algorithm (tuned for baseball)
------------------------------
  - K = 6 base, scaled by margin-of-victory multiplier
  - MOV multiplier = (|mov| + 1)^0.7 / (3.0 + 0.004 * |elo_diff_winner|)
      Adapted from NBA/FiveThirtyEight: smaller constants because run margins
      are much tighter than point margins in basketball.
  - Home advantage = 24 Elo points (MLB home teams win ~54% historically)
  - Season regression = pull 1/3 toward 1500 at the start of each new season
  - New team initial Elo = 1500

Usage
-----
  python -m mlb_pipeline.compute_elo
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import pandas as pd
import psycopg2

from mlb_pipeline.db import PG_DSN

log = logging.getLogger("mlb_pipeline.compute_elo")

INIT_ELO: float = 1500.0
HOME_ADV: float = 24.0     # MLB home-field ≈ 54% win rate → ~24 Elo pts
K_BASE: float = 6.0        # Lower than NBA (20) because 162 games vs 82
REGRESSION: float = 1 / 3  # Fraction pulled toward mean at new season start

_DDL = """
CREATE TABLE IF NOT EXISTS raw.mlb_elo (
    game_slug    TEXT   NOT NULL,
    team_abbr    TEXT   NOT NULL,
    season       TEXT   NOT NULL,
    game_date_et DATE   NOT NULL,
    elo_pre      FLOAT  NOT NULL,
    elo_post     FLOAT  NOT NULL,
    PRIMARY KEY (game_slug, team_abbr)
);
"""

_UPSERT = """
INSERT INTO raw.mlb_elo (game_slug, team_abbr, season, game_date_et, elo_pre, elo_post)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (game_slug, team_abbr) DO UPDATE SET
    season       = EXCLUDED.season,
    game_date_et = EXCLUDED.game_date_et,
    elo_pre      = EXCLUDED.elo_pre,
    elo_post     = EXCLUDED.elo_post;
"""


@dataclass(frozen=True)
class EloConfig:
    pg_dsn: str = PG_DSN


def _expected_home_win(home_elo: float, away_elo: float) -> float:
    """Win probability for the home team (home advantage included)."""
    return 1.0 / (1.0 + math.pow(10.0, (away_elo - (home_elo + HOME_ADV)) / 400.0))


def _mov_multiplier(mov: float, elo_diff_winner: float) -> float:
    """Margin-of-victory multiplier adapted for baseball's lower scoring.

    Compared to NBA version:
      - Exponent 0.7 (vs 0.8) to dampen extreme blowouts
      - +1 (vs +3) because run margins cluster near 1-4
      - Denominator constants scaled down proportionally
    """
    return (abs(mov) + 1.0) ** 0.7 / (3.0 + 0.004 * abs(elo_diff_winner))


def _elo_update(
    home_elo: float,
    away_elo: float,
    home_score: float,
    away_score: float,
) -> tuple[float, float]:
    """Return (home_elo_post, away_elo_post) after one game."""
    home_win = (
        1.0 if home_score > away_score
        else (0.5 if home_score == away_score else 0.0)
    )
    exp_home = _expected_home_win(home_elo, away_elo)

    mov = home_score - away_score
    if mov != 0:
        elo_diff_winner = home_elo - away_elo if mov > 0 else away_elo - home_elo
    else:
        elo_diff_winner = 0.0
    mult = _mov_multiplier(mov, elo_diff_winner)

    delta = K_BASE * mult * (home_win - exp_home)
    return home_elo + delta, away_elo - delta


def compute_all_elo(conn) -> list[tuple]:
    """Compute Elo for all completed MLB games. Returns rows for raw.mlb_elo."""
    df = pd.read_sql(
        """
        SELECT game_slug, season, game_date_et,
               home_team_abbr, away_team_abbr,
               home_score, away_score
        FROM   raw.mlb_games
        WHERE  status = 'final'
          AND  home_score IS NOT NULL
          AND  away_score IS NOT NULL
        ORDER  BY game_date_et, game_slug
        """,
        conn,
    )

    if df.empty:
        log.warning("No completed games found in raw.mlb_games")
        return []

    df["game_date_et"] = pd.to_datetime(df["game_date_et"])

    elo: dict[str, float] = {}
    seasons_seen: set[str] = set()
    rows: list[tuple] = []

    for _, row in df.iterrows():
        season = row["season"]
        home = row["home_team_abbr"]
        away = row["away_team_abbr"]

        # Season regression: first time we encounter this season, regress all teams
        if season not in seasons_seen:
            seasons_seen.add(season)
            for team in list(elo.keys()):
                elo[team] = INIT_ELO + (elo[team] - INIT_ELO) * (1.0 - REGRESSION)

        # Initialise new teams at league average
        if home not in elo:
            elo[home] = INIT_ELO
        if away not in elo:
            elo[away] = INIT_ELO

        home_elo_pre = elo[home]
        away_elo_pre = elo[away]

        home_elo_post, away_elo_post = _elo_update(
            home_elo_pre, away_elo_pre,
            float(row["home_score"]), float(row["away_score"]),
        )

        elo[home] = home_elo_post
        elo[away] = away_elo_post

        gd = row["game_date_et"].date()
        rows.append((row["game_slug"], home, season, gd, home_elo_pre, home_elo_post))
        rows.append((row["game_slug"], away, season, gd, away_elo_pre, away_elo_post))

    log.info("Computed Elo for %d game-team rows (%d games)", len(rows), len(rows) // 2)
    return rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = EloConfig()
    log.info("Computing MLB Elo ratings (MOV-adjusted, K=%s, home_adv=%s)…", K_BASE, HOME_ADV)

    with psycopg2.connect(cfg.pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
        log.info("Ensured raw.mlb_elo table exists.")

        rows = compute_all_elo(conn)

        if not rows:
            log.warning("No Elo rows to insert — exiting.")
            return

        with conn.cursor() as cur:
            cur.execute("DELETE FROM raw.mlb_elo")
            cur.executemany(_UPSERT, rows)
        conn.commit()

    log.info("Done. Replaced raw.mlb_elo with %d rows.", len(rows))


if __name__ == "__main__":
    main()
