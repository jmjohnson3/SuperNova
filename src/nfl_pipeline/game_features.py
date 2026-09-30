"""NFL game-level feature generation for spread/total models."""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.markets import normalize_team
from nfl_pipeline.schema import ensure_schema

log = logging.getLogger("nfl_pipeline.game_features")
warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
    category=UserWarning,
)


@dataclass(frozen=True)
class GameFeatureConfig:
    pg_dsn: str = PG_DSN
    min_season: int | None = None


ROLL_WINDOWS = (3, 5, 10)
FEATURE_COLUMNS = [
    "game_id", "season", "week", "season_type", "game_date_et", "start_ts_utc",
    "home_team_abbr", "away_team_abbr", "home_score", "away_score",
    "home_margin", "total_points_actual", "market_spread_home", "market_total",
    "roof", "surface", "temp", "wind", "home_game_number", "away_game_number",
    "home_rest_days", "away_rest_days",
    *[f"home_{stat}_avg_{window}" for stat in ("pf", "pa", "margin", "total") for window in ROLL_WINDOWS],
    *[f"away_{stat}_avg_{window}" for stat in ("pf", "pa", "margin", "total") for window in ROLL_WINDOWS],
    *[f"home_{stat}_avg_{window}" for stat in ("plays", "pass_rate", "yards_per_play", "yards_per_pass", "yards_per_carry", "tds_per_play", "red_zone_td_rate", "red_zone_plays") for window in ROLL_WINDOWS],
    *[f"away_{stat}_avg_{window}" for stat in ("plays", "pass_rate", "yards_per_play", "yards_per_pass", "yards_per_carry", "tds_per_play", "red_zone_td_rate", "red_zone_plays") for window in ROLL_WINDOWS],
    "home_qb_injury_risk", "away_qb_injury_risk",
    "home_ol_injury_score", "away_ol_injury_score",
    "home_skill_injury_score", "away_skill_injury_score",
    "home_total_injury_score", "away_total_injury_score",
]


def _clean_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _load_games(conn, cfg: GameFeatureConfig) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            game_id, season, week, season_type, game_date_et, start_ts_utc,
            UPPER(home_team_abbr) AS home_team_abbr,
            UPPER(away_team_abbr) AS away_team_abbr,
            CASE WHEN status='final' THEN home_score::float END AS home_score,
            CASE WHEN status='final' THEN away_score::float END AS away_score,
            -spread_line::float AS market_spread_home,
            total_line::float AS market_total,
            roof, surface, temp::float AS temp, wind::float AS wind
        FROM raw.nfl_games
        WHERE game_date_et IS NOT NULL
          AND home_team_abbr IS NOT NULL
          AND away_team_abbr IS NOT NULL
          AND (%(min_season)s IS NULL OR season >= %(min_season)s)
        ORDER BY game_date_et, COALESCE(start_ts_utc, game_date_et::timestamptz), game_id
        """,
        conn,
        params={"min_season": cfg.min_season},
    )


def _load_team_usage(conn, cfg: GameFeatureConfig) -> dict[tuple[str, str], dict[str, float]]:
    rows = pd.read_sql(
        """
        SELECT
            game_id,
            UPPER(team_abbr) AS team_abbr,
            SUM(COALESCE(pass_attempts, 0))::float AS pass_attempts,
            SUM(COALESCE(carries, 0))::float AS carries,
            SUM(COALESCE(passing_yards, 0))::float AS passing_yards,
            SUM(COALESCE(rushing_yards, 0))::float AS rushing_yards,
            SUM(COALESCE(passing_tds, 0) + COALESCE(rushing_tds, 0))::float AS offensive_tds,
            SUM(COALESCE(red_zone_pass_attempts, 0) + COALESCE(red_zone_carries, 0))::float AS red_zone_plays,
            SUM(COALESCE(red_zone_pass_tds, 0) + COALESCE(red_zone_rush_tds, 0))::float AS red_zone_tds
        FROM raw.nfl_player_gamelogs
        WHERE game_id IS NOT NULL
          AND team_abbr IS NOT NULL
          AND (%(min_season)s IS NULL OR season >= %(min_season)s)
        GROUP BY game_id, UPPER(team_abbr)
        """,
        conn,
        params={"min_season": cfg.min_season},
    )
    out: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in rows.iterrows():
        game_id = str(row.get("game_id") or "")
        team = normalize_team(row.get("team_abbr"))
        if not game_id or not team:
            continue
        pass_attempts = _clean_float(row.get("pass_attempts")) or 0.0
        carries = _clean_float(row.get("carries")) or 0.0
        plays = pass_attempts + carries
        yards = (_clean_float(row.get("passing_yards")) or 0.0) + (_clean_float(row.get("rushing_yards")) or 0.0)
        passing_yards = _clean_float(row.get("passing_yards")) or 0.0
        rushing_yards = _clean_float(row.get("rushing_yards")) or 0.0
        offensive_tds = _clean_float(row.get("offensive_tds")) or 0.0
        red_zone_plays = _clean_float(row.get("red_zone_plays")) or 0.0
        red_zone_tds = _clean_float(row.get("red_zone_tds")) or 0.0
        out[(game_id, team)] = {
            "plays": plays if plays > 0 else None,
            "pass_rate": pass_attempts / plays if plays > 0 else None,
            "yards_per_play": yards / plays if plays > 0 else None,
            "yards_per_pass": passing_yards / pass_attempts if pass_attempts > 0 else None,
            "yards_per_carry": rushing_yards / carries if carries > 0 else None,
            "tds_per_play": offensive_tds / plays if plays > 0 else None,
            "red_zone_plays": red_zone_plays if red_zone_plays > 0 else None,
            "red_zone_td_rate": red_zone_tds / red_zone_plays if red_zone_plays > 0 else None,
        }
    return out


def _load_team_injury_context(conn, cfg: GameFeatureConfig) -> dict[tuple[int, int, str], dict[str, float]]:
    rows = pd.read_sql(
        """
        WITH observed AS (
            SELECT DISTINCT ON (g.season,g.week,o.payload->>'team_abbr',o.payload->>'player_id')
                (jsonb_populate_record(NULL::raw.nfl_injuries,o.payload)).*
            FROM raw.nfl_context_observations o
            JOIN raw.nfl_games g ON g.season=(o.payload->>'season')::integer
              AND g.week=(o.payload->>'week')::integer
              AND o.payload->>'team_abbr' IN (g.home_team_abbr,g.away_team_abbr)
              AND o.observed_at<LEAST(g.start_ts_utc,NOW())
            WHERE o.kind='nfl_injuries'
            ORDER BY g.season,g.week,o.payload->>'team_abbr',o.payload->>'player_id',o.observed_at DESC
        ), scored AS (
            SELECT
                season,
                week,
                UPPER(team_abbr) AS team_abbr,
                UPPER(COALESCE(position, '')) AS position,
                GREATEST(
                    CASE
                        WHEN LOWER(COALESCE(report_status, '')) ~ 'injured reserve|reserve|out' THEN 1.00
                        WHEN LOWER(COALESCE(report_status, '')) LIKE '%%doubtful%%' THEN 0.85
                        WHEN LOWER(COALESCE(report_status, '')) LIKE '%%questionable%%' THEN 0.45
                        ELSE 0.0
                    END,
                    CASE
                        WHEN LOWER(COALESCE(practice_status, '')) ~ 'did not participate|dnp' THEN 0.65
                        WHEN LOWER(COALESCE(practice_status, '')) LIKE '%%limited%%' THEN 0.25
                        ELSE 0.0
                    END
                )::float AS injury_score
            FROM observed
            WHERE season IS NOT NULL
              AND week IS NOT NULL
              AND team_abbr IS NOT NULL
              AND (%(min_season)s IS NULL OR season >= %(min_season)s)
        )
        SELECT
            season,
            week,
            team_abbr,
            MAX(CASE WHEN position = 'QB' THEN injury_score ELSE 0.0 END)::float AS qb_injury_risk,
            SUM(CASE WHEN position IN ('C', 'G', 'T', 'OL', 'OG', 'OT') THEN injury_score ELSE 0.0 END)::float AS ol_injury_score,
            SUM(CASE WHEN position IN ('RB', 'WR', 'TE') THEN injury_score ELSE 0.0 END)::float AS skill_injury_score,
            SUM(injury_score)::float AS total_injury_score
        FROM scored
        GROUP BY season, week, team_abbr
        """,
        conn,
        params={"min_season": cfg.min_season},
    )
    out: dict[tuple[int, int, str], dict[str, float]] = {}
    for _, row in rows.iterrows():
        team = normalize_team(row.get("team_abbr"))
        if not team:
            continue
        try:
            key = (int(row["season"]), int(row["week"]), team)
        except Exception:
            continue
        out[key] = {
            "qb_injury_risk": _clean_float(row.get("qb_injury_risk")) or 0.0,
            "ol_injury_score": _clean_float(row.get("ol_injury_score")) or 0.0,
            "skill_injury_score": _clean_float(row.get("skill_injury_score")) or 0.0,
            "total_injury_score": _clean_float(row.get("total_injury_score")) or 0.0,
        }
    return out


def _team_history_features(history: list[dict[str, Any]], prefix: str, game_date: pd.Timestamp | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        f"{prefix}_game_number": len(history),
        f"{prefix}_rest_days": None,
    }
    if history and game_date is not None and not pd.isna(game_date):
        last_date = pd.to_datetime(history[-1]["game_date_et"], errors="coerce")
        if not pd.isna(last_date):
            out[f"{prefix}_rest_days"] = int((game_date.normalize() - last_date.normalize()).days)
    hist_df = pd.DataFrame(history)
    for stat in ("pf", "pa", "margin", "total", "plays", "pass_rate", "yards_per_play", "yards_per_pass", "yards_per_carry", "tds_per_play", "red_zone_td_rate", "red_zone_plays"):
        values = pd.to_numeric(hist_df.get(stat, pd.Series(dtype=float)), errors="coerce").dropna()
        for window in ROLL_WINDOWS:
            out[f"{prefix}_{stat}_avg_{window}"] = float(values.tail(window).mean()) if len(values) else None
    return out


def build_game_feature_frame(conn, cfg: GameFeatureConfig) -> pd.DataFrame:
    games = _load_games(conn, cfg)
    if games.empty:
        return games
    team_usage = _load_team_usage(conn, cfg)
    team_injuries = _load_team_injury_context(conn, cfg)
    games = games.copy()
    games["home_team_abbr"] = games["home_team_abbr"].map(normalize_team)
    games["away_team_abbr"] = games["away_team_abbr"].map(normalize_team)
    games["game_date_et"] = pd.to_datetime(games["game_date_et"], errors="coerce")
    histories: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []

    for _, date_games in games.groupby(games["game_date_et"].dt.date, sort=True):
        completed_to_add: list[tuple[str, dict[str, Any]]] = []
        for _, game in date_games.sort_values(["start_ts_utc", "game_id"], na_position="last").iterrows():
            home = normalize_team(game.get("home_team_abbr"))
            away = normalize_team(game.get("away_team_abbr"))
            if not home or not away:
                continue
            game_date = pd.to_datetime(game.get("game_date_et"), errors="coerce")
            home_score = _clean_float(game.get("home_score"))
            away_score = _clean_float(game.get("away_score"))
            try:
                injury_key_home = (int(game.get("season")), int(game.get("week")), home)
                injury_key_away = (int(game.get("season")), int(game.get("week")), away)
            except Exception:
                injury_key_home = None
                injury_key_away = None
            home_inj = team_injuries.get(injury_key_home, {}) if injury_key_home else {}
            away_inj = team_injuries.get(injury_key_away, {}) if injury_key_away else {}
            home_margin = None
            total_points = None
            if home_score is not None and away_score is not None:
                home_margin = home_score - away_score
                total_points = home_score + away_score
            row = {
                "game_id": game.get("game_id"),
                "season": game.get("season"),
                "week": game.get("week"),
                "season_type": game.get("season_type"),
                "game_date_et": game_date.date() if not pd.isna(game_date) else None,
                "start_ts_utc": game.get("start_ts_utc"),
                "home_team_abbr": home,
                "away_team_abbr": away,
                "home_score": home_score,
                "away_score": away_score,
                "home_margin": home_margin,
                "total_points_actual": total_points,
                "market_spread_home": _clean_float(game.get("market_spread_home")),
                "market_total": _clean_float(game.get("market_total")),
                "roof": game.get("roof"),
                "surface": game.get("surface"),
                "temp": _clean_float(game.get("temp")),
                "wind": _clean_float(game.get("wind")),
                "home_qb_injury_risk": home_inj.get("qb_injury_risk"),
                "away_qb_injury_risk": away_inj.get("qb_injury_risk"),
                "home_ol_injury_score": home_inj.get("ol_injury_score"),
                "away_ol_injury_score": away_inj.get("ol_injury_score"),
                "home_skill_injury_score": home_inj.get("skill_injury_score"),
                "away_skill_injury_score": away_inj.get("skill_injury_score"),
                "home_total_injury_score": home_inj.get("total_injury_score"),
                "away_total_injury_score": away_inj.get("total_injury_score"),
            }
            row.update(_team_history_features(histories.get(home, []), "home", game_date))
            row.update(_team_history_features(histories.get(away, []), "away", game_date))
            rows.append(row)

            if home_score is not None and away_score is not None:
                home_usage = team_usage.get((str(game.get("game_id")), home), {})
                away_usage = team_usage.get((str(game.get("game_id")), away), {})
                completed_to_add.append((home, {
                    "game_date_et": game_date,
                    "pf": home_score,
                    "pa": away_score,
                    "margin": home_score - away_score,
                    "total": home_score + away_score,
                    "plays": home_usage.get("plays"),
                    "pass_rate": home_usage.get("pass_rate"),
                    "yards_per_play": home_usage.get("yards_per_play"),
                    "yards_per_pass": home_usage.get("yards_per_pass"),
                    "yards_per_carry": home_usage.get("yards_per_carry"),
                    "tds_per_play": home_usage.get("tds_per_play"),
                    "red_zone_td_rate": home_usage.get("red_zone_td_rate"),
                    "red_zone_plays": home_usage.get("red_zone_plays"),
                }))
                completed_to_add.append((away, {
                    "game_date_et": game_date,
                    "pf": away_score,
                    "pa": home_score,
                    "margin": away_score - home_score,
                    "total": home_score + away_score,
                    "plays": away_usage.get("plays"),
                    "pass_rate": away_usage.get("pass_rate"),
                    "yards_per_play": away_usage.get("yards_per_play"),
                    "yards_per_pass": away_usage.get("yards_per_pass"),
                    "yards_per_carry": away_usage.get("yards_per_carry"),
                    "tds_per_play": away_usage.get("tds_per_play"),
                    "red_zone_td_rate": away_usage.get("red_zone_td_rate"),
                    "red_zone_plays": away_usage.get("red_zone_plays"),
                }))
        for team, rec in completed_to_add:
            histories.setdefault(team, []).append(rec)

    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[FEATURE_COLUMNS]


def _row_tuple(row: pd.Series) -> tuple:
    out: list[Any] = []
    for col in FEATURE_COLUMNS:
        value = row.get(col)
        try:
            if pd.isna(value):
                value = None
        except Exception:
            pass
        out.append(value)
    return tuple(out)


def write_game_features(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = [_row_tuple(row) for _, row in df.iterrows()]
    value_cols = [col for col in FEATURE_COLUMNS if col != "game_id"]
    update_sql = ", ".join(f"{col} = EXCLUDED.{col}" for col in value_cols)
    # Skip no-op rewrites so updated_at_utc keeps meaning "last content change".
    changed_sql = (
        f"({', '.join(f'features.nfl_game_training_features.{col}' for col in value_cols)})"
        f" IS DISTINCT FROM ({', '.join(f'EXCLUDED.{col}' for col in value_cols)})"
    )
    sql = f"""
        INSERT INTO features.nfl_game_training_features ({", ".join(FEATURE_COLUMNS)})
        VALUES %s
        ON CONFLICT (game_id) DO UPDATE SET
            {update_sql},
            updated_at_utc = NOW()
        WHERE {changed_sql}
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    return len(rows)


def build_and_write(cfg: GameFeatureConfig) -> dict[str, Any]:
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_schema(conn)
        df = build_game_feature_frame(conn, cfg)
        rows = write_game_features(conn, df)
    return {"status": "ok", "game_feature_rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NFL game-level training features")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--min-season", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    print(json.dumps(build_and_write(GameFeatureConfig(pg_dsn=args.pg_dsn, min_season=args.min_season)), indent=2))


if __name__ == "__main__":
    main()
