"""Grade locked NFL game and player-prop predictions."""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN

log = logging.getLogger("nfl_pipeline.grade_predictions")


@dataclass(frozen=True)
class GradeConfig:
    pg_dsn: str = PG_DSN
    game_date: date | None = None


def _clean_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _profit(result: str | None, price: Any) -> float | None:
    if result in {"push", "void_nonparticipant"}:
        return 0.0
    if result != "win":
        return -1.0 if result == "loss" else None
    price_f = _clean_float(price)
    if price_f is None or price_f == 0:
        return None
    return price_f / 100.0 if price_f > 0 else 100.0 / abs(price_f)


def _grade_side(actual: float | None, line: float | None, side: str | None) -> str | None:
    if actual is None or line is None or side not in {"over", "under", "home", "away"}:
        return None
    value = actual + line if side in {"home", "away"} else actual - line
    if abs(value) < 1e-9:
        return "push"
    if side in {"over", "home", "away"}:
        return "win" if value > 0 else "loss"
    return "win" if value < 0 else "loss"


def grade_game_predictions(conn, cfg: GradeConfig) -> int:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                p.id, p.game_date_et, p.game_id, p.market, p.side, p.book,
                p.line::float AS line, p.price,
                g.home_score::float AS home_score,
                g.away_score::float AS away_score
            FROM bets.nfl_game_predictions p
            JOIN raw.nfl_games g ON g.game_id = p.game_id
            WHERE (%(game_date)s IS NULL OR p.game_date_et = %(game_date)s)
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
              AND g.status = 'final'
            """,
            {"game_date": cfg.game_date},
        )
        rows = cur.fetchall()
    out_rows: list[tuple] = []
    for row in rows:
        home_score = _clean_float(row["home_score"])
        away_score = _clean_float(row["away_score"])
        if home_score is None or away_score is None:
            continue
        home_margin = home_score - away_score
        total = home_score + away_score
        side = str(row["side"] or "").lower()
        actual = home_margin if side == "home" else -home_margin if side == "away" else total
        result = _grade_side(actual, _clean_float(row["line"]), side)
        out_rows.append((
            row["id"], row["game_date_et"], row["game_id"], row["market"], row["side"],
            row["book"], row["line"], row["price"], home_margin, total,
            result or "pending", _profit(result, row["price"]),
        ))
    if out_rows:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO bets.nfl_game_prediction_results (
                    prediction_id, game_date_et, game_id, market, side, book,
                    line, price, actual_home_margin, actual_total_points,
                    result, profit_per_unit
                )
                VALUES %s
                ON CONFLICT (prediction_id) DO UPDATE SET
                    actual_home_margin = EXCLUDED.actual_home_margin,
                    actual_total_points = EXCLUDED.actual_total_points,
                    result = EXCLUDED.result,
                    profit_per_unit = EXCLUDED.profit_per_unit,
                    updated_at_utc = NOW()
                WHERE (bets.nfl_game_prediction_results.actual_home_margin,
                       bets.nfl_game_prediction_results.actual_total_points,
                       bets.nfl_game_prediction_results.result,
                       bets.nfl_game_prediction_results.profit_per_unit)
                  IS DISTINCT FROM (EXCLUDED.actual_home_margin, EXCLUDED.actual_total_points,
                                    EXCLUDED.result, EXCLUDED.profit_per_unit)
                """,
                out_rows,
                page_size=1000,
            )
        conn.commit()
    return len(out_rows)


def grade_player_prop_predictions(conn, cfg: GradeConfig) -> int:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                p.id, p.game_date_et, p.game_id, p.player_id, p.player_name,
                p.stat, p.side, p.line::float AS line, p.price,
                l.passing_yards::float AS passing_yards,
                l.passing_tds::float AS passing_tds,
                l.rushing_yards::float AS rushing_yards,
                l.rushing_tds::float AS rushing_tds,
                l.receiving_yards::float AS receiving_yards,
                l.receiving_tds::float AS receiving_tds,
                l.offense_snaps::float AS offense_snaps,
                prior.result AS prior_result,
                (COALESCE(l.pass_attempts,0)+COALESCE(l.carries,0)+COALESCE(l.targets,0))::float AS offensive_actions
            FROM bets.nfl_player_prop_predictions p
            JOIN raw.nfl_player_gamelogs l
              ON l.game_id = p.game_id
             AND l.player_id = p.player_id
             AND l.team_abbr = p.team_abbr
            JOIN raw.nfl_games g ON g.game_id = p.game_id AND g.status = 'final'
            LEFT JOIN bets.nfl_player_prop_prediction_results prior ON prior.prediction_id=p.id
            WHERE (%(game_date)s IS NULL OR p.game_date_et = %(game_date)s)
              AND p.side IN ('over', 'under')
              AND p.line IS NOT NULL
            """,
            {"game_date": cfg.game_date},
        )
        rows = cur.fetchall()
    out_rows: list[tuple] = []
    for row in rows:
        stat = str(row["stat"] or "")
        actual = _clean_float(row.get(stat))
        result = _grade_side(actual, _clean_float(row["line"]), str(row["side"] or "").lower())
        if not ((row.get('offense_snaps') or 0)>0 or (row.get('offensive_actions') or 0)>0):
            # Do not turn an unverified appearance into a winning under.
            actual, result = None, ('void_nonparticipant' if row.get('prior_result')=='void_nonparticipant'
                                    else 'pending_participation_review')
        out_rows.append((
            row["id"], row["game_date_et"], row["game_id"], row["player_id"], row["player_name"],
            stat, row["side"], row["line"], row["price"], actual,
            result or "pending", _profit(result, row["price"]),
        ))
    if out_rows:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO bets.nfl_player_prop_prediction_results (
                    prediction_id, game_date_et, game_id, player_id, player_name,
                    stat, side, line, price, actual_stat, result, profit_per_unit
                )
                VALUES %s
                ON CONFLICT (prediction_id) DO UPDATE SET
                    actual_stat = EXCLUDED.actual_stat,
                    result = EXCLUDED.result,
                    profit_per_unit = EXCLUDED.profit_per_unit,
                    updated_at_utc = NOW()
                WHERE (bets.nfl_player_prop_prediction_results.actual_stat,
                       bets.nfl_player_prop_prediction_results.result,
                       bets.nfl_player_prop_prediction_results.profit_per_unit)
                  IS DISTINCT FROM (EXCLUDED.actual_stat, EXCLUDED.result, EXCLUDED.profit_per_unit)
                """,
                out_rows,
                page_size=1000,
            )
        conn.commit()
    return len(out_rows)


def grade(cfg: GradeConfig) -> dict[str, Any]:
    with psycopg2.connect(cfg.pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout='120s'")
            cur.execute("SET lock_timeout='5s'")
        game_rows = grade_game_predictions(conn, cfg)
        prop_rows = grade_player_prop_predictions(conn, cfg)
    return {"status": "ok", "game_predictions_graded": game_rows, "prop_predictions_graded": prop_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade NFL predictions against final results")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    result = grade(GradeConfig(
        pg_dsn=args.pg_dsn,
        game_date=date.fromisoformat(args.date) if args.date else None,
    ))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
