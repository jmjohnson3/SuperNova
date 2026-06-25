# src/mlb_pipeline/modeling/diagnose_market_weakness.py
"""
Diagnose weak MLB betting markets by side, bucket, price, and calibration bias.

Use this before disabling any market. It answers "why is this weak?" with the
same saved bet-side fields that update_outcomes.py grades.

Usage:
    python -m mlb_pipeline.modeling.diagnose_market_weakness
    python -m mlb_pipeline.modeling.diagnose_market_weakness --days 180 --min-bets 20
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Iterable

import psycopg2
import psycopg2.extras

from mlb_pipeline.db import PG_DSN
def _american_profit(price) -> float:
    if price is None:
        return 100.0 / 110.0
    p = float(price)
    if p > 0:
        return p / 100.0
    if p < 0:
        return 100.0 / abs(p)
    return 100.0 / 110.0


def _unit_profit(covered, price) -> float:
    return _american_profit(price) if bool(covered) else -1.0


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100.0:.1f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _avg(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _print_table(title: str, rows: list[dict], columns: list[tuple[str, str, int]]) -> None:
    print(f"\n{title}")
    print("  " + "  ".join(label.ljust(width) for label, _, width in columns))
    print("  " + "  ".join(("-" * width) for _, _, width in columns))
    for row in rows:
        parts = []
        for _, key, width in columns:
            value = row.get(key, "")
            parts.append(str(value).ljust(width))
        print("  " + "  ".join(parts))


def _summarize_groups(
    rows: list[dict],
    *,
    side_key: str,
    covered_key: str,
    price_key: str,
    edge_key: str,
    bucket_key: str | None = None,
    min_bets: int = 1,
) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        side = row.get(side_key)
        covered = row.get(covered_key)
        if side is None or covered is None:
            continue
        bucket = str(row.get(bucket_key) or "all") if bucket_key else "all"
        groups.setdefault((str(side), bucket), []).append(row)

    out = []
    for (side, bucket), subset in sorted(groups.items()):
        n = len(subset)
        if n < min_bets:
            continue
        wins = sum(1 for r in subset if r[covered_key])
        profits = [_unit_profit(r[covered_key], r.get(price_key)) for r in subset]
        out.append({
            "side": side,
            "bucket": bucket,
            "n": n,
            "wr": _pct(wins / n),
            "roi": _pct(sum(profits) / n),
            "avg_edge": _num(_avg(abs(float(r[edge_key])) for r in subset if r.get(edge_key) is not None)),
            "avg_price": _num(_avg(float(r[price_key]) for r in subset if r.get(price_key) is not None), 0),
        })
    return out


def _load_game_rows(conn, cutoff: str) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                game_date_et,
                pred_run_diff,
                pred_total,
                actual_run_diff,
                actual_total,
                market_run_line,
                market_total,
                edge_run_line,
                edge_total,
                run_line_bet_side,
                total_bet_side,
                run_line_covered,
                total_covered,
                market_rl_price,
                market_total_price,
                clv_rl_price,
                clv_total_price,
                CASE
                    WHEN run_line_bet_side = 'away' AND market_run_line > 0 THEN 'away favorite'
                    WHEN market_rl_price < -180 THEN 'heavy juice'
                    WHEN pred_run_diff <= -3.0 THEN 'pred away by 3+'
                    WHEN pred_run_diff < 0 THEN 'pred away close'
                    WHEN pred_run_diff < 1.5 THEN 'pred home close'
                    ELSE 'pred home by 1.5+'
                END AS run_line_bucket,
                CASE
                    WHEN market_total >= 10.5 THEN '10.5+'
                    WHEN market_total >= 9.5 THEN '9.5-10.0'
                    WHEN market_total >= 8.5 THEN '8.5-9.0'
                    ELSE '<=8.0'
                END AS total_bucket
            FROM bets.mlb_game_predictions
            WHERE game_date_et >= %s
              AND (
                    (run_line_bet_side IS NOT NULL AND run_line_covered IS NOT NULL)
                 OR (total_bet_side IS NOT NULL AND total_covered IS NOT NULL)
              )
            """,
            (cutoff,),
        )
        return list(cur.fetchall())


def _load_prop_rows(conn, cutoff: str) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                stat,
                COALESCE(bet_side, CASE WHEN edge >= 0 THEN 'over' ELSE 'under' END) AS side,
                CASE
                    WHEN line_bucket IS NOT NULL AND line_bucket <> 'unknown' THEN line_bucket
                    WHEN stat = 'batter_total_bases' AND book_line < 1.0 THEN 'TB 0.5'
                    WHEN stat = 'batter_total_bases' AND book_line < 2.0 THEN 'TB 1.5'
                    WHEN stat = 'batter_total_bases' THEN 'TB 2.5+'
                    WHEN stat = 'batter_hits' AND book_line < 1.0 THEN 'H 0.5'
                    WHEN stat = 'batter_hits' AND book_line < 2.0 THEN 'H 1.5'
                    WHEN stat = 'batter_hits' THEN 'H 2.5+'
                    WHEN stat = 'pitcher_strikeouts' AND book_line < 4.5 THEN 'K <4.5'
                    WHEN stat = 'pitcher_strikeouts' AND book_line < 6.5 THEN 'K 4.5-6.0'
                    WHEN stat = 'pitcher_strikeouts' AND book_line < 8.5 THEN 'K 6.5-8.0'
                    WHEN stat = 'pitcher_strikeouts' THEN 'K 8.5+'
                    WHEN stat = 'batter_home_runs' AND book_line < 1.0 THEN 'HR 0.5'
                    WHEN stat = 'batter_home_runs' THEN 'HR 1.5+'
                    ELSE 'unknown'
                END AS bucket,
                COALESCE(
                    edge_type,
                    CASE
                        WHEN pred_prob_over IS NOT NULL THEN 'probability'
                        ELSE 'count'
                    END
                ) AS edge_type,
                over_hit,
                book_line,
                edge,
                bet_price,
                pred_count,
                pred_value,
                actual_value,
                CASE
                    WHEN COALESCE(bet_side, CASE WHEN edge >= 0 THEN 'over' ELSE 'under' END) = 'over'
                        THEN over_hit
                    ELSE NOT over_hit
                END AS covered
            FROM bets.mlb_prop_predictions
            WHERE game_date_et >= %s
              AND over_hit IS NOT NULL
              AND edge IS NOT NULL
              AND stat IN ('pitcher_strikeouts', 'batter_hits', 'batter_total_bases', 'batter_home_runs')
            """,
            (cutoff,),
        )
        return list(cur.fetchall())


def _prop_summaries(rows: list[dict], min_bets: int) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (
            str(row["stat"]),
            str(row["side"]),
            str(row["bucket"]),
            str(row.get("edge_type") or "unknown"),
        )
        groups.setdefault(key, []).append(row)

    out = []
    for (stat, side, bucket, edge_type), subset in sorted(groups.items()):
        n = len(subset)
        if n < min_bets:
            continue
        wins = sum(1 for r in subset if r["covered"])
        profits = [_unit_profit(r["covered"], r.get("bet_price")) for r in subset]
        pred_minus_actual = _avg(
            float((r.get("pred_count") if r.get("pred_count") is not None else r.get("pred_value"))) - float(r["actual_value"])
            for r in subset
            if r.get("actual_value") is not None
            and (r.get("pred_count") is not None or r.get("pred_value") is not None)
        )
        out.append({
            "stat": stat,
            "side": side,
            "bucket": bucket,
            "edge_type": edge_type,
            "n": n,
            "wr": _pct(wins / n),
            "roi": _pct(sum(profits) / n),
            "avg_edge": _num(_avg(abs(float(r["edge"])) for r in subset if r.get("edge") is not None)),
            "pred_bias": _num(pred_minus_actual),
        })
    return out


def run(days: int, min_bets: int) -> None:
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with psycopg2.connect(PG_DSN) as conn:
        game_rows = _load_game_rows(conn, cutoff)
        prop_rows = _load_prop_rows(conn, cutoff)

    print(f"\nMLB MARKET WEAKNESS DIAGNOSIS since {cutoff} ({days}d)")
    print("ROI uses stored American prices when present; otherwise assumes -110.")

    _print_table(
        "Game Run Line by Side",
        _summarize_groups(
            game_rows,
            side_key="run_line_bet_side",
            covered_key="run_line_covered",
            price_key="market_rl_price",
            edge_key="edge_run_line",
            min_bets=min_bets,
        ),
        [("Side", "side", 8), ("Bucket", "bucket", 12), ("Bets", "n", 6), ("WR", "wr", 8), ("ROI", "roi", 9), ("AvgEdge", "avg_edge", 8), ("AvgPx", "avg_price", 7)],
    )
    _print_table(
        "Game Run Line Buckets",
        _summarize_groups(
            game_rows,
            side_key="run_line_bet_side",
            covered_key="run_line_covered",
            price_key="market_rl_price",
            edge_key="edge_run_line",
            bucket_key="run_line_bucket",
            min_bets=min_bets,
        ),
        [("Side", "side", 8), ("Bucket", "bucket", 19), ("Bets", "n", 6), ("WR", "wr", 8), ("ROI", "roi", 9), ("AvgEdge", "avg_edge", 8), ("AvgPx", "avg_price", 7)],
    )
    _print_table(
        "Game Totals by Side",
        _summarize_groups(
            game_rows,
            side_key="total_bet_side",
            covered_key="total_covered",
            price_key="market_total_price",
            edge_key="edge_total",
            min_bets=min_bets,
        ),
        [("Side", "side", 8), ("Bucket", "bucket", 12), ("Bets", "n", 6), ("WR", "wr", 8), ("ROI", "roi", 9), ("AvgEdge", "avg_edge", 8), ("AvgPx", "avg_price", 7)],
    )
    _print_table(
        "Game Total Buckets",
        _summarize_groups(
            game_rows,
            side_key="total_bet_side",
            covered_key="total_covered",
            price_key="market_total_price",
            edge_key="edge_total",
            bucket_key="total_bucket",
            min_bets=1,
        ),
        [("Side", "side", 8), ("Bucket", "bucket", 12), ("Bets", "n", 6), ("WR", "wr", 8), ("ROI", "roi", 9), ("AvgEdge", "avg_edge", 8), ("AvgPx", "avg_price", 7)],
    )
    _print_table(
        "Player Props by Stat/Side/Bucket",
        _prop_summaries(prop_rows, min_bets),
        [
            ("Stat", "stat", 20),
            ("Side", "side", 7),
            ("Bucket", "bucket", 10),
            ("Type", "edge_type", 11),
            ("Bets", "n", 6),
            ("WR", "wr", 8),
            ("ROI", "roi", 9),
            ("AvgEdge", "avg_edge", 8),
            ("PredBias", "pred_bias", 8),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose weak MLB betting markets")
    parser.add_argument("--days", type=int, default=180, help="Lookback window in days")
    parser.add_argument("--min-bets", type=int, default=20, help="Minimum rows per group")
    args = parser.parse_args()
    run(days=args.days, min_bets=args.min_bets)


if __name__ == "__main__":
    main()
