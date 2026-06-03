"""Generate MLB prop failure diagnostics from replay history.

This report is meant to answer "why is this market weak?" by slicing graded
prop picks by stat, side, line bucket, price bucket, model family, team, player,
and probability buckets.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from .prop_replay import ensure_prop_replay_schema
from .side_recalibration import price_bucket

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"


@dataclass(frozen=True)
class DiagnosticConfig:
    pg_dsn: str = _PG_DSN
    lookback_days: int = 365
    min_rows: int = 10
    top_n: int = 20
    out_file: Path | None = None


SQL = """
SELECT
    run_id,
    game_date_et,
    game_slug,
    player_id,
    player_name,
    team_abbr,
    stat,
    side,
    line_bucket,
    model_family,
    market_line,
    market_price,
    model_prob_side,
    prob_edge_vs_market,
    ev,
    bankroll_candidate,
    bankroll_reasons,
    result_status,
    won,
    push,
    profit_units,
    clv_line,
    clv_price
FROM bets.mlb_prop_prediction_replay
WHERE game_date_et >= %(cutoff)s
  AND result_status = 'graded'
  AND side IN ('over','under')
"""


def _load(cfg: DiagnosticConfig) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)).isoformat()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_prop_replay_schema(conn)
        df = pd.read_sql(SQL, conn, params={"cutoff": cutoff})
    if df.empty:
        return df
    df["price_bucket"] = df["market_price"].map(price_bucket)
    df["priced"] = pd.to_numeric(df["market_price"], errors="coerce").notna()
    p = pd.to_numeric(df["model_prob_side"], errors="coerce")
    bins = [0, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 1.01]
    labels = ["<45", "45-50", "50-55", "55-60", "60-65", "65-70", "70+"]
    df["prob_bucket"] = pd.cut(p, bins=bins, labels=labels, include_lowest=True)
    df["settled"] = ~df["push"].fillna(False).astype(bool)
    df["won_num"] = df["won"].fillna(False).astype(bool).astype(float)
    df["profit_units"] = pd.to_numeric(df["profit_units"], errors="coerce")
    df["clv_line"] = pd.to_numeric(df["clv_line"], errors="coerce")
    df["clv_price"] = pd.to_numeric(df["clv_price"], errors="coerce")
    df["model_prob_side"] = p
    return df


def _summarize(df: pd.DataFrame, cols: list[str], cfg: DiagnosticConfig) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = (
        df.groupby(cols, dropna=False, observed=True)
        .agg(
            n=("won", "size"),
            settled=("settled", "sum"),
            priced=("priced", "sum"),
            wins=("won_num", "sum"),
            units=("profit_units", "sum"),
            avg_prob=("model_prob_side", "mean"),
            avg_ev=("ev", "mean"),
            avg_line_clv=("clv_line", "mean"),
            avg_price_clv=("clv_price", "mean"),
            clv_price_beat=("clv_price", lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean()),
        )
        .reset_index()
    )
    g = g[g["settled"] >= cfg.min_rows].copy()
    if g.empty:
        return g
    g["win_rate"] = g["wins"] / g["settled"]
    g["priced_rate"] = g["priced"] / g["settled"]
    g["roi"] = g["units"] / g["priced"].replace(0, np.nan)
    g["cal_error"] = g["win_rate"] - g["avg_prob"]
    return g.sort_values(["roi", "n"], ascending=[True, False])


def _summarize_bookability(df: pd.DataFrame, cols: list[str], cfg: DiagnosticConfig) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = (
        df.groupby(cols, dropna=False, observed=True)
        .agg(
            n=("won", "size"),
            settled=("settled", "sum"),
            priced=("priced", "sum"),
            wins=("won_num", "sum"),
            avg_prob=("model_prob_side", "mean"),
        )
        .reset_index()
    )
    g = g[g["settled"] >= cfg.min_rows].copy()
    if g.empty:
        return g
    g["priced_rate"] = g["priced"] / g["settled"]
    g["win_rate"] = g["wins"] / g["settled"]
    return g.sort_values(["priced_rate", "n"], ascending=[True, False])


def _fmt_table(df: pd.DataFrame, cols: list[str], top_n: int) -> str:
    if df.empty:
        return "_No buckets met the sample threshold._\n"
    out = df.head(top_n).copy()
    for c in ["win_rate", "priced_rate", "roi", "avg_prob", "avg_ev", "cal_error", "clv_price_beat"]:
        if c in out:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{float(v)*100:+.1f}%" if c in {"roi", "avg_ev", "cal_error"} else f"{float(v)*100:.1f}%")
    for c in ["avg_line_clv", "avg_price_clv", "units"]:
        if c in out:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{float(v):+.2f}")
    show_cols = cols + [
        "n",
        "settled",
        "priced",
        "priced_rate",
        "win_rate",
        "units",
        "roi",
        "avg_prob",
        "cal_error",
        "avg_line_clv",
        "avg_price_clv",
        "clv_price_beat",
    ]
    show_cols = [c for c in show_cols if c in out.columns]
    view = out[show_cols].fillna("").astype(str)
    header = "| " + " | ".join(show_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(show_cols)) + " |"
    rows = [
        "| " + " | ".join(str(row[c]).replace("|", "\\|") for c in show_cols) + " |"
        for _, row in view.iterrows()
    ]
    return "\n".join([header, sep, *rows]) + "\n"


def build_report(cfg: DiagnosticConfig) -> str:
    df = _load(cfg)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    lines = [
        "# MLB Prop Failure Diagnostics",
        "",
        f"- Generated: {generated}",
        f"- Lookback days: {cfg.lookback_days}",
        f"- Graded rows: {len(df)}",
        f"- Min bucket rows: {cfg.min_rows}",
        "",
    ]
    if df.empty:
        lines.append("_No graded replay rows found. Run backfill_prop_prediction_replay.py with --grade first._")
        return "\n".join(lines)
    sections = [
        ("Worst Stat / Side", ["stat", "side"]),
        ("Worst Line Buckets", ["stat", "side", "line_bucket"]),
        ("Worst Price Buckets", ["stat", "side", "price_bucket"]),
        ("Worst Model Families", ["stat", "side", "model_family"]),
        ("Worst Probability Buckets", ["stat", "side", "prob_bucket"]),
        ("Worst Teams", ["stat", "side", "team_abbr"]),
        ("Worst Players", ["stat", "side", "player_name"]),
        ("Worst Bankroll Reasons", ["stat", "side", "bankroll_reasons"]),
    ]
    for title, cols in sections:
        lines.extend([f"## {title}", ""])
        lines.append(_fmt_table(_summarize(df, cols, cfg), cols, cfg.top_n))
        lines.append("")
    lines.extend(["## Bookability Gaps", ""])
    bookability_cols = ["stat", "side", "price_bucket"]
    lines.append(_fmt_table(_summarize_bookability(df, bookability_cols, cfg), bookability_cols, cfg.top_n))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MLB prop failure diagnostics")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out-file", default=None)
    args = parser.parse_args()
    cfg = DiagnosticConfig(
        pg_dsn=args.pg_dsn,
        lookback_days=args.lookback_days,
        min_rows=args.min_rows,
        top_n=args.top_n,
        out_file=Path(args.out_file) if args.out_file else None,
    )
    report = build_report(cfg)
    if cfg.out_file:
        cfg.out_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_file.write_text(report, encoding="utf-8")
        print(f"Wrote {cfg.out_file}")
    else:
        print(report)


if __name__ == "__main__":
    main()
