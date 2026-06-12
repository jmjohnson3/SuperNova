"""Post-mortem report for a single MLB prop slate."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2
import psycopg2.extras

from .side_recalibration import price_bucket, prop_line_bucket, prop_line_surface

_ET = ZoneInfo("America/New_York")
_PG_DSN = "postgresql://josh:password@localhost:5432/nba"


@dataclass(frozen=True)
class SlatePostmortemConfig:
    pg_dsn: str = _PG_DSN
    report_date: date | None = None
    min_ev: float = 0.02
    top_n: int = 30
    out: str | None = None


SQL = """
SELECT
    game_date_et,
    game_slug,
    player_id,
    player_name,
    team_abbr,
    stat,
    bet_side AS side,
    book_line::float AS line,
    bet_price::float AS price,
    bookmaker_key,
    model_family,
    pred_count::float AS pred_count,
    pred_prob_over::float AS pred_prob_over,
    ev::float AS ev,
    bankroll_candidate,
    bankroll_tier,
    bankroll_reasons,
    stake_pct::float AS stake_pct,
    actual_value::float AS actual_value,
    over_hit,
    closing_line::float AS closing_line,
    closing_price::float AS closing_price,
    clv_line::float AS clv_line,
    clv_price::float AS clv_price,
    beat_clv_line,
    beat_clv_price,
    COALESCE(clv_valid, false) AS clv_valid,
    clv_status,
    clv_unknown_reason,
    COALESCE(is_active, TRUE) AS is_active,
    stale_reason,
    prop_offer_id,
    prediction_key
FROM bets.mlb_prop_predictions
WHERE game_date_et = %(report_date)s
  AND bet_side IN ('over','under')
  AND book_line IS NOT NULL
"""


def _ensure_prediction_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE IF EXISTS bets.mlb_prop_predictions
                ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE,
                ADD COLUMN IF NOT EXISTS stale_reason TEXT,
                ADD COLUMN IF NOT EXISTS clv_valid BOOLEAN,
                ADD COLUMN IF NOT EXISTS clv_status TEXT,
                ADD COLUMN IF NOT EXISTS clv_unknown_reason TEXT;
            """
        )
    conn.commit()


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return pd.DataFrame([dict(row) for row in cur.fetchall()])


def _american_profit_mult(price: Any) -> float | None:
    p = _clean_float(price)
    if p is None or p == 0:
        return None
    return p / 100.0 if p > 0 else 100.0 / abs(p)


def _side_prob(row: pd.Series) -> float | None:
    p_over = _clean_float(row.get("pred_prob_over"))
    if p_over is None:
        return None
    return p_over if row.get("side") == "over" else 1.0 - p_over


def _side_result(row: pd.Series) -> tuple[bool | None, bool]:
    actual = _clean_float(row.get("actual_value"))
    line = _clean_float(row.get("line"))
    side = row.get("side")
    if actual is None or line is None or side not in {"over", "under"}:
        return None, False
    if abs(actual - line) <= 1e-9:
        return None, True
    over = actual > line
    return (over if side == "over" else not over), False


def _profit_units(row: pd.Series, won: bool | None, push: bool) -> float | None:
    if push:
        return 0.0
    if won is None:
        return None
    mult = _american_profit_mult(row.get("price"))
    if mult is None:
        return None
    return mult if won else -1.0


def _load(cfg: SlatePostmortemConfig) -> pd.DataFrame:
    report_date = cfg.report_date or datetime.now(_ET).date()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        _ensure_prediction_schema(conn)
        df = _query_df(conn, SQL, {"report_date": report_date})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    df["line_bucket"] = [prop_line_bucket(stat, line) for stat, line in zip(df["stat"], df["line"])]
    df["price_bucket"] = df["price"].map(price_bucket)
    df["line_surface"] = [
        prop_line_surface(stat, side, line)
        for stat, side, line in zip(df["stat"], df["side"], df["line"])
    ]
    df["model_prob_side"] = df.apply(_side_prob, axis=1)
    results = df.apply(_side_result, axis=1)
    df["won"] = [r[0] for r in results]
    df["push"] = [r[1] for r in results]
    df["profit_units"] = [
        _profit_units(row, won, push)
        for (_, row), won, push in zip(df.iterrows(), df["won"], df["push"])
    ]
    return df


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0, "graded": 0, "wins": 0, "losses": 0, "pushes": 0,
            "units": None, "roi": None, "win_rate": None,
            "avg_clv_price": None, "clv_price_beat_rate": None, "clv_price_rows": 0,
        }
    settled = df[df["won"].notna() & ~df["push"].fillna(False).astype(bool)].copy()
    wins = int(settled["won"].astype(bool).sum()) if not settled.empty else 0
    losses = int(len(settled) - wins)
    profit = pd.to_numeric(df["profit_units"], errors="coerce").dropna()
    valid_clv = df["clv_valid"].fillna(False).astype(bool)
    clv_price = pd.to_numeric(df.loc[valid_clv, "clv_price"], errors="coerce").dropna()
    beat = pd.to_numeric(df.loc[valid_clv, "beat_clv_price"], errors="coerce").dropna()
    return {
        "rows": int(len(df)),
        "graded": int(len(settled)),
        "wins": wins,
        "losses": losses,
        "pushes": int(df["push"].fillna(False).astype(bool).sum()),
        "units": float(profit.sum()) if not profit.empty else None,
        "roi": float(profit.mean()) if not profit.empty else None,
        "win_rate": float(wins / (wins + losses)) if wins + losses else None,
        "avg_clv_price": float(clv_price.mean()) if not clv_price.empty else None,
        "clv_price_beat_rate": float(beat.mean()) if not beat.empty else None,
        "clv_price_rows": int(len(clv_price)),
    }


def _fmt_pct(value: Any, *, signed: bool = False) -> str:
    v = _clean_float(value)
    if v is None:
        return "-"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def _fmt_num(value: Any, digits: int = 2, *, signed: bool = False) -> str:
    v = _clean_float(value)
    if v is None:
        return "-"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(title for title, _ in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for _, key in cols) + " |")
    return "\n".join(lines)


def _metric_row(label: str, df: pd.DataFrame) -> dict[str, Any]:
    s = _summary(df)
    return {
        "bucket": label,
        "rows": s["rows"],
        "graded": s["graded"],
        "record": f"{s['wins']}-{s['losses']}-{s['pushes']}",
        "win_rate": _fmt_pct(s["win_rate"]),
        "units": _fmt_num(s["units"], 2, signed=True),
        "roi": _fmt_pct(s["roi"], signed=True),
        "avg_clv": _fmt_num(s["avg_clv_price"], 2, signed=True),
        "clv_beat": f"{_fmt_pct(s['clv_price_beat_rate'])} ({s['clv_price_rows']})",
    }


def _group_rows(df: pd.DataFrame, cols: list[str], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    for keys, group in df.groupby(cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        label = "|".join(str(k) for k in keys)
        rows.append(_metric_row(label, group))
    rows.sort(key=lambda r: (-int(r["graded"]), r["bucket"]))
    return rows[:limit]


def _top_rows(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    work["_ev"] = pd.to_numeric(work["ev"], errors="coerce").fillna(-999.0)
    work = work.sort_values("_ev", ascending=False).head(limit)
    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        rows.append({
            "player": row.get("player_name"),
            "stat": row.get("stat"),
            "side": row.get("side"),
            "line": _fmt_num(row.get("line"), 1),
            "price": _fmt_num(row.get("price"), 0, signed=True),
            "ev": _fmt_pct(row.get("ev"), signed=True),
            "result": "pending" if row.get("won") is None else ("win" if bool(row.get("won")) else "loss"),
            "clv": _fmt_num(row.get("clv_price"), 2, signed=True),
            "reasons": row.get("bankroll_reasons") or "",
        })
    return rows


def build_report(cfg: SlatePostmortemConfig) -> str:
    report_date = cfg.report_date or datetime.now(_ET).date()
    df = _load(SlatePostmortemConfig(**{**cfg.__dict__, "report_date": report_date}))
    active = df[df["is_active"].fillna(True).astype(bool)] if not df.empty else df
    bankroll = active[active["bankroll_candidate"].fillna(False).astype(bool)] if not active.empty else active
    positive = active[pd.to_numeric(active.get("ev"), errors="coerce").fillna(-999.0) >= cfg.min_ev] if not active.empty else active
    passed_positive = positive[~positive["bankroll_candidate"].fillna(False).astype(bool)] if not positive.empty else positive
    stale = df[~df["is_active"].fillna(True).astype(bool)] if not df.empty else df
    rows = [
        _metric_row("active_all", active),
        _metric_row("bankroll", bankroll),
        _metric_row("paper_positive_ev", passed_positive),
        _metric_row("stale_locked_rows", stale),
    ]
    lines = [
        "# MLB Prop Slate Post-Mortem",
        "",
        f"Date: {report_date}",
        f"Rows: {len(df)} total, {len(active)} active, {len(stale)} stale locked",
        "",
        "## Slate Summary",
        "",
        _table(
            rows,
            [
                ("Bucket", "bucket"),
                ("Rows", "rows"),
                ("Graded", "graded"),
                ("Record", "record"),
                ("Win%", "win_rate"),
                ("Units", "units"),
                ("ROI", "roi"),
                ("Avg CLV", "avg_clv"),
                ("CLV Beat", "clv_beat"),
            ],
        ),
        "",
        "## By Market / Surface",
        "",
        _table(
            _group_rows(active, ["stat", "side", "line_surface"], cfg.top_n),
            [
                ("Bucket", "bucket"),
                ("Rows", "rows"),
                ("Graded", "graded"),
                ("Record", "record"),
                ("ROI", "roi"),
                ("Avg CLV", "avg_clv"),
                ("CLV Beat", "clv_beat"),
            ],
        ),
        "",
        "## Top Positive-EV Research Rows",
        "",
        _table(
            _top_rows(passed_positive, cfg.top_n),
            [
                ("Player", "player"),
                ("Stat", "stat"),
                ("Side", "side"),
                ("Line", "line"),
                ("Price", "price"),
                ("EV", "ev"),
                ("Result", "result"),
                ("CLV", "clv"),
                ("Reasons", "reasons"),
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop slate post-mortem report")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--date", default=None)
    parser.add_argument("--min-ev", type=float, default=0.02)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = SlatePostmortemConfig(
        pg_dsn=args.pg_dsn,
        report_date=date.fromisoformat(args.date) if args.date else None,
        min_ev=args.min_ev,
        top_n=args.top_n,
        out=args.out,
    )
    report = build_report(cfg)
    if cfg.out:
        out = Path(cfg.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
