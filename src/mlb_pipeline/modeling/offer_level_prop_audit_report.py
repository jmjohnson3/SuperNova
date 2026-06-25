"""Offer-level MLB prop audit report.

This report audits the new one-row-per-offer prediction surface.  It is meant
to answer: which exact market/side/line/price/book/model buckets are actually
winning, calibrated, and beating close?
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2
import psycopg2.extras

from .side_recalibration import price_bucket, prop_line_bucket

_ET = ZoneInfo("America/New_York")
from mlb_pipeline.db import PG_DSN as _PG_DSN
@dataclass(frozen=True)
class OfferAuditConfig:
    dsn: str = _PG_DSN
    start_date: date | None = None
    end_date: date | None = None
    lookback_days: int = 60
    min_bucket_rows: int = 5
    out: str | None = None
    json_out: str | None = None
    top_n: int = 50


SQL = """
SELECT
    id,
    prediction_key,
    prop_offer_id,
    prop_offer_source_row_id,
    game_date_et,
    game_slug,
    player_id,
    player_name,
    team_abbr,
    stat,
    bet_side AS side,
    book_line::float AS line,
    COALESCE(line_bucket, 'unknown') AS line_bucket,
    bookmaker_key,
    bet_price::float AS price,
    over_price::float AS over_price,
    under_price::float AS under_price,
    model_family,
    edge_type,
    pred_count::float AS pred_count,
    pred_prob_over::float AS pred_prob_over,
    ev::float AS ev,
    bankroll_tier,
    bankroll_candidate,
    bankroll_reasons,
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
    bet_link
FROM bets.mlb_prop_predictions
WHERE game_date_et BETWEEN %(start_date)s AND %(end_date)s
  AND stat IN ('pitcher_strikeouts', 'batter_hits', 'batter_total_bases', 'batter_home_runs')
  AND bet_side IN ('over', 'under')
  AND book_line IS NOT NULL
ORDER BY game_date_et DESC, stat, player_name, line, side, bookmaker_key
"""


def _ensure_prediction_columns(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE IF EXISTS bets.mlb_prop_predictions
                ADD COLUMN IF NOT EXISTS prediction_key TEXT,
                ADD COLUMN IF NOT EXISTS prop_offer_id TEXT,
                ADD COLUMN IF NOT EXISTS prop_offer_source_row_id BIGINT,
                ADD COLUMN IF NOT EXISTS model_family TEXT,
                ADD COLUMN IF NOT EXISTS edge_type TEXT,
                ADD COLUMN IF NOT EXISTS bet_price NUMERIC,
                ADD COLUMN IF NOT EXISTS closing_line NUMERIC,
                ADD COLUMN IF NOT EXISTS closing_price INTEGER,
                ADD COLUMN IF NOT EXISTS clv_line NUMERIC,
                ADD COLUMN IF NOT EXISTS clv_price NUMERIC,
                ADD COLUMN IF NOT EXISTS beat_clv_line BOOLEAN,
                ADD COLUMN IF NOT EXISTS beat_clv_price BOOLEAN,
                ADD COLUMN IF NOT EXISTS clv_valid BOOLEAN,
                ADD COLUMN IF NOT EXISTS clv_status TEXT,
                ADD COLUMN IF NOT EXISTS clv_unknown_reason TEXT,
                ADD COLUMN IF NOT EXISTS bet_link TEXT;
            """
        )


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


def _american_profit_mult(price: Any) -> float | None:
    p = _clean_float(price)
    if p is None or p == 0:
        return None
    return p / 100.0 if p > 0 else 100.0 / abs(p)


def _side_prob(row: pd.Series) -> float | None:
    p_over = _clean_float(row.get("pred_prob_over"))
    side = row.get("side")
    if p_over is None or side not in {"over", "under"}:
        return None
    return p_over if side == "over" else 1.0 - p_over


def _side_result(row: pd.Series) -> tuple[bool | None, bool]:
    actual = _clean_float(row.get("actual_value"))
    line = _clean_float(row.get("line"))
    side = row.get("side")
    if actual is None or line is None or side not in {"over", "under"}:
        return None, False
    push = abs(actual - line) <= 1e-9
    if push:
        return None, True
    over_hit = actual > line
    return (over_hit if side == "over" else not over_hit), False


def _profit_units(won: bool | None, push: bool, price: Any) -> float | None:
    if push:
        return 0.0
    if won is None:
        return None
    mult = _american_profit_mult(price)
    if mult is None:
        return None
    return mult if won else -1.0


def _is_tail_alt(row: pd.Series) -> bool:
    stat = row.get("stat")
    side = row.get("side")
    line = _clean_float(row.get("line"))
    if side != "over" or line is None:
        return False
    if stat == "batter_hits":
        return line >= 2.5
    if stat == "batter_total_bases":
        return line >= 3.5
    if stat == "batter_home_runs":
        return line >= 1.5
    return False


def _load(cfg: OfferAuditConfig) -> pd.DataFrame:
    end_date = cfg.end_date or datetime.now(_ET).date()
    start_date = cfg.start_date or (end_date - timedelta(days=max(1, cfg.lookback_days) - 1))
    with psycopg2.connect(cfg.dsn) as conn:
        _ensure_prediction_columns(conn)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(SQL, {"start_date": start_date, "end_date": end_date})
            rows = [dict(r) for r in cur.fetchall()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    df["model_prob_side"] = df.apply(_side_prob, axis=1)
    results = df.apply(_side_result, axis=1)
    df["won"] = [r[0] for r in results]
    df["push"] = [r[1] for r in results]
    df["profit_units"] = [
        _profit_units(won, push, price)
        for won, push, price in zip(df["won"], df["push"], df["price"])
    ]
    df["price_bucket"] = df["price"].map(price_bucket)
    df["line_bucket"] = [
        prop_line_bucket(stat, line)
        for stat, line in zip(df["stat"], df["line"])
    ]
    df["tail_alt"] = df.apply(_is_tail_alt, axis=1)
    return df


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _summary(group: pd.DataFrame) -> dict[str, Any]:
    settled = group[group["won"].notna() & ~group["push"].fillna(False).astype(bool)].copy()
    priced = group["profit_units"].notna()
    wins = int(settled["won"].astype(bool).sum()) if not settled.empty else 0
    losses = int(len(settled) - wins)
    units = float(pd.to_numeric(group.loc[priced, "profit_units"], errors="coerce").sum()) if priced.any() else None
    avg_prob = _mean(group["model_prob_side"])
    win_rate = float(wins / (wins + losses)) if wins + losses else None
    brier = None
    if not settled.empty:
        p = pd.to_numeric(settled["model_prob_side"], errors="coerce")
        y = settled["won"].astype(bool).astype(float)
        valid = p.notna()
        if valid.any():
            brier = float(((p[valid] - y[valid]) ** 2).mean())
    valid_clv = group["clv_valid"].fillna(False).astype(bool)
    clv_price = pd.to_numeric(group.loc[valid_clv, "clv_price"], errors="coerce").dropna()
    clv_line = pd.to_numeric(group.loc[valid_clv, "clv_line"], errors="coerce").dropna()
    return {
        "rows": int(len(group)),
        "graded": int(len(settled)),
        "wins": wins,
        "losses": losses,
        "pushes": int(group["push"].fillna(False).astype(bool).sum()),
        "win_rate": win_rate,
        "units": units,
        "roi": (units / int(priced.sum())) if units is not None and int(priced.sum()) else None,
        "avg_prob": avg_prob,
        "calibration_error": (win_rate - avg_prob) if win_rate is not None and avg_prob is not None else None,
        "brier": brier,
        "avg_ev": _mean(group["ev"]),
        "avg_clv_price": float(clv_price.mean()) if not clv_price.empty else None,
        "clv_price_beat_rate": float((clv_price > 0).mean()) if not clv_price.empty else None,
        "clv_price_rows": int(len(clv_price)),
        "avg_clv_line": float(clv_line.mean()) if not clv_line.empty else None,
        "clv_line_beat_rate": float((clv_line > 0).mean()) if not clv_line.empty else None,
        "clv_line_rows": int(len(clv_line)),
    }


def _fmt_pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    v = _clean_float(value)
    if v is None:
        return "-"
    text = f"{v * 100:+.{digits}f}%" if signed else f"{v * 100:.{digits}f}%"
    return text


def _fmt_num(value: Any, digits: int = 2, signed: bool = False) -> str:
    v = _clean_float(value)
    if v is None:
        return "-"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    out = [
        "| " + " | ".join(title for title, _key in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(key, "")) for _title, key in columns) + " |")
    return "\n".join(out)


def _bucket_rows(df: pd.DataFrame, group_cols: list[str], min_rows: int, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    for keys, group in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        s = _summary(group)
        if s["rows"] < min_rows and s["graded"] < min_rows:
            continue
        rec = {col: key for col, key in zip(group_cols, keys)}
        rec.update({
            "rows": s["rows"],
            "graded": s["graded"],
            "record": f"{s['wins']}-{s['losses']}-{s['pushes']}",
            "roi": _fmt_pct(s["roi"], signed=True),
            "win_rate": _fmt_pct(s["win_rate"]),
            "avg_prob": _fmt_pct(s["avg_prob"]),
            "cal_error": _fmt_pct(s["calibration_error"], signed=True),
            "brier": _fmt_num(s["brier"], 3),
            "avg_ev": _fmt_pct(s["avg_ev"], signed=True),
            "avg_clv_price": _fmt_num(s["avg_clv_price"], 2, signed=True),
            "clv_price_beat": _fmt_pct(s["clv_price_beat_rate"]),
            "clv_price_rows": s["clv_price_rows"],
        })
        rows.append(rec)
    rows.sort(key=lambda r: (-int(r["rows"]), str(r.get("stat", "")), str(r.get("side", ""))))
    return rows[:limit]


def _offer_rows(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    work["_ev_sort"] = pd.to_numeric(work["ev"], errors="coerce").fillna(-999.0)
    work = work.sort_values(["tail_alt", "_ev_sort"], ascending=[False, False]).head(limit)
    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        s = _summary(pd.DataFrame([row]))
        rows.append({
            "date": row.get("game_date_et"),
            "offer_id": row.get("prop_offer_id"),
            "player": row.get("player_name"),
            "stat": row.get("stat"),
            "side": row.get("side"),
            "line": _fmt_num(row.get("line"), 1),
            "price": _fmt_num(row.get("price"), 0, signed=True),
            "book": row.get("bookmaker_key"),
            "family": row.get("model_family"),
            "prob": _fmt_pct(row.get("model_prob_side")),
            "ev": _fmt_pct(row.get("ev"), signed=True),
            "result": f"{s['wins']}-{s['losses']}-{s['pushes']}" if s["graded"] else "pending",
            "clv_price": _fmt_num(row.get("clv_price"), 2, signed=True),
            "tail": "yes" if row.get("tail_alt") else "",
        })
    return rows


def build_payload(cfg: OfferAuditConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _load(cfg)
    overall = _summary(df) if not df.empty else _summary(pd.DataFrame(columns=["won", "push", "profit_units", "model_prob_side", "ev", "clv_price", "clv_line", "clv_valid"]))
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "overall": overall,
        "by_bucket": _bucket_rows(
            df,
            ["stat", "side", "line_bucket", "price_bucket", "bookmaker_key", "model_family"],
            cfg.min_bucket_rows,
            cfg.top_n,
        ),
        "by_offer": _offer_rows(df, cfg.top_n),
        "tail_alt_buckets": _bucket_rows(
            df[df["tail_alt"]] if not df.empty else df,
            ["stat", "side", "line_bucket", "price_bucket", "bookmaker_key", "model_family"],
            1,
            cfg.top_n,
        ),
    }
    return df, payload


def build_report(cfg: OfferAuditConfig) -> str:
    _df, payload = build_payload(cfg)
    overall = payload["overall"]
    lines = [
        "# MLB Offer-Level Prop Audit",
        "",
        f"Rows: {payload['rows']}",
        f"Record: {overall['wins']}-{overall['losses']}-{overall['pushes']}",
        f"ROI: {_fmt_pct(overall['roi'], signed=True)}",
        f"Brier: {_fmt_num(overall['brier'], 3)}",
        f"Calibration error: {_fmt_pct(overall['calibration_error'], signed=True)}",
        f"Avg price CLV: {_fmt_num(overall['avg_clv_price'], 2, signed=True)}",
        f"Price CLV beat rate: {_fmt_pct(overall['clv_price_beat_rate'])} ({overall['clv_price_rows']} rows)",
        "",
        "## Exact Offer Buckets",
        "",
        _table(
            payload["by_bucket"],
            [
                ("Stat", "stat"),
                ("Side", "side"),
                ("Line", "line_bucket"),
                ("Price", "price_bucket"),
                ("Book", "bookmaker_key"),
                ("Family", "model_family"),
                ("Rows", "rows"),
                ("Graded", "graded"),
                ("Record", "record"),
                ("ROI", "roi"),
                ("Win%", "win_rate"),
                ("Avg Prob", "avg_prob"),
                ("Cal Err", "cal_error"),
                ("Brier", "brier"),
                ("CLV", "avg_clv_price"),
                ("CLV Beat", "clv_price_beat"),
            ],
        ),
        "",
        "## Tail Alt-Line Buckets",
        "",
        _table(
            payload["tail_alt_buckets"],
            [
                ("Stat", "stat"),
                ("Line", "line_bucket"),
                ("Price", "price_bucket"),
                ("Book", "bookmaker_key"),
                ("Family", "model_family"),
                ("Rows", "rows"),
                ("Record", "record"),
                ("ROI", "roi"),
                ("Cal Err", "cal_error"),
                ("Brier", "brier"),
                ("CLV", "avg_clv_price"),
            ],
        ),
        "",
        "## Individual Offer Rows",
        "",
        _table(
            payload["by_offer"],
            [
                ("Date", "date"),
                ("Offer", "offer_id"),
                ("Player", "player"),
                ("Stat", "stat"),
                ("Side", "side"),
                ("Line", "line"),
                ("Price", "price"),
                ("Book", "book"),
                ("Family", "family"),
                ("Prob", "prob"),
                ("EV", "ev"),
                ("Result", "result"),
                ("CLV", "clv_price"),
                ("Tail", "tail"),
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _parse_args() -> OfferAuditConfig:
    parser = argparse.ArgumentParser(description="Build offer-level MLB prop audit report")
    parser.add_argument("--dsn", default=_PG_DSN)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--min-bucket-rows", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--out", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    return OfferAuditConfig(
        dsn=args.dsn,
        start_date=date.fromisoformat(args.start_date) if args.start_date else None,
        end_date=date.fromisoformat(args.end_date) if args.end_date else None,
        lookback_days=args.lookback_days,
        min_bucket_rows=args.min_bucket_rows,
        top_n=args.top_n,
        out=args.out,
        json_out=args.json_out,
    )


def main() -> None:
    cfg = _parse_args()
    report = build_report(cfg)
    if cfg.json_out:
        _df, payload = build_payload(cfg)
        path = Path(cfg.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if cfg.out:
        path = Path(cfg.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
