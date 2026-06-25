"""Report locked MLB model picks and graded accuracy/calibration."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import create_engine, text

_ET = ZoneInfo("America/New_York")
from mlb_pipeline.db import PG_DSN as _PG_DSN
@dataclass(frozen=True)
class ModelPickReportConfig:
    report_date: date
    start_date: date
    end_date: date
    dsn: str
    out: str | None = None
    top_n: int = 30
    min_bucket_rows: int = 5


LEDGER_SQL = """
SELECT
    id,
    inserted_at_utc,
    source,
    game_date_et,
    game_slug,
    market,
    stat,
    side,
    label,
    team_abbr,
    opponent_abbr,
    home_team_abbr,
    away_team_abbr,
    player_id,
    player_name,
    bookmaker_key,
    market_line,
    bet_line,
    market_price,
    link,
    pred_value,
    pred_count,
    model_prob,
    edge,
    edge_type,
    ev,
    kelly_fraction,
    model_tier,
    warning_reasons,
    stake_pct,
    result_status,
    won,
    push,
    profit_units,
    actual_value,
    actual_home_score,
    actual_away_score,
    actual_run_diff,
    actual_total,
    over_hit,
    closing_line,
    closing_price,
    clv_line,
    clv_price,
    graded_at_utc,
    grade_source
FROM bets.mlb_model_pick_ledger
WHERE game_date_et BETWEEN :start_date AND :end_date
ORDER BY game_date_et DESC, inserted_at_utc DESC, id DESC
"""


TODAY_SQL = LEDGER_SQL.replace(
    "WHERE game_date_et BETWEEN :start_date AND :end_date",
    "WHERE game_date_et = :report_date",
)


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return bool(value)


def _fmt_num(value, digits: int = 2, *, signed: bool = False) -> str:
    val = _as_float(value)
    if val is None:
        return "-"
    return f"{val:+.{digits}f}" if signed else f"{val:.{digits}f}"


def _fmt_pct(value, digits: int = 1, *, signed: bool = False) -> str:
    val = _as_float(value)
    if val is None:
        return "-"
    return f"{val:+.{digits}f}%" if signed else f"{val:.{digits}f}%"


def _fmt_prob(value) -> str:
    val = _as_float(value)
    if val is None:
        return "-"
    return f"{val * 100:.1f}%"


def _fmt_price(value) -> str:
    val = _as_float(value)
    if val is None:
        return "-"
    return f"{int(round(val)):+d}"


def _markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_None._"
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def _price_bucket(price) -> str:
    price = _as_float(price)
    if price is None:
        return "missing_price"
    if price > 0:
        return "plus_money"
    if price >= -129:
        return "fair_lay"
    if price >= -149:
        return "lay_130_149"
    if price >= -180:
        return "lay_150_180"
    return "heavy_lay"


def _edge_bucket(edge) -> str:
    edge = abs(_as_float(edge) or 0.0)
    if edge < 0.25:
        return "edge_lt_0.25"
    if edge < 0.5:
        return "edge_0.25_0.5"
    if edge < 1.0:
        return "edge_0.5_1"
    if edge < 1.5:
        return "edge_1_1.5"
    if edge < 2.0:
        return "edge_1.5_2"
    if edge < 3.0:
        return "edge_2_3"
    return "edge_3plus"


def _ev_bucket(ev) -> str:
    ev = _as_float(ev)
    if ev is None:
        return "missing_ev"
    if ev < 0.02:
        return "ev_lt_2"
    if ev < 0.05:
        return "ev_2_5"
    if ev < 0.10:
        return "ev_5_10"
    if ev < 0.20:
        return "ev_10_20"
    return "ev_20plus"


def _line_bucket(row: pd.Series) -> str:
    market = str(row.get("market") or "")
    line = _as_float(row.get("market_line"))
    if line is None:
        return "missing_line"
    if market == "pitcher_strikeouts":
        if line < 4.5:
            return "K <4.5"
        if line < 6.5:
            return "K 4.5-6"
        if line < 8.5:
            return "K 6.5-8"
        return "K 8.5+"
    if market == "batter_total_bases":
        if line < 1.0:
            return "TB 0.5"
        if line < 2.0:
            return "TB 1.5"
        return "TB 2.5+"
    if market == "batter_hits":
        if line < 1.0:
            return "H 0.5"
        if line < 2.0:
            return "H 1.5"
        return "H 2.5+"
    if market == "batter_home_runs":
        return "HR 0.5" if line < 1.0 else "HR 1.5+"
    if market == "run_line":
        return f"RL {line:+.1f}"
    if market == "total":
        if line < 8.0:
            return "total <8"
        if line < 9.5:
            return "total 8-9"
        if line < 11.0:
            return "total 9.5-10.5"
        return "total 11+"
    return "other"


def _result(row: pd.Series) -> str:
    if str(row.get("result_status") or "pending") != "graded":
        return "pending"
    if _as_bool(row.get("push")):
        return "PUSH"
    return "WIN" if _as_bool(row.get("won")) else "LOSS"


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["game_date_et"] = pd.to_datetime(out["game_date_et"]).dt.date
    for col in [
        "market_line",
        "bet_line",
        "market_price",
        "pred_value",
        "pred_count",
        "model_prob",
        "edge",
        "ev",
        "kelly_fraction",
        "stake_pct",
        "profit_units",
        "actual_value",
        "actual_run_diff",
        "actual_total",
        "closing_line",
        "closing_price",
        "clv_line",
        "clv_price",
    ]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["market_side"] = out.apply(
        lambda r: f"{r.get('source')}:{r.get('market')}/{r.get('side')}",
        axis=1,
    )
    out["price_bucket"] = out["market_price"].map(_price_bucket)
    out["edge_bucket"] = out["edge"].map(_edge_bucket)
    out["ev_bucket"] = out["ev"].map(_ev_bucket)
    out["line_bucket"] = out.apply(_line_bucket, axis=1)
    out["result"] = out.apply(_result, axis=1)
    return out


def _metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "locked": 0,
            "graded": 0,
            "pending": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": None,
            "units": 0.0,
            "roi": None,
            "avg_ev": None,
            "avg_prob": None,
            "cal_error": None,
            "brier": None,
            "avg_line_clv": None,
            "line_clv_beat": None,
            "avg_price_clv": None,
            "price_clv_beat": None,
        }
    graded = df[df["result_status"].astype(str) == "graded"].copy()
    settled = graded[~graded["push"].fillna(False).astype(bool)].copy()
    wins = int(settled["won"].fillna(False).astype(bool).sum()) if not settled.empty else 0
    losses = int(len(settled) - wins)
    pushes = int(graded["push"].fillna(False).astype(bool).sum()) if not graded.empty else 0
    units = float(pd.to_numeric(graded["profit_units"], errors="coerce").fillna(0.0).sum()) if not graded.empty else 0.0
    all_probs = pd.to_numeric(df["model_prob"], errors="coerce").dropna() if "model_prob" in df else pd.Series(dtype=float)
    settled_probs = pd.to_numeric(settled["model_prob"], errors="coerce").dropna() if not settled.empty else pd.Series(dtype=float)
    obs = settled.loc[settled_probs.index, "won"].astype(bool).astype(float) if not settled_probs.empty else pd.Series(dtype=float)
    brier = float(((settled_probs - obs) ** 2).mean()) if not settled_probs.empty else None
    line_clv = pd.to_numeric(graded["clv_line"], errors="coerce").dropna() if not graded.empty else pd.Series(dtype=float)
    price_clv = pd.to_numeric(graded["clv_price"], errors="coerce").dropna() if not graded.empty else pd.Series(dtype=float)
    return {
        "locked": int(len(df)),
        "graded": int(len(graded)),
        "pending": int((df["result_status"].astype(str) == "pending").sum()),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": (wins / (wins + losses) * 100.0) if wins + losses else None,
        "units": units,
        "roi": (units / len(settled) * 100.0) if len(settled) else None,
        "avg_ev": float(pd.to_numeric(df["ev"], errors="coerce").mean() * 100.0) if "ev" in df else None,
        "avg_prob": float(all_probs.mean() * 100.0) if not all_probs.empty else None,
        "cal_error": float((obs.mean() - settled_probs.mean()) * 100.0) if not settled_probs.empty else None,
        "brier": brier,
        "avg_line_clv": float(line_clv.mean()) if not line_clv.empty else None,
        "line_clv_beat": float((line_clv > 0).mean() * 100.0) if not line_clv.empty else None,
        "avg_price_clv": float(price_clv.mean()) if not price_clv.empty else None,
        "price_clv_beat": float((price_clv > 0).mean() * 100.0) if not price_clv.empty else None,
    }


def _metric_row(label: str, df: pd.DataFrame) -> dict:
    m = _metrics(df)
    return {
        "bucket": label,
        "locked": m["locked"],
        "graded": m["graded"],
        "pending": m["pending"],
        "record": f"{m['wins']}-{m['losses']}-{m['pushes']}",
        "win_rate": _fmt_pct(m["win_rate"]),
        "units": _fmt_num(m["units"], 2, signed=True),
        "roi": _fmt_pct(m["roi"], signed=True),
        "avg_ev": _fmt_pct(m["avg_ev"], signed=True),
        "avg_prob": _fmt_pct(m["avg_prob"]),
        "cal_error": _fmt_pct(m["cal_error"], signed=True),
        "brier": _fmt_num(m["brier"], 3),
        "line_clv": _fmt_num(m["avg_line_clv"], 2, signed=True),
        "line_clv_beat": _fmt_pct(m["line_clv_beat"]),
        "price_clv": _fmt_pct(m["avg_price_clv"], 2, signed=True),
        "price_clv_beat": _fmt_pct(m["price_clv_beat"]),
    }


def _breakdown_rows(df: pd.DataFrame, col: str, cfg: ModelPickReportConfig) -> list[dict]:
    if df.empty or col not in df:
        return []
    rows = []
    for key, sub in df.groupby(col, dropna=False, sort=True):
        if len(sub) < cfg.min_bucket_rows and int((sub["result_status"].astype(str) == "graded").sum()) < cfg.min_bucket_rows:
            continue
        rows.append(_metric_row(str(key), sub))
    rows.sort(key=lambda r: (-int(r["locked"]), r["bucket"]))
    return rows


def _today_rows(today: pd.DataFrame, cfg: ModelPickReportConfig) -> list[dict]:
    if today.empty:
        return []
    work = today.copy()
    work["_score"] = pd.to_numeric(work["ev"], errors="coerce").fillna(-999.0)
    work["_edge_abs"] = pd.to_numeric(work["edge"], errors="coerce").abs().fillna(0.0)
    work["_bankroll"] = work["model_tier"].astype(str).eq("bankroll_candidate")
    work["_has_model_edge"] = ~work["warning_reasons"].fillna("").astype(str).str.contains("no_model_edge")
    work = work.sort_values(["_bankroll", "_has_model_edge", "_score", "_edge_abs"], ascending=False).head(cfg.top_n)
    rows = []
    for _, r in work.iterrows():
        rows.append({
            "date": r.get("game_date_et"),
            "pick": r.get("label") or f"{r.get('market')} {r.get('side')}",
            "market": r.get("market_side"),
            "tier": r.get("model_tier") or "",
            "line": _fmt_num(r.get("bet_line"), 1, signed=r.get("market") == "run_line"),
            "price": _fmt_price(r.get("market_price")),
            "prob": _fmt_prob(r.get("model_prob")),
            "edge": _fmt_num(abs(_as_float(r.get("edge")) or 0.0), 2, signed=True),
            "ev": _fmt_pct((_as_float(r.get("ev")) or 0.0) * 100.0, signed=True),
            "result": r.get("result"),
            "warnings": r.get("warning_reasons") or "",
        })
    return rows


def _calibration_rows(df: pd.DataFrame, cfg: ModelPickReportConfig) -> list[dict]:
    if df.empty:
        return []
    graded = df[
        (df["result_status"].astype(str) == "graded")
        & ~df["push"].fillna(False).astype(bool)
        & df["model_prob"].notna()
    ].copy()
    if graded.empty:
        return []
    bins = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01]
    labels = ["<50", "50-55", "55-60", "60-65", "65-70", "70-80", "80+"]
    graded["prob_bin"] = pd.cut(graded["model_prob"], bins=bins, labels=labels, include_lowest=True, right=False)
    rows = []
    for (market_side, prob_bin), sub in graded.groupby(["market_side", "prob_bin"], observed=True):
        if len(sub) < cfg.min_bucket_rows:
            continue
        p = pd.to_numeric(sub["model_prob"], errors="coerce")
        obs = sub["won"].astype(bool).astype(float)
        rows.append({
            "market": market_side,
            "prob_bin": prob_bin,
            "n": len(sub),
            "avg_prob": _fmt_pct(float(p.mean() * 100.0)),
            "actual": _fmt_pct(float(obs.mean() * 100.0)),
            "error": _fmt_pct(float((obs.mean() - p.mean()) * 100.0), signed=True),
            "brier": _fmt_num(float(((p - obs) ** 2).mean()), 3),
        })
    rows.sort(key=lambda r: (r["market"], str(r["prob_bin"])))
    return rows


def build_report(cfg: ModelPickReportConfig) -> str:
    engine = create_engine(cfg.dsn)
    ledger = _enrich(pd.read_sql(text(LEDGER_SQL), engine, params={"start_date": cfg.start_date, "end_date": cfg.end_date}))
    today = _enrich(pd.read_sql(text(TODAY_SQL), engine, params={"report_date": cfg.report_date}))

    lines = [
        "# MLB Model Pick Ledger",
        "",
        f"Report date: {cfg.report_date}",
        f"Window: {cfg.start_date} to {cfg.end_date}",
        "Source: `bets.mlb_model_pick_ledger` only. This tracks every locked model pick, not just bankroll candidates.",
        "",
        "## Summary",
        "",
        _markdown_table(
            [
                _metric_row("All model picks", ledger),
                _metric_row("Games", ledger[ledger["source"] == "game"] if not ledger.empty else ledger),
                _metric_row("Props", ledger[ledger["source"] == "prop"] if not ledger.empty else ledger),
            ],
            [
                ("Bucket", "bucket"),
                ("Locked", "locked"),
                ("Graded", "graded"),
                ("Pending", "pending"),
                ("Record", "record"),
                ("Win%", "win_rate"),
                ("Units", "units"),
                ("ROI", "roi"),
                ("Avg EV", "avg_ev"),
                ("Avg Prob", "avg_prob"),
                ("Cal Err", "cal_error"),
                ("Brier", "brier"),
                ("Avg CLV", "line_clv"),
                ("Beat CLV", "line_clv_beat"),
                ("Avg Price CLV", "price_clv"),
                ("Beat Price CLV", "price_clv_beat"),
            ],
        ),
        "",
        "## Today's Locked Picks",
        "",
        _markdown_table(
            _today_rows(today, cfg),
            [
            ("Date", "date"),
            ("Pick", "pick"),
            ("Market", "market"),
            ("Tier", "tier"),
            ("Line", "line"),
            ("Price", "price"),
            ("Prob", "prob"),
                ("Edge", "edge"),
                ("EV", "ev"),
                ("Result", "result"),
                ("Warnings", "warnings"),
            ],
        ),
        "",
        "## Market/Side Accuracy",
        "",
        _markdown_table(
            _breakdown_rows(ledger, "market_side", cfg),
            [
                ("Market", "bucket"),
                ("Locked", "locked"),
                ("Graded", "graded"),
                ("Record", "record"),
                ("Win%", "win_rate"),
                ("Units", "units"),
                ("ROI", "roi"),
                ("Avg EV", "avg_ev"),
                ("Cal Err", "cal_error"),
                ("Brier", "brier"),
                ("Avg CLV", "line_clv"),
                ("Beat Price CLV", "price_clv_beat"),
            ],
        ),
        "",
        "## Line Buckets",
        "",
        _markdown_table(
            _breakdown_rows(ledger, "line_bucket", cfg),
            [
                ("Line Bucket", "bucket"),
                ("Locked", "locked"),
                ("Graded", "graded"),
                ("Record", "record"),
                ("Win%", "win_rate"),
                ("Units", "units"),
                ("ROI", "roi"),
                ("Avg EV", "avg_ev"),
                ("Cal Err", "cal_error"),
            ],
        ),
        "",
        "## Price Buckets",
        "",
        _markdown_table(
            _breakdown_rows(ledger, "price_bucket", cfg),
            [
                ("Price Bucket", "bucket"),
                ("Locked", "locked"),
                ("Graded", "graded"),
                ("Record", "record"),
                ("Win%", "win_rate"),
                ("Units", "units"),
                ("ROI", "roi"),
                ("Avg EV", "avg_ev"),
                ("Beat Price CLV", "price_clv_beat"),
            ],
        ),
        "",
        "## EV Buckets",
        "",
        _markdown_table(
            _breakdown_rows(ledger, "ev_bucket", cfg),
            [
                ("EV Bucket", "bucket"),
                ("Locked", "locked"),
                ("Graded", "graded"),
                ("Record", "record"),
                ("Win%", "win_rate"),
                ("Units", "units"),
                ("ROI", "roi"),
                ("Avg EV", "avg_ev"),
                ("Cal Err", "cal_error"),
            ],
        ),
        "",
        "## Probability Calibration",
        "",
        _markdown_table(
            _calibration_rows(ledger, cfg),
            [
                ("Market", "market"),
                ("Prob Bin", "prob_bin"),
                ("N", "n"),
                ("Avg Prob", "avg_prob"),
                ("Actual", "actual"),
                ("Error", "error"),
                ("Brier", "brier"),
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _parse_dates(args) -> tuple[date, date, date]:
    report_date = date.fromisoformat(args.date) if args.date else datetime.now(_ET).date()
    end_date = date.fromisoformat(args.end_date) if args.end_date else report_date
    start_date = (
        date.fromisoformat(args.start_date)
        if args.start_date
        else end_date - timedelta(days=max(0, args.lookback_days - 1))
    )
    if start_date > end_date:
        raise SystemExit("--start-date cannot be after --end-date")
    return report_date, start_date, end_date


def main() -> None:
    parser = argparse.ArgumentParser(description="Report locked MLB model-pick accuracy and calibration")
    parser.add_argument("--date", default=None, help="Report date YYYY-MM-DD ET. Defaults to today.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--dsn", default=_PG_DSN)
    parser.add_argument("--out", default=None)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--min-bucket-rows", type=int, default=5)
    args = parser.parse_args()

    report_date, start_date, end_date = _parse_dates(args)
    cfg = ModelPickReportConfig(
        report_date=report_date,
        start_date=start_date,
        end_date=end_date,
        dsn=args.dsn,
        out=args.out,
        top_n=args.top_n,
        min_bucket_rows=args.min_bucket_rows,
    )
    report = build_report(cfg)
    out = cfg.out
    if out is None:
        out = str(Path("reports") / f"mlb_model_pick_ledger_{report_date}.md")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
