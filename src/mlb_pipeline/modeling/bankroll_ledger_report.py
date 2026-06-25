"""Report locked MLB bankroll ledger picks and graded performance.

Unlike saved prediction audits, this reads only bets.mlb_bankroll_ledger. That
table is append-only by pick key, so reruns do not rewrite the released bet.
"""
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
class LedgerReportConfig:
    report_date: date
    start_date: date
    end_date: date
    dsn: str
    out: str | None = None
    daily_only: bool = False
    global_cap_pct: float = 0.02
    min_graded_bets: int = 50
    min_roi_pct: float = 2.0
    min_price_clv_rate_pct: float = 50.0


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
    bankroll_tier,
    bankroll_reasons,
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
FROM bets.mlb_bankroll_ledger
WHERE game_date_et BETWEEN :start_date AND :end_date
ORDER BY game_date_et DESC, inserted_at_utc DESC, id DESC
"""


REPORT_DATE_SQL = """
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
    bankroll_tier,
    bankroll_reasons,
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
FROM bets.mlb_bankroll_ledger
WHERE game_date_et = :report_date
ORDER BY source, market, label, id
"""


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


def _fmt_units(value) -> str:
    return _fmt_num(value, 2, signed=True)


def _fmt_stake_pct(value) -> str:
    val = _as_float(value)
    if val is None:
        return "-"
    return f"{val * 100:.2f}%"


def _fmt_result(row: pd.Series) -> str:
    status = str(row.get("result_status") or "pending")
    if status != "graded":
        return status
    if _as_bool(row.get("push")):
        return "PUSH"
    if _as_bool(row.get("won")):
        return "WIN"
    return "LOSS"


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
    if edge < 0.5:
        return "edge_lt_0.5"
    if edge < 1.0:
        return "edge_0.5_1"
    if edge < 1.5:
        return "edge_1_1.5"
    if edge < 2.0:
        return "edge_1.5_2"
    if edge < 3.0:
        return "edge_2_3"
    return "edge_3plus"


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


def _load_frame(engine, sql: str, params: dict) -> pd.DataFrame:
    df = pd.read_sql(text(sql), engine, params=params)
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in (
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
        "actual_home_score",
        "actual_away_score",
        "actual_run_diff",
        "actual_total",
        "closing_line",
        "closing_price",
        "clv_line",
        "clv_price",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _record_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "risked": 0,
            "units": 0.0,
            "roi_pct": None,
            "win_pct": None,
            "avg_line_clv": None,
            "line_clv_n": 0,
            "line_clv_beat_pct": None,
            "avg_price_clv": None,
            "price_clv_n": 0,
            "price_clv_beat_pct": None,
        }
    graded = df[df["result_status"].astype(str) == "graded"].copy()
    if graded.empty:
        return _record_metrics(pd.DataFrame())
    push_mask = graded["push"].fillna(False).astype(bool)
    win_mask = graded["won"].fillna(False).astype(bool) & ~push_mask
    loss_mask = ~win_mask & ~push_mask
    risked = int((~push_mask).sum())
    units = float(graded["profit_units"].fillna(0.0).sum())
    roi_pct = (units / risked * 100.0) if risked else None
    win_pct = (int(win_mask.sum()) / risked * 100.0) if risked else None

    line_clv = graded["clv_line"].dropna()
    price_clv = graded["clv_price"].dropna()
    return {
        "bets": int(len(graded)),
        "wins": int(win_mask.sum()),
        "losses": int(loss_mask.sum()),
        "pushes": int(push_mask.sum()),
        "risked": risked,
        "units": units,
        "roi_pct": roi_pct,
        "win_pct": win_pct,
        "avg_line_clv": float(line_clv.mean()) if not line_clv.empty else None,
        "line_clv_n": int(line_clv.size),
        "line_clv_beat_pct": float((line_clv > 0).mean() * 100.0) if not line_clv.empty else None,
        "avg_price_clv": float(price_clv.mean()) if not price_clv.empty else None,
        "price_clv_n": int(price_clv.size),
        "price_clv_beat_pct": float((price_clv > 0).mean() * 100.0) if not price_clv.empty else None,
    }


def _metrics_row(label: str, df: pd.DataFrame) -> dict:
    m = _record_metrics(df)
    return {
        "bucket": label,
        "bets": m["bets"],
        "w_l_p": f"{m['wins']}-{m['losses']}-{m['pushes']}",
        "units": _fmt_units(m["units"]),
        "roi": _fmt_pct(m["roi_pct"], signed=True),
        "win_pct": _fmt_pct(m["win_pct"]),
        "avg_line_clv": _fmt_num(m["avg_line_clv"], 2, signed=True),
        "line_clv": (
            "-"
            if m["line_clv_n"] == 0
            else f"{_fmt_pct(m['line_clv_beat_pct'])} ({m['line_clv_n']})"
        ),
        "avg_price_clv": _fmt_pct(m["avg_price_clv"], 2, signed=True),
        "price_clv": (
            "-"
            if m["price_clv_n"] == 0
            else f"{_fmt_pct(m['price_clv_beat_pct'])} ({m['price_clv_n']})"
        ),
    }


def _trust_status(df: pd.DataFrame, cfg: LedgerReportConfig) -> tuple[str, list[str]]:
    m = _record_metrics(df)
    reasons: list[str] = []
    status = "HOLD"
    if m["bets"] < cfg.min_graded_bets:
        reasons.append(f"only {m['bets']} graded locked bets, need {cfg.min_graded_bets}")
    if m["risked"] > 0 and m["units"] <= 0:
        reasons.append("net units are not positive")
    if m["roi_pct"] is not None and m["roi_pct"] < cfg.min_roi_pct:
        reasons.append(f"ROI {_fmt_pct(m['roi_pct'])} below target {_fmt_pct(cfg.min_roi_pct)}")
    if m["price_clv_n"] > 0 and (m["price_clv_beat_pct"] or 0.0) < cfg.min_price_clv_rate_pct:
        reasons.append(
            f"price CLV beat rate {_fmt_pct(m['price_clv_beat_pct'])} below target "
            f"{_fmt_pct(cfg.min_price_clv_rate_pct)}"
        )
    if not reasons and m["bets"] >= cfg.min_graded_bets:
        status = "TRUST-BUT-MONITOR"
        reasons.append("minimum locked-ledger gates are passing")
    elif m["bets"] == 0:
        reasons.append("no graded locked bets yet")
    return status, reasons


def _pending_rows(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    rows = []
    for _, row in df.iterrows():
        source = str(row.get("source") or "")
        market = str(row.get("market") or "")
        side = str(row.get("side") or "")
        label = str(row.get("label") or "")
        price = _fmt_price(row.get("market_price"))
        bet_line = _fmt_num(row.get("bet_line"), 1)
        stake = _fmt_stake_pct(row.get("stake_pct"))
        p_model = _fmt_prob(row.get("model_prob"))
        raw_edge = _as_float(row.get("edge"))
        edge = "-" if raw_edge is None else _fmt_num(abs(raw_edge), 2, signed=True)
        link = row.get("link")
        bet_text = label
        if link:
            bet_text = f"[{label}]({link})"
        rows.append({
            "source": source,
            "market": market,
            "side": side,
            "bet": bet_text,
            "line": bet_line,
            "price": price,
            "p_model": p_model,
            "edge": edge,
            "stake": stake,
            "status": _fmt_result(row),
        })
    return rows


def _graded_daily_rows(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    rows = []
    graded = df[df["result_status"].astype(str) == "graded"]
    for game_date, sub in graded.groupby("game_date_et", sort=True):
        m = _record_metrics(sub)
        exposure = float(sub["stake_pct"].fillna(0.0).sum()) * 100.0
        rows.append({
            "date": game_date,
            "bets": m["bets"],
            "w_l_p": f"{m['wins']}-{m['losses']}-{m['pushes']}",
            "units": _fmt_units(m["units"]),
            "roi": _fmt_pct(m["roi_pct"], signed=True),
            "exposure": _fmt_pct(exposure),
        })
    return rows


def _breakdown_rows(df: pd.DataFrame, field: str, *, min_bets: int = 1) -> list[dict]:
    if df.empty:
        return []
    graded = df[df["result_status"].astype(str) == "graded"].copy()
    if graded.empty:
        return []
    if field == "source_market_side":
        graded["_bucket"] = (
            graded["source"].astype(str)
            + " / "
            + graded["market"].astype(str)
            + " / "
            + graded["side"].astype(str)
        )
    elif field == "price_bucket":
        graded["_bucket"] = graded["market_price"].map(_price_bucket)
    elif field == "edge_bucket":
        graded["_bucket"] = graded["edge"].map(_edge_bucket)
    else:
        graded["_bucket"] = graded[field].fillna("missing").astype(str)

    rows = []
    for bucket, sub in graded.groupby("_bucket", sort=True):
        if len(sub) < min_bets:
            continue
        rows.append(_metrics_row(str(bucket), sub))
    rows.sort(key=lambda r: (int(r["bets"]), r["bucket"]), reverse=True)
    return rows


def _reason_rows(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    counts: dict[str, int] = {}
    for reasons in df["bankroll_reasons"].fillna("ok"):
        parts = [p.strip() for p in str(reasons or "ok").split(";") if p.strip()]
        if not parts:
            parts = ["ok"]
        for part in parts:
            counts[part] = counts.get(part, 0) + 1
    return [{"reason": k, "locked_bets": v} for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]


def _no_bet_days(df: pd.DataFrame, cfg: LedgerReportConfig) -> tuple[int, list[date]]:
    if df.empty:
        all_days = [d.date() for d in pd.date_range(cfg.start_date, cfg.end_date, freq="D")]
        return len(all_days), all_days
    active_start = max(cfg.start_date, min(df["game_date_et"].tolist()))
    all_days = [d.date() for d in pd.date_range(active_start, cfg.end_date, freq="D")]
    bet_days = set(df["game_date_et"].tolist())
    missing = [day for day in all_days if day not in bet_days]
    return len(missing), missing


def _daily_exposure_rows(df: pd.DataFrame, cfg: LedgerReportConfig) -> list[dict]:
    if df.empty:
        return []
    grouped = (
        df.groupby("game_date_et", sort=True)
        .agg(locked_bets=("id", "count"), exposure_pct=("stake_pct", "sum"))
        .reset_index()
    )
    rows = []
    for _, row in grouped.iterrows():
        exposure = float(row["exposure_pct"] or 0.0)
        rows.append({
            "date": row["game_date_et"],
            "locked_bets": int(row["locked_bets"]),
            "exposure": _fmt_pct(exposure * 100.0),
            "cap": _fmt_pct(cfg.global_cap_pct * 100.0),
            "status": "OVER CAP" if exposure > cfg.global_cap_pct + 1e-12 else "ok",
        })
    return rows


def _build_daily_summary(report_day: pd.DataFrame, cfg: LedgerReportConfig) -> list[str]:
    pending = report_day[report_day["result_status"].astype(str) == "pending"] if not report_day.empty else report_day
    exposure = float(report_day["stake_pct"].fillna(0.0).sum()) if not report_day.empty else 0.0
    lines = [
        f"## Daily Ledger Summary - {cfg.report_date}",
        "",
        f"- Locked bankroll bets: {len(report_day)}",
        f"- Pending bankroll bets: {len(pending)}",
        f"- Daily exposure: {_fmt_pct(exposure * 100.0)} of {_fmt_pct(cfg.global_cap_pct * 100.0)} cap",
    ]
    if exposure > cfg.global_cap_pct + 1e-12:
        lines.append(f"- GLOBAL CAP WARNING: over cap by {_fmt_pct((exposure - cfg.global_cap_pct) * 100.0)}")
    lines.extend([
        "",
        _markdown_table(
            _pending_rows(report_day),
            [
                ("Source", "source"),
                ("Market", "market"),
                ("Side", "side"),
                ("Bet", "bet"),
                ("Line", "line"),
                ("Price", "price"),
                ("Model P", "p_model"),
                ("Edge", "edge"),
                ("Stake", "stake"),
                ("Status", "status"),
            ],
        ),
    ])
    return lines


def build_report(cfg: LedgerReportConfig) -> str:
    engine = create_engine(cfg.dsn)
    ledger = _load_frame(
        engine,
        LEDGER_SQL,
        {"start_date": cfg.start_date, "end_date": cfg.end_date},
    )
    report_day = _load_frame(engine, REPORT_DATE_SQL, {"report_date": cfg.report_date})

    lines: list[str] = [
        "# MLB Locked Bankroll Ledger Report",
        "",
        f"Range: {cfg.start_date} to {cfg.end_date}",
        "Source: `bets.mlb_bankroll_ledger` only.",
        "",
    ]
    lines.extend(_build_daily_summary(report_day, cfg))
    if cfg.daily_only:
        return "\n".join(lines) + "\n"

    graded = ledger[ledger["result_status"].astype(str) == "graded"] if not ledger.empty else ledger
    pending = ledger[ledger["result_status"].astype(str) == "pending"] if not ledger.empty else ledger
    status, reasons = _trust_status(graded, cfg)
    no_bet_count, no_bet_days = _no_bet_days(ledger, cfg)
    total = _record_metrics(graded)
    first_locked_date = min(ledger["game_date_et"].tolist()) if not ledger.empty else None

    lines.extend([
        "",
        "## Trust Status",
        "",
        f"**{status}**",
        "",
        *[f"- {reason}" for reason in reasons],
        "",
        "## Ledger Health",
        "",
        f"- First locked bet date in range: {first_locked_date or '-'}",
        f"- Locked bets in range: {len(ledger)}",
        f"- Graded bets: {total['bets']}",
        f"- Pending bets: {len(pending)}",
        f"- No-bet days: {no_bet_count}",
        f"- Net units: {_fmt_units(total['units'])}",
        f"- ROI: {_fmt_pct(total['roi_pct'], signed=True)}",
        f"- W-L-P: {total['wins']}-{total['losses']}-{total['pushes']}",
        "",
        "## Overall Performance",
        "",
        _markdown_table(
            [_metrics_row("All graded bankroll bets", graded)],
            [
                ("Bucket", "bucket"),
                ("Bets", "bets"),
                ("W-L-P", "w_l_p"),
                ("Units", "units"),
                ("ROI", "roi"),
                ("Win %", "win_pct"),
                ("Avg Line CLV", "avg_line_clv"),
                ("Beat Line CLV", "line_clv"),
                ("Avg Price CLV", "avg_price_clv"),
                ("Beat Price CLV", "price_clv"),
            ],
        ),
        "",
        "## Performance By Market",
        "",
        _markdown_table(
            _breakdown_rows(ledger, "source_market_side"),
            [
                ("Bucket", "bucket"),
                ("Bets", "bets"),
                ("W-L-P", "w_l_p"),
                ("Units", "units"),
                ("ROI", "roi"),
                ("Win %", "win_pct"),
                ("Avg Line CLV", "avg_line_clv"),
                ("Beat Line CLV", "line_clv"),
                ("Avg Price CLV", "avg_price_clv"),
                ("Beat Price CLV", "price_clv"),
            ],
        ),
        "",
        "## Performance By Price",
        "",
        _markdown_table(
            _breakdown_rows(ledger, "price_bucket"),
            [
                ("Bucket", "bucket"),
                ("Bets", "bets"),
                ("W-L-P", "w_l_p"),
                ("Units", "units"),
                ("ROI", "roi"),
                ("Win %", "win_pct"),
                ("Avg Line CLV", "avg_line_clv"),
                ("Beat Line CLV", "line_clv"),
                ("Avg Price CLV", "avg_price_clv"),
                ("Beat Price CLV", "price_clv"),
            ],
        ),
        "",
        "## Performance By Edge Bucket",
        "",
        _markdown_table(
            _breakdown_rows(ledger, "edge_bucket"),
            [
                ("Bucket", "bucket"),
                ("Bets", "bets"),
                ("W-L-P", "w_l_p"),
                ("Units", "units"),
                ("ROI", "roi"),
                ("Win %", "win_pct"),
                ("Avg Line CLV", "avg_line_clv"),
                ("Beat Line CLV", "line_clv"),
                ("Avg Price CLV", "avg_price_clv"),
                ("Beat Price CLV", "price_clv"),
            ],
        ),
        "",
        "## Daily Results",
        "",
        _markdown_table(
            _graded_daily_rows(ledger),
            [
                ("Date", "date"),
                ("Bets", "bets"),
                ("W-L-P", "w_l_p"),
                ("Units", "units"),
                ("ROI", "roi"),
                ("Exposure", "exposure"),
            ],
        ),
        "",
        "## Daily Exposure",
        "",
        _markdown_table(
            _daily_exposure_rows(ledger, cfg),
            [
                ("Date", "date"),
                ("Locked Bets", "locked_bets"),
                ("Exposure", "exposure"),
                ("Cap", "cap"),
                ("Status", "status"),
            ],
        ),
        "",
        "## Bankroll Reasons",
        "",
        _markdown_table(
            _reason_rows(ledger),
            [("Reason", "reason"), ("Locked Bets", "locked_bets")],
        ),
        "",
        "## No-Bet Days",
        "",
    ])
    if no_bet_days:
        preview = ", ".join(str(day) for day in no_bet_days[:20])
        suffix = f" ... +{len(no_bet_days) - 20} more" if len(no_bet_days) > 20 else ""
        lines.append(preview + suffix)
    else:
        lines.append("_None._")
    return "\n".join(lines) + "\n"


def _default_dates(args) -> tuple[date, date, date]:
    report_date = date.fromisoformat(args.date) if args.date else datetime.now(_ET).date()
    if args.start_date:
        start_date = date.fromisoformat(args.start_date)
    else:
        start_date = report_date - timedelta(days=max(0, args.lookback_days - 1))
    end_date = date.fromisoformat(args.end_date) if args.end_date else report_date
    if start_date > end_date:
        raise SystemExit("--start-date cannot be after --end-date")
    return report_date, start_date, end_date


def main() -> None:
    parser = argparse.ArgumentParser(description="Report locked MLB bankroll ledger bets")
    parser.add_argument("--date", default=None, help="Daily summary date YYYY-MM-DD ET. Defaults to today.")
    parser.add_argument("--start-date", default=None, help="Report range start YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Report range end YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=60, help="Default range length when start-date is omitted")
    parser.add_argument("--dsn", default=_PG_DSN)
    parser.add_argument("--out", default=None, help="Optional markdown output path")
    parser.add_argument("--daily-only", action="store_true", help="Print only the daily locked bet summary")
    parser.add_argument("--global-cap-pct", type=float, default=0.02, help="Daily cap as fraction, 0.02 = 2%%")
    parser.add_argument("--min-graded-bets", type=int, default=50)
    parser.add_argument("--min-roi-pct", type=float, default=2.0)
    parser.add_argument("--min-price-clv-rate-pct", type=float, default=50.0)
    args = parser.parse_args()

    report_date, start_date, end_date = _default_dates(args)
    cfg = LedgerReportConfig(
        report_date=report_date,
        start_date=start_date,
        end_date=end_date,
        dsn=args.dsn,
        out=args.out,
        daily_only=bool(args.daily_only),
        global_cap_pct=float(args.global_cap_pct),
        min_graded_bets=int(args.min_graded_bets),
        min_roi_pct=float(args.min_roi_pct),
        min_price_clv_rate_pct=float(args.min_price_clv_rate_pct),
    )
    try:
        report = build_report(cfg)
    except Exception as exc:
        raise SystemExit(
            "Could not build bankroll ledger report. "
            "Run predictions once so bets.mlb_bankroll_ledger exists. "
            f"Detail: {exc}"
        ) from exc

    if cfg.out:
        path = Path(cfg.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        print(f"Wrote {path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
