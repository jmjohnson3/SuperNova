"""Audit current real-money MLB bet gates against saved predictions.

This is a saved-prediction audit, not a locked-ledger walk-forward. Rows can be
overwritten by reruns, so use this as a gate-health report until a ledger stores
every Discord candidate exactly once.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import create_engine, text

_ET = ZoneInfo("America/New_York")
_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_PROP_THRESHOLDS = Path(__file__).resolve().parent / "models" / "player_props" / "prop_thresholds.json"


@dataclass(frozen=True)
class AuditConfig:
    start_date: date
    end_date: date
    dsn: str
    out: str | None = None
    prop_thresholds_file: Path | None = _PROP_THRESHOLDS
    global_cap_pct: float = 0.02
    max_stake_pct: float = 0.005
    min_bets: int = 20
    max_run_line_lay_price: float = -180.0
    max_away_dog_lay_price: float = -130.0
    max_prop_lay_price: float = -180.0
    min_ev: float = 0.02
    threshold_strikeouts: float = 2.0
    threshold_hits: float = 0.75
    threshold_total_bases: float = 1.5
    threshold_home_runs_over: float = 0.05
    threshold_home_runs_under: float = 0.45


GAME_SQL = """
SELECT
    game_date_et,
    game_slug,
    home_team_abbr,
    away_team_abbr,
    run_line_bet_side,
    market_run_line,
    edge_run_line,
    win_prob_rl,
    market_rl_price,
    run_line_covered,
    clv_run_line,
    clv_rl_price,
    total_bet_side,
    market_total,
    edge_total,
    win_prob_total,
    market_total_price,
    total_covered,
    clv_total,
    clv_total_price
FROM bets.mlb_game_predictions
WHERE game_date_et BETWEEN :start_date AND :end_date
  AND (
      (run_line_bet_side IS NOT NULL AND run_line_covered IS NOT NULL)
      OR
      (total_bet_side IS NOT NULL AND total_covered IS NOT NULL)
  )
"""


PROP_SQL = """
SELECT
    game_date_et,
    game_slug,
    player_name,
    team_abbr,
    stat,
    pred_value,
    pred_count,
    pred_prob_over,
    book_line,
    edge,
    edge_type,
    model_family,
    bet_side,
    line_bucket,
    over_price,
    under_price,
    bet_price,
    breakeven_prob,
    ev,
    kelly_fraction,
    over_hit
FROM bets.mlb_prop_predictions
WHERE game_date_et BETWEEN :start_date AND :end_date
  AND edge IS NOT NULL
  AND over_hit IS NOT NULL
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


def _american_profit_mult(price: float | None) -> float | None:
    price = _as_float(price)
    if price is None or price == 0:
        return None
    if price > 0:
        return price / 100.0
    return 100.0 / abs(price)


def _profit(won: bool, price: float | None) -> float | None:
    mult = _american_profit_mult(price)
    if mult is None:
        return None
    return mult if won else -1.0


def _price_bucket(price: float | None) -> str:
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


def _edge_bucket(edge: float | None, market: str) -> str:
    edge = abs(_as_float(edge) or 0.0)
    if market == "run_line":
        if edge < 2.0:
            return "edge_lt_2"
        if edge < 3.0:
            return "edge_2_3"
        return "edge_3plus"
    if market == "total":
        if edge < 2.0:
            return "edge_lt_2"
        if edge < 3.0:
            return "edge_2_3"
        return "edge_3plus"
    if edge < 0.5:
        return "edge_lt_0.5"
    if edge < 1.0:
        return "edge_0.5_1"
    if edge < 1.5:
        return "edge_1_1.5"
    if edge < 2.0:
        return "edge_1.5_2"
    return "edge_2plus"


def _run_line_side_bucket(side: str, line: float | None) -> str:
    line = _as_float(line)
    if line is None:
        return f"{side}_unknown"
    if side == "home":
        return "home_dog" if line > 0 else "home_fav"
    if side == "away":
        return "away_dog" if line < 0 else "away_fav"
    return "unknown"


def _standard_run_line(line: float | None) -> bool:
    line = _as_float(line)
    return line is not None and abs(abs(line) - 1.5) <= 1e-9


def _side_from_edge(row: pd.Series) -> str | None:
    side = row.get("bet_side")
    if isinstance(side, str) and side:
        return side.lower()
    edge = _as_float(row.get("edge"))
    if edge is None or abs(edge) <= 1e-12:
        return None
    return "over" if edge > 0 else "under"


def _prop_side_won(side: str | None, over_hit) -> bool | None:
    if side not in {"over", "under"} or over_hit is None or pd.isna(over_hit):
        return None
    hit = bool(over_hit)
    return hit if side == "over" else not hit


def _prop_threshold_pass(stat: str, side: str | None, edge: float | None, cfg: AuditConfig) -> bool:
    edge = _as_float(edge)
    if side not in {"over", "under"} or edge is None:
        return False
    abs_edge = abs(edge)
    if stat == "pitcher_strikeouts":
        return abs_edge >= cfg.threshold_strikeouts
    if stat == "batter_hits":
        return side == "over" and abs_edge >= cfg.threshold_hits
    if stat == "batter_total_bases":
        return abs_edge >= cfg.threshold_total_bases
    if stat == "batter_home_runs":
        return edge >= cfg.threshold_home_runs_over if side == "over" else abs_edge >= cfg.threshold_home_runs_under
    return False


def _prop_side_blocked(stat: str, side: str | None) -> str | None:
    if side == "under" and stat in {"batter_hits", "batter_home_runs"}:
        return "unbookable_under"
    if stat == "batter_home_runs":
        return "hr_longshot_variance"
    if stat == "batter_total_bases" and side == "over":
        return "weak_over_bucket"
    return None


def _load_games(engine, cfg: AuditConfig) -> pd.DataFrame:
    raw = pd.read_sql(
        text(GAME_SQL),
        engine,
        params={"start_date": cfg.start_date, "end_date": cfg.end_date},
    )
    rows: list[dict] = []
    for _, row in raw.iterrows():
        if pd.notna(row.get("run_line_bet_side")) and pd.notna(row.get("run_line_covered")):
            side = str(row["run_line_bet_side"]).lower()
            price = _as_float(row.get("market_rl_price"))
            line = _as_float(row.get("market_run_line"))
            reasons: list[str] = []
            if not _standard_run_line(line):
                reasons.append("non_standard_run_line")
            if price is None:
                reasons.append("missing_price")
            elif price < cfg.max_run_line_lay_price:
                reasons.append("heavy_juice")
            if side == "away" and line is not None and line > 0:
                reasons.append("away_favorite_run_line")
            if side == "away" and line is not None and line < 0 and price is not None and price < cfg.max_away_dog_lay_price:
                reasons.append("away_dog_lay_price")
            won = bool(row["run_line_covered"])
            rows.append({
                "game_date_et": row["game_date_et"],
                "source": "game",
                "market": "run_line",
                "side": side,
                "bucket": _run_line_side_bucket(side, line),
                "line_bucket": "standard_1.5" if _standard_run_line(line) else "alt_line",
                "price_bucket": _price_bucket(price),
                "edge_bucket": _edge_bucket(row.get("edge_run_line"), "run_line"),
                "label": f"{row['away_team_abbr']}@{row['home_team_abbr']} run_line {side}",
                "price": price,
                "edge": _as_float(row.get("edge_run_line")),
                "won": won,
                "profit": _profit(won, price),
                "clv": _as_float(row.get("clv_run_line")),
                "price_clv": _as_float(row.get("clv_rl_price")),
                "qualifies": not reasons,
                "reason": "; ".join(reasons) if reasons else "ok",
                "stake_pct": cfg.max_stake_pct,
                "priority": abs(_as_float(row.get("edge_run_line")) or 0.0),
            })
        if pd.notna(row.get("total_bet_side")) and pd.notna(row.get("total_covered")):
            side = str(row["total_bet_side"]).lower()
            price = _as_float(row.get("market_total_price"))
            reasons = []
            if side == "under":
                reasons.append("total_under_disabled")
            if price is None:
                reasons.append("missing_price")
            elif price < cfg.max_run_line_lay_price:
                reasons.append("heavy_juice")
            won = bool(row["total_covered"])
            rows.append({
                "game_date_et": row["game_date_et"],
                "source": "game",
                "market": "total",
                "side": side,
                "bucket": f"total_{side}",
                "line_bucket": "total",
                "price_bucket": _price_bucket(price),
                "edge_bucket": _edge_bucket(row.get("edge_total"), "total"),
                "label": f"{row['away_team_abbr']}@{row['home_team_abbr']} total {side}",
                "price": price,
                "edge": _as_float(row.get("edge_total")),
                "won": won,
                "profit": _profit(won, price),
                "clv": _as_float(row.get("clv_total")),
                "price_clv": _as_float(row.get("clv_total_price")),
                "qualifies": not reasons,
                "reason": "; ".join(reasons) if reasons else "ok",
                "stake_pct": cfg.max_stake_pct,
                "priority": abs(_as_float(row.get("edge_total")) or 0.0),
            })
    return pd.DataFrame(rows)


def _load_props(engine, cfg: AuditConfig) -> pd.DataFrame:
    raw = pd.read_sql(
        text(PROP_SQL),
        engine,
        params={"start_date": cfg.start_date, "end_date": cfg.end_date},
    )
    rows: list[dict] = []
    for _, row in raw.iterrows():
        stat = str(row.get("stat"))
        side = _side_from_edge(row)
        edge = _as_float(row.get("edge"))
        price = _as_float(row.get("bet_price"))
        if price is None:
            price = _as_float(row.get("over_price") if side == "over" else row.get("under_price"))
        ev = _as_float(row.get("ev"))
        kelly = _as_float(row.get("kelly_fraction"))
        side_block = _prop_side_blocked(stat, side)
        reasons: list[str] = []
        if not _prop_threshold_pass(stat, side, edge, cfg):
            reasons.append("no_model_edge")
        if side_block:
            reasons.append(side_block)
        if price is None:
            reasons.append("missing_price")
        elif price < cfg.max_prop_lay_price:
            reasons.append("heavy_juice")
        if ev is None:
            reasons.append("missing_ev")
        elif ev < cfg.min_ev:
            reasons.append("ev_below_min")
        if kelly is None or kelly <= 0:
            reasons.append("zero_kelly")
        won = _prop_side_won(side, row.get("over_hit"))
        if won is None:
            continue
        line_bucket = str(row.get("line_bucket") or row.get("book_line") or "unknown")
        rows.append({
            "game_date_et": row["game_date_et"],
            "source": "prop",
            "market": stat,
            "side": side or "unknown",
            "bucket": f"{stat}_{side}",
            "line_bucket": line_bucket,
            "price_bucket": _price_bucket(price),
            "edge_bucket": _edge_bucket(edge, stat),
            "label": f"{row.get('player_name')} {stat} {side}",
            "price": price,
            "edge": edge,
            "won": won,
            "profit": _profit(won, price),
            "clv": None,
            "price_clv": None,
            "qualifies": not reasons,
            "reason": "; ".join(reasons) if reasons else "ok",
            "stake_pct": cfg.max_stake_pct,
            "priority": (ev if ev is not None else abs(edge or 0.0)),
        })
    return pd.DataFrame(rows)


def _apply_daily_cap(df: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    if df.empty:
        df["selected"] = []
        return df
    out = df.copy()
    out["selected"] = False
    candidates = out[out["qualifies"]].sort_values(
        ["game_date_et", "priority"],
        ascending=[True, False],
    )
    for day, group in candidates.groupby("game_date_et"):
        used = 0.0
        for idx, row in group.iterrows():
            stake = float(row.get("stake_pct") or cfg.max_stake_pct)
            if used + stake <= cfg.global_cap_pct + 1e-12:
                out.at[idx, "selected"] = True
                used += stake
    return out


def _summary(df: pd.DataFrame) -> dict:
    n = int(len(df))
    wins = int(df["won"].sum()) if n else 0
    profit = pd.to_numeric(df["profit"], errors="coerce")
    profit_n = int(profit.notna().sum())
    roi = float(profit.mean() * 100.0) if profit_n else float("nan")
    return {
        "n": n,
        "wins": wins,
        "losses": n - wins,
        "win_pct": (wins / n * 100.0) if n else float("nan"),
        "roi": roi,
        "avg_clv": float(pd.to_numeric(df.get("clv"), errors="coerce").mean()) if n and "clv" in df else float("nan"),
        "avg_price_clv": float(pd.to_numeric(df.get("price_clv"), errors="coerce").mean()) if n and "price_clv" in df else float("nan"),
    }


def _fmt_pct(value: float) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.1f}%"


def _fmt_num(value: float) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.2f}"


def _markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    headers = [h for h, _ in columns]
    keys = [k for _, k in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)


def _bucket_rows(df: pd.DataFrame, cfg: AuditConfig, by: list[str], limit: int = 30) -> list[dict]:
    if df.empty:
        return []
    rows: list[dict] = []
    for key, group in df.groupby(by, dropna=False):
        if len(group) < cfg.min_bets:
            continue
        s = _summary(group)
        if not isinstance(key, tuple):
            key = (key,)
        rec = {name: val for name, val in zip(by, key)}
        rec.update({
            "bets": s["n"],
            "w_l": f"{s['wins']}-{s['losses']}",
            "win_pct": _fmt_pct(s["win_pct"]),
            "roi": _fmt_pct(s["roi"]),
            "avg_clv": _fmt_num(s["avg_clv"]),
            "flag": "OK" if s["roi"] > 0 else "FAIL",
        })
        rows.append(rec)
    rows.sort(key=lambda r: (r["flag"] != "FAIL", -r["bets"]))
    return rows[:limit]


def build_report(cfg: AuditConfig) -> str:
    engine = create_engine(cfg.dsn)
    games = _load_games(engine, cfg)
    props = _load_props(engine, cfg)
    all_rows = pd.concat([games, props], ignore_index=True) if not games.empty or not props.empty else pd.DataFrame()
    if not all_rows.empty:
        all_rows["game_date_et"] = pd.to_datetime(all_rows["game_date_et"]).dt.date
    audited = _apply_daily_cap(all_rows, cfg) if not all_rows.empty else all_rows

    selected = audited[audited["selected"]] if not audited.empty else audited
    pre_cap = audited[audited["qualifies"]] if not audited.empty else audited
    paper = audited[~audited["qualifies"]] if not audited.empty else audited

    days = pd.date_range(cfg.start_date, cfg.end_date, freq="D").date
    daily_counts = (
        selected.groupby("game_date_et").size().to_dict()
        if not selected.empty else {}
    )
    no_bet_days = sum(1 for d in days if int(daily_counts.get(d, 0)) == 0)
    avg_bets = (len(selected) / len(days)) if len(days) else 0.0

    overall = _summary(selected) if not selected.empty else _summary(pd.DataFrame(columns=["won", "profit"]))
    pre = _summary(pre_cap) if not pre_cap.empty else _summary(pd.DataFrame(columns=["won", "profit"]))

    daily_rows = []
    for d in sorted(days, reverse=True)[:21]:
        day_df = selected[selected["game_date_et"] == d] if not selected.empty else selected
        s = _summary(day_df) if not day_df.empty else {"n": 0, "wins": 0, "losses": 0, "win_pct": float("nan"), "roi": float("nan")}
        daily_rows.append({
            "date": d.isoformat(),
            "bets": s["n"],
            "w_l": f"{s['wins']}-{s['losses']}" if s["n"] else "0-0",
            "roi": _fmt_pct(s["roi"]),
        })

    market_rows = _bucket_rows(selected, cfg, ["source", "market", "side"], limit=40)
    price_rows = _bucket_rows(selected, cfg, ["source", "market", "side", "price_bucket"], limit=40)
    reject_rows = []
    if not paper.empty:
        for (source, market, reason), group in paper.groupby(["source", "market", "reason"], dropna=False):
            reject_rows.append({
                "source": source,
                "market": market,
                "reason": reason,
                "rows": len(group),
            })
        reject_rows.sort(key=lambda r: -r["rows"])
        reject_rows = reject_rows[:25]

    lines = [
        f"# MLB Real-Money Audit ({cfg.start_date} to {cfg.end_date})",
        "",
        "Scope: saved prediction rows re-filtered through current real-money gates.",
        "Limitation: this is not a locked-ledger walk-forward; reruns can overwrite historical predictions.",
        "",
        "## Overall",
        "",
        f"- Pre-cap qualifying bets: {pre['n']} ({pre['wins']}-{pre['losses']}, ROI {_fmt_pct(pre['roi'])})",
        f"- Selected after daily cap: {overall['n']} ({overall['wins']}-{overall['losses']}, ROI {_fmt_pct(overall['roi'])})",
        f"- Bet days: {len(days) - no_bet_days}/{len(days)}",
        f"- No-bet days: {no_bet_days}/{len(days)}",
        f"- Avg selected bets/day: {avg_bets:.2f}",
        f"- Global cap: {cfg.global_cap_pct * 100:.2f}% per day",
        f"- Prop thresholds: K {cfg.threshold_strikeouts:g}, H {cfg.threshold_hits:g}, "
        f"TB {cfg.threshold_total_bases:g}, HR over {cfg.threshold_home_runs_over:g}",
        "",
        "## Recent Daily Volume",
        "",
        _markdown_table(daily_rows, [("Date", "date"), ("Bets", "bets"), ("W-L", "w_l"), ("ROI", "roi")]),
        "",
        "## Selected Buckets",
        "",
        _markdown_table(
            market_rows,
            [
                ("Source", "source"),
                ("Market", "market"),
                ("Side", "side"),
                ("Bets", "bets"),
                ("W-L", "w_l"),
                ("Win%", "win_pct"),
                ("ROI", "roi"),
                ("Avg CLV", "avg_clv"),
                ("Flag", "flag"),
            ],
        ),
        "",
        "## Price Buckets",
        "",
        _markdown_table(
            price_rows,
            [
                ("Source", "source"),
                ("Market", "market"),
                ("Side", "side"),
                ("Price", "price_bucket"),
                ("Bets", "bets"),
                ("W-L", "w_l"),
                ("ROI", "roi"),
                ("Flag", "flag"),
            ],
        ),
        "",
        "## Top Rejection Reasons",
        "",
        _markdown_table(reject_rows, [("Source", "source"), ("Market", "market"), ("Reason", "reason"), ("Rows", "rows")]),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> AuditConfig:
    parser = argparse.ArgumentParser(description="Audit current MLB real-money gates")
    parser.add_argument("--dsn", default=_PG_DSN)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--min-bets", type=int, default=20)
    parser.add_argument("--out", default=None, help="Optional markdown file path")
    parser.add_argument("--global-cap-pct", type=float, default=0.02)
    parser.add_argument("--prop-thresholds-file", default=str(_PROP_THRESHOLDS))
    args = parser.parse_args()

    end_date = date.fromisoformat(args.end_date) if args.end_date else datetime.now(_ET).date()
    start_date = (
        date.fromisoformat(args.start_date)
        if args.start_date
        else end_date - timedelta(days=max(1, args.days) - 1)
    )
    cfg = AuditConfig(
        start_date=start_date,
        end_date=end_date,
        dsn=args.dsn,
        out=args.out,
        prop_thresholds_file=Path(args.prop_thresholds_file) if args.prop_thresholds_file else None,
        global_cap_pct=args.global_cap_pct,
        min_bets=args.min_bets,
    )
    if cfg.prop_thresholds_file and cfg.prop_thresholds_file.exists():
        try:
            payload = json.loads(cfg.prop_thresholds_file.read_text(encoding="utf-8"))
            cfg = replace(
                cfg,
                threshold_strikeouts=float(payload.get("threshold_strikeouts", cfg.threshold_strikeouts)),
                threshold_hits=float(payload.get("threshold_hits", cfg.threshold_hits)),
                threshold_total_bases=float(payload.get("threshold_total_bases", cfg.threshold_total_bases)),
                threshold_home_runs_over=float(payload.get("threshold_home_runs_over", cfg.threshold_home_runs_over)),
                threshold_home_runs_under=float(payload.get("threshold_home_runs_under", cfg.threshold_home_runs_under)),
                min_ev=float(payload.get("min_ev", cfg.min_ev)),
            )
        except Exception:
            pass
    return cfg


def main() -> None:
    cfg = parse_args()
    report = build_report(cfg)
    out = cfg.out
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        print(f"Wrote {path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
