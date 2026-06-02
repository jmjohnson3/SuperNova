"""Diagnose MLB prop model picks, warnings, and recalibration candidates.

This report is intentionally diagnostic. It explains which positive-EV prop rows
the model is selecting, which warnings are attached, and which historical
buckets have enough evidence to consider recalibration.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

_ET = ZoneInfo("America/New_York")
_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_THRESHOLDS_FILE = _MODEL_DIR / "prop_thresholds.json"
_BREAKEVEN_PROB = 0.524
_DISABLED_THRESHOLD = 900.0

_DEFAULT_THRESHOLDS = {
    "threshold_strikeouts": 2.0,
    "threshold_strikeouts_over": None,
    "threshold_strikeouts_under": None,
    "threshold_hits": 0.75,
    "threshold_total_bases": 1.5,
    "threshold_total_bases_over": None,
    "threshold_total_bases_under": None,
    "threshold_home_runs_over": 0.05,
    "threshold_home_runs_under": 0.45,
    "threshold_clf": 0.03,
    "min_ev": 0.02,
}

_FD_OVER_ONLY = {"batter_hits", "batter_home_runs"}


@dataclass(frozen=True)
class DiagnosticConfig:
    report_date: date
    start_date: date
    end_date: date
    dsn: str
    out: str | None = None
    top_n: int = 40
    min_train_bets: int = 40
    min_holdout_bets: int = 20
    holdout_days: int = 21
    max_lay_price: float = -180.0
    min_ev: float = 0.02
    thresholds_file: Path = _THRESHOLDS_FILE


TODAY_SQL = """
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
    bankroll_tier,
    bankroll_candidate,
    bankroll_reasons,
    stake_pct
FROM bets.mlb_prop_predictions
WHERE game_date_et = :report_date
  AND edge IS NOT NULL
"""


HISTORICAL_SQL = """
SELECT
    game_date_et,
    player_name,
    team_abbr,
    stat,
    pred_value,
    pred_count,
    pred_prob_over,
    book_line,
    edge,
    edge_type,
    bet_side,
    line_bucket,
    over_price,
    under_price,
    bet_price,
    ev,
    over_hit,
    actual_value
FROM bets.mlb_prop_predictions
WHERE game_date_et BETWEEN :start_date AND :end_date
  AND edge IS NOT NULL
  AND over_hit IS NOT NULL
  AND stat IN (
      'pitcher_strikeouts',
      'batter_hits',
      'batter_total_bases',
      'batter_home_runs'
  )
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


def _american_profit_mult(price) -> float | None:
    price = _as_float(price)
    if price is None or price == 0:
        return None
    return price / 100.0 if price > 0 else 100.0 / abs(price)


def _profit_units(won: bool, price) -> float | None:
    mult = _american_profit_mult(price)
    if mult is None:
        return None
    return mult if won else -1.0


def _side_from_row(row: pd.Series) -> str | None:
    side = row.get("bet_side")
    if isinstance(side, str) and side.lower() in {"over", "under"}:
        return side.lower()
    edge = _as_float(row.get("edge"))
    if edge is None or abs(edge) <= 1e-12:
        return None
    return "over" if edge > 0 else "under"


def _side_won(side: str | None, over_hit) -> bool | None:
    if side not in {"over", "under"}:
        return None
    try:
        if pd.isna(over_hit):
            return None
    except Exception:
        pass
    hit = bool(over_hit)
    return hit if side == "over" else not hit


def _load_thresholds(path: Path) -> tuple[dict[str, float], dict]:
    thresholds = dict(_DEFAULT_THRESHOLDS)
    diagnostics: dict = {}
    if not path.exists():
        return thresholds, diagnostics
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return thresholds, diagnostics
    for key in thresholds:
        if key in payload:
            if payload[key] is None:
                thresholds[key] = None
            else:
                try:
                    thresholds[key] = float(payload[key])
                except Exception:
                    pass
    diagnostics = payload.get("_diagnostics", {}) if isinstance(payload, dict) else {}
    return thresholds, diagnostics


def _threshold_name(stat: str, side: str | None, edge_type: str | None) -> str:
    if edge_type == "probability":
        return "threshold_clf"
    if stat == "pitcher_strikeouts":
        if side == "over":
            return "threshold_strikeouts_over"
        if side == "under":
            return "threshold_strikeouts_under"
        return "threshold_strikeouts"
    if stat == "batter_hits":
        return "threshold_hits"
    if stat == "batter_total_bases":
        if side == "over":
            return "threshold_total_bases_over"
        if side == "under":
            return "threshold_total_bases_under"
        return "threshold_total_bases"
    if stat == "batter_home_runs":
        return "threshold_home_runs_over" if side != "under" else "threshold_home_runs_under"
    return "unknown"


def _threshold_pass(row: pd.Series, thresholds: dict[str, float], *, use_defaults: bool = False) -> bool:
    side = _side_from_row(row)
    edge = _as_float(row.get("edge"))
    if side is None or edge is None:
        return False
    stat = str(row.get("stat") or "")
    edge_type = str(row.get("edge_type") or "")
    name = _threshold_name(stat, side, edge_type)
    source = _DEFAULT_THRESHOLDS if use_defaults else thresholds
    threshold = source.get(name)
    if threshold is None:
        fallback_name = "threshold_strikeouts" if stat == "pitcher_strikeouts" else (
            "threshold_total_bases" if stat == "batter_total_bases" else name
        )
        threshold = source.get(fallback_name)
    if threshold is None or threshold >= _DISABLED_THRESHOLD:
        return False
    if stat == "batter_home_runs" and side == "over":
        return edge >= threshold
    return abs(edge) >= threshold


def _block_reasons(row: pd.Series, thresholds: dict[str, float], cfg: DiagnosticConfig) -> list[str]:
    reasons: list[str] = []
    stored = row.get("bankroll_reasons")
    if isinstance(stored, str) and stored.strip():
        reasons.extend([part.strip() for part in stored.split(";") if part.strip()])
    side = _side_from_row(row)
    stat = str(row.get("stat") or "")
    edge_type = str(row.get("edge_type") or "")
    edge = _as_float(row.get("edge"))
    ev = _as_float(row.get("ev"))
    ev_qualified = ev is not None and ev >= cfg.min_ev
    if side is None:
        reasons.append("missing_side")
    if edge is None:
        reasons.append("missing_edge")

    name = _threshold_name(stat, side, edge_type)
    threshold = thresholds.get(name)
    if threshold is None:
        fallback_name = "threshold_strikeouts" if stat == "pitcher_strikeouts" else (
            "threshold_total_bases" if stat == "batter_total_bases" else name
        )
        threshold = thresholds.get(fallback_name)
    if threshold is None:
        reasons.append("unknown_threshold")
    elif threshold >= _DISABLED_THRESHOLD:
        if not ev_qualified:
            reasons.append(f"threshold_disabled:{name}")
    elif edge is not None and not _threshold_pass(row, thresholds):
        if not ev_qualified:
            reasons.append(f"below_edge_threshold:{name}")

    if side == "under" and stat in _FD_OVER_ONLY:
        reasons.append("unbookable_under")
    if stat == "batter_home_runs":
        reasons.append("hr_longshot_variance")

    price = _as_float(row.get("bet_price"))
    if price is None:
        reasons.append("missing_price")
    elif price < cfg.max_lay_price:
        reasons.append("heavy_juice")

    if ev is None:
        reasons.append("missing_ev")
    elif ev < cfg.min_ev:
        reasons.append("ev_below_min")

    kelly = _as_float(row.get("kelly_fraction"))
    if kelly is None or kelly <= 0:
        reasons.append("zero_kelly")

    deduped: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason and reason not in seen:
            seen.add(reason)
            deduped.append(reason)
    return deduped


def _load_df(engine, sql: str, params: dict) -> pd.DataFrame:
    df = pd.read_sql(text(sql), engine, params=params)
    if df.empty:
        return df
    if "game_date_et" in df:
        df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in (
        "pred_value",
        "pred_count",
        "pred_prob_over",
        "book_line",
        "edge",
        "over_price",
        "under_price",
        "bet_price",
        "breakeven_prob",
        "ev",
        "kelly_fraction",
        "stake_pct",
        "actual_value",
    ):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _candidate_thresholds(edges: pd.Series) -> list[float]:
    vals = pd.to_numeric(edges, errors="coerce").abs().dropna()
    if vals.empty:
        return []
    base = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50]
    qs = [float(np.quantile(vals, q)) for q in (0.5, 0.6, 0.7, 0.8, 0.9)]
    return sorted({round(v, 3) for v in [*base, *qs] if v >= 0})


def _bucket_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "wins": 0, "losses": 0, "wr": None, "units": 0.0, "roi": None}
    wins = 0
    losses = 0
    units = 0.0
    for _, row in df.iterrows():
        won = _side_won(row.get("_side"), row.get("over_hit"))
        price = row.get("_price", row.get("bet_price"))
        profit = _profit_units(bool(won), price) if won is not None else None
        if profit is None:
            continue
        if won:
            wins += 1
        else:
            losses += 1
        units += profit
    risked = wins + losses
    return {
        "n": risked,
        "wins": wins,
        "losses": losses,
        "wr": (wins / risked * 100.0) if risked else None,
        "units": units,
        "roi": (units / risked * 100.0) if risked else None,
    }


def _pick_recalibration_threshold(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    cfg: DiagnosticConfig,
) -> dict:
    cands = _candidate_thresholds(train["edge"])
    if not cands:
        return {"reason": "no_candidates"}
    best = None
    for threshold in cands:
        sub = train[train["edge"].abs() >= threshold]
        summary = _bucket_summary(sub)
        if summary["n"] < cfg.min_train_bets:
            continue
        rec = {"threshold": threshold, **summary}
        if best is None or (rec["roi"] or -999) > (best["roi"] or -999):
            best = rec
    if best is None:
        return {"reason": "train_sample_too_small"}
    ho = holdout[holdout["edge"].abs() >= best["threshold"]]
    ho_summary = _bucket_summary(ho)
    return {"reason": "scanned", "train": best, "holdout": ho_summary}


def _threshold_status_rows(thresholds: dict[str, float], diagnostics: dict) -> list[dict]:
    rows = []
    diag_map = {
        "threshold_strikeouts": "pitcher_strikeouts",
        "threshold_strikeouts_over": "pitcher_strikeouts_over",
        "threshold_strikeouts_under": "pitcher_strikeouts_under",
        "threshold_hits": "batter_hits",
        "threshold_total_bases": "batter_total_bases",
        "threshold_total_bases_over": "batter_total_bases_over",
        "threshold_total_bases_under": "batter_total_bases_under",
        "threshold_home_runs_over": "batter_home_runs",
        "threshold_home_runs_under": "batter_home_runs",
        "threshold_clf": "classifier_probability",
    }
    for key in [
        "threshold_strikeouts",
        "threshold_strikeouts_over",
        "threshold_strikeouts_under",
        "threshold_hits",
        "threshold_total_bases",
        "threshold_total_bases_over",
        "threshold_total_bases_under",
        "threshold_home_runs_over",
        "threshold_home_runs_under",
        "threshold_clf",
        "min_ev",
    ]:
        value = thresholds.get(key)
        diag = diagnostics.get(diag_map.get(key, key), {})
        rows.append({
            "threshold": key,
            "value": _fmt_num(value, 3),
            "status": "disabled" if value is not None and value >= _DISABLED_THRESHOLD else "active",
            "optimizer_reason": diag.get("reason", "-") if isinstance(diag, dict) else "-",
        })
    return rows


def _today_summary_rows(today: pd.DataFrame, thresholds: dict[str, float], cfg: DiagnosticConfig) -> list[dict]:
    if today.empty:
        return []
    rows = []
    for stat, sub in today.groupby("stat", sort=True):
        ev = pd.to_numeric(sub["ev"], errors="coerce")
        edge = pd.to_numeric(sub["edge"], errors="coerce").abs()
        bankroll = sub["bankroll_candidate"].fillna(False).astype(bool) if "bankroll_candidate" in sub else pd.Series(False, index=sub.index)
        warnings = 0
        for _, row in sub.iterrows():
            if _block_reasons(row, thresholds, cfg):
                warnings += 1
        rows.append({
            "stat": stat,
            "rows": len(sub),
            "positive_ev": int((ev >= cfg.min_ev).sum()),
            "bankroll": int(bankroll.sum()),
            "warnings": warnings,
            "max_edge": _fmt_num(edge.max(), 3),
            "max_ev": _fmt_num(ev.max(), 4),
        })
    return rows


def _reason_rows(today: pd.DataFrame, thresholds: dict[str, float], cfg: DiagnosticConfig) -> list[dict]:
    counts: dict[tuple[str, str], int] = {}
    if today.empty:
        return []
    for _, row in today.iterrows():
        stat = str(row.get("stat") or "")
        for reason in _block_reasons(row, thresholds, cfg) or ["ok"]:
            key = (stat, reason)
            counts[key] = counts.get(key, 0) + 1
    return [
        {"stat": stat, "reason": reason, "rows": rows}
        for (stat, reason), rows in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ]


def _watchlist_rows(
    today: pd.DataFrame,
    thresholds: dict[str, float],
    cfg: DiagnosticConfig,
    *,
    use_defaults: bool = False,
) -> list[dict]:
    if today.empty:
        return []
    rows = []
    for _, row in today.iterrows():
        ev = _as_float(row.get("ev"))
        if ev is None or ev < cfg.min_ev:
            continue
        side = _side_from_row(row)
        reasons = _block_reasons(row, thresholds, cfg)
        if use_defaults:
            if not _threshold_pass(row, thresholds, use_defaults=True):
                continue
            hard_reasons = [
                r for r in reasons
                if not r.startswith("threshold_disabled") and not r.startswith("below_edge_threshold")
            ]
            if hard_reasons:
                continue
            reasons = ["threshold_reopen_needed"]
        elif not reasons:
            continue
        p_over = _as_float(row.get("pred_prob_over"))
        rows.append({
            "player": row.get("player_name"),
            "team": row.get("team_abbr"),
            "stat": row.get("stat"),
            "side": side or "-",
            "line": _fmt_num(row.get("book_line"), 1),
            "price": _fmt_price(row.get("bet_price")),
            "pred": _fmt_num(row.get("pred_count") if pd.notna(row.get("pred_count")) else row.get("pred_value"), 3),
            "p_over": _fmt_pct(p_over * 100.0 if p_over is not None else None),
            "edge": _fmt_num(abs(_as_float(row.get("edge")) or 0.0), 3, signed=True),
            "ev": _fmt_pct(ev * 100.0, signed=True),
            "reasons": "; ".join(reasons),
        })
    rows.sort(key=lambda r: float(str(r["ev"]).replace("%", "").replace("+", "")) if r["ev"] != "-" else -999, reverse=True)
    return rows[: cfg.top_n]


def _prepare_historical(hist: pd.DataFrame, cfg: DiagnosticConfig) -> pd.DataFrame:
    if hist.empty:
        return hist
    out = hist.copy()
    out["_side"] = out.apply(_side_from_row, axis=1)
    out["_raw_price"] = pd.to_numeric(out["bet_price"], errors="coerce")
    out["_price"] = out["_raw_price"].fillna(-110.0)
    out["_edge_type"] = out["edge_type"].fillna("count").astype(str)
    out = out[out["_side"].isin(["over", "under"])]
    out = out[out["_raw_price"].isna() | (out["_raw_price"] >= cfg.max_lay_price)]
    out = out[~((out["stat"].isin(_FD_OVER_ONLY)) & (out["_side"] == "under"))]
    return out


def _recalibration_rows(hist: pd.DataFrame, thresholds: dict[str, float], cfg: DiagnosticConfig) -> list[dict]:
    hist = _prepare_historical(hist, cfg)
    if hist.empty:
        return []
    split = cfg.end_date - timedelta(days=cfg.holdout_days)
    train = hist[hist["game_date_et"] < split]
    holdout = hist[hist["game_date_et"] >= split]
    rows = []
    for (stat, side, edge_type), tr in train.groupby(["stat", "_side", "_edge_type"], sort=True):
        ho = holdout[
            (holdout["stat"] == stat)
            & (holdout["_side"] == side)
            & (holdout["_edge_type"] == edge_type)
        ]
        rec = _pick_recalibration_threshold(tr, ho, cfg)
        name = _threshold_name(str(stat), str(side), str(edge_type))
        current = thresholds.get(name)
        if current is None:
            if stat == "pitcher_strikeouts":
                current = thresholds.get("threshold_strikeouts")
            elif stat == "batter_total_bases":
                current = thresholds.get("threshold_total_bases")
        disabled = current is not None and current >= _DISABLED_THRESHOLD
        train_rec = rec.get("train") or {}
        holdout_rec = rec.get("holdout") or {}
        holdout_n = int(holdout_rec.get("n") or 0)
        holdout_roi = holdout_rec.get("roi")
        train_roi = train_rec.get("roi")
        if rec.get("reason") != "scanned":
            recommendation = rec.get("reason")
        elif (train_roi or -999.0) <= 0:
            recommendation = "KEEP_DISABLED_TRAIN_BAD" if disabled else "ACTIVE_TRAIN_BAD"
        elif holdout_n < cfg.min_holdout_bets:
            recommendation = "KEEP_DISABLED_SAMPLE" if disabled else "ACTIVE_SAMPLE_LOW"
        elif (holdout_roi or -999.0) <= 0:
            recommendation = "KEEP_DISABLED_BAD" if disabled else "ACTIVE_WEAK"
        elif disabled:
            recommendation = "SIDE_SPECIFIC_REOPEN_CANDIDATE"
        else:
            recommendation = "ACTIVE_OK"
        rows.append({
            "bucket": f"{stat}/{side}/{edge_type}",
            "current_threshold": "disabled" if disabled else _fmt_num(current, 3),
            "suggested_threshold": _fmt_num(train_rec.get("threshold"), 3),
            "train_n": int(train_rec.get("n") or 0),
            "train_w_l": f"{int(train_rec.get('wins') or 0)}-{int(train_rec.get('losses') or 0)}",
            "train_roi": _fmt_pct(train_rec.get("roi"), signed=True),
            "holdout_n": holdout_n,
            "holdout_w_l": f"{int(holdout_rec.get('wins') or 0)}-{int(holdout_rec.get('losses') or 0)}",
            "holdout_roi": _fmt_pct(holdout_roi, signed=True),
            "recommendation": recommendation,
        })
    priority = {
        "REOPEN_CANDIDATE": 0,
        "ACTIVE_OK": 1,
        "ACTIVE_WEAK": 2,
        "KEEP_DISABLED_BAD": 3,
        "KEEP_DISABLED_SAMPLE": 4,
        "ACTIVE_SAMPLE_LOW": 5,
    }
    rows.sort(key=lambda r: (priority.get(str(r["recommendation"]), 9), -int(r["holdout_n"]), r["bucket"]))
    return rows


def build_report(cfg: DiagnosticConfig) -> str:
    engine = create_engine(cfg.dsn)
    thresholds, diagnostics = _load_thresholds(cfg.thresholds_file)
    thresholds["min_ev"] = cfg.min_ev
    today = _load_df(engine, TODAY_SQL, {"report_date": cfg.report_date})
    hist = _load_df(
        engine,
        HISTORICAL_SQL,
        {"start_date": cfg.start_date, "end_date": cfg.end_date},
    )
    watchlist = _watchlist_rows(today, thresholds, cfg)
    fallback = _watchlist_rows(today, thresholds, cfg, use_defaults=True)
    recalibration = _recalibration_rows(hist, thresholds, cfg)

    lines = [
        "# MLB Prop Model Pick Diagnostics",
        "",
        f"Date: {cfg.report_date}",
        f"Historical window: {cfg.start_date} to {cfg.end_date}",
        "Purpose: explain positive-EV model picks, attached warnings, and recalibration candidates.",
        "",
        "## Threshold Status",
        "",
        _markdown_table(
            _threshold_status_rows(thresholds, diagnostics),
            [
                ("Threshold", "threshold"),
                ("Value", "value"),
                ("Status", "status"),
                ("Optimizer Reason", "optimizer_reason"),
            ],
        ),
        "",
        "## Today By Stat",
        "",
        _markdown_table(
            _today_summary_rows(today, thresholds, cfg),
            [
                ("Stat", "stat"),
                ("Rows", "rows"),
                ("Positive EV", "positive_ev"),
                ("Bankroll", "bankroll"),
                ("Warnings", "warnings"),
                ("Max Edge", "max_edge"),
                ("Max EV", "max_ev"),
            ],
        ),
        "",
        "## Warning Reasons",
        "",
        _markdown_table(
            _reason_rows(today, thresholds, cfg),
            [("Stat", "stat"), ("Reason", "reason"), ("Rows", "rows")],
        ),
        "",
        "## Positive-EV Warning Watchlist",
        "",
        _markdown_table(
            watchlist,
            [
                ("Player", "player"),
                ("Team", "team"),
                ("Stat", "stat"),
                ("Side", "side"),
                ("Line", "line"),
                ("Price", "price"),
                ("Pred", "pred"),
                ("P(over)", "p_over"),
                ("Edge", "edge"),
                ("EV", "ev"),
                ("Reasons", "reasons"),
            ],
        ),
        "",
        "## Fallback Reopen Test",
        "",
        "These would clear the old/default edge threshold and hard filters today. Treat this as historical context, not an instruction to suppress the model pick.",
        "",
        _markdown_table(
            fallback,
            [
                ("Player", "player"),
                ("Team", "team"),
                ("Stat", "stat"),
                ("Side", "side"),
                ("Line", "line"),
                ("Price", "price"),
                ("Pred", "pred"),
                ("P(over)", "p_over"),
                ("Edge", "edge"),
                ("EV", "ev"),
                ("Reasons", "reasons"),
            ],
        ),
        "",
        "## Recalibration Scan",
        "",
        "Buckets are stat/side/edge_type. SIDE_SPECIFIC_REOPEN_CANDIDATE means the current threshold is disabled, train and holdout have enough sample, and holdout ROI is positive.",
        "",
        _markdown_table(
            recalibration,
            [
                ("Bucket", "bucket"),
                ("Current", "current_threshold"),
                ("Suggested", "suggested_threshold"),
                ("Train N", "train_n"),
                ("Train W-L", "train_w_l"),
                ("Train ROI", "train_roi"),
                ("Holdout N", "holdout_n"),
                ("Holdout W-L", "holdout_w_l"),
                ("Holdout ROI", "holdout_roi"),
                ("Recommendation", "recommendation"),
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _parse_dates(args) -> tuple[date, date, date]:
    report_date = date.fromisoformat(args.date) if args.date else datetime.now(_ET).date()
    end_date = date.fromisoformat(args.end_date) if args.end_date else report_date - timedelta(days=1)
    start_date = (
        date.fromisoformat(args.start_date)
        if args.start_date
        else end_date - timedelta(days=max(0, args.lookback_days - 1))
    )
    if start_date > end_date:
        raise SystemExit("--start-date cannot be after --end-date")
    return report_date, start_date, end_date


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose MLB prop model picks, warnings, and recalibration buckets")
    parser.add_argument("--date", default=None, help="Prediction date YYYY-MM-DD ET. Defaults to today.")
    parser.add_argument("--start-date", default=None, help="Historical scan start YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Historical scan end YYYY-MM-DD. Defaults to date-1.")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--holdout-days", type=int, default=21)
    parser.add_argument("--min-train-bets", type=int, default=40)
    parser.add_argument("--min-holdout-bets", type=int, default=20)
    parser.add_argument("--max-lay-price", type=float, default=-180.0)
    parser.add_argument("--min-ev", type=float, default=0.02)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--dsn", default=_PG_DSN)
    parser.add_argument("--thresholds-file", default=str(_THRESHOLDS_FILE))
    parser.add_argument("--out", default=None, help="Optional markdown output path")
    args = parser.parse_args()

    report_date, start_date, end_date = _parse_dates(args)
    cfg = DiagnosticConfig(
        report_date=report_date,
        start_date=start_date,
        end_date=end_date,
        dsn=args.dsn,
        out=args.out,
        top_n=int(args.top_n),
        min_train_bets=int(args.min_train_bets),
        min_holdout_bets=int(args.min_holdout_bets),
        holdout_days=int(args.holdout_days),
        max_lay_price=float(args.max_lay_price),
        min_ev=float(args.min_ev),
        thresholds_file=Path(args.thresholds_file),
    )
    report = build_report(cfg)
    if cfg.out:
        path = Path(cfg.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        print(f"Wrote {path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
