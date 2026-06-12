"""Real-money readiness report for MLB player props.

The report turns locked offer-level evidence into exact bucket trust scores and
explains why today's active positive-EV rows did or did not become bankroll
candidates.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from .side_recalibration import price_bucket, prop_line_bucket, prop_line_surface

_ET = ZoneInfo("America/New_York")
_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_OPEN_LADDER_TIERS = {"micro", "starter", "bankroll"}


@dataclass(frozen=True)
class ReadinessConfig:
    pg_dsn: str = _PG_DSN
    report_date: date | None = None
    lookback_days: int = 365
    min_bankroll_rows: int = 150
    min_paper_rows: int = 50
    min_clv_rows: int = 30
    min_roi: float = 0.0
    min_clv_beat_rate: float = 0.55
    min_avg_clv_price: float = 0.0
    max_abs_calibration_error: float = 0.05
    max_player_share: float = 0.12
    max_team_share: float = 0.25
    max_day_share: float = 0.35
    min_ev: float = 0.02
    top_n: int = 30
    model_dir: Path = _MODEL_DIR
    json_out: str = "prop_bucket_trust_scores.json"
    out: str | None = None


ACTIVE_SQL = """
SELECT
    id,
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
    edge_type,
    pred_count::float AS pred_count,
    pred_prob_over::float AS pred_prob_over,
    ev::float AS ev,
    bankroll_candidate,
    bankroll_tier,
    bankroll_reasons,
    stake_pct::float AS stake_pct,
    prop_offer_id,
    prediction_key,
    COALESCE(is_active, TRUE) AS is_active,
    stale_reason
FROM bets.mlb_prop_predictions
WHERE game_date_et = %(report_date)s
  AND COALESCE(is_active, TRUE) IS TRUE
  AND bet_side IN ('over','under')
  AND book_line IS NOT NULL
"""


HISTORY_SQL = """
SELECT
    game_date_et,
    player_id,
    player_name,
    team_abbr,
    market AS stat,
    side,
    market_line::float AS line,
    market_price::float AS price,
    bookmaker_key,
    model_family,
    model_prob_side::float AS model_prob_side,
    ev::float AS ev,
    CASE WHEN won IS TRUE THEN 1 ELSE 0 END AS won,
    COALESCE(push, false) AS push,
    profit_units::float AS profit_units,
    clv_price::float AS clv_price,
    CASE
        WHEN beat_clv_price IS TRUE THEN 1
        WHEN beat_clv_price IS FALSE THEN 0
        ELSE NULL
    END AS beat_clv_price,
    clv_line::float AS clv_line,
    CASE
        WHEN beat_clv_line IS TRUE THEN 1
        WHEN beat_clv_line IS FALSE THEN 0
        ELSE NULL
    END AS beat_clv_line,
    COALESCE(clv_valid, false) AS clv_valid,
    clv_status,
    clv_unknown_reason
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND side IN ('over','under')
  AND market_line IS NOT NULL
  AND won IS NOT NULL
"""


def _ensure_prediction_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE IF EXISTS bets.mlb_prop_predictions
                ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE,
                ADD COLUMN IF NOT EXISTS stale_reason TEXT,
                ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
            """
        )
    conn.commit()


def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1 FROM information_schema.tables
              WHERE table_schema = %s AND table_name = %s
            )
            """,
            (schema, table),
        )
        return bool(cur.fetchone()[0])


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return pd.DataFrame([dict(row) for row in cur.fetchall()])


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


def _side_prob(row: pd.Series) -> float | None:
    p_over = _clean_float(row.get("pred_prob_over"))
    side = row.get("side")
    if p_over is None:
        return None
    return p_over if side == "over" else 1.0 - p_over


def _bucket_key(stat: str, side: str, line_surface: str, line_bucket: str, price_bucket_value: str, book: str) -> str:
    return "|".join([
        str(stat or "*"),
        str(side or "*"),
        str(line_surface or "*"),
        str(line_bucket or "*"),
        str(price_bucket_value or "*"),
        str(book or "*"),
    ])


def _decorate(df: pd.DataFrame, *, active_rows: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["line_bucket"] = [
        prop_line_bucket(stat, line)
        for stat, line in zip(out["stat"], out["line"])
    ]
    out["price_bucket"] = out["price"].map(price_bucket)
    out["line_surface"] = [
        prop_line_surface(stat, side, line)
        for stat, side, line in zip(out["stat"], out["side"], out["line"])
    ]
    out["bookmaker_key"] = out["bookmaker_key"].fillna("unknown").astype(str)
    out["bucket_key"] = [
        _bucket_key(stat, side, surf, lb, pb, book)
        for stat, side, surf, lb, pb, book in zip(
            out["stat"], out["side"], out["line_surface"], out["line_bucket"],
            out["price_bucket"], out["bookmaker_key"],
        )
    ]
    if active_rows:
        out["model_prob_side"] = out.apply(_side_prob, axis=1)
    return out


def _load(cfg: ReadinessConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    report_date = cfg.report_date or datetime.now(_ET).date()
    cutoff = report_date - timedelta(days=max(1, cfg.lookback_days) - 1)
    with psycopg2.connect(cfg.pg_dsn) as conn:
        _ensure_prediction_schema(conn)
        active = _query_df(conn, ACTIVE_SQL, {"report_date": report_date})
        if _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            hist = _query_df(conn, HISTORY_SQL, {"cutoff": cutoff})
        else:
            hist = pd.DataFrame()
    if not active.empty:
        active["game_date_et"] = pd.to_datetime(active["game_date_et"]).dt.date
        active = _decorate(active, active_rows=True)
    if not hist.empty:
        hist["game_date_et"] = pd.to_datetime(hist["game_date_et"]).dt.date
        for col in ["line", "price", "model_prob_side", "ev", "won", "profit_units", "clv_price", "beat_clv_price", "clv_line", "beat_clv_line"]:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")
        hist["push"] = hist["push"].fillna(False).astype(bool)
        hist = _decorate(hist)
    return active, hist


def _share(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df.columns:
        return None
    counts = df[col].dropna().value_counts()
    if counts.empty:
        return None
    return float(counts.iloc[0] / len(df))


def _bucket_summary(group: pd.DataFrame, cfg: ReadinessConfig) -> dict[str, Any]:
    settled = group.loc[~group["push"].fillna(False).astype(bool)].copy()
    wins = int(pd.to_numeric(settled["won"], errors="coerce").fillna(0).sum())
    losses = int(len(settled) - wins)
    priced = group["profit_units"].notna()
    units = float(pd.to_numeric(group.loc[priced, "profit_units"], errors="coerce").sum()) if priced.any() else None
    roi = units / int(priced.sum()) if units is not None and int(priced.sum()) else None
    p = pd.to_numeric(settled["model_prob_side"], errors="coerce")
    y = pd.to_numeric(settled["won"], errors="coerce")
    valid = p.notna() & y.notna()
    brier = float(((p[valid] - y[valid]) ** 2).mean()) if valid.any() else None
    win_rate = float(wins / (wins + losses)) if wins + losses else None
    avg_prob = float(p[valid].mean()) if valid.any() else None
    cal_error = win_rate - avg_prob if win_rate is not None and avg_prob is not None else None
    valid_clv = group["clv_valid"].fillna(False).astype(bool)
    clv_price = pd.to_numeric(group.loc[valid_clv, "clv_price"], errors="coerce").dropna()
    clv_beat = pd.to_numeric(group.loc[valid_clv, "beat_clv_price"], errors="coerce").dropna()
    max_player_share = _share(group, "player_id")
    max_team_share = _share(group, "team_abbr")
    max_day_share = _share(group, "game_date_et")
    reasons: list[str] = []
    if len(settled) < cfg.min_bankroll_rows:
        reasons.append("sample_too_small")
    if len(clv_price) < cfg.min_clv_rows:
        reasons.append("no_clv_history")
    if roi is None or roi <= cfg.min_roi:
        reasons.append("roi_not_positive")
    if clv_beat.empty or float(clv_beat.mean()) < cfg.min_clv_beat_rate:
        reasons.append("clv_beat_rate_low")
    if clv_price.empty or float(clv_price.mean()) <= cfg.min_avg_clv_price:
        reasons.append("avg_clv_not_positive")
    if cal_error is None or abs(cal_error) > cfg.max_abs_calibration_error:
        reasons.append("calibration_not_ready")
    if max_player_share is not None and max_player_share > cfg.max_player_share:
        reasons.append("player_concentration")
    if max_team_share is not None and max_team_share > cfg.max_team_share:
        reasons.append("team_concentration")
    if max_day_share is not None and max_day_share > cfg.max_day_share:
        reasons.append("day_concentration")

    if not reasons:
        status = "bankroll"
    elif len(settled) >= cfg.min_paper_rows and roi is not None and brier is not None:
        status = "paper"
    elif len(settled) > 0:
        status = "watch"
    else:
        status = "closed"

    score = 100
    score -= 30 if "sample_too_small" in reasons else 0
    score -= 25 if "no_clv_history" in reasons else 0
    score -= 20 if "roi_not_positive" in reasons else 0
    score -= 20 if "clv_beat_rate_low" in reasons else 0
    score -= 15 if "calibration_not_ready" in reasons else 0
    score -= 10 if any(r.endswith("concentration") for r in reasons) else 0
    return {
        "rows": int(len(group)),
        "graded": int(len(settled)),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "units": units,
        "roi": roi,
        "avg_prob": avg_prob,
        "calibration_error": cal_error,
        "brier": brier,
        "avg_clv_price": float(clv_price.mean()) if not clv_price.empty else None,
        "clv_price_rows": int(len(clv_price)),
        "clv_price_beat_rate": float(clv_beat.mean()) if not clv_beat.empty else None,
        "unique_players": int(group["player_id"].dropna().nunique()),
        "unique_teams": int(group["team_abbr"].dropna().nunique()),
        "unique_dates": int(group["game_date_et"].dropna().nunique()),
        "max_player_share": max_player_share,
        "max_team_share": max_team_share,
        "max_day_share": max_day_share,
        "status": status,
        "raw_readiness_status": status,
        "trust_score": max(0, min(100, score)),
        "reasons": reasons,
    }


def _load_reopen_policy(model_dir: Path) -> dict[str, dict[str, Any]]:
    path = Path(model_dir) / "prop_bucket_reopen_policy.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, rec in (payload.get("ladder_buckets") or {}).items():
        out[str(key)] = dict(rec or {})
    for key, rec in (payload.get("reopen_buckets") or {}).items():
        out[str(key)] = dict(rec or {})
    return out


def _apply_ladder_policy(scores: dict[str, dict[str, Any]], cfg: ReadinessConfig) -> None:
    policy = _load_reopen_policy(cfg.model_dir)
    for key, rec in scores.items():
        ladder = policy.get(str(key)) or {}
        ladder_tier = str(ladder.get("ladder_tier") or "watch")
        desired_tier = str(ladder.get("desired_ladder_tier") or "watch")
        rec["ladder_tier"] = ladder_tier
        rec["desired_ladder_tier"] = desired_tier
        rec["reopen_policy_status"] = ladder.get("status", "closed")
        rec["promotion_source"] = ladder.get("promotion_source", "closed")
        rec["bootstrap_micro_eligible"] = bool(ladder.get("bootstrap_micro_eligible"))
        rec["bootstrap_micro_reasons"] = list(ladder.get("bootstrap_micro_reasons") or [])
        rec["raw_readiness_status"] = rec.get("status", "closed")
        if ladder_tier in _OPEN_LADDER_TIERS:
            rec["status"] = ladder_tier
        elif rec.get("status") == "bankroll":
            rec["status"] = "paper"
            rec.setdefault("reasons", []).append("bucket_not_reopened")


def _trust_scores(hist: pd.DataFrame, cfg: ReadinessConfig) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    if hist.empty:
        return scores
    group_cols = ["stat", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"]
    for values, group in hist.groupby(group_cols, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        key = _bucket_key(*[str(v) for v in values])
        rec = dict(zip(group_cols, values))
        rec["key"] = key
        rec.update(_bucket_summary(group, cfg))
        scores[key] = rec
    return scores


def _split_reasons(text: Any) -> list[str]:
    if text is None:
        return []
    parts = []
    for chunk in str(text).replace(",", ";").split(";"):
        clean = chunk.strip()
        if clean:
            parts.append(clean)
    return parts


def _row_block_reasons(row: pd.Series, trust: dict[str, Any] | None, cfg: ReadinessConfig) -> list[str]:
    reasons = _split_reasons(row.get("bankroll_reasons"))
    if _clean_float(row.get("ev")) is None:
        reasons.append("missing_ev")
    elif float(row.get("ev") or 0.0) < cfg.min_ev:
        reasons.append("ev_below_min")
    if _clean_float(row.get("price")) is None:
        reasons.append("book_or_price_missing")
    if trust is None:
        reasons.append("no_bucket_history")
    elif trust.get("status") not in _OPEN_LADDER_TIERS:
        reasons.extend(trust.get("reasons") or [])
        reasons.append("bucket_not_reopened")
    return sorted(dict.fromkeys(reasons))


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


def build_payload(cfg: ReadinessConfig) -> dict[str, Any]:
    report_date = cfg.report_date or datetime.now(_ET).date()
    active, hist = _load(ReadinessConfig(**{**cfg.__dict__, "report_date": report_date}))
    scores = _trust_scores(hist, cfg)
    _apply_ladder_policy(scores, cfg)
    active_rows = []
    reason_counter: Counter[str] = Counter()
    positive_ev = active[pd.to_numeric(active.get("ev"), errors="coerce").fillna(-999.0) >= cfg.min_ev] if not active.empty else active
    for _, row in positive_ev.iterrows():
        trust = scores.get(str(row.get("bucket_key")))
        reasons = _row_block_reasons(row, trust, cfg)
        if not bool(row.get("bankroll_candidate")):
            reason_counter.update(reasons or ["not_bankroll_candidate"])
        active_rows.append({
            "player": row.get("player_name"),
            "team": row.get("team_abbr"),
            "stat": row.get("stat"),
            "side": row.get("side"),
            "line": _fmt_num(row.get("line"), 1),
            "price": _fmt_num(row.get("price"), 0, signed=True),
            "book": row.get("bookmaker_key"),
            "surface": row.get("line_surface"),
            "ev": _fmt_pct(row.get("ev"), signed=True),
            "bankroll": bool(row.get("bankroll_candidate")),
            "trust": (trust or {}).get("status", "closed"),
            "score": (trust or {}).get("trust_score", 0),
            "reasons": "; ".join(reasons),
        })
    active_rows.sort(key=lambda r: (bool(r["bankroll"]), float(str(r["ev"]).replace("%", "").replace("+", "") or -999)), reverse=True)

    score_rows = list(scores.values())
    score_rows.sort(key=lambda r: (-int(r.get("trust_score") or 0), -int(r.get("graded") or 0), str(r.get("key"))))
    payload = {
        "generated_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds"),
        "report_date": str(report_date),
        "lookback_days": cfg.lookback_days,
        "active_rows": int(len(active)),
        "active_positive_ev_rows": int(len(positive_ev)),
        "active_bankroll_rows": (
            0 if active.empty else int(active["bankroll_candidate"].fillna(False).astype(bool).sum())
        ),
        "history_rows": int(len(hist)),
        "bucket_scores": scores,
        "reason_counts": dict(reason_counter.most_common()),
        "top_active_rows": active_rows[: cfg.top_n],
        "top_bucket_rows": [
            {
                "key": row["key"],
                "status": row["status"],
                "ladder": row.get("ladder_tier", "watch"),
                "source": row.get("promotion_source", "closed"),
                "score": row["trust_score"],
                "graded": row["graded"],
                "roi": _fmt_pct(row.get("roi"), signed=True),
                "brier": _fmt_num(row.get("brier"), 3),
                "cal_error": _fmt_pct(row.get("calibration_error"), signed=True),
                "clv_rows": row.get("clv_price_rows"),
                "clv_beat": _fmt_pct(row.get("clv_price_beat_rate")),
                "reasons": "; ".join(row.get("reasons") or []),
            }
            for row in score_rows[: cfg.top_n]
        ],
    }
    return payload


def build_report(cfg: ReadinessConfig) -> str:
    payload = build_payload(cfg)
    lines = [
        "# MLB Prop Real-Money Readiness",
        "",
        f"Date: {payload['report_date']}",
        f"Active offer rows: {payload['active_rows']}",
        f"Positive-EV active rows: {payload['active_positive_ev_rows']}",
        f"Bankroll rows: {payload['active_bankroll_rows']}",
        f"History rows: {payload['history_rows']}",
        "",
        "## No-Bet Reasons",
        "",
        _table(
            [{"reason": k, "rows": v} for k, v in payload["reason_counts"].items()],
            [("Reason", "reason"), ("Rows", "rows")],
        ),
        "",
        "## Active Positive-EV Rows",
        "",
        _table(
            payload["top_active_rows"],
            [
                ("Player", "player"),
                ("Stat", "stat"),
                ("Side", "side"),
                ("Line", "line"),
                ("Price", "price"),
                ("Book", "book"),
                ("Surface", "surface"),
                ("EV", "ev"),
                ("Bankroll", "bankroll"),
                ("Trust", "trust"),
                ("Score", "score"),
                ("Reasons", "reasons"),
            ],
        ),
        "",
        "## Top Bucket Trust Scores",
        "",
        _table(
            payload["top_bucket_rows"],
            [
                ("Bucket", "key"),
                ("Status", "status"),
                ("Ladder", "ladder"),
                ("Source", "source"),
                ("Score", "score"),
                ("Graded", "graded"),
                ("ROI", "roi"),
                ("Brier", "brier"),
                ("Cal Err", "cal_error"),
                ("CLV Rows", "clv_rows"),
                ("CLV Beat", "clv_beat"),
                ("Reasons", "reasons"),
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop real-money readiness report")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--date", default=None)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--min-bankroll-rows", type=int, default=150)
    parser.add_argument("--min-paper-rows", type=int, default=50)
    parser.add_argument("--min-clv-rows", type=int, default=30)
    parser.add_argument("--min-ev", type=float, default=0.02)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--json-out", default="prop_bucket_trust_scores.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = ReadinessConfig(
        pg_dsn=args.pg_dsn,
        report_date=date.fromisoformat(args.date) if args.date else None,
        lookback_days=args.lookback_days,
        min_bankroll_rows=args.min_bankroll_rows,
        min_paper_rows=args.min_paper_rows,
        min_clv_rows=args.min_clv_rows,
        min_ev=args.min_ev,
        top_n=args.top_n,
        model_dir=Path(args.model_dir),
        json_out=args.json_out,
        out=args.out,
    )
    payload = build_payload(cfg)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    (cfg.model_dir / cfg.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = build_report(cfg)
    if cfg.out:
        out_path = Path(cfg.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
