"""Line/book-specific calibration for TB over 1.5.

This focuses on the exact weak-but-close area in the micro ledger: common
total-bases overs, especially DraftKings TB 1.5 true-paired offers.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

from .prop_training_groups import dedupe_locked_offer_rows
from .side_recalibration import price_bucket

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

_MIN_PRIOR_ROWS = 80
_MIN_GROUP_ROWS = 120
_SHRINK_ROWS = 220.0
_BINS = (
    (0.00, 0.40, "p<40"),
    (0.40, 0.55, "p40_55"),
    (0.55, 0.70, "p55_70"),
    (0.70, 1.01, "p70_plus"),
)

SQL = """
SELECT
    id,
    replay_id,
    source_created_at,
    game_date_et,
    game_slug,
    player_id,
    player_name,
    market,
    side,
    bookmaker_key,
    market_line::float AS market_line,
    market_price::float AS market_price,
    COALESCE(price_bucket, 'missing_price') AS price_bucket,
    COALESCE(model_family, 'unknown') AS model_family,
    model_prob_side::float AS model_prob_side,
    CASE WHEN won IS TRUE THEN 1.0 WHEN won IS FALSE THEN 0.0 ELSE NULL END AS target,
    COALESCE(push, false) AS push,
    clv_valid,
    clv_price::float AS clv_price,
    COALESCE(true_pair_flag::float, CASE WHEN pair_quality IN ('same_book','cross_book') THEN 1.0 ELSE 0.0 END) AS true_pair_flag,
    COALESCE(synthetic_pair_flag::float, CASE WHEN pair_quality = 'synthetic' THEN 1.0 ELSE 0.0 END) AS synthetic_pair_flag,
    pair_quality,
    prop_offer_id
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market = 'batter_total_bases'
  AND side = 'over'
  AND market_line = 1.5
  AND model_prob_side IS NOT NULL
  AND won IS NOT NULL
  AND COALESCE(push, false) IS FALSE
ORDER BY game_date_et, source_created_at, id
"""


def _table_exists(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (table_name,))
        return bool(cur.fetchone()[0])


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _prob_bin(value: Any) -> str:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return "missing_prob"
    if not math.isfinite(p):
        return "missing_prob"
    for lo, hi, label in _BINS:
        if lo <= p < hi:
            return label
    return "p70_plus" if p >= 0.70 else "missing_prob"


def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    return math.log(p / (1.0 - p))


def _inv_logit(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(z)))))


def _apply_offset(prob: pd.Series, offset: float) -> pd.Series:
    return prob.map(lambda p: _inv_logit(_logit(float(p)) + float(offset))).clip(1e-6, 1.0 - 1e-6)


def _brier(target: pd.Series, prob: pd.Series) -> float | None:
    work = pd.DataFrame({"target": target, "prob": prob}).dropna()
    if work.empty:
        return None
    return float(np.square(work["target"] - work["prob"].clip(1e-6, 1.0 - 1e-6)).mean())


def _group_key(row: pd.Series, level: str) -> str:
    book = str(row.get("bookmaker_key") or "unknown").lower()
    pb = str(row.get("price_bucket") or "missing_price")
    if level == "book_price":
        return f"{book}|{pb}"
    if level == "book":
        return f"{book}|*"
    return "*|*"


def _candidate_keys(book: str, pb: str) -> list[str]:
    book = str(book or "unknown").lower()
    pb = str(pb or "missing_price")
    return [f"{book}|{pb}", f"{book}|*", f"*|{pb}", "*|*"]


def _fit_offsets(prior: pd.DataFrame) -> dict[str, Any]:
    pred_rate = float(prior["model_prob_side"].mean())
    actual_rate = float(prior["target"].mean())
    shrink = len(prior) / (len(prior) + _SHRINK_ROWS)
    global_offset = shrink * (_logit(actual_rate) - _logit(pred_rate))
    bins: dict[str, dict[str, Any]] = {}
    for bin_key, group in prior.groupby("prob_bin"):
        if len(group) < max(25, _MIN_PRIOR_ROWS // 4):
            continue
        bin_pred = float(group["model_prob_side"].mean())
        bin_actual = float(group["target"].mean())
        bin_shrink = len(group) / (len(group) + _SHRINK_ROWS)
        bins[str(bin_key)] = {
            "rows": int(len(group)),
            "pred_rate": bin_pred,
            "actual_rate": bin_actual,
            "offset": float(bin_shrink * (_logit(bin_actual) - _logit(bin_pred))),
            "blend_weight": float(min(0.85, bin_shrink)),
        }
    return {
        "rows": int(len(prior)),
        "pred_rate": pred_rate,
        "actual_rate": actual_rate,
        "offset": float(global_offset),
        "blend_weight": float(min(0.85, shrink)),
        "bins": bins,
    }


def _apply_calibrator(rows: pd.DataFrame, cal: dict[str, Any]) -> pd.Series:
    raw = rows["model_prob_side"].astype(float).clip(1e-6, 1.0 - 1e-6)
    out = _apply_offset(raw, float(cal.get("offset") or 0.0))
    for bin_key, bin_cal in (cal.get("bins") or {}).items():
        mask = rows["prob_bin"].eq(bin_key)
        if not mask.any():
            continue
        out.loc[mask] = _apply_offset(raw.loc[mask], float(bin_cal.get("offset") or 0.0))
    return out.clip(1e-6, 1.0 - 1e-6)


def _walk_forward_group(group: pd.DataFrame) -> tuple[pd.Series, list[dict[str, Any]]]:
    dates = sorted(group["game_date_et"].dropna().unique())
    pred = pd.Series(np.nan, index=group.index, dtype=float)
    folds: list[dict[str, Any]] = []
    for day in dates:
        current_mask = group["game_date_et"].eq(day)
        prior = group.loc[group["game_date_et"].lt(day)]
        if len(prior) < _MIN_PRIOR_ROWS:
            continue
        cal = _fit_offsets(prior)
        pred.loc[group.index[current_mask]] = _apply_calibrator(group.loc[current_mask], cal)
        folds.append({
            "date": str(day),
            "prior_rows": int(len(prior)),
            "prior_pred_rate": cal.get("pred_rate"),
            "prior_actual_rate": cal.get("actual_rate"),
            "prior_offset": cal.get("offset"),
            "holdout_rows": int(current_mask.sum()),
        })
    return pred, folds


def _metrics(group: pd.DataFrame, calibrated: pd.Series) -> dict[str, Any]:
    valid = calibrated.notna()
    target = group.loc[valid, "target"].astype(float)
    raw = group.loc[valid, "model_prob_side"].astype(float)
    cal = calibrated.loc[valid].astype(float)
    clv_valid = group.loc[valid, "clv_valid"].astype(bool)
    clv_price = pd.to_numeric(group.loc[valid, "clv_price"], errors="coerce")
    raw_brier = _brier(target, raw)
    cal_brier = _brier(target, cal)
    return {
        "rows": int(len(group)),
        "holdout_rows": int(valid.sum()),
        "win_rate": float(target.mean()) if len(target) else None,
        "avg_raw_prob": float(raw.mean()) if len(raw) else None,
        "avg_calibrated_prob": float(cal.mean()) if len(cal) else None,
        "raw_brier": raw_brier,
        "calibrated_brier": cal_brier,
        "brier_gain": raw_brier - cal_brier if raw_brier is not None and cal_brier is not None else None,
        "raw_calibration_error": float(raw.mean() - target.mean()) if len(raw) else None,
        "calibrated_calibration_error": float(cal.mean() - target.mean()) if len(cal) else None,
        "clv_rows": int(clv_valid.sum()),
        "clv_beat_rate": float((clv_price.loc[clv_valid] > 0).mean()) if clv_valid.any() else None,
        "avg_clv_price": float(clv_price.loc[clv_valid].mean()) if clv_valid.any() else None,
    }


def _load_rows(pg_dsn: str, lookback_days: int) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, int(lookback_days)))
    with psycopg2.connect(pg_dsn) as conn:
        if not _table_exists(conn, "features.mlb_prop_market_training_examples"):
            return pd.DataFrame()
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in ("market_line", "market_price", "model_prob_side", "target", "true_pair_flag", "synthetic_pair_flag"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["bookmaker_key"] = df["bookmaker_key"].fillna("unknown").astype(str).str.lower()
    df["price_bucket"] = [
        pb if isinstance(pb, str) and pb and pb != "missing_price" else price_bucket(price)
        for pb, price in zip(df["price_bucket"], df["market_price"])
    ]
    df = df.loc[
        df["target"].notna()
        & df["model_prob_side"].between(0.0, 1.0)
        & df["true_pair_flag"].fillna(0.0).ge(0.5)
        & df["synthetic_pair_flag"].fillna(0.0).lt(0.5)
    ].copy()
    df = dedupe_locked_offer_rows(df)
    df["prob_bin"] = df["model_prob_side"].map(_prob_bin)
    return df


def build(*, pg_dsn: str = PG_DSN, lookback_days: int = 365) -> dict[str, Any]:
    df = _load_rows(pg_dsn, lookback_days)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if not df.empty else "no_rows",
        "usage": "tb_1_5_over_line_book_probability_calibration",
        "lookback_days": int(lookback_days),
        "rows": int(len(df)),
        "target": "batter_total_bases|over|TB 1.5|true_pair_only",
        "calibrators": {},
        "groups": [],
    }
    if df.empty:
        _write_outputs(payload)
        return payload

    calibrated_by_level: dict[str, pd.Series] = {}
    fold_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for level in ("book_price", "book", "global"):
        series = pd.Series(np.nan, index=df.index, dtype=float)
        for key, group in df.groupby(df.apply(lambda row: _group_key(row, level), axis=1)):
            if len(group) < _MIN_GROUP_ROWS and level != "global":
                continue
            cal_pred, folds = _walk_forward_group(group)
            series.loc[group.index] = cal_pred
            fold_rows[str(key)].extend(folds)
        calibrated_by_level[level] = series

    group_records = []
    calibrators: dict[str, Any] = {}
    for level in ("book_price", "book", "global"):
        for key, group in df.groupby(df.apply(lambda row: _group_key(row, level), axis=1)):
            if len(group) < _MIN_GROUP_ROWS and level != "global":
                continue
            cal_series = calibrated_by_level[level].loc[group.index]
            rec = _metrics(group, cal_series)
            rec.update({
                "key": str(key),
                "level": level,
                "bookmaker_key": str(key).split("|")[0],
                "price_bucket": str(key).split("|")[1] if "|" in str(key) else "*",
                "enabled": bool(
                    rec.get("holdout_rows", 0) >= 40
                    and rec.get("brier_gain") is not None
                    and float(rec["brier_gain"]) > 0.0
                    and abs(float(rec.get("calibrated_calibration_error") or 0.0))
                       <= abs(float(rec.get("raw_calibration_error") or 0.0)) + 0.01
                ),
                "folds": fold_rows.get(str(key), [])[-20:],
            })
            final_cal = _fit_offsets(group)
            final_cal.update({
                "key": str(key),
                "level": level,
                "enabled": rec["enabled"],
                "holdout_brier_gain": rec.get("brier_gain"),
                "holdout_rows": rec.get("holdout_rows"),
                "bookmaker_key": rec["bookmaker_key"],
                "price_bucket": rec["price_bucket"],
                "line_bucket": "TB 1.5",
                "market": "batter_total_bases",
                "side": "over",
            })
            calibrators[str(key)] = final_cal
            group_records.append(rec)

    # A convenient aggregate for the exact DK target the user asked about.
    dk_keys = _candidate_keys("draftkings", "plus_100_149") + _candidate_keys("draftkings", "fair_lay")
    payload["draftkings_focus_keys"] = [key for key in dk_keys if key in calibrators]
    payload["calibrators"] = calibrators
    group_records.sort(key=lambda row: (row["level"], row["bookmaker_key"], row["price_bucket"]))
    payload["groups"] = group_records
    payload["enabled_count"] = sum(1 for rec in calibrators.values() if rec.get("enabled"))
    _write_outputs(payload)
    return payload


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.1f}%"
    except Exception:
        return "-"


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Prop TB 1.5 Line Calibration",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        "Target: `batter_total_bases | over | TB 1.5 | true-paired offers only`",
        "",
        f"- Rows: {payload.get('rows', 0)}",
        f"- Enabled calibrators: {payload.get('enabled_count', 0)}",
        f"- DraftKings focus keys: {', '.join(payload.get('draftkings_focus_keys') or []) or '-'}",
        "",
        "## Holdout Calibration By Book / Price",
        "",
        "| Level | Key | Rows | Holdout | Win | Raw P | Cal P | Raw Brier | Cal Brier | Gain | Raw Cal Err | Cal Err | CLV Beat | Avg CLV | Enabled |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("groups") or []:
        lines.append(
            f"| {row.get('level')} | {row.get('key')} | {row.get('rows')} | {row.get('holdout_rows')} | "
            f"{_pct(row.get('win_rate'))} | {_pct(row.get('avg_raw_prob'))} | {_pct(row.get('avg_calibrated_prob'))} | "
            f"{_fmt(row.get('raw_brier'), 5)} | {_fmt(row.get('calibrated_brier'), 5)} | "
            f"{_fmt(row.get('brier_gain'), 5)} | {_pct(row.get('raw_calibration_error'))} | "
            f"{_pct(row.get('calibrated_calibration_error'))} | {_pct(row.get('clv_beat_rate'))} | "
            f"{_fmt(row.get('avg_clv_price'))} | {bool(row.get('enabled'))} |"
        )
    lines.extend([
        "",
        "## Probability Bins",
        "",
        "| Key | Bin | Rows | Pred | Actual | Offset | Blend |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for key, cal in (payload.get("calibrators") or {}).items():
        if not cal.get("enabled"):
            continue
        for bin_key, rec in (cal.get("bins") or {}).items():
            lines.append(
                f"| {key} | {bin_key} | {rec.get('rows')} | {_pct(rec.get('pred_rate'))} | "
                f"{_pct(rec.get('actual_rate'))} | {_fmt(rec.get('offset'))} | {_fmt(rec.get('blend_weight'), 2)} |"
            )
    return "\n".join(lines) + "\n"


def _write_outputs(payload: dict[str, Any]) -> tuple[Path, Path]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _MODEL_DIR / "prop_tb15_line_calibrators.json"
    report_path = _REPORT_DIR / "mlb_prop_tb15_line_calibration_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, _render(payload))
    payload["report_path"] = str(report_path)
    payload["json_path"] = str(json_path)
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TB over 1.5 line/book calibration report")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    args = parser.parse_args()
    payload = build(pg_dsn=args.pg_dsn, lookback_days=args.lookback_days)
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows"),
        "enabled_count": payload.get("enabled_count", 0),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
