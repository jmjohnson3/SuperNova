"""Audit immutable lock-time hitter and pitcher opportunity projections."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2

from .prop_real_money_eligibility import (
    PROP_REAL_MONEY_ELIGIBILITY_START_DATE,
    parse_eligibility_start_date,
)
from mlb_pipeline.db import PG_DSN as _PG_DSN

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"


@dataclass(frozen=True)
class ProspectiveOpportunityConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_prospective_opportunity_audit.json"
    report_file: str = "mlb_prop_prospective_opportunity_audit_latest.md"
    eligibility_start_date: date = PROP_REAL_MONEY_ELIGIBILITY_START_DATE


SQL = """
WITH actuals AS (
    SELECT
        replay_id,
        MAX(actual_pa::float) AS actual_pa,
        MAX(actual_bf::float) AS actual_bf,
        MAX(actual_ip::float) AS actual_ip,
        MAX(actual_pitch_count_proxy::float) AS actual_pitch_count,
        MAX(low_pa_flag::float) AS low_pa_flag
    FROM features.mlb_prop_market_training_examples
    WHERE game_date_et >= %(eligibility_start)s
      AND result_status = 'graded'
    GROUP BY replay_id
)
SELECT
    r.id AS replay_id,
    r.game_date_et,
    r.game_slug,
    r.player_id,
    r.player_name,
    r.stat AS market,
    r.run_started_at_utc,
    COALESCE(r.locked_at_utc, r.source_created_at, r.run_started_at_utc) AS locked_at_utc,
    g.start_ts_utc,
    r.opportunity_context,
    a.actual_pa,
    a.actual_bf,
    a.actual_ip,
    a.actual_pitch_count,
    a.low_pa_flag
FROM bets.mlb_prop_prediction_replay r
LEFT JOIN actuals a ON a.replay_id = r.id
JOIN raw.mlb_games g
  ON g.game_slug = r.game_slug
 AND g.status = 'final'
WHERE r.game_date_et >= %(eligibility_start)s
  AND r.result_status = 'graded'
  AND r.stat IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND r.opportunity_context IS NOT NULL
  AND r.opportunity_context <> '{}'::jsonb
ORDER BY r.game_date_et, r.game_slug, r.player_id, r.stat, locked_at_utc
"""


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


def _load(cfg: ProspectiveOpportunityConfig) -> pd.DataFrame:
    with psycopg2.connect(cfg.pg_dsn) as conn:
        required = (
            _table_exists(conn, "bets", "mlb_prop_prediction_replay")
            and _table_exists(conn, "features", "mlb_prop_market_training_examples")
        )
        if not required:
            return pd.DataFrame()
        with conn.cursor() as cur:
            cur.execute(SQL, {"eligibility_start": cfg.eligibility_start_date})
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    for col in ("locked_at_utc", "run_started_at_utc", "start_ts_utc"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    contexts = df["opportunity_context"].apply(lambda value: value if isinstance(value, dict) else {})
    df["context_version"] = contexts.apply(lambda value: str(value.get("context_version") or "missing"))
    df["context_captured_at"] = pd.to_datetime(
        contexts.apply(lambda value: value.get("captured_at")), utc=True, errors="coerce"
    )
    for name in (
        "projected_pa", "baseline_projected_pa", "two_part_projected_pa",
        "low_pa_probability", "projected_bf", "projected_ip", "projected_pitch_count",
    ):
        df[name] = pd.to_numeric(contexts.apply(lambda value, key=name: value.get(key)), errors="coerce")
    for name in ("actual_pa", "actual_bf", "actual_ip", "actual_pitch_count", "low_pa_flag"):
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df["lock_before_start"] = df["locked_at_utc"].notna() & (
        df["start_ts_utc"].isna() | (df["locked_at_utc"] < df["start_ts_utc"])
    )
    df["context_before_start"] = df["context_captured_at"].notna() & (
        df["start_ts_utc"].isna() | (df["context_captured_at"] < df["start_ts_utc"])
    )
    df["context_near_lock"] = df["context_captured_at"].notna() & df["locked_at_utc"].notna() & (
        (df["context_captured_at"] - df["locked_at_utc"]).abs().dt.total_seconds() <= 15 * 60
    )
    df = df.sort_values("locked_at_utc").drop_duplicates(
        ["game_slug", "player_id", "market"], keep="first"
    )
    return df


def _error_metrics(df: pd.DataFrame, predicted: str, actual: str) -> dict[str, Any]:
    work = df[[predicted, actual]].dropna()
    if work.empty:
        return {"rows": 0, "mae": None, "rmse": None, "bias": None}
    error = work[predicted].astype(float) - work[actual].astype(float)
    return {
        "rows": int(len(work)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(error.mean()),
    }


def _low_pa_metrics(df: pd.DataFrame) -> dict[str, Any]:
    work = df[["low_pa_probability", "low_pa_flag"]].dropna()
    if work.empty:
        return {"rows": 0, "brier": None, "actual_rate": None, "predicted_rate": None}
    probability = work["low_pa_probability"].astype(float).clip(1e-6, 1.0 - 1e-6)
    target = work["low_pa_flag"].astype(float)
    return {
        "rows": int(len(work)),
        "brier": float(np.mean(np.square(probability - target))),
        "actual_rate": float(target.mean()),
        "predicted_rate": float(probability.mean()),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}f}" if math.isfinite(number) else "-"


def _date_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game_date, group in df.groupby("game_date_et", dropna=False):
        hitter = group.loc[group["market"].ne("pitcher_strikeouts")]
        pitcher = group.loc[group["market"].eq("pitcher_strikeouts")]
        rows.append({
            "game_date_et": str(game_date),
            "player_markets": int(len(group)),
            "immutable_rate": float((group["lock_before_start"] & group["context_before_start"] & group["context_near_lock"]).mean()),
            "pa": _error_metrics(hitter, "projected_pa", "actual_pa"),
            "bf": _error_metrics(pitcher, "projected_bf", "actual_bf"),
            "pitch_count": _error_metrics(pitcher, "projected_pitch_count", "actual_pitch_count"),
        })
    return rows


def build_report(cfg: ProspectiveOpportunityConfig) -> dict[str, Any]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load(cfg)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "eligibility_start_date": cfg.eligibility_start_date.isoformat(),
        "source": "immutable bets.mlb_prop_prediction_replay.opportunity_context",
        "rows": int(len(df)),
        "status": "ready" if not df.empty else "no_graded_immutable_rows",
    }
    if not df.empty:
        immutable = df["lock_before_start"] & df["context_before_start"] & df["context_near_lock"]
        cohort = df.loc[immutable].copy()
        hitters = cohort.loc[cohort["market"].ne("pitcher_strikeouts")]
        pitchers = cohort.loc[cohort["market"].eq("pitcher_strikeouts")]
        payload.update({
            "unique_dates": int(cohort["game_date_et"].nunique()),
            "immutable_rows": int(immutable.sum()),
            "immutable_rate": float(immutable.mean()),
            "timing": {
                "lock_before_start_rate": float(df["lock_before_start"].mean()),
                "context_before_start_rate": float(df["context_before_start"].mean()),
                "context_near_lock_rate": float(df["context_near_lock"].mean()),
            },
            "context_versions": {str(k): int(v) for k, v in cohort["context_version"].value_counts().items()},
            "hitter": {
                "pa": _error_metrics(hitters, "projected_pa", "actual_pa"),
                "baseline_pa": _error_metrics(hitters, "baseline_projected_pa", "actual_pa"),
                "two_part_pa": _error_metrics(hitters, "two_part_projected_pa", "actual_pa"),
                "low_pa": _low_pa_metrics(hitters),
            },
            "pitcher": {
                "bf": _error_metrics(pitchers, "projected_bf", "actual_bf"),
                "innings": _error_metrics(pitchers, "projected_ip", "actual_ip"),
                "pitch_count": _error_metrics(pitchers, "projected_pitch_count", "actual_pitch_count"),
            },
            "by_date": _date_rows(cohort),
        })
    out_path = cfg.model_dir / cfg.out_file
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    report_path = _REPORT_DIR / cfg.report_file
    hitter = payload.get("hitter") or {}
    pitcher = payload.get("pitcher") or {}
    timing = payload.get("timing") or {}
    lines = [
        "# MLB Prospective Opportunity Audit",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Eligibility start: {payload['eligibility_start_date']}",
        f"Status: {payload['status']}",
        f"Immutable player-market rows: {payload.get('immutable_rows', 0)} / {payload.get('rows', 0)}",
        f"Clean prospective dates: {payload.get('unique_dates', 0)}",
        "",
        "## Timing Integrity",
        "",
        f"- Lock before first pitch: {_fmt(timing.get('lock_before_start_rate'))}",
        f"- Context captured before first pitch: {_fmt(timing.get('context_before_start_rate'))}",
        f"- Context captured within 15 minutes of lock: {_fmt(timing.get('context_near_lock_rate'))}",
        "",
        "## Opportunity Accuracy",
        "",
        "| Target | Rows | MAE | RMSE | Bias |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, rec in (
        ("Hitter PA v3", hitter.get("pa") or {}),
        ("Hitter PA baseline", hitter.get("baseline_pa") or {}),
        ("Hitter PA two-part", hitter.get("two_part_pa") or {}),
        ("Pitcher BF", pitcher.get("bf") or {}),
        ("Pitcher pitch count", pitcher.get("pitch_count") or {}),
        ("Pitcher innings", pitcher.get("innings") or {}),
    ):
        lines.append(
            f"| {label} | {rec.get('rows', 0)} | {_fmt(rec.get('mae'))} | "
            f"{_fmt(rec.get('rmse'))} | {_fmt(rec.get('bias'))} |"
        )
    low_pa = hitter.get("low_pa") or {}
    lines.extend([
        "",
        f"Low-PA classifier: {low_pa.get('rows', 0)} rows, Brier {_fmt(low_pa.get('brier'))}, "
        f"actual {_fmt(low_pa.get('actual_rate'))}, predicted {_fmt(low_pa.get('predicted_rate'))}.",
        "",
        "## By Date",
        "",
        "| Date | Rows | Immutable | PA MAE | BF MAE | Pitch MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for rec in payload.get("by_date") or []:
        lines.append(
            f"| {rec['game_date_et']} | {rec['player_markets']} | {_fmt(rec.get('immutable_rate'))} | "
            f"{_fmt((rec.get('pa') or {}).get('mae'))} | {_fmt((rec.get('bf') or {}).get('mae'))} | "
            f"{_fmt((rec.get('pitch_count') or {}).get('mae'))} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit immutable prospective prop opportunity projections")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--eligibility-start-date", default=PROP_REAL_MONEY_ELIGIBILITY_START_DATE.isoformat())
    args = parser.parse_args()
    payload = build_report(ProspectiveOpportunityConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        eligibility_start_date=parse_eligibility_start_date(args.eligibility_start_date),
    ))
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows", 0),
        "immutable_rows": payload.get("immutable_rows", 0),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
