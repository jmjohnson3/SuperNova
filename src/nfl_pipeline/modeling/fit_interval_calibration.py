"""Fit release-bound asymmetric yardage intervals from immutable locked forecasts.

The calibrator changes the full forecast curve, not just displayed bounds.  It
uses expanding NFL-week folds and only labels graded before the next test lock.
Betting approval is deliberately outside this artifact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling.chronological_calibration import forward_folds
from nfl_pipeline.modeling.predict_player_props import INTERVAL_CALIBRATION_CONTRACT

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "src" / "nfl_pipeline" / "modeling" / "models" / "player_props" / "interval_calibration.json"
REPORT = ROOT / "reports" / "nfl_asymmetric_interval_calibration_latest.json"
YARDAGE_STATS = ("passing_yards", "rushing_yards", "receiving_yards")
TARGET_TAIL = 0.10
MIN_FORWARD_ROWS = 100
MIN_EVALUATED_FOLDS = 2
MIN_INTERVAL_SCORE_GAIN = 0.02
SCALE_GRID = np.round(np.arange(1.0, 3.0001, 0.01), 2)


def load_rows(stats: tuple[str, ...] = YARDAGE_STATS) -> pd.DataFrame:
    sql = """
        SELECT p.game_date_et,p.season,p.week,p.game_id,p.player_id,p.stat,p.line,p.projection,
               p.projection_p10 AS p10,p.projection_p50 AS p50,p.projection_p90 AS p90,
               p.created_at_utc AS locked_at,p.is_current,p.model_version,
               r.actual_stat AS actual,r.graded_at_utc AS label_available_at,
               COALESCE((p.forecast_payload->'probability_trace'->>'model_projection')::numeric,
                        p.projection) AS model_projection
        FROM bets.nfl_player_prop_predictions p
        JOIN bets.nfl_player_prop_prediction_results r ON r.prediction_id=p.id
        WHERE p.integrity_version='nfl-asof-v2'
          AND p.stat = ANY(%s)
          AND p.projection_p10 IS NOT NULL AND p.projection_p50 IS NOT NULL
          AND p.projection_p90 IS NOT NULL AND r.actual_stat IS NOT NULL
    """
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        frame = pd.read_sql(sql, conn, params=(list(stats),))
    if frame.empty:
        return frame
    for column in ("line", "projection", "p10", "p50", "p90", "actual", "model_projection"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["p10", "p50", "p90", "actual", "locked_at", "label_available_at"])
    frame["distance"] = (frame.line - frame.model_projection).abs()
    # A player-game outcome is one unit of evidence. Prefer the current immutable
    # decision, then the offered line nearest the original model forecast.
    frame = frame.sort_values(
        ["is_current", "distance", "locked_at"], ascending=[False, True, True]
    ).drop_duplicates(["game_id", "player_id", "stat"])
    frame["key"] = frame.game_id.astype(str) + "|" + frame.player_id.astype(str) + "|" + frame.stat
    return frame.reset_index(drop=True)


def _tail_scale(frame: pd.DataFrame, lower: bool) -> float:
    if frame.empty:
        return 1.0
    if lower:
        denominator = (frame.p50 - frame.p10).replace(0.0, np.nan)
        ratio = (frame.p50 - frame.actual) / denominator
    else:
        denominator = (frame.p90 - frame.p50).replace(0.0, np.nan)
        ratio = (frame.actual - frame.p50) / denominator
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return 1.0
    for scale in SCALE_GRID:
        if float((ratio > scale).mean()) <= TARGET_TAIL:
            return float(scale)
    return float(SCALE_GRID[-1])


def fit_scales(frame: pd.DataFrame) -> tuple[float, float]:
    return _tail_scale(frame, True), _tail_scale(frame, False)


def apply_scales(frame: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    result = frame.copy()
    result["lo"] = np.maximum(0.0, result.p50 + lower * (result.p10 - result.p50))
    result["hi"] = np.maximum(0.0, result.p50 + upper * (result.p90 - result.p50))
    return result


def interval_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"rows": 0}
    below = frame.actual < frame.lo
    above = frame.actual > frame.hi
    alpha = 0.20
    score = ((frame.hi - frame.lo)
             + (2.0 / alpha) * (frame.lo - frame.actual).clip(lower=0.0)
             + (2.0 / alpha) * (frame.actual - frame.hi).clip(lower=0.0))
    return {
        "rows": int(len(frame)),
        "coverage": float((~(below | above)).mean()),
        "below_p10": float(below.mean()),
        "above_p90": float(above.mean()),
        "mean_interval_score": float(score.mean()),
        "mean_width": float((frame.hi - frame.lo).mean()),
    }


def fit_stat(frame: pd.DataFrame) -> dict:
    folds, base_tests, calibrated_tests = [], [], []
    for training, testing, metadata in forward_folds(frame):
        if len(training) < 20 or testing.empty:
            folds.append({**metadata, "status": "insufficient_earlier_available_labels"})
            continue
        lower, upper = fit_scales(training)
        base = apply_scales(testing, 1.0, 1.0)
        calibrated = apply_scales(testing, lower, upper)
        base_tests.append(base)
        calibrated_tests.append(calibrated)
        folds.append({**metadata, "status": "evaluated", "training_rows": int(len(training)),
                      "lower_tail_scale": lower, "upper_tail_scale": upper,
                      "base": interval_metrics(base), "calibrated": interval_metrics(calibrated)})
    base = interval_metrics(pd.concat(base_tests, ignore_index=True)) if base_tests else {"rows": 0}
    calibrated = interval_metrics(pd.concat(calibrated_tests, ignore_index=True)) if calibrated_tests else {"rows": 0}
    evaluated_folds = sum(fold["status"] == "evaluated" for fold in folds)
    base_score = base.get("mean_interval_score")
    calibrated_score = calibrated.get("mean_interval_score")
    score_gain = ((base_score - calibrated_score) / base_score
                  if base_score and calibrated_score is not None else None)
    coverage_better = (
        calibrated.get("coverage") is not None
        and abs(calibrated["coverage"] - 0.80) <= abs(base.get("coverage", 0.0) - 0.80)
    )
    accepted = bool(
        evaluated_folds >= MIN_EVALUATED_FOLDS
        and calibrated.get("rows", 0) >= MIN_FORWARD_ROWS
        and score_gain is not None and score_gain >= MIN_INTERVAL_SCORE_GAIN
        and coverage_better
        and calibrated.get("below_p10", 1.0) <= 0.12
        and calibrated.get("above_p90", 1.0) <= 0.12
    )
    lower, upper = fit_scales(frame)
    return {
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "lower_tail_scale": lower,
        "upper_tail_scale": upper,
        "forward_base": base,
        "forward_calibrated": calibrated,
        "relative_interval_score_gain": score_gain,
        "evaluated_folds": evaluated_folds,
        "folds": folds,
    }


def build(stats: tuple[str, ...] = YARDAGE_STATS) -> dict:
    frame = load_rows(stats)
    results = {stat: fit_stat(group.copy()) for stat, group in frame.groupby("stat")}
    accepted = {
        stat: {
            "lower_tail_scale": rec["lower_tail_scale"],
            "upper_tail_scale": rec["upper_tail_scale"],
            "evidence": {k: rec[k] for k in (
                "forward_base", "forward_calibrated", "relative_interval_score_gain", "evaluated_folds"
            )},
        }
        for stat, rec in results.items() if rec["accepted"]
    }
    now = datetime.now(timezone.utc).isoformat()
    return clean({
        "contract": INTERVAL_CALIBRATION_CONTRACT,
        "version": "asymmetric-interval-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "fitted_at": now,
        "training_through": str(frame.game_date_et.max()) if not frame.empty else None,
        "model_versions": sorted(frame.model_version.dropna().astype(str).unique().tolist()) if not frame.empty else [],
        "target_coverage": 0.80,
        "method": "expanding-week asymmetric conformal scaling of immutable p10/p50/p90",
        "stats": accepted,
        "all_stat_results": results,
        "deployment_approved": bool(accepted),
        "betting_approved": False,
        "scope": "forecast_distribution_only",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stat", action="append", choices=YARDAGE_STATS)
    parser.add_argument("--write", action="store_true", help="Install only chronologically accepted stat calibrators")
    args = parser.parse_args()
    result = build(tuple(args.stat or YARDAGE_STATS))
    atomic_json(REPORT, result)
    if args.write:
        if not result["deployment_approved"]:
            raise SystemExit("No interval calibrator passed chronological deployment gates")
        atomic_json(ARTIFACT, {k: v for k, v in result.items() if k != "all_stat_results"})
    for stat, rec in result["all_stat_results"].items():
        print(stat, rec["status"], rec["lower_tail_scale"], rec["upper_tail_scale"],
              rec["relative_interval_score_gain"])
    print("report", REPORT)
    if args.write:
        print("artifact", ARTIFACT)


if __name__ == "__main__":
    main()
