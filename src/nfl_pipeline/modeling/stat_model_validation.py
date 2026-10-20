"""Prospective validation for independently deployable NFL stat components.

Immutable pre-kickoff player-game shadows validate point forecasts.  Exact
same-offer FanDuel shadows are additionally required for probability evidence.
Forecast deployment and betting approval are deliberately separate contracts.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.forecast_horizon import horizon
from nfl_pipeline.modeling.evaluation import clustered_gain, probability_metrics, projection_metrics
from nfl_pipeline.modeling.live_scoring_replay import validate_lock
from nfl_pipeline.modeling.stat_model_registry import activate, clear_candidate, load_registry

ROOT = Path(__file__).resolve().parents[3]
REPORT_JSON = ROOT / "reports" / "nfl_stat_model_validation_latest.json"
REPORT_MD = ROOT / "reports" / "nfl_stat_model_validation_latest.md"
CONTRACT = "nfl-stat-model-prospective-validation-v2"
SELECTED_TIERS = {"micro_projection", "locked_micro", "starter", "bankroll"}


def load_rows(day: date | None = None) -> list[dict[str, Any]]:
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        cur.execute("SET LOCAL statement_timeout='90s'")
        cur.execute(
            """
            SELECT p.id,p.game_date_et,p.game_id,p.player_id,p.player_name,p.stat,
                   p.side,p.line::float,p.price,p.book,p.tier,p.created_at_utc,
                   p.is_current,p.forecast_payload,p.forecast_horizon_minutes::float,
                   p.forecast_horizon_bucket,g.season,g.week,g.status,
                   g.start_ts_utc,o.fetched_at_utc AS offer_fetched_at,
                   r.actual_stat::float AS actual,r.result AS graded_result
              FROM bets.nfl_player_prop_predictions p
              JOIN raw.nfl_games g USING(game_id)
              LEFT JOIN odds.nfl_player_prop_lines o ON o.id=p.offer_id
              LEFT JOIN bets.nfl_player_prop_prediction_results r ON r.prediction_id=p.id
             WHERE p.integrity_version='nfl-asof-v2'
               AND (p.forecast_payload ? 'stat_model_challenger'
                    OR p.forecast_payload ? 'stat_model_challengers')
               AND (%(day)s IS NULL OR p.game_date_et=%(day)s)
             ORDER BY p.created_at_utc,p.id
            """,
            {"day": day},
        )
        return [dict(row) for row in cur.fetchall()]


def _over_probability(probability: Any, side: Any) -> float | None:
    try:
        value = float(probability)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or not 0 <= value <= 1:
        return None
    side = str(side or "").lower()
    return value if side == "over" else 1 - value if side == "under" else None


def _eligible_shadow(
    record: dict[str, Any], payload: dict[str, Any], shadow: dict[str, Any],
    entry: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if set(shadow.get("outputs") or ()) != set(entry.get("outputs") or ()):
        return None, "shadow_output_contract_mismatch"
    if entry.get("registered_at"):
        try:
            created_at = pd.Timestamp(record["created_at_utc"])
            registered_at = pd.Timestamp(entry["registered_at"])
        except (KeyError, TypeError, ValueError):
            return None, "invalid_candidate_registration_time"
        if created_at < registered_at:
            return None, "not_post_registration"
    try:
        training_end = date.fromisoformat(str(entry["training_end"])[:10])
    except (KeyError, ValueError):
        return None, "invalid_candidate_training_end"
    if record["game_date_et"] <= training_end:
        return None, "not_post_training"
    actual = record.get("actual")
    if actual is None or not np.isfinite(float(actual)):
        return None, "awaiting_settlement"
    probability_issue = None
    try:
        record_line = float(record["line"]) if record.get("line") is not None else None
        shadow_line = float(shadow["line"]) if shadow.get("line") is not None else None
    except (TypeError, ValueError):
        record_line = shadow_line = None
    if not shadow.get("same_offer"):
        probability_issue = "not_same_offer_shadow"
    elif str(record.get("book") or "").lower() != "fanduel":
        probability_issue = "not_fanduel_offer"
    elif record_line is None or shadow_line != record_line:
        probability_issue = "shadow_line_mismatch"
    elif str(shadow.get("side") or "").lower() not in {"over", "under"}:
        probability_issue = "shadow_side_missing"
    probability_eligible = probability_issue is None
    production_over = (_over_probability(payload.get("probability"), payload.get("side"))
                       if probability_eligible else None)
    candidate_over = (_over_probability(shadow.get("probability"), shadow.get("side"))
                      if probability_eligible else None)
    market_over = (_over_probability(payload.get("market_no_vig_probability"), payload.get("side"))
                   if probability_eligible else None)
    horizon_minutes = record.get("forecast_horizon_minutes")
    horizon_bucket = record.get("forecast_horizon_bucket")
    if horizon_minutes is None or not horizon_bucket:
        horizon_minutes, horizon_bucket = horizon(
            record.get("start_ts_utc"),
            payload.get("prediction_context_cutoff_utc") or record.get("created_at_utc"),
        )
    return {
        "component_id": shadow["component_id"],
        "outputs": tuple(entry.get("outputs") or ()),
        "season": int(record["season"]),
        "week": int(record["week"]),
        "game_id": record["game_id"],
        "player_id": record["player_id"],
        "stat": record["stat"],
        "line": record_line,
        "actual": float(actual),
        "production_projection": float(payload["projection"]),
        "candidate_projection": float(shadow["projection"]),
        "candidate_median": float(shadow["median"]),
        "candidate_p10": float(shadow["p10"]),
        "candidate_p90": float(shadow["p90"]),
        "production_over": production_over,
        "candidate_over": candidate_over,
        "market_over": market_over,
        "probability_eligible": probability_eligible,
        "probability_issue": probability_issue,
        "selected": probability_eligible and str(record.get("tier") or "") in SELECTED_TIERS,
        "forecast_horizon_minutes": horizon_minutes,
        "forecast_horizon_bucket": horizon_bucket,
        "created_at_utc": record["created_at_utc"],
    }, None


def _eligible(
    record: dict[str, Any], registry: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    error = validate_lock(record)
    if error:
        return [], [error]
    if record.get("graded_result") in {"void_nonparticipant", "pending_participation_review"}:
        return [], ["unverified_participation"]
    payload = record.get("forecast_payload") or {}
    components = (registry.get("candidates") or {}).get(record["stat"]) or {}
    if components.get("component_id"):
        components = {str(components["component_id"]): components}
    shadows = payload.get("stat_model_challengers") or {}
    if not isinstance(shadows, dict):
        shadows = {}
    legacy = payload.get("stat_model_challenger") or {}
    if not shadows and legacy.get("component_id"):
        shadows = {str(legacy["component_id"]): legacy}
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for component_id, shadow in shadows.items():
        entry = components.get(component_id)
        if not entry or shadow.get("component_id") != component_id:
            continue
        row, shadow_error = _eligible_shadow(record, payload, shadow, entry)
        if shadow_error:
            errors.append(shadow_error)
        elif row is not None:
            rows.append(row)
    if not rows and not errors:
        errors.append("not_current_registered_candidate")
    return rows, errors


def _scope_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"offers": 0, "player_games": 0, "weeks": 0}
    point = (
        frame.sort_values("created_at_utc")
        .drop_duplicates(["component_id", "game_id", "player_id", "stat"], keep="last")
    )
    result: dict[str, Any] = {
        "offers": int(len(frame)),
        "player_games": int(len(point)),
        "weeks": int(len(point[["season", "week"]].drop_duplicates())),
        "candidate_projection": projection_metrics(
            point.actual, point.candidate_projection, point.candidate_p10, point.candidate_p90
        ),
        "production_projection": projection_metrics(point.actual, point.production_projection),
        "candidate_median": projection_metrics(point.actual, point.candidate_median),
    }
    binary = frame.loc[frame.probability_eligible & frame.line.notna() & frame.actual.ne(frame.line)].copy()
    if binary.empty:
        result["probability_rows"] = 0
        result["probability_player_games"] = 0
        result["probability_weeks"] = 0
        return result
    binary["outcome_over"] = binary.actual.gt(binary.line).astype(float)
    binary = binary.drop_duplicates(
        ["component_id", "game_id", "player_id", "stat", "line"], keep="last"
    )
    binary["weight"] = 1 / binary.groupby(
        ["component_id", "game_id", "player_id", "stat"]
    ).line.transform("count")
    result["probability_rows"] = int(len(binary))
    result["probability_player_games"] = int(len(binary[["game_id", "player_id", "stat"]].drop_duplicates()))
    result["probability_weeks"] = int(len(binary[["season", "week"]].drop_duplicates()))
    for label, column in (
        ("candidate_probability", "candidate_over"),
        ("production_probability", "production_over"),
        ("market_probability", "market_over"),
    ):
        result[label] = probability_metrics(binary[column], binary.outcome_over, binary.weight)
    paired = binary.dropna(subset=["candidate_over", "production_over"]).copy()
    if len(paired):
        paired["reference_error"] = (paired.production_over - paired.outcome_over) ** 2
        paired["challenger_error"] = (paired.candidate_over - paired.outcome_over) ** 2
        grouped = paired.groupby(
            ["season", "week", "game_id", "player_id"]
        )[["reference_error", "challenger_error"]].mean().reset_index()
        result["candidate_vs_production_brier_gain"] = clustered_gain(grouped)
    return result


def _deployment_review(frame: pd.DataFrame, entry: dict[str, Any]) -> dict[str, Any]:
    horizon_metrics = {
        str(bucket): _scope_metrics(group)
        for bucket, group in frame.groupby("forecast_horizon_bucket", dropna=False)
    } if not frame.empty else {}
    primary_horizon = next(
        (
            bucket for bucket in ("t20", "t120", "t24h", "early", "unknown")
            if horizon_metrics.get(bucket, {}).get("player_games", 0) >= 100
            and horizon_metrics.get(bucket, {}).get("weeks", 0) >= 3
        ),
        None,
    )
    review_frame = (
        frame.loc[frame.forecast_horizon_bucket.eq(primary_horizon)].copy()
        if primary_horizon is not None else frame
    )
    metrics = _scope_metrics(review_frame)
    metrics["primary_horizon"] = primary_horizon
    metrics["by_horizon"] = horizon_metrics
    outputs = set(entry.get("outputs") or ())
    enough = (primary_horizon is not None and metrics["player_games"] >= 100
              and metrics["weeks"] >= 3)
    enough_probability = (primary_horizon is not None
                          and metrics.get("probability_player_games", 0) >= 100
                          and metrics.get("probability_weeks", 0) >= 3)
    review_complete = bool(enough and ("probability" not in outputs or enough_probability))
    approved: list[str] = []
    blockers: list[str] = []
    if not enough:
        blockers.append("needs_100_player_games_and_3_independent_weeks")
    if primary_horizon is None:
        blockers.append("needs_fixed_horizon_evidence")
    if frame.empty:
        return {
            "metrics": metrics,
            "approved_outputs": approved,
            "deployment_approved": False,
            "review_complete": False,
            "betting_approved": False,
            "blockers": blockers,
        }
    point = (
        review_frame.sort_values("created_at_utc")
        .drop_duplicates(["component_id", "game_id", "player_id", "stat"], keep="last")
    )
    if not point.empty:
        point["reference_error"] = (point.production_projection - point.actual) ** 2
        point["challenger_error"] = (point.candidate_projection - point.actual) ** 2
        squared_gain = clustered_gain(point)
        point["reference_abs"] = abs(point.production_projection - point.actual)
        point["challenger_abs"] = abs(point.candidate_median - point.actual)
        median_gain = clustered_gain(point, challenger="challenger_abs", reference="reference_abs")
        metrics["candidate_vs_production_squared_error_gain"] = squared_gain
        metrics["candidate_median_vs_production_mae_gain"] = median_gain
        c, p = metrics["candidate_projection"], metrics["production_projection"]
        if "expected_mean" in outputs and enough and c["rmse"] < p["rmse"] and abs(c["bias"]) <= abs(p["bias"]) and (squared_gain.get("lower_95") or 0) > 0:
            approved.append("expected_mean")
        elif "expected_mean" in outputs:
            blockers.append("expected_mean_not_proven_prospectively")
        cm = metrics["candidate_median"]
        if "median" in outputs and enough and cm["mae"] < p["mae"] and (median_gain.get("lower_95") or 0) > 0:
            approved.append("median")
        elif "median" in outputs:
            blockers.append("median_not_proven_prospectively")
    if "probability" in outputs:
        cp = metrics.get("candidate_probability", {})
        pp = metrics.get("production_probability", {})
        mp = metrics.get("market_probability", {})
        coverage = metrics.get("candidate_projection", {}).get("coverage_80")
        gain = metrics.get("candidate_vs_production_brier_gain", {})
        if (enough_probability and cp.get("rows", 0) >= 100 and cp.get("brier", 1) < pp.get("brier", 0)
                and cp.get("brier", 1) < mp.get("brier", 0)
                and cp.get("calibration_error", 1) <= pp.get("calibration_error", 0)
                and coverage is not None and .75 <= coverage <= .85
                and (gain.get("lower_95") or 0) > 0):
            approved.append("probability")
        else:
            blockers.append("probability_not_better_than_production_and_market")
    all_outputs_approved = bool(outputs) and outputs.issubset(approved)
    deployment_approved = bool(approved)
    if not deployment_approved and not blockers:
        blockers.append("registered_outputs_not_all_approved")
    return {
        "metrics": metrics,
        "approved_outputs": approved,
        "deployment_approved": deployment_approved,
        "review_complete": review_complete,
        "all_outputs_approved": all_outputs_approved,
        "betting_approved": False,
        "blockers": sorted(set(blockers)),
    }


def build_report(records: list[dict[str, Any]], registry: dict[str, Any]) -> dict[str, Any]:
    exclusions: Counter[str] = Counter()
    probability_exclusions: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for record in records:
        rows, errors = _eligible(record, registry)
        for error in errors:
            exclusions[error] += 1
        for row in rows:
            eligible.append(row)
            if row.get("probability_issue"):
                probability_exclusions[row["probability_issue"]] += 1
    frame = pd.DataFrame(eligible)
    eligible_offer_rows = int(frame.probability_eligible.sum()) if not frame.empty else 0
    cohorts: dict[str, Any] = {}
    for stat, components in (registry.get("candidates") or {}).items():
        if components.get("component_id"):
            components = {str(components["component_id"]): components}
        for component_id, entry in components.items():
            group = frame.loc[
                frame.component_id.eq(component_id) & frame.stat.eq(stat)
            ].copy() if not frame.empty else pd.DataFrame()
            review = _deployment_review(group, entry)
            review["all_offers"] = _scope_metrics(group)
            review["selected_picks"] = (
                _scope_metrics(group.loc[group.selected])
                if not group.empty else _scope_metrics(group)
            )
            cohorts[component_id] = {
                "stat": stat,
                "component_id": component_id,
                "registered_outputs": entry["outputs"],
                **review,
            }
    return {
        "contract": CONTRACT,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "status": "evaluated" if eligible else "waiting_for_settled_player_game_shadows",
        "records_scanned": len(records),
        "eligible_player_game_rows": len(eligible),
        "eligible_shadow_rows": len(eligible),
        "eligible_offer_rows": eligible_offer_rows,
        "exclusions": dict(exclusions),
        "probability_exclusions": dict(probability_exclusions),
        "cohorts": cohorts,
        "active_components_unchanged": True,
        "betting_approval": False,
        "limitations": [
            "Only forecasts captured after candidate registration can become prospective evidence.",
            "Point forecasts use immutable player-game shadows; probabilities additionally require the identical FanDuel offer.",
            "Multiple lines on one player-game receive inverse multiplicity weight.",
            "Selected-pick metrics use immutable production tiers; candidates do not receive a synthetic betting history.",
            "Forecast deployment does not authorize cash or change bankroll policy.",
        ],
    }


def _update_evidence(
    report: dict[str, Any], registry: dict[str, Any], activate_passed: bool
) -> dict[str, list[dict[str, Any]]]:
    changes: dict[str, list[dict[str, Any]]] = {"activated": [], "rejected": []}
    for component_id, cohort in report["cohorts"].items():
        stat = cohort["stat"]
        components = (registry.get("candidates") or {}).get(stat) or {}
        if components.get("component_id"):
            components = {str(components["component_id"]): components}
        entry = components.get(component_id)
        if not entry:
            continue
        path = Path(entry["evidence"])
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["prospective_validation"] = cohort["metrics"]
        evidence["approved_outputs"] = cohort["approved_outputs"]
        evidence["deployment_approved"] = cohort["deployment_approved"]
        evidence["review_complete"] = cohort["review_complete"]
        evidence["betting_approved"] = False
        evidence["blockers"] = cohort["blockers"]
        evidence["reviewed_at"] = report["built_at"]
        atomic_json(path, evidence)
        if activate_passed and cohort["deployment_approved"]:
            activation = activate(
                stat, cohort["approved_outputs"], component_id=component_id
            )
            changes["activated"].append(activation)
            # Once every registered output has enough prospective evidence,
            # retire any outputs that failed instead of carrying a stale trial
            # forever. Activated outputs remain untouched.
            remaining = activation.get("remaining_candidate_outputs") or []
            if cohort["review_complete"] and remaining:
                if clear_candidate(stat, component_id=component_id):
                    changes["rejected"].append({
                        "stat": stat,
                        "component_id": component_id,
                        "outputs": remaining,
                        "reason": "prospective_review_complete_outputs_not_approved",
                    })
        elif activate_passed and cohort["review_complete"]:
            if clear_candidate(stat, component_id=component_id):
                changes["rejected"].append({
                    "stat": stat,
                    "component_id": component_id,
                    "outputs": list(entry.get("outputs") or ()),
                    "reason": "prospective_review_complete_no_outputs_approved",
                })
    return changes


def _write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# NFL Per-Stat Model Validation",
        "",
        f"Status: {report['status']}",
        f"Eligible player-game shadow rows: {report['eligible_player_game_rows']}",
        "",
        "| Stat | Component | Registered outputs | Player-games | Weeks | Candidate / production RMSE | Candidate / production Brier | Candidate / market Brier | Deployment |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for component_id, cohort in report["cohorts"].items():
        stat = cohort["stat"]
        m = cohort["all_offers"]
        cp, pp, mp = m.get("candidate_probability", {}), m.get("production_probability", {}), m.get("market_probability", {})
        cproj, pproj = m.get("candidate_projection", {}), m.get("production_projection", {})
        fmt = lambda value, digits=3: "n/a" if value is None else f"{value:.{digits}f}"
        lines.append(
            f"| {stat} | {component_id} | {', '.join(cohort['registered_outputs'])} | "
            f"{m.get('player_games', 0)} | {m.get('weeks', 0)} | "
            f"{fmt(cproj.get('rmse'))} / {fmt(pproj.get('rmse'))} | "
            f"{fmt(cp.get('brier'), 4)} / {fmt(pp.get('brier'), 4)} | "
            f"{fmt(cp.get('brier'), 4)} / {fmt(mp.get('brier'), 4)} | "
            f"{'approved' if cohort['deployment_approved'] else 'shadow'} |"
        )
    lines += ["", "## Exclusions", ""]
    lines += [f"- {key}: {value}" for key, value in sorted(report["exclusions"].items())] or ["- none"]
    lines += ["", "## Probability-Only Exclusions", ""]
    lines += [f"- {key}: {value}" for key, value in sorted(report["probability_exclusions"].items())] or ["- none"]
    lines += ["", "Forecast deployment remains separate from betting approval.", ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def run(day: date | None = None, *, activate_passed: bool = False) -> dict[str, Any]:
    registry = load_registry()
    report = build_report(load_rows(day), registry)
    changes = _update_evidence(report, registry, activate_passed)
    report.update(changes)
    report["active_components_unchanged"] = not bool(report["activated"])
    report["candidate_registry_unchanged"] = not bool(report["activated"] or report["rejected"])
    atomic_json(REPORT_JSON, report)
    _write_markdown(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="ET date YYYY-MM-DD")
    parser.add_argument("--activate-passed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(date.fromisoformat(args.date) if args.date else None,
                         activate_passed=args.activate_passed), indent=2, default=str))


if __name__ == "__main__":
    main()
