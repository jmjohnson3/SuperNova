"""Train and register the isolated receiving opportunity/efficiency candidate.

This command never changes production or betting approval. It replaces the
current receiving shadow trial only after all historical output contracts pass,
then starts a new prospective cohort at registration time.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from nfl_pipeline.integrity import MODEL_ROOT, atomic_joblib, atomic_json, freeze_production
from nfl_pipeline.modeling.challenger_models import POSITIONS
from nfl_pipeline.modeling.stat_model_registry import load_registry, register_candidate
from nfl_pipeline.modeling.stat_runtime import TeamOpportunityRuntime
from nfl_pipeline.modeling.team_workload import load_history, prepare_history
from nfl_pipeline.modeling.train_accuracy_challengers import ROOT, load_training
from nfl_pipeline.modeling.train_accuracy_components import evaluate_team, prospective_outputs

log = logging.getLogger(__name__)

STAT = "receiving_yards"
CONTRACT = "nfl-receiving-opportunity-efficiency-v2"
FAMILY_ROOT = MODEL_ROOT / "receiving_opportunity_efficiency"
REQUIRED_OUTPUTS = {"expected_mean", "median", "probability"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source() -> tuple[dict, dict, Path]:
    manifest_path = MODEL_ROOT / "challengers" / "latest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_root = MODEL_ROOT / "challengers" / manifest["run_id"]
    model_path = source_root / "models.joblib"
    if _sha(model_path) != manifest["sha256"]:
        raise RuntimeError("Source challenger checksum mismatch")
    models = joblib.load(model_path)
    source = joblib.load(source_root / f"{STAT}_oof.joblib")
    return models, source, source_root


def _load_team_history(players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cache_path = MODEL_ROOT / "accuracy_components" / "team_history_cache.joblib"
    contract = "team-history-v3-lagged-usage"
    through = str(players.game_date_et.max())
    cached = joblib.load(cache_path) if cache_path.exists() else None
    if (isinstance(cached, dict) and cached.get("contract") == contract
            and cached.get("through") == through):
        return cached["teams"], cached["roster"], {
            "cache": str(cache_path), "contract": contract, "through": through,
            "source_rows": cached.get("source_rows"),
        }

    raw = load_history()
    raw = raw.loc[raw.game_date_et <= players.game_date_et.max()]
    teams, roster = prepare_history(raw)
    payload = {
        "contract": contract,
        "through": through,
        "teams": teams,
        "roster": roster,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_rows": len(raw),
    }
    atomic_joblib(cache_path, payload)
    return teams, roster, {
        "cache": str(cache_path), "contract": contract, "through": through,
        "source_rows": len(raw),
    }


def _write_report(report: dict) -> None:
    atomic_json(ROOT / "reports" / "nfl_receiving_opportunity_efficiency_latest.json", report)
    gates = report["historical_screen"]
    mean = gates["expected_mean"]
    median = gates["median"]
    probability = gates["probability"]
    lines = [
        "# NFL Receiving Opportunity + Efficiency v2",
        "",
        f"Candidate: {report['run_id']}",
        f"Frozen production: {report['production_release']}",
        f"Training through: {report['training_end']}",
        "",
        "This is a forecast shadow candidate, not a betting approval.",
        "",
        "## Historical chronological screen",
        "",
        "| Output | Candidate | Reference | Pass |",
        "|---|---:|---:|---|",
        f"| Expected yards RMSE | {mean['metrics']['rmse']:.3f} | {mean['reference']['rmse']:.3f} | {mean['pass']} |",
        f"| Typical outcome MAE | {median['metrics']['mae']:.3f} | {mean['reference']['mae']:.3f} | {median['pass']} |",
        f"| Proxy-line Brier | {probability['metrics']['brier']:.4f} | {probability['reference']['brier']:.4f} | {probability['pass']} |",
        f"| Calibration error | {probability['metrics']['calibration_error']:.2%} | {probability['reference']['calibration_error']:.2%} | {probability['pass']} |",
        f"| Nominal 80% coverage | {probability['coverage_80']:.2%} | 80.00% | {probability['pass']} |",
        "",
        "## What changed",
        "",
        f"- Team-allocation weight selected chronologically: {report['opportunity_blend']:.0%}.",
        f"- Current-season opportunity weighting selected chronologically: {report['season_recency_strength']:.0%}.",
        "- The remaining opportunity weight comes from the player's stable pregame role history.",
        f"- Receiving efficiency is estimated separately with {report['efficiency_shrinkage_strength']:.0%} empirical shrinkage.",
        "- Tail calibration is mean preserving, so uncertainty cannot silently inflate expected yards.",
        f"- Optional probability calibration enabled: {report['probability_calibrator_enabled']}.",
        f"- Optional tail repair enabled: {report['tail_repair_enabled']}.",
        "- The prospective scorer now uses the exact validated distribution path.",
        "",
        "## Prospective status",
        "",
        f"- Registered outputs: {', '.join(report['registered_outputs'])}.",
        f"- Superseded shadow candidates: {', '.join(report['superseded_candidates']) or 'none'}.",
        "- Acceptance starts after this registration. Previously reviewed games are not reused as prospective proof.",
        "- Required next evidence: identical FanDuel lines, at least 100 player-games and 3 independent weeks, plus selected-pick validation.",
        "- Production changed: no.",
        "- Betting approved: no.",
    ]
    path = ROOT / "reports" / "nfl_receiving_opportunity_efficiency_latest.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict:
    freeze = freeze_production(
        "Receiving opportunity/efficiency candidate; production remains frozen"
    )
    production_path = MODEL_ROOT / "active_release.json"
    production_before = production_path.read_bytes()
    source_models, source, source_root = _load_source()
    if source_models.get("production_release") != freeze["release_id"]:
        raise RuntimeError("Source challenger belongs to another production release")

    players, _ = load_training()
    training_end = pd.Timestamp(source_models["training_end"]).date()
    players = players.loc[
        (players.game_date_et <= training_end) & players.position.isin(POSITIONS[STAT])
    ].copy()
    teams, roster, history = _load_team_history(players)
    seasons = tuple(sorted(set(source["player_games"].season)))
    log.info(
        "Evaluating %s player-games through %s against frozen %s",
        len(players), training_end, freeze["release_id"],
    )

    gates, rows, lines, bundle = evaluate_team(
        players, teams, roster, STAT, source, seasons
    )
    outputs = prospective_outputs(
        "team_receiving_yards", gates, {"stat": STAT, "bundle": bundle}
    )
    if set(outputs) != REQUIRED_OUTPUTS:
        missing = sorted(REQUIRED_OUTPUTS - set(outputs))
        raise RuntimeError(
            "Receiving v2 failed its historical output contracts: " + ", ".join(missing)
        )

    run_id = datetime.now(timezone.utc).strftime("receiving-oe-v2-%Y%m%dT%H%M%S%fZ")
    root = FAMILY_ROOT / run_id
    artifact = {
        "contract": CONTRACT,
        "run_id": run_id,
        "stat": STAT,
        "runtime": TeamOpportunityRuntime(bundle, STAT),
        "production_release": freeze["release_id"],
        "training_end": str(training_end),
        "source_run": source_models["run_id"],
        "source_root": str(source_root),
        "betting_approved": False,
    }
    model_path = root / "model.joblib"
    atomic_joblib(model_path, artifact)
    model_sha = _sha(model_path)
    atomic_joblib(root / "oof.joblib", {"player_games": rows, "proxy_lines": lines})

    registry_before = load_registry()
    previous = sorted((registry_before.get("candidates") or {}).get(STAT, {}))
    evidence = {
        "contract": "nfl-stat-candidate-evidence-v1",
        "stat": STAT,
        "component": "receiving_opportunity_efficiency_v2",
        "artifact_sha256": model_sha,
        "historical_screen_outputs": outputs,
        "approved_outputs": [],
        "deployment_approved": False,
        "betting_approved": False,
        "prospective_acceptance_start": "candidate registration timestamp",
        "blockers": [
            "prospective_identical_fanduel_line_validation_required",
            "needs_100_player_games_and_3_independent_weeks",
            "selected_pick_probability_validation_required",
        ],
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    evidence_path = root / "evidence.json"
    atomic_json(evidence_path, evidence)
    entry = register_candidate(
        STAT, model_path, outputs, evidence=evidence_path, replace_existing=True
    )

    report = {
        "contract": CONTRACT,
        "run_id": run_id,
        "production_release": freeze["release_id"],
        "training_end": str(training_end),
        "source_run": source_models["run_id"],
        "source_oof": str(source_root / f"{STAT}_oof.joblib"),
        "history": history,
        "training_player_games": len(players),
        "historical_screen": gates,
        "registered_outputs": sorted(outputs),
        "registered_at": entry["registered_at"],
        "component_id": entry["component_id"],
        "artifact": str(model_path),
        "artifact_sha256": model_sha,
        "opportunity_blend": float(bundle["opportunity_blend"]),
        "efficiency_shrinkage_strength": float(bundle["rate"].strength),
        "probability_calibrator_enabled": bool(bundle["calibrator"].enabled),
        "tail_repair_enabled": bool(bundle["tail"].enabled),
        "structure_tuning": bundle["structure_tuning"],
        "season_recency_strength": float(bundle.get("season_recency_strength", 0.0)),
        "season_recency_validation": bundle.get("season_recency_validation"),
        "player_specific_uncertainty_groups": len(
            getattr(bundle["uncertainty"], "role_quantiles", {})
        ),
        "mean_preserving_distribution": bool(bundle["mean_preserving_distribution"]),
        "superseded_candidates": previous,
        "prospective_evidence_reused": False,
        "production_changed": False,
        "betting_approved": False,
    }
    atomic_json(root / "report.json", report)
    atomic_json(FAMILY_ROOT / "latest.json", {
        "run_id": run_id,
        "production_release": freeze["release_id"],
        "sha256": model_sha,
    })
    _write_report(report)
    if production_path.read_bytes() != production_before:
        raise RuntimeError("Production release changed while training receiving v2")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    result = run()
    print(json.dumps({
        "run_id": result["run_id"],
        "component_id": result["component_id"],
        "registered_outputs": result["registered_outputs"],
        "production_changed": result["production_changed"],
        "betting_approved": result["betting_approved"],
    }, indent=2))
