"""Per-target forecast deployment registry.

The monolithic release remains the fallback for every stat.  A component may
replace only explicitly approved outputs for one target; forecast deployment
never implies betting approval.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from nfl_pipeline.integrity import MODEL_ROOT, active_release, atomic_json

CONTRACT = "nfl-stat-model-registry-v3"
V2_CONTRACT = "nfl-stat-model-registry-v2"
LEGACY_CONTRACT = "nfl-stat-model-registry-v1"
PATH = MODEL_ROOT / "stat_model_registry.json"
OUTPUTS = {"expected_mean", "median", "probability", "interval"}
STATS = {
    "passing_yards", "rushing_yards", "receiving_yards", "receptions",
    "passing_tds", "rushing_tds", "receiving_tds",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty() -> dict[str, Any]:
    release = active_release()
    if not release:
        raise RuntimeError("NFL stat registry requires an active base release")
    return {
        "contract": CONTRACT,
        "base_release": release["release_id"],
        "active": {},
        "candidates": {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _migrate(doc: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Move legacy registries to output-level active and multi-candidate trials."""
    if doc.get("contract") == CONTRACT:
        return doc, False
    if doc.get("contract") not in {LEGACY_CONTRACT, V2_CONTRACT}:
        raise ValueError("NFL stat model registry contract mismatch")
    active = doc.get("active") or {}
    if doc.get("contract") == LEGACY_CONTRACT:
        active = {
            stat: {
                output: dict(entry, outputs=[output])
                for output in entry.get("outputs") or ()
            }
            for stat, entry in active.items()
        }
    candidates: dict[str, dict[str, Any]] = {}
    for stat, value in (doc.get("candidates") or {}).items():
        if value.get("component_id"):
            candidates[stat] = {str(value["component_id"]): value}
        else:
            candidates[stat] = value
    migrated = dict(doc, contract=CONTRACT, active=active, candidates=candidates)
    migrated["updated_at"] = datetime.now(timezone.utc).isoformat()
    return migrated, True


def load_registry() -> dict[str, Any]:
    doc = json.loads(PATH.read_text(encoding="utf-8")) if PATH.exists() else _empty()
    doc, migrated = _migrate(doc)
    release = active_release()
    if not release or doc.get("base_release") != release.get("release_id"):
        raise ValueError("NFL stat model registry base release mismatch")
    for stat, outputs in (doc.get("active") or {}).items():
        if not isinstance(outputs, dict):
            raise ValueError("NFL active stat outputs must be a mapping")
        for output, entry in outputs.items():
            if output not in OUTPUTS or entry.get("outputs") != [output]:
                raise ValueError("NFL active stat output ownership is invalid")
            _validate_entry(stat, entry)
    for stat, components in (doc.get("candidates") or {}).items():
        if not isinstance(components, dict):
            raise ValueError("NFL stat candidates must be a component mapping")
        owned: set[str] = set()
        for component_id, entry in components.items():
            if component_id != entry.get("component_id"):
                raise ValueError("NFL stat candidate component identity mismatch")
            _validate_entry(stat, entry)
            overlap = owned.intersection(entry.get("outputs") or ())
            if overlap:
                raise ValueError("NFL stat candidate outputs have multiple owners")
            owned.update(entry.get("outputs") or ())
    if migrated:
        atomic_json(PATH, doc)
    return doc


def _validate_entry(stat: str, entry: dict[str, Any]) -> None:
    if stat not in STATS or entry.get("stat") != stat:
        raise ValueError("Invalid stat model registry target")
    outputs = set(entry.get("outputs") or [])
    if not outputs or not outputs.issubset(OUTPUTS):
        raise ValueError("Invalid stat model output contract")
    path = Path(str(entry.get("artifact") or "")).resolve()
    try:
        path.relative_to(MODEL_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Stat model artifact is outside the model root") from exc
    if not path.is_file() or _sha(path) != entry.get("sha256"):
        raise ValueError("Stat model artifact missing or checksum mismatch")
    if entry.get("base_release") != active_release()["release_id"]:
        raise ValueError("Stat model component belongs to another base release")


@lru_cache(maxsize=16)
def _load(path: str, sha256: str) -> dict[str, Any]:
    artifact_path = Path(path)
    if _sha(artifact_path) != sha256:
        raise ValueError("Stat model artifact checksum mismatch")
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("Stat model artifact is not a mapping")
    return artifact


def active_components() -> dict[str, dict[str, tuple[dict[str, Any], dict[str, Any]]]]:
    result: dict[str, dict[str, tuple[dict[str, Any], dict[str, Any]]]] = {}
    for stat, outputs in (load_registry().get("active") or {}).items():
        result[stat] = {}
        for output, entry in outputs.items():
            artifact = _load(entry["artifact"], entry["sha256"])
            if artifact.get("stat") != stat or artifact.get("production_release") != entry["base_release"]:
                raise ValueError("Stat model artifact identity mismatch")
            result[stat][output] = (entry, artifact)
    return result


def candidate_components() -> dict[str, dict[str, tuple[dict[str, Any], dict[str, Any]]]]:
    result: dict[str, dict[str, tuple[dict[str, Any], dict[str, Any]]]] = {}
    for stat, components in (load_registry().get("candidates") or {}).items():
        result[stat] = {}
        for component_id, entry in components.items():
            artifact = _load(entry["artifact"], entry["sha256"])
            if artifact.get("stat") != stat or artifact.get("production_release") != entry["base_release"]:
                raise ValueError("Stat model candidate identity mismatch")
            result[stat][component_id] = (entry, artifact)
    return result


def score_component(component: tuple[dict[str, Any], dict[str, Any]], frame, production_projection):
    """Score a full player roster; candidate output is capture-only until activated."""
    entry, artifact = component
    runtime = artifact.get("runtime")
    if runtime is None or not hasattr(runtime, "score"):
        raise ValueError("Stat model artifact has no runtime scorer")
    from nfl_pipeline.modeling.receiving_paper_trial import generation_attestation
    before = generation_attestation(entry["component_id"], entry.get("sha256"))
    score = runtime.score(frame, production_projection)
    from nfl_pipeline.modeling.stat_runtime import validate_score
    validate_score(score, len(frame))
    score["component_id"] = entry["component_id"]
    score["outputs"] = list(entry["outputs"])
    score["artifact_sha256"] = entry.get("sha256")
    after = generation_attestation(entry["component_id"], entry.get("sha256"))
    if (before is None) != (after is None) or (before and
            any(before[key] != after[key] for key in ("registration_sha256", "artifact_sha256", "scoring_fingerprint"))):
        raise ValueError("Frozen experiment changed during shadow generation")
    score["experiment_generation"] = dict(after, started_at=before["verified_at"]) if after else None
    score["betting_approved"] = False
    return score


def register_candidate(
    stat: str,
    artifact: str | Path,
    outputs: list[str] | tuple[str, ...],
    *,
    evidence: str | Path,
    replace_existing: bool = False,
) -> dict[str, Any]:
    path = Path(artifact).resolve()
    review_path = Path(evidence).resolve()
    payload = joblib.load(path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    release = active_release()
    if stat not in STATS or payload.get("stat") != stat:
        raise ValueError("Candidate target does not match artifact")
    if payload.get("production_release") != release["release_id"]:
        raise ValueError("Candidate was not evaluated against active production")
    if review.get("artifact_sha256") != _sha(path) or review.get("stat") != stat:
        raise ValueError("Candidate evidence does not identify this artifact")
    requested_outputs = sorted(set(outputs))
    if not requested_outputs or not set(requested_outputs).issubset(OUTPUTS):
        raise ValueError("Candidate must own at least one valid stat output")
    doc = load_registry()
    candidates = doc.setdefault("candidates", {}).setdefault(stat, {})
    component_id = str(payload.get("run_id") or path.parent.name)
    overlapping = [
        existing
        for existing in candidates.values()
        if set(existing.get("outputs") or ()).intersection(requested_outputs)
    ]
    if overlapping and not replace_existing:
        # Prospective cohorts are immutable. A nightly retrain may add a
        # disjoint output trial, but it must not reset an output already under
        # review before that trial reaches its decision threshold.
        return dict(sorted(overlapping, key=lambda item: item["component_id"])[0])
    if replace_existing:
        for existing_id, existing in list(candidates.items()):
            remaining = set(existing.get("outputs") or ()) - set(requested_outputs)
            if remaining:
                candidates[existing_id] = dict(existing, outputs=sorted(remaining))
            elif set(existing.get("outputs") or ()).intersection(requested_outputs):
                del candidates[existing_id]
    entry = {
        "stat": stat,
        "component_id": component_id,
        "contract": str(payload.get("contract") or "unknown"),
        "artifact": str(path),
        "sha256": _sha(path),
        "outputs": requested_outputs,
        "base_release": release["release_id"],
        "training_end": str(payload.get("training_end") or ""),
        "evidence": str(review_path),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "betting_approved": False,
    }
    _validate_entry(stat, entry)
    candidates[component_id] = entry
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(PATH, doc)
    _load.cache_clear()
    return entry


def clear_candidate(stat: str, *, component_id: str | None = None) -> bool:
    """Remove a failed shadow candidate without touching an active component."""
    doc = load_registry()
    candidates = doc.setdefault("candidates", {})
    components = candidates.get(stat)
    if not components:
        return False
    if component_id is None:
        del candidates[stat]
    else:
        if component_id not in components:
            return False
        del components[component_id]
        if not components:
            del candidates[stat]
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(PATH, doc)
    _load.cache_clear()
    return True


def activate(
    stat: str,
    outputs: list[str] | tuple[str, ...] | None = None,
    *,
    component_id: str | None = None,
) -> dict[str, Any]:
    """Activate only the independently approved outputs of a component."""
    doc = load_registry()
    components = (doc.get("candidates") or {}).get(stat) or {}
    if not components:
        raise ValueError("No registered candidate for stat")
    if component_id is None:
        if len(components) != 1:
            raise ValueError("component_id is required when a stat has multiple candidates")
        component_id = next(iter(components))
    entry = components.get(component_id)
    if not entry:
        raise ValueError("No registered candidate with that component_id")
    review = json.loads(Path(entry["evidence"]).read_text(encoding="utf-8"))
    if not review.get("deployment_approved"):
        raise ValueError("Stat model deployment evidence has not passed")
    approved = set(review.get("approved_outputs") or [])
    requested = set(outputs or approved)
    if not requested:
        raise ValueError("No approved stat model outputs requested")
    if not requested.issubset(set(entry["outputs"])):
        raise ValueError("Requested output does not belong to the candidate")
    if not requested.issubset(approved):
        raise ValueError("Evidence does not approve every requested output")
    activated_at = datetime.now(timezone.utc).isoformat()
    active = doc.setdefault("active", {}).setdefault(stat, {})
    for output in sorted(requested):
        active[output] = dict(
            entry,
            outputs=[output],
            activated_at=activated_at,
            betting_approved=False,
        )
    remaining = set(entry["outputs"]) - requested
    if remaining:
        components[component_id] = dict(entry, outputs=sorted(remaining))
    else:
        components.pop(component_id, None)
        if not components:
            doc.setdefault("candidates", {}).pop(stat, None)
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(PATH, doc)
    _load.cache_clear()
    return {
        "stat": stat,
        "component_id": entry["component_id"],
        "activated_outputs": sorted(requested),
        "remaining_candidate_outputs": sorted(remaining),
        "activated_at": activated_at,
    }


def describe() -> dict[str, Any]:
    doc = load_registry()
    return {
        "contract": CONTRACT,
        "base_release": doc["base_release"],
        "active": doc.get("active") or {},
        "candidates": doc.get("candidates") or {},
        "betting_approval_separate": True,
    }
