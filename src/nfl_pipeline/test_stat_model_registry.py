import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from nfl_pipeline.modeling import stat_model_registry as registry
from nfl_pipeline.modeling.predict_player_props import (
    _apply_active_expected_mean, _score_stat_candidates, _score_stat_component,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_registry_migrates_active_component_to_output_ownership():
    entry = {"outputs": ["expected_mean", "probability"], "component_id": "c1"}
    migrated, changed = registry._migrate({
        "contract": registry.LEGACY_CONTRACT,
        "active": {"receiving_yards": entry},
    })
    assert changed
    assert migrated["contract"] == registry.CONTRACT
    assert migrated["active"]["receiving_yards"]["expected_mean"]["outputs"] == ["expected_mean"]
    assert migrated["active"]["receiving_yards"]["probability"]["outputs"] == ["probability"]


def test_v2_registry_migrates_singular_candidate_to_component_ownership():
    entry = {"outputs": ["expected_mean"], "component_id": "c1"}
    migrated, changed = registry._migrate({
        "contract": registry.V2_CONTRACT,
        "active": {},
        "candidates": {"receiving_yards": entry},
    })
    assert changed
    assert migrated["candidates"]["receiving_yards"] == {"c1": entry}


def test_activation_moves_only_approved_output(monkeypatch, tmp_path):
    model_path = tmp_path / "receiving.joblib"
    joblib.dump({"stat": "receiving_yards", "production_release": "base"}, model_path)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps({
        "deployment_approved": True,
        "approved_outputs": ["expected_mean"],
    }), encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    entry = {
        "stat": "receiving_yards",
        "component_id": "receiving-v2",
        "artifact": str(model_path),
        "sha256": _sha(model_path),
        "outputs": ["expected_mean", "probability"],
        "base_release": "base",
        "evidence": str(evidence_path),
    }
    registry_path.write_text(json.dumps({
        "contract": registry.CONTRACT,
        "base_release": "base",
        "active": {},
        "candidates": {"receiving_yards": {"receiving-v2": entry}},
    }), encoding="utf-8")
    monkeypatch.setattr(registry, "PATH", registry_path)
    monkeypatch.setattr(registry, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(registry, "active_release", lambda: {"release_id": "base"})
    registry._load.cache_clear()

    result = registry.activate(
        "receiving_yards", ["expected_mean"], component_id="receiving-v2"
    )
    saved = json.loads(registry_path.read_text(encoding="utf-8"))

    assert result["activated_outputs"] == ["expected_mean"]
    assert result["remaining_candidate_outputs"] == ["probability"]
    assert saved["active"]["receiving_yards"]["expected_mean"]["outputs"] == ["expected_mean"]
    assert saved["candidates"]["receiving_yards"]["receiving-v2"]["outputs"] == ["probability"]
    assert saved["active"]["receiving_yards"]["expected_mean"]["betting_approved"] is False


def test_live_scoring_merges_independently_owned_outputs(monkeypatch):
    def fake_score(component, frame, production):
        output = component[0]["outputs"][0]
        base = {
            "component_id": component[0]["component_id"],
            "outputs": [output],
            "mean": np.array([60.0]),
            "median": np.array([55.0]),
            "values": np.array([[20.0, 80.0]]),
            "weights": np.array([[0.5, 0.5]]),
            "p10": np.array([20.0]),
            "p90": np.array([80.0]),
        }
        if output == "probability":
            base["mean"] = np.array([50.0])
        return base

    monkeypatch.setattr(registry, "score_component", fake_score)
    artifact = {"active_stat_components": {"receiving_yards": {
        "expected_mean": ({"component_id": "mean-v2", "outputs": ["expected_mean"]}, {}),
        "probability": ({"component_id": "cdf-v3", "outputs": ["probability"]}, {}),
    }}}
    scored = _score_stat_component(
        artifact, "active_stat_components", "receiving_yards",
        pd.DataFrame({"player_id": ["p1"]}), np.array([45.0]),
    )
    assert scored["mean"][0] == 60.0
    assert scored["values"][0, 1] == 80.0
    assert scored["component_ids"] == {"expected_mean": "mean-v2", "probability": "cdf-v3"}


def test_median_only_owner_cannot_change_expected_projection():
    production = np.array([44.0])
    median_only = {
        'outputs': ['median'], 'mean': np.array([80.0]), 'median': np.array([35.0]),
    }
    np.testing.assert_array_equal(
        _apply_active_expected_mean(production, median_only), production)
    mean_owner = dict(median_only, outputs=['expected_mean'])
    np.testing.assert_array_equal(
        _apply_active_expected_mean(production, mean_owner), np.array([80.0]))


def test_nightly_registration_preserves_an_in_progress_prospective_candidate(monkeypatch, tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({
        "contract": registry.CONTRACT,
        "base_release": "base",
        "active": {},
        "candidates": {},
    }), encoding="utf-8")
    monkeypatch.setattr(registry, "PATH", registry_path)
    monkeypatch.setattr(registry, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(registry, "active_release", lambda: {"release_id": "base"})
    registry._load.cache_clear()

    def candidate(name):
        model_path = tmp_path / f"{name}.joblib"
        joblib.dump({
            "stat": "receiving_yards",
            "production_release": "base",
            "run_id": name,
            "training_end": "2026-09-01",
        }, model_path)
        evidence_path = tmp_path / f"{name}.json"
        evidence_path.write_text(json.dumps({
            "stat": "receiving_yards",
            "artifact_sha256": _sha(model_path),
        }), encoding="utf-8")
        return model_path, evidence_path

    first_model, first_evidence = candidate("first")
    second_model, second_evidence = candidate("second")
    first = registry.register_candidate(
        "receiving_yards", first_model, ["expected_mean"], evidence=first_evidence
    )
    second = registry.register_candidate(
        "receiving_yards", second_model, ["expected_mean"], evidence=second_evidence
    )
    saved = json.loads(registry_path.read_text(encoding="utf-8"))

    assert first["component_id"] == "first"
    assert second["component_id"] == "first"
    assert saved["candidates"]["receiving_yards"]["first"]["component_id"] == "first"
    assert not registry.clear_candidate("receiving_yards", component_id="second")
    assert registry.clear_candidate("receiving_yards", component_id="first")


def test_disjoint_probability_candidate_runs_beside_pinned_point_candidate(monkeypatch, tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({
        "contract": registry.CONTRACT,
        "base_release": "base",
        "active": {},
        "candidates": {},
    }), encoding="utf-8")
    monkeypatch.setattr(registry, "PATH", registry_path)
    monkeypatch.setattr(registry, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(registry, "active_release", lambda: {"release_id": "base"})
    registry._load.cache_clear()

    def candidate(name):
        model_path = tmp_path / f"{name}.joblib"
        joblib.dump({
            "stat": "receiving_yards", "production_release": "base",
            "run_id": name, "training_end": "2026-09-01",
        }, model_path)
        evidence_path = tmp_path / f"{name}.json"
        evidence_path.write_text(json.dumps({
            "stat": "receiving_yards", "artifact_sha256": _sha(model_path),
        }), encoding="utf-8")
        return model_path, evidence_path

    point_model, point_evidence = candidate("point")
    probability_model, probability_evidence = candidate("probability")
    registry.register_candidate(
        "receiving_yards", point_model, ["expected_mean", "median"], evidence=point_evidence
    )
    registry.register_candidate(
        "receiving_yards", probability_model, ["probability"], evidence=probability_evidence
    )
    saved = json.loads(registry_path.read_text(encoding="utf-8"))

    assert set(saved["candidates"]["receiving_yards"]) == {"point", "probability"}
    assert saved["candidates"]["receiving_yards"]["point"]["outputs"] == ["expected_mean", "median"]
    assert saved["candidates"]["receiving_yards"]["probability"]["outputs"] == ["probability"]


def test_explicit_point_replacement_preserves_disjoint_probability_trial(monkeypatch, tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({
        "contract": registry.CONTRACT,
        "base_release": "base",
        "active": {},
        "candidates": {},
    }), encoding="utf-8")
    monkeypatch.setattr(registry, "PATH", registry_path)
    monkeypatch.setattr(registry, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(registry, "active_release", lambda: {"release_id": "base"})
    registry._load.cache_clear()

    def candidate(name):
        model_path = tmp_path / f"{name}.joblib"
        joblib.dump({
            "stat": "receiving_yards", "production_release": "base",
            "run_id": name, "training_end": "2026-10-12",
        }, model_path)
        evidence_path = tmp_path / f"{name}.json"
        evidence_path.write_text(json.dumps({
            "stat": "receiving_yards", "artifact_sha256": _sha(model_path),
        }), encoding="utf-8")
        return model_path, evidence_path

    old_model, old_evidence = candidate("old-all")
    point_model, point_evidence = candidate("new-point")
    registry.register_candidate(
        "receiving_yards", old_model,
        ["expected_mean", "median", "probability"], evidence=old_evidence,
    )
    registry.register_candidate(
        "receiving_yards", point_model, ["expected_mean", "median"],
        evidence=point_evidence, replace_existing=True,
    )
    saved = json.loads(registry_path.read_text(encoding="utf-8"))

    assert saved["candidates"]["receiving_yards"]["old-all"]["outputs"] == ["probability"]
    assert saved["candidates"]["receiving_yards"]["new-point"]["outputs"] == [
        "expected_mean", "median",
    ]


def test_live_scoring_captures_every_disjoint_candidate(monkeypatch):
    def fake_score(component, frame, production):
        component_id = component[0]["component_id"]
        return {"component_id": component_id, "outputs": component[0]["outputs"]}

    monkeypatch.setattr(registry, "score_component", fake_score)
    artifact = {"candidate_stat_components": {"receiving_yards": {
        "point": ({"component_id": "point", "outputs": ["expected_mean", "median"]}, {}),
        "probability": ({"component_id": "probability", "outputs": ["probability"]}, {}),
    }}}

    scored = _score_stat_candidates(
        artifact, "receiving_yards", pd.DataFrame({"player_id": ["p1"]}), np.array([45.0])
    )

    assert set(scored) == {"point", "probability"}
    assert scored["point"]["outputs"] == ["expected_mean", "median"]
    assert scored["probability"]["outputs"] == ["probability"]


def test_drifted_candidate_is_withheld_without_stopping_predictions(monkeypatch):
    # A frozen-experiment drift in one research shadow stopped every player
    # prediction on 19 October. Candidates are capture-only, so a drifted one must
    # withhold its own evidence and leave the rest of the run alone.
    def fake_score(component, frame, production):
        entry = component[0]
        if entry["component_id"] == "drifted":
            raise ValueError("Frozen experiment changed; restore the pinned version")
        return {"component_id": entry["component_id"], "outputs": entry["outputs"]}

    monkeypatch.setattr(registry, "score_component", fake_score)
    artifact = {"candidate_stat_components": {"receiving_yards": {
        "drifted": ({"component_id": "drifted", "outputs": ["probability"]}, {}),
        "healthy": ({"component_id": "healthy", "outputs": ["expected_mean"]}, {}),
    }}}

    scored = _score_stat_candidates(
        artifact, "receiving_yards", pd.DataFrame({"player_id": ["p1"]}), np.array([45.0])
    )

    assert set(scored) == {"healthy"}  # the drifted shadow records nothing at all
    assert scored["healthy"]["outputs"] == ["expected_mean"]
