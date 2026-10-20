from datetime import datetime, timezone
import json

import pytest

from nfl_pipeline import experiment_registry as registry


def test_registration_is_idempotent_and_refuses_source_or_rule_rollover(tmp_path):
    source = tmp_path / "scorer.py"
    source.write_text("probability = 0.55\n")
    store = tmp_path / "registry"
    kwargs = dict(root=tmp_path, store=store)
    first = registry.register_experiment("paper-v2", {"min_ev": .03}, {"scorer": source}, **kwargs)
    assert registry.register_experiment("paper-v2", {"min_ev": .03}, {"scorer": source}, **kwargs) == first
    assert first["betting_approved"] is False
    with pytest.raises(ValueError, match="cannot be replaced"):
        registry.register_experiment("paper-v2", {"min_ev": .02}, {"scorer": source}, **kwargs)
    source.write_text("probability = 0.6\n")
    with pytest.raises(ValueError, match="Frozen experiment changed"):
        registry.check_experiment("paper-v2", **kwargs)
    assert (store / "paper-v2/sources/scorer.py").read_text() == "probability = 0.55\n"
    assert registry.status(**kwargs)[0]["status"] == "blocked"


def test_missing_artifact_tampered_registration_and_outside_paths_fail(tmp_path):
    source = tmp_path / "model.joblib"
    source.write_bytes(b"frozen model")
    kwargs = dict(root=tmp_path, store=tmp_path / "registry")
    doc = registry.register_experiment("paper-v2", {}, {"model": source}, **kwargs)
    source.unlink()
    with pytest.raises(ValueError, match="Frozen experiment changed"):
        registry.check_experiment("paper-v2", **kwargs)
    path = kwargs["store"] / "paper-v2/registration.json"
    doc["betting_approved"] = True
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="checksum/contract"):
        registry.load_experiment("paper-v2", **kwargs)
    with pytest.raises(ValueError):
        registry.register_experiment("../escape", {}, {"model": source}, **kwargs)
    with pytest.raises(ValueError):
        registry.register_experiment("outside", {}, {"model": tmp_path.parent / "outside"}, **kwargs)


def test_runtime_change_is_not_silently_accepted(tmp_path, monkeypatch):
    source = tmp_path / "model.json"
    source.write_text("{}")
    kwargs = dict(root=tmp_path, store=tmp_path / "registry")
    monkeypatch.setattr(registry, "runtime_versions", lambda: {"numpy": "1"})
    registry.register_experiment("paper-v2", {}, {"model": source}, **kwargs)
    monkeypatch.setattr(registry, "runtime_versions", lambda: {"numpy": "2"})
    with pytest.raises(ValueError, match="runtime_versions"):
        registry.register_experiment("paper-v2", {}, {"model": source}, **kwargs)


def test_new_active_overlay_cannot_silently_change_an_absent_input(tmp_path):
    source = tmp_path / "model.json"
    source.write_text("{}")
    absent = tmp_path / "active_overlay.json"
    kwargs = dict(root=tmp_path, store=tmp_path / "registry")
    registry.register_experiment("paper-v2", {}, {"model": source},
                                 absent_paths={"overlay": absent}, **kwargs)
    absent.write_text("{}")
    with pytest.raises(ValueError, match="overlay"):
        registry.check_experiment("paper-v2", **kwargs)


def test_source_closure_tracks_imports_without_executing_code(tmp_path):
    package = tmp_path / "src/nfl_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "scorer.py").write_text("from . import helper\nraise RuntimeError('must not execute')\n")
    (package / "helper.py").write_text("from nfl_pipeline import nested\n")
    (package / "nested.py").write_text("import numpy\n")
    files = registry.source_dependencies("nfl_pipeline.scorer", root=tmp_path)
    assert {"nfl_pipeline", "nfl_pipeline.scorer", "nfl_pipeline.helper", "nfl_pipeline.nested"} <= files.keys()
    assert registry.source_dependencies("nfl_pipeline.nested", root=tmp_path).keys() == {"nfl_pipeline", "nfl_pipeline.nested"}
