from __future__ import annotations

import joblib

from .model_release import (
    DEFAULT_HITTER_RATE_SHADOW_RELEASE_ID,
    ensure_hitter_production_artifact,
    ensure_hitter_rate_live_artifact,
    hitter_rate_shadow_release_id,
)


def test_hitter_production_artifact_is_pinned_once(tmp_path) -> None:
    source = tmp_path / "hitter_player_game_outcome_models.joblib"
    joblib.dump({"models": {"marker": "first"}}, source)
    production = ensure_hitter_production_artifact(tmp_path)
    assert production is not None
    first = joblib.load(production)
    assert first["models"]["marker"] == "first"
    assert first["artifact_role"] == "production_frozen"

    joblib.dump({"models": {"marker": "nightly-challenger"}}, source)
    assert ensure_hitter_production_artifact(tmp_path) == production
    assert joblib.load(production)["models"]["marker"] == "first"


def test_hitter_rate_shadow_release_is_stable_and_overridable(monkeypatch) -> None:
    monkeypatch.delenv("MLB_HITTER_RATE_SHADOW_RELEASE_ID", raising=False)
    assert hitter_rate_shadow_release_id() == DEFAULT_HITTER_RATE_SHADOW_RELEASE_ID
    monkeypatch.setenv("MLB_HITTER_RATE_SHADOW_RELEASE_ID", "shadow-test-r2")
    assert hitter_rate_shadow_release_id() == "shadow-test-r2"


def test_hitter_rate_live_artifact_pins_accepted_heads_once(tmp_path) -> None:
    source = tmp_path / "hitter_player_game_outcome_models.challenger.joblib"
    joblib.dump({
        "model_release_id": "hitter-live-test-r1",
        "models": {
            "hits_count_model": "model",
            "hits_count_blend_alpha": 0.5,
            "hr_count_model": "model",
            "hr_count_blend_alpha": 0.0,
        },
        "recommendation": {"direct_hits_count_repair_enabled": True},
    }, source)
    live = ensure_hitter_rate_live_artifact(tmp_path)
    assert live is not None
    first = joblib.load(live)
    assert first["artifact_role"] == "live_forecast_frozen"
    assert first["live_forecast_accepted_heads"] == ["hits"]

    joblib.dump({
        "models": {"hits_count_model": "model", "hits_count_blend_alpha": 0.0},
        "recommendation": {},
    }, source)
    assert ensure_hitter_rate_live_artifact(tmp_path) == live
    assert joblib.load(live)["live_forecast_accepted_heads"] == ["hits"]
