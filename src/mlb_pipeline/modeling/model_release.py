"""Stable prospective model-release and evaluation-phase identifiers."""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_via


DEFAULT_CANONICAL_FORECAST_PHASE = "day_pregame"
DEFAULT_HITTER_MODEL_RELEASE_ID = "hitter-pg-2026-07-06-r2"
DEFAULT_HITTER_RATE_SHADOW_RELEASE_ID = "hitter-rate-shadow-2026-07-09-r1"
DEFAULT_PITCHER_MODEL_RELEASE_ID = "pitcher-k-2026-07-03-r1"
DEFAULT_GAME_MODEL_RELEASE_ID = "game-2026-07-03-r1"
HITTER_PRODUCTION_ARTIFACT = "hitter_player_game_outcome_models.production.joblib"
HITTER_CHALLENGER_ARTIFACT = "hitter_player_game_outcome_models.challenger.joblib"
HITTER_RATE_LIVE_ARTIFACT = "hitter_player_game_outcome_models.live_forecast.joblib"


def canonical_forecast_phase() -> str:
    return (
        os.getenv("MLB_CANONICAL_FORECAST_PHASE", "").strip()
        or DEFAULT_CANONICAL_FORECAST_PHASE
    )


def hitter_model_release_id(artifact: Mapping[str, Any] | None = None) -> str:
    """Return a recipe release ID that does not change on nightly retraining."""
    configured = os.getenv("MLB_HITTER_MODEL_RELEASE_ID", "").strip()
    if configured:
        return configured
    artifact_release = str((artifact or {}).get("model_release_id") or "").strip()
    return artifact_release or DEFAULT_HITTER_MODEL_RELEASE_ID


def hitter_rate_shadow_release_id() -> str:
    """Stable prospective ID for hits/HR rate challengers kept out of production."""
    return (
        os.getenv("MLB_HITTER_RATE_SHADOW_RELEASE_ID", "").strip()
        or DEFAULT_HITTER_RATE_SHADOW_RELEASE_ID
    )


def hitter_production_artifact_path(model_dir: Path) -> Path:
    return Path(model_dir) / HITTER_PRODUCTION_ARTIFACT


def hitter_rate_live_artifact_path(model_dir: Path) -> Path:
    return Path(model_dir) / HITTER_RATE_LIVE_ARTIFACT


def _accepted_count_head(artifact: Mapping[str, Any], prefix: str) -> bool:
    models = artifact.get("models") or {}
    model = models.get(f"{prefix}_count_model")
    try:
        alpha = float(models.get(f"{prefix}_count_blend_alpha") or 0.0)
    except (TypeError, ValueError):
        alpha = 0.0
    if model is None or alpha <= 0.0:
        return False
    rec = artifact.get("recommendation") or {}
    metric = (artifact.get("metrics") or {}).get(f"direct_{prefix}_count_repair") or {}
    try:
        gain = float(rec.get(f"direct_{prefix}_count_repair_mae_gain") or 0.0)
    except (TypeError, ValueError):
        gain = 0.0
    return bool(rec.get(f"direct_{prefix}_count_repair_enabled") or metric.get("enabled") or gain > 0.0)


def ensure_hitter_production_artifact(model_dir: Path) -> Path | None:
    """Pin the current hitter artifact once; nightly trainers write challengers."""
    try:
        import joblib
    except Exception:
        return None
    model_dir = Path(model_dir)
    target = hitter_production_artifact_path(model_dir)
    if target.exists():
        return target
    source = model_dir / "hitter_player_game_outcome_models.joblib"
    if not source.exists():
        return None
    artifact = joblib.load(source)
    release_id = hitter_model_release_id()
    frozen_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    artifact["model_release_id"] = release_id
    artifact["artifact_role"] = "production_frozen"
    artifact["production_frozen_at_utc"] = frozen_at
    atomic_write_via(
        target,
        lambda temp: joblib.dump(artifact, temp),
        attempts=6,
        retry_all_errors=True,
    )
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_write_json(model_dir / "hitter_player_game_outcome_models.production.json", {
        "model_release_id": release_id,
        "artifact_role": "production_frozen",
        "production_frozen_at_utc": frozen_at,
        "source_artifact": source.name,
        "sha256": digest,
        "minimum_prospective_dates": 5,
        "promotion": "manual_after_projection_and_betting_audits",
    })
    return target


def ensure_hitter_rate_live_artifact(model_dir: Path) -> Path | None:
    """Pin accepted hits/HR challenger heads for stable live forecast scoring."""
    try:
        import joblib
    except Exception:
        return None
    model_dir = Path(model_dir)
    target = hitter_rate_live_artifact_path(model_dir)
    if target.exists():
        return target
    source = model_dir / HITTER_CHALLENGER_ARTIFACT
    if not source.exists():
        source = model_dir / "hitter_player_game_outcome_models.joblib"
    if not source.exists():
        return None
    artifact = joblib.load(source)
    accepted_heads = [
        prefix for prefix in ("hits", "hr")
        if _accepted_count_head(artifact, prefix)
    ]
    if not accepted_heads:
        return None
    release_id = hitter_model_release_id(artifact)
    frozen_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    artifact["model_release_id"] = release_id
    artifact["artifact_role"] = "live_forecast_frozen"
    artifact["live_forecast_frozen_at_utc"] = frozen_at
    artifact["live_forecast_accepted_heads"] = accepted_heads
    artifact["live_forecast_source_artifact"] = source.name
    atomic_write_via(
        target,
        lambda temp: joblib.dump(artifact, temp),
        attempts=6,
        retry_all_errors=True,
    )
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_write_json(model_dir / "hitter_player_game_outcome_models.live_forecast.json", {
        "model_release_id": release_id,
        "artifact_role": "live_forecast_frozen",
        "live_forecast_frozen_at_utc": frozen_at,
        "source_artifact": source.name,
        "accepted_heads": accepted_heads,
        "sha256": digest,
        "usage": "live_forecast_scoring_only",
        "betting_promotion": "separate_exact_bucket_micro_gates_required",
    })
    return target


def pitcher_model_release_id() -> str:
    return (
        os.getenv("MLB_PITCHER_MODEL_RELEASE_ID", "").strip()
        or DEFAULT_PITCHER_MODEL_RELEASE_ID
    )


def game_model_release_id() -> str:
    return (
        os.getenv("MLB_GAME_MODEL_RELEASE_ID", "").strip()
        or DEFAULT_GAME_MODEL_RELEASE_ID
    )
