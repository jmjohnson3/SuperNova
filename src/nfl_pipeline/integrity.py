"""Shared forecast identity, time and artifact contracts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

FEATURE_CONTRACT = "nfl-asof-v2"
MODEL_ROOT = Path(__file__).resolve().parent / "modeling" / "models"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(payload, name)
        # Verify readability before publishing. Never truncate the previous release.
        joblib.load(name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def active_release() -> dict[str, Any] | None:
    pinned = os.getenv('NFL_MODEL_RELEASE_ID')
    if pinned and (Path(pinned).name != pinned or '..' in pinned):
        raise RuntimeError('Invalid pinned NFL release ID')
    path = MODEL_ROOT / 'releases' / pinned / 'manifest.json' if pinned else MODEL_ROOT / "active_release.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("feature_contract") != FEATURE_CONTRACT:
        raise RuntimeError("NFL model release has an incompatible feature contract")
    return result


def release_artifact(kind: str) -> dict[str, Any] | None:
    manifest = active_release()
    if manifest is None:
        return None
    path = MODEL_ROOT / "releases" / manifest["release_id"] / (kind + ".joblib")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][kind]:
        raise RuntimeError(f"NFL {kind} release checksum mismatch")
    return joblib.load(path)


def production_freeze() -> dict[str, Any] | None:
    path = MODEL_ROOT / 'production_freeze.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def freeze_production(reason: str) -> dict[str, Any]:
    manifest = active_release()
    if not manifest:
        raise RuntimeError('Cannot freeze an absent NFL release')
    current = production_freeze()
    if current:
        if current['release_id'] != manifest['release_id'] or current['sha256'] != manifest['sha256']:
            raise RuntimeError('Production has changed since its recorded freeze')
        return current
    record = {**manifest, 'frozen_at': datetime.now(timezone.utc).isoformat(), 'reason': reason}
    atomic_json(MODEL_ROOT / 'production_freeze.json', record)
    return record


def assert_publication_allowed() -> None:
    if production_freeze():
        raise RuntimeError('NFL production is frozen; train a challenger without --publish')


def nfl_season(day: Any = None) -> int:
    """NFL season a date belongs to: Jan/Feb playoff dates belong to the previous year's season."""
    if day is None:
        from zoneinfo import ZoneInfo
        day = datetime.now(ZoneInfo("America/New_York")).date()
    return day.year if day.month >= 3 else day.year - 1


def utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def decision_key(row: dict[str, Any], *, at: datetime) -> str:
    fields = ("game_date_et", "game_id", "player_id", "stat", "market", "side", "line", "book",
              "price", "projection", "probability", "model_version", "offer_id")
    payload = {key: row.get(key) for key in fields}
    payload["decision_at"] = utc(at).isoformat()
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
