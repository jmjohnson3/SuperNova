from __future__ import annotations

import errno
from pathlib import Path

import pytest

from mlb_pipeline import atomic_io


def test_atomic_write_text_replaces_target_without_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("old", encoding="utf-8")

    atomic_io.atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_retries_windows_invalid_argument(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("old", encoding="utf-8")
    real_replace = atomic_io.os.replace
    calls = 0

    def flaky_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.EINVAL, "transient Windows handle error")
        return real_replace(source, destination)

    monkeypatch.setattr(atomic_io.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_io.time, "sleep", lambda _seconds: None)

    atomic_io.atomic_write_text(target, "recovered", attempts=2)

    assert calls == 2
    assert target.read_text(encoding="utf-8") == "recovered"


def test_atomic_writer_failure_preserves_previous_artifact(tmp_path: Path) -> None:
    target = tmp_path / "model.txt"
    target.write_text("known-good", encoding="utf-8")

    def fail(temporary: Path) -> None:
        temporary.write_text("partial", encoding="utf-8")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        atomic_io.atomic_write_via(target, fail)

    assert target.read_text(encoding="utf-8") == "known-good"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_lightgbm_save_uses_temporary_sibling(tmp_path: Path) -> None:
    target = tmp_path / "total_q90_lgb.txt"

    class Booster:
        saved_path: Path | None = None

        def save_model(self, path: str) -> None:
            self.saved_path = Path(path)
            self.saved_path.write_text("booster", encoding="utf-8")

    booster = Booster()
    atomic_io.atomic_save_lightgbm_booster(booster, target)

    assert booster.saved_path is not None
    assert booster.saved_path != target
    assert booster.saved_path.parent == target.parent
    assert target.read_text(encoding="utf-8") == "booster"
