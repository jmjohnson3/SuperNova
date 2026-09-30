"""Atomic, retrying filesystem writes for scheduled pipeline artifacts."""
from __future__ import annotations

import errno
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable


_RETRYABLE_ERRNOS = {errno.EACCES, errno.EBUSY, errno.EINVAL, errno.EPERM}
_RETRYABLE_WINERRORS = {5, 32, 33, 87}


def _retryable_os_error(exc: Exception) -> bool:
    return isinstance(exc, OSError) and (
        exc.errno in _RETRYABLE_ERRNOS
        or getattr(exc, "winerror", None) in _RETRYABLE_WINERRORS
    )


def _temporary_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def atomic_write_via(
    path: str | Path,
    writer: Callable[[Path], None],
    *,
    attempts: int = 8,
    initial_delay_s: float = 0.05,
    retry_all_errors: bool = False,
) -> Path:
    """Write to a unique sibling and atomically replace the destination."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        temporary = _temporary_path(target)
        try:
            writer(temporary)
            if temporary.exists():
                with temporary.open("r+b") as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(temporary, target)
            return target
        except Exception as exc:
            last_error = exc
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt + 1 >= max(1, attempts) or not (
                retry_all_errors or _retryable_os_error(exc)
            ):
                raise
            time.sleep(initial_delay_s * (2 ** attempt))
    assert last_error is not None
    raise last_error


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    attempts: int = 8,
) -> Path:
    def write(temporary: Path) -> None:
        with temporary.open("x", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    return atomic_write_via(path, write, attempts=attempts)


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    default: Callable[[Any], Any] | None = None,
    allow_nan: bool = True,
    attempts: int = 8,
) -> Path:
    text = json.dumps(
        payload,
        indent=indent,
        default=default,
        allow_nan=allow_nan,
    )
    return atomic_write_text(path, text, attempts=attempts)


def atomic_save_lightgbm_booster(
    booster: Any,
    path: str | Path,
    *,
    attempts: int = 6,
) -> Path:
    return atomic_write_via(
        path,
        lambda temporary: booster.save_model(str(temporary)),
        attempts=attempts,
        initial_delay_s=0.10,
        retry_all_errors=True,
    )
