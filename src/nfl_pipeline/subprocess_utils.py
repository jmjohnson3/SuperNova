"""Subprocess helpers for NFL scheduler-safe steps."""
from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence


def _kill_process_tree(proc: subprocess.Popen, *, timeout_s: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
                check=False,
            )
            return
        except Exception:
            pass
    try:
        proc.kill()
    except ProcessLookupError:
        return


def run_subprocess(
    cmd: Sequence[str],
    *,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: int | None = None,
) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        list(cmd),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return proc.returncode, stdout or "", stderr or ""
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc, timeout_s=5)
            stdout, stderr = "", ""
        msg = f"Timed out after {timeout_s}s; killed process tree rooted at PID {proc.pid}"
        return 124, stdout or "", "\n".join(part for part in ((stderr or "").strip(), msg) if part)
