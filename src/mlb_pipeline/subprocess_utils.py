"""Subprocess helpers for scheduler-safe pipeline steps."""
from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping, Sequence


def kill_process_tree(proc: subprocess.Popen, *, timeout_s: float = 30.0) -> None:
    """Terminate a process and its children.

    On Windows, ``subprocess`` timeouts can kill only the launcher process
    while leaving the real Python child running.  The scheduler runs through a
    venv launcher on this machine, so use ``taskkill /T`` there.
    """
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


def run_subprocess_tree(
    cmd: Sequence[str],
    *,
    timeout_s: int | None,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    encoding: str | None = None,
) -> tuple[int, str, str, float]:
    """Run a command and kill the full process tree on timeout."""
    t0 = time.perf_counter()
    kwargs: dict[str, object] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "cwd": cwd,
        "env": dict(env) if env is not None else None,
    }
    if encoding:
        kwargs["encoding"] = encoding
    proc = subprocess.Popen(list(cmd), **kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return proc.returncode, stdout or "", stderr or "", time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            kill_process_tree(proc, timeout_s=5)
            stdout, stderr = "", ""
        secs = time.perf_counter() - t0
        msg = f"Timed out after {timeout_s}s; killed process tree rooted at PID {proc.pid}"
        stderr_text = "\n".join(part for part in ((stderr or "").strip(), msg) if part)
        return 124, stdout or "", stderr_text, secs
