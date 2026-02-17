# src/nba_pipeline/modeling/run_daily_and_notify.py
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx

log = logging.getLogger("nba_pipeline.modeling.run_daily_and_notify")

# Prefer env var; fallback to the hardcoded value you already have.
DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1461420766108319858/LenBk50YR1eS1isFMSOzE8gMWgSgBTSYmU4Ac1unf2SOo_kPSGk71afBqbBiQDuUZwD3",
)

DISCORD_LIMIT = 1900  # Discord hard limit 2000; keep buffer.


def _repo_root_from_this_file() -> Path:
    """
    This file is: <repo>/src/nba_pipeline/modeling/run_daily_and_notify.py
    Repo root is: <repo>
    """
    return Path(__file__).resolve().parents[3]


def _src_dir_from_this_file() -> Path:
    """
    <repo>/src
    """
    return Path(__file__).resolve().parents[2]


async def discord_post(content: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        log.warning("DISCORD_WEBHOOK_URL not set; skipping Discord post.")
        return

    content = content[:DISCORD_LIMIT]
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(DISCORD_WEBHOOK_URL, json={"content": content})
        r.raise_for_status()


def _chunk_text(s: str, limit: int = DISCORD_LIMIT) -> list[str]:
    """
    Splits a long message into chunks under Discord limit.
    Attempts to split on line breaks for readability.
    """
    s = s.strip()
    if not s:
        return []

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for line in s.splitlines():
        # +1 for newline
        add_len = len(line) + 1
        if cur and cur_len + add_len > limit:
            chunks.append("\n".join(cur).rstrip())
            cur = [line]
            cur_len = len(line)
        else:
            cur.append(line)
            cur_len += add_len

    if cur:
        chunks.append("\n".join(cur).rstrip())

    return chunks


def tail(s: str, n_lines: int = 25) -> str:
    lines = s.splitlines()
    return "\n".join(lines[-n_lines:])


@dataclass(frozen=True)
class StepResult:
    label: str
    module: str
    rc: int
    stdout: str
    stderr: str


def run_module(mod: str) -> tuple[int, str, str]:
    """
    Runs: <current venv python> -m <mod>
    Ensures repo/src is on PYTHONPATH for the subprocess.
    """
    repo_root = _repo_root_from_this_file()
    src_dir = _src_dir_from_this_file()

    env = os.environ.copy()
    # Ensure `import nba_pipeline` works in subprocess:
    env["PYTHONPATH"] = str(src_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # IMPORTANT: use current interpreter (venv), not "python"
    p = subprocess.run(
        [sys.executable, "-m", mod],
        text=True,
        capture_output=True,
        check=False,
        cwd=str(repo_root),
        env=env,
    )
    return p.returncode, p.stdout or "", p.stderr or ""


def _format_section(title: str, body: str) -> str:
    body = body.strip()
    if not body:
        return ""
    return f"**{title}**\n```text\n{body}\n```"


async def _post_long(title_line: str, text_block: str) -> None:
    """
    Posts a title + chunked text body in multiple Discord messages.
    """
    chunks = _chunk_text(text_block, limit=DISCORD_LIMIT - len(title_line) - 5)
    if not chunks:
        await discord_post(title_line)
        return

    # First message includes the title; subsequent messages are continuation.
    await discord_post(f"{title_line}\n{chunks[0]}")
    for i, c in enumerate(chunks[1:], start=2):
        await discord_post(f"{title_line} _(cont. {i})_\n{c}")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    started = time.time()

    await discord_post("🏀 **SuperNovaBets** — predictions run starting…")

    steps: list[tuple[str, str]] = [
        ("Predict games", "nba_pipeline.modeling.predict_today"),
        ("Predict player props", "nba_pipeline.modeling.predict_player_props"),
        ("Scan alt lines grid", "nba_pipeline.modeling.scan_alt_lines_grid"),
    ]

    results: list[StepResult] = []

    try:
        for label, mod in steps:
            log.info("Running step: %s (%s)", label, mod)
            rc, out, err = run_module(mod)

            results.append(StepResult(label=label, module=mod, rc=rc, stdout=out, stderr=err))

            log.info("[%s] rc=%s", mod, rc)
            if out.strip():
                log.info("[%s stdout]\n%s", mod, out)
            if err.strip():
                log.warning("[%s stderr]\n%s", mod, err)

            if rc != 0:
                msg = (
                    f"❌ **Run failed** at: `{label}` (`{mod}`)\n"
                    f"{_format_section('stderr (tail)', tail(err))}"
                )
                await discord_post(msg)
                raise RuntimeError(f"Step failed: {label} ({mod}) rc={rc}")

        # If we got here, all steps succeeded.
        dur = time.time() - started

        # Build a clean “one report” message, chunked.
        report_lines: list[str] = []
        report_lines.append(f"✅ **SuperNovaBets** — predictions complete in {dur:.1f}s")
        report_lines.append("")
        report_lines.append(f"Interpreter: `{sys.executable}`")
        report_lines.append("")

        for r in results:
            # Keep stdout pretty but not insane; predict scripts already format nicely.
            stdout = (r.stdout or "").strip()
            if stdout:
                report_lines.append(f"## {r.label}")
                report_lines.append(stdout)
                report_lines.append("")

        report = "\n".join(report_lines).strip()

        # Post it chunked (plain text; your predict outputs already look good in Discord monospace)
        await _post_long("🏀 **SuperNovaBets — Daily Slate Output**", report)

    except Exception as e:
        log.exception("Daily run crashed")
        await discord_post(f"❌ **Run crashed**: `{type(e).__name__}` — {e}")


if __name__ == "__main__":
    asyncio.run(main())
