# src/nba_pipeline/run_daily_and_notify.py
"""
Run predict_today + scan_alt_lines_grid (and optionally predict_player_props),
then post each output as its own formatted Discord message.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

log = logging.getLogger("nba_pipeline.run_daily_and_notify")

DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1461420766108319858/LenBk50YR1eS1isFMSOzE8gMWgSgBTSYmU4Ac1unf2SOo_kPSGk71afBqbBiQDuUZwD3",
)

DISCORD_LIMIT = 1950  # hard Discord cap is 2000; keep buffer


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    # this file: <repo>/src/nba_pipeline/run_daily_and_notify.py
    return Path(__file__).resolve().parents[2]


def run_module(mod: str) -> tuple[int, str, str]:
    """Run `python -m <mod>` and return (returncode, stdout, stderr)."""
    env = os.environ.copy()
    src_dir = str(_repo_root() / "src")
    env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    p = subprocess.run(
        [sys.executable, "-m", mod],
        text=True,
        capture_output=True,
        check=False,
        cwd=str(_repo_root()),
        env=env,
    )
    return p.returncode, p.stdout or "", p.stderr or ""


# ---------------------------------------------------------------------------
# Discord posting
# ---------------------------------------------------------------------------
async def _post(content: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        log.warning("DISCORD_WEBHOOK_URL not set; skipping.")
        return

    async with httpx.AsyncClient(timeout=20) as client:
        for attempt in range(4):
            try:
                r = await client.post(DISCORD_WEBHOOK_URL, json={"content": content})
                if r.status_code in (200, 204):
                    return
                if r.status_code == 429 and attempt < 3:
                    retry_after = float(r.json().get("retry_after", 1.5))
                    log.warning("Discord rate-limited. Waiting %.1fs…", retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                r.raise_for_status()
            except httpx.TimeoutException:
                if attempt >= 3:
                    raise
                await asyncio.sleep(2.0 * (attempt + 1))


def _build_chunks(header: str, body: str) -> list[str]:
    """
    Split header + body into Discord messages ≤ DISCORD_LIMIT chars each.
    Each chunk is wrapped in a code block so output renders in monospace.
    """
    body = body.strip()
    if not body:
        return [header]

    CODE_WRAP = 8  # len("```\n" + "\n```")

    first_budget  = DISCORD_LIMIT - len(header) - 1 - CODE_WRAP
    cont_budget   = DISCORD_LIMIT - CODE_WRAP

    def flush(lines: list[str], is_first: bool) -> str:
        block = "\n".join(lines)
        if is_first:
            return f"{header}\n```\n{block}\n```"
        return f"```\n{block}\n```"

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    is_first = True
    budget = first_budget

    for line in body.splitlines():
        needed = len(line) + (1 if current else 0)  # +1 for newline separator
        if current and current_len + needed > budget:
            chunks.append(flush(current, is_first))
            current = []
            current_len = 0
            is_first = False
            budget = cont_budget

        current.append(line)
        current_len += needed

    if current:
        chunks.append(flush(current, is_first))

    return chunks


async def post_section(header: str, body: str) -> None:
    """Post a titled section, chunked to fit Discord's limit."""
    for chunk in _build_chunks(header, body):
        await _post(chunk)
        await asyncio.sleep(0.4)  # be polite between chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    started = time.time()
    await _post("🏀 **SuperNovaBets** — running today's slate…")

    # Steps: (emoji + title, module, post_on_success)
    steps = [
        ("🏀 **Game Predictions**",    "nba_pipeline.modeling.predict_today",        True),
        ("📊 **Alt Line Scan**",        "nba_pipeline.modeling.scan_alt_lines_grid",   True),
        ("🎯 **Player Prop Projections**", "nba_pipeline.modeling.predict_player_props", True),
    ]

    all_ok = True

    for title, mod, post_output in steps:
        log.info("Running %s…", mod)
        rc, stdout, stderr = run_module(mod)

        if rc != 0:
            all_ok = False
            err_tail = "\n".join(stderr.strip().splitlines()[-30:])
            await post_section(
                f"❌ **{title} FAILED** (`{mod}`)",
                err_tail or "(no stderr)",
            )
            log.error("[%s] failed rc=%s\n%s", mod, rc, stderr)
            # non-fatal: continue to next step so we still post whatever succeeded
            continue

        log.info("[%s] OK", mod)

        if post_output and stdout.strip():
            await post_section(title, stdout.strip())

    elapsed = time.time() - started
    status = "✅ All steps OK" if all_ok else "⚠️ Some steps failed (see above)"
    await _post(f"{status} — finished in {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
