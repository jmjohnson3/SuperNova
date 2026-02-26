# src/nba_pipeline/run_daily_and_notify.py
"""
Full SuperNovaBets daily pipeline with Discord notifications.

Order:
  1. Odds crawler       — post status
  2. MSF crawler        — post status (injuries, lineups, box scores)
  3. Parse + load       — post status
  4. Compute Elo        — post status (non-critical)
  5. Train game models  — post status
  6. Train prop models  — post status
  7. Game predictions   — post full output
  8. Alt line scan      — post full output
  9. Player prop projections — post full output

Critical steps (1, 2, 3, 5, 6) halt the pipeline on failure so we don't post
stale predictions. Non-critical steps (4, 7, 8, 9) log + continue.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger("nba_pipeline.run_daily_and_notify")

DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1461420766108319858/LenBk50YR1eS1isFMSOzE8gMWgSgBTSYmU4Ac1unf2SOo_kPSGk71afBqbBiQDuUZwD3",
)

DISCORD_LIMIT = 1950  # Discord hard cap is 2000; keep a buffer


# ---------------------------------------------------------------------------
# Pipeline step definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    label: str          # human-readable name shown in Discord
    module: str         # python -m <module>
    critical: bool      # halt pipeline on failure
    post_output: bool   # True = post stdout to Discord, False = status only
    timeout_s: int      # subprocess timeout in seconds


STEPS: list[Step] = [
    Step("Odds Crawler",              "nba_pipeline.crawler_oddsapi",                    critical=True,  post_output=False, timeout_s=900),
    Step("MSF Crawler",               "nba_pipeline.crawler",                            critical=True,  post_output=False, timeout_s=3600),
    Step("Parse + Load",              "nba_pipeline.parse_all",                          critical=True,  post_output=False, timeout_s=1800),
    Step("Elo Ratings",               "nba_pipeline.compute_elo",                        critical=False, post_output=False, timeout_s=300),
    Step("Train Game Models",         "nba_pipeline.modeling.train_game_models",          critical=True,  post_output=False, timeout_s=3600),
    Step("Train Player Prop Models",  "nba_pipeline.modeling.train_player_prop_models",   critical=True,  post_output=False, timeout_s=3600),
    Step("Game Predictions",          "nba_pipeline.modeling.predict_today",              critical=False, post_output=True,  timeout_s=120),
    Step("Alt Line Scan",             "nba_pipeline.modeling.scan_alt_lines_grid",         critical=False, post_output=True,  timeout_s=120),
    Step("Player Prop Projections",   "nba_pipeline.modeling.predict_player_props",        critical=False, post_output=True,  timeout_s=300),
]


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    # this file: <repo>/src/nba_pipeline/run_daily_and_notify.py
    return Path(__file__).resolve().parents[2]


def run_module(mod: str, timeout_s: int) -> tuple[int, str, str]:
    """Run `python -m <mod>` and return (returncode, stdout, stderr).
    Returns rc=124 on timeout (mirrors Unix timeout behaviour).
    """
    env = os.environ.copy()
    src_dir = str(_repo_root() / "src")
    env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"   # ensure subprocess stdout is UTF-8

    try:
        p = subprocess.run(
            [sys.executable, "-m", mod],
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
            cwd=str(_repo_root()),
            env=env,
            timeout=timeout_s,
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        log.error("[%s] timed out after %ds", mod, timeout_s)
        return 124, "", f"Timed out after {timeout_s}s"


# ---------------------------------------------------------------------------
# Discord helpers
# ---------------------------------------------------------------------------
async def _post(content: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        log.warning("DISCORD_WEBHOOK_URL not set; skipping Discord post.")
        return

    async with httpx.AsyncClient(timeout=20) as client:
        for attempt in range(4):
            try:
                r = await client.post(DISCORD_WEBHOOK_URL, json={"content": content})
                if r.status_code in (200, 204):
                    return
                if r.status_code == 429 and attempt < 3:
                    retry_after = float(r.json().get("retry_after", 1.5))
                    log.warning("Discord rate-limited — waiting %.1fs", retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                r.raise_for_status()
            except httpx.TimeoutException:
                if attempt >= 3:
                    raise
                await asyncio.sleep(2.0 * (attempt + 1))


def _build_chunks(header: str, body: str) -> list[str]:
    """Split header + body into ≤ DISCORD_LIMIT messages wrapped in code blocks."""
    body = body.strip()
    if not body:
        return [header]

    CODE_WRAP = 8  # ``` \n ... \n ```
    first_budget = DISCORD_LIMIT - len(header) - 1 - CODE_WRAP
    cont_budget  = DISCORD_LIMIT - CODE_WRAP

    def flush(lines: list[str], is_first: bool) -> str:
        block = "\n".join(lines)
        return f"{header}\n```\n{block}\n```" if is_first else f"```\n{block}\n```"

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    is_first = True
    budget = first_budget

    for line in body.splitlines():
        needed = len(line) + (1 if current else 0)
        if current and current_len + needed > budget:
            chunks.append(flush(current, is_first))
            current, current_len, is_first, budget = [], 0, False, cont_budget
        current.append(line)
        current_len += needed

    if current:
        chunks.append(flush(current, is_first))

    return chunks


async def _post_section(header: str, body: str) -> None:
    for chunk in _build_chunks(header, body):
        await _post(chunk)
        await asyncio.sleep(0.4)


async def _post_status(step: Step, secs: float, ok: bool, detail: str = "") -> None:
    icon = "✅" if ok else "❌"
    msg = f"{icon} **{step.label}** — {'done' if ok else 'FAILED'} in {secs:.0f}s"
    if detail:
        msg += f"\n```\n{detail[:800]}\n```"
    await _post(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    wall_start = time.time()
    await _post("🏀 **SuperNovaBets** — daily pipeline starting…")

    results: list[tuple[str, bool, float]] = []   # (label, ok, secs)
    halted = False

    for step in STEPS:
        if halted:
            results.append((step.label, False, 0.0))
            continue

        log.info("▶ %s (%s)", step.label, step.module)
        t0 = time.time()
        rc, stdout, stderr = run_module(step.module, step.timeout_s)
        secs = time.time() - t0
        ok = rc == 0

        log.info("[%s] rc=%d %.0fs", step.module, rc, secs)
        if stderr.strip():
            log.warning("[%s stderr]\n%s", step.module, stderr[-2000:])

        results.append((step.label, ok, secs))

        if not ok:
            err_tail = "\n".join(stderr.strip().splitlines()[-25:])
            if step.post_output:
                # prediction step failed — post the error where the output would go
                await _post_section(f"❌ **{step.label} FAILED**", err_tail or "(no output)")
            else:
                await _post_status(step, secs, ok=False, detail=err_tail)

            if step.critical:
                await _post(f"🛑 **Pipeline halted** — `{step.label}` is required. Skipping remaining steps.")
                halted = True
            continue

        # Step succeeded
        if step.post_output:
            header = {
                "Game Predictions":        "🏀 **Game Predictions**",
                "Alt Line Scan":           "📊 **Alt Line Scan**",
                "Player Prop Projections": "🎯 **Player Prop Projections**",
            }.get(step.label, f"**{step.label}**")

            if stdout.strip():
                await _post_section(header, stdout.strip())
            else:
                await _post(f"{header}\n_(no output for today's slate)_")
        else:
            await _post_status(step, secs, ok=True)

    # Summary
    total = time.time() - wall_start
    lines = [f"{'✅' if ok else ('⏭️' if secs == 0.0 else '❌')} {label} ({secs:.0f}s)" for label, ok, secs in results]
    summary_body = "\n".join(lines)
    all_ok = all(ok for _, ok, _ in results)
    icon = "✅" if all_ok else ("🛑" if halted else "⚠️")
    await _post_section(f"{icon} **Pipeline complete** — {total:.0f}s total", summary_body)


if __name__ == "__main__":
    asyncio.run(main())
