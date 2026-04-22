"""
mlb_pipeline.run_daily_and_notify
==================================
Full MLB daily pipeline with Discord notifications.

Steps:
  1. MLB Stats API crawler        — schedule + boxscores
  2. MSF crawler                  — injuries/lineups (non-critical; 403 on game data)
  3. Odds API crawler             — run lines, totals, prop lines
  4. Statcast crawler             — Baseball Savant exit velo/barrel/expected stats (non-critical)
  5. Statcast extended crawler    — spray angle, sprint speed, pitcher arsenal (non-critical)
  6. Parse + load                 — parse_all (parsers + SQL views + mat views)
  7. Elo ratings                  — MOV-adjusted team Elo (non-critical)
  8. Train game models            — XGBoost + LightGBM
  9. Train player prop models     — XGBoost + LightGBM (non-critical)
  10. Game predictions            — post full output to Discord
  11. Player prop projections     — post full output to Discord

Set env var:
  MLB_DISCORD_WEBHOOK_URL   — Discord webhook URL for the #mlb channel
  DISCORD_FORMAT=1          — set automatically by this script for prediction steps
  MLB_POST_PREDICTION_CARDS — set to 1/true/yes to post PNG cards to Discord
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

log = logging.getLogger("mlb_pipeline.run_daily_and_notify")

MLB_DISCORD_WEBHOOK_URL = os.getenv(
    "MLB_DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1487880251886403596/fB9WT_Krl2QdOV8MD6o0Pzdp-BgnsJ8wISAJ6-Xi0wMVQfViVjbKU2wV4VC9f52Iwo9n",
)
MLB_POST_PREDICTION_CARDS = os.getenv("MLB_POST_PREDICTION_CARDS", "0").strip().lower() in {
    "1", "true", "yes", "on",
}

DISCORD_LIMIT = 1950


# ---------------------------------------------------------------------------
# Pipeline step definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    label: str
    module: str
    args: tuple[str, ...] = ()
    critical: bool = True
    post_output: bool = False
    timeout_s: int = 14400


STEPS: list[Step] = [
    Step("MLB Stats API Crawler",  "mlb_pipeline.crawler_statsapi",
         args=("--season", "2026-regular"),   critical=True,  post_output=False),
    Step("MSF Crawler",            "mlb_pipeline.crawler",          critical=False, post_output=False),
    Step("Odds Crawler",           "mlb_pipeline.crawler_oddsapi",  critical=True,  post_output=False),
    Step("Statcast Crawler",          "mlb_pipeline.crawler_statcast",          critical=False, post_output=False),
    Step("Statcast Extended Crawler", "mlb_pipeline.crawler_statcast_extended", critical=False, post_output=False),
    Step("Parse + Load",              "mlb_pipeline.parse_all",                 critical=True,  post_output=False),
    Step("Elo Ratings",            "mlb_pipeline.compute_elo",      critical=False, post_output=False),
    Step("Train Game Models",      "mlb_pipeline.modeling.train_game_models",
         critical=True, post_output=False),
    Step("Train Player Prop Models", "mlb_pipeline.modeling.train_player_prop_models",
         critical=False, post_output=False),
    Step("Train Binary Prop Classifiers", "mlb_pipeline.modeling.train_binary_prop_models",
         critical=False, post_output=False),
    Step("Game Predictions",       "mlb_pipeline.modeling.predict_today",
         critical=False, post_output=True),
    Step("Player Prop Projections", "mlb_pipeline.modeling.predict_player_props",
         critical=False, post_output=True),
]


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_module(mod: str, args: tuple[str, ...], timeout_s: int) -> tuple[int, str, str]:
    env = os.environ.copy()
    src_dir = str(_repo_root() / "src")
    env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["DISCORD_FORMAT"] = "1"

    try:
        p = subprocess.run(
            [sys.executable, "-m", mod, *args],
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
    if not MLB_DISCORD_WEBHOOK_URL:
        log.warning("MLB_DISCORD_WEBHOOK_URL not set; printing to stdout instead.")
        print(content)
        return

    async with httpx.AsyncClient(timeout=20) as client:
        for attempt in range(4):
            try:
                r = await client.post(MLB_DISCORD_WEBHOOK_URL, json={"content": content})
                if r.status_code in (200, 204):
                    return
                if r.status_code == 429 and attempt < 3:
                    retry_after = float(r.json().get("retry_after", 1.5))
                    log.warning("Discord rate-limited — waiting %.1fs", retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                # Transient Discord-side failures (e.g., 5xx) should not halt the pipeline.
                if 500 <= r.status_code < 600 and attempt < 3:
                    wait_s = 1.5 * (attempt + 1)
                    log.warning("Discord server error %d — retrying in %.1fs", r.status_code, wait_s)
                    await asyncio.sleep(wait_s)
                    continue
                if 400 <= r.status_code < 500:
                    log.error("Discord post failed (%d): %s", r.status_code, (r.text or "")[:400])
                    return
                if attempt < 3:
                    wait_s = 1.5 * (attempt + 1)
                    log.warning("Discord post failed (%d) — retrying in %.1fs", r.status_code, wait_s)
                    await asyncio.sleep(wait_s)
                    continue
                log.error("Discord post failed after retries (%d): %s", r.status_code, (r.text or "")[:400])
                return
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                if attempt >= 3:
                    log.error("Discord post request failed after retries: %s", exc)
                    return
                wait_s = 2.0 * (attempt + 1)
                log.warning("Discord post request error (%s) — retrying in %.1fs", exc.__class__.__name__, wait_s)
                await asyncio.sleep(wait_s)


def _build_rich_chunks(header: str, body: str) -> list[str]:
    body = body.strip()
    if not body:
        return [header]

    first_budget = DISCORD_LIMIT - len(header) - 1
    cont_budget  = DISCORD_LIMIT

    def flush(lines: list[str], is_first: bool) -> str:
        block = "\n".join(lines)
        return f"{header}\n{block}" if is_first else block

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
    for chunk in _build_rich_chunks(header, body):
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
    import argparse
    parser = argparse.ArgumentParser(description="MLB daily pipeline with Discord notifications")
    parser.add_argument("--skip-crawl",   action="store_true")
    parser.add_argument("--skip-parse",   action="store_true")
    parser.add_argument("--skip-train",   action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--date", default=None, help="Game date YYYY-MM-DD (ET)")
    args = parser.parse_args()

    from datetime import date as _date, datetime as _datetime
    from zoneinfo import ZoneInfo as _ZI
    et_day = _date.fromisoformat(args.date) if args.date else _datetime.now(_ZI("America/New_York")).date()

    _CRAWL_MODULES  = {"mlb_pipeline.crawler_statsapi", "mlb_pipeline.crawler",
                       "mlb_pipeline.crawler_oddsapi", "mlb_pipeline.crawler_statcast"}
    _PARSE_MODULES  = {"mlb_pipeline.parse_all", "mlb_pipeline.compute_elo"}
    _TRAIN_MODULES  = {"mlb_pipeline.modeling.train_game_models",
                       "mlb_pipeline.modeling.train_player_prop_models",
                       "mlb_pipeline.modeling.train_binary_prop_models"}
    _PREDICT_MODULES = {"mlb_pipeline.modeling.predict_today",
                        "mlb_pipeline.modeling.predict_player_props"}

    def _should_skip(step: Step) -> bool:
        if args.skip_crawl   and step.module in _CRAWL_MODULES:   return True
        if args.skip_parse   and step.module in _PARSE_MODULES:   return True
        if args.skip_train   and step.module in _TRAIN_MODULES:   return True
        if args.skip_predict and step.module in _PREDICT_MODULES: return True
        return False

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    wall_start = time.time()
    await _post("⚾ **SuperNovaBets MLB** — daily pipeline starting…")

    results: list[tuple[str, bool, float]] = []
    halted = False

    for step in STEPS:
        if _should_skip(step):
            log.info("⏭ Skipping %s", step.label)
            continue
        if halted:
            results.append((step.label, False, 0.0))
            continue

        log.info("▶ %s (%s)", step.label, step.module)
        t0 = time.time()
        rc, stdout, stderr = run_module(step.module, step.args, step.timeout_s)
        secs = time.time() - t0
        ok = rc == 0

        log.info("[%s] rc=%d %.0fs", step.module, rc, secs)
        if stderr.strip():
            log.warning("[%s stderr]\n%s", step.module, stderr[-2000:])

        results.append((step.label, ok, secs))

        if not ok:
            err_tail = "\n".join(stderr.strip().splitlines()[-25:])
            if step.post_output:
                await _post_section(f"❌ **{step.label} FAILED**", err_tail or "(no output)")
            else:
                await _post_status(step, secs, ok=False, detail=err_tail)
            if step.critical:
                await _post(f"🛑 **MLB pipeline halted** — `{step.label}` is required.")
                halted = True
            continue

        if step.post_output:
            header = {
                "Game Predictions":       "⚾ **MLB Game Predictions**",
                "Player Prop Projections": "⚾ **MLB Player Props**",
            }.get(step.label, f"**{step.label}**")
            if stdout.strip():
                await _post_section(header, stdout.strip())
            else:
                await _post(f"{header}\n_(no games for today's slate)_")
        else:
            await _post_status(step, secs, ok=True)

    # ── Optional prediction cards (PNG images) ─────────────────────────────
    predict_ok = not halted and not args.skip_predict
    if predict_ok and MLB_POST_PREDICTION_CARDS:
        try:
            from mlb_pipeline.modeling.generate_cards import generate_and_post as _gen_cards
            await _gen_cards(MLB_DISCORD_WEBHOOK_URL, et_day)
            log.info("Prediction cards posted to Discord.")
        except Exception as _card_exc:
            log.warning("Could not generate/post prediction cards: %s", _card_exc)
    elif predict_ok:
        log.info("Skipping Discord prediction cards (MLB_POST_PREDICTION_CARDS is disabled).")

    total = time.time() - wall_start
    lines = [
        f"{'✅' if ok else ('⏭️' if secs == 0.0 else '❌')} {label} ({secs:.0f}s)"
        for label, ok, secs in results
    ]
    all_ok = all(ok for _, ok, _ in results)
    icon = "✅" if all_ok else ("🛑" if halted else "⚠️")
    await _post_section(f"{icon} **MLB pipeline complete** — {total:.0f}s total", "\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
