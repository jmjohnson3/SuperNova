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
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from mlb_pipeline.subprocess_utils import run_subprocess_tree

log = logging.getLogger("mlb_pipeline.run_daily_and_notify")

def _load_discord_webhook_url() -> str:
    """Load the webhook even from stale Windows shells.

    Windows user/machine env changes are not visible to already-open terminals.
    The scheduled batch files hydrate the variable, but direct
    ``python -m mlb_pipeline.run_daily_and_notify`` runs should also work.
    """
    value = os.getenv("MLB_DISCORD_WEBHOOK_URL")
    if value:
        return value
    if os.name != "nt":
        return ""
    try:
        import winreg

        for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(root, "Environment") as key:
                    saved, _ = winreg.QueryValueEx(key, "MLB_DISCORD_WEBHOOK_URL")
                    if saved:
                        return str(saved)
            except OSError:
                continue
    except Exception:
        return ""
    return ""


MLB_DISCORD_WEBHOOK_URL = _load_discord_webhook_url()
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
    fail_task_on_error: bool = False


STEPS: list[Step] = [
    Step("MLB Stats API Crawler",  "mlb_pipeline.crawler_statsapi",
         args=("--season", "2026-regular"),   critical=True,  post_output=False),
    Step("MSF Crawler",            "mlb_pipeline.crawler",          critical=False, post_output=False),
    Step("Odds Crawler",           "mlb_pipeline.crawler_oddsapi",
         args=("--force-props",), critical=True, post_output=False),
    Step("Statcast Crawler",          "mlb_pipeline.crawler_statcast",          critical=False, post_output=False),
    Step("Statcast Extended Crawler", "mlb_pipeline.crawler_statcast_extended", critical=False, post_output=False),
    Step("Parse + Load",              "mlb_pipeline.parse_all",                 critical=True,  post_output=False),
    Step("Elo Ratings",            "mlb_pipeline.compute_elo",      critical=False, post_output=False),
    Step("Build Prop Offer Links", "mlb_pipeline.modeling.build_prop_offer_links_table",
         critical=False, post_output=False),
    Step("Grade Outcomes + Ledgers", "mlb_pipeline.modeling.update_outcomes",
         critical=False, post_output=False, timeout_s=300),
    Step("Grade Prop Shadow Replay", "mlb_pipeline.modeling.grade_prop_prediction_replay",
         critical=False, post_output=False, timeout_s=300),
    Step("Game Predictions",       "mlb_pipeline.modeling.predict_today",
         critical=False, post_output=True),
    Step("Player Prop Projections", "mlb_pipeline.modeling.predict_player_props",
         critical=False, post_output=True),
    Step("Shadow-Lock Prop Predictions", "mlb_pipeline.modeling.shadow_lock_prop_predictions",
         args=("--phase", "morning"), critical=True, post_output=False, timeout_s=600,
         fail_task_on_error=True),
    Step("Prop Snapshot Coverage", "mlb_pipeline.modeling.prop_snapshot_coverage_report",
         critical=False, post_output=False, timeout_s=120),
    Step("Offer-Level Prop Audit", "mlb_pipeline.modeling.offer_level_prop_audit_report",
         args=("--lookback-days", "90", "--min-bucket-rows", "5", "--top-n", "25"),
         critical=False, post_output=False, timeout_s=120),
    Step("Prop Real-Money Readiness", "mlb_pipeline.modeling.prop_real_money_readiness_report",
         args=("--lookback-days", "90", "--top-n", "25"),
         critical=False, post_output=False, timeout_s=120),
    Step("Prop Real-Money Kill Switch", "mlb_pipeline.modeling.prop_real_money_kill_switch",
         critical=False, post_output=False, timeout_s=60, fail_task_on_error=True),
    Step("Prop Bucket Promotion", "mlb_pipeline.modeling.prop_bucket_promotion_report",
         args=("--lookback-days", "365", "--top-n", "25"),
         critical=False, post_output=False, timeout_s=120),
    Step("Prop Miss Diagnostic", "mlb_pipeline.modeling.prop_miss_diagnostic_report",
         critical=False, post_output=False, timeout_s=180),
    Step("Prop Bucket Repair", "mlb_pipeline.modeling.prop_bucket_repair_report",
         critical=False, post_output=False, timeout_s=180),
    Step("Prop Target Quality", "mlb_pipeline.modeling.prop_target_quality_report",
         critical=False, post_output=False, timeout_s=180),
    Step("Prop Slate Post-Mortem", "mlb_pipeline.modeling.prop_slate_postmortem_report",
         args=("--top-n", "25"),
         critical=False, post_output=False, timeout_s=120),
    # Train after predictions so a slow model refresh cannot delay today's slate.
    # The newly written artifacts are used by the next prediction run.
    Step("Train Game Models",      "mlb_pipeline.modeling.train_game_models",
         critical=True, post_output=False),
    Step("Train Player Prop Models", "mlb_pipeline.modeling.train_player_prop_models",
         args=("--skip-optuna",), critical=False, post_output=False),
    Step("Train Binary Prop Classifiers", "mlb_pipeline.modeling.train_binary_prop_models",
         critical=False, post_output=False),
    Step("Build Prop Market History Table", "mlb_pipeline.modeling.build_prop_market_history_table",
         critical=False, post_output=False, timeout_s=600),
    Step("Train Prop Market Side Priors", "mlb_pipeline.modeling.train_prop_market_side_priors",
         critical=False, post_output=False, timeout_s=600),
    Step("Build Prop Market Training Table", "mlb_pipeline.modeling.build_prop_market_training_table",
         args=("--include-pending", "--ensure-schema"), critical=False, post_output=False,
         timeout_s=1800, fail_task_on_error=True),
    Step("Build Hitter Player-Game Training Table", "mlb_pipeline.modeling.build_hitter_player_game_training_table",
         critical=False, post_output=False, timeout_s=600),
    Step("Hitter Event Feature Ablation", "mlb_pipeline.modeling.hitter_event_feature_ablation_report",
         critical=False, post_output=False, timeout_s=900),
    Step("Train Hitter Player-Game Outcome Models", "mlb_pipeline.modeling.train_hitter_player_game_outcome_models",
         critical=False, post_output=False, timeout_s=900),
    Step("Train Prop Side Recalibrators", "mlb_pipeline.modeling.train_prop_side_recalibrators",
         critical=False, post_output=False),
    Step("Train Prop Betting Layer", "mlb_pipeline.modeling.train_prop_betting_layer",
         critical=False, post_output=False),
    Step("Train Prop Direct Side Models", "mlb_pipeline.modeling.train_prop_direct_side_models",
         critical=False, post_output=False),
    Step("Train Prop Opportunity Models", "mlb_pipeline.modeling.train_prop_opportunity_models",
         critical=False, post_output=False, timeout_s=600),
    Step("Train Prop Bookability Model", "mlb_pipeline.modeling.train_prop_bookability_model",
         critical=False, post_output=False, timeout_s=600),
    Step("Train Prop Market-Residual Models", "mlb_pipeline.modeling.train_prop_market_residual_models",
         critical=False, post_output=False, timeout_s=600),
    Step("Train Prop Distribution Models", "mlb_pipeline.modeling.train_prop_distribution_models",
         critical=False, post_output=False, timeout_s=600),
    Step("Compare Prop Probability Variants", "mlb_pipeline.modeling.compare_prop_probability_variants",
         critical=False, post_output=False),
    Step("Prop Opportunity Feature Report", "mlb_pipeline.modeling.prop_opportunity_feature_report",
         args=("--lookback-days", "30"),
         critical=False, post_output=False, timeout_s=600),
    Step("Train Prop Bucket Reopen Policy", "mlb_pipeline.modeling.train_prop_bucket_reopen_policy",
         critical=False, post_output=False),
    Step("Prop Walk-Forward Accuracy", "mlb_pipeline.modeling.prop_walk_forward_accuracy_report",
         critical=False, post_output=False, timeout_s=600),
    Step("Prop Shadow Selector", "mlb_pipeline.modeling.prop_shadow_selector",
         critical=False, post_output=False, timeout_s=300, fail_task_on_error=True),
    Step("Optimize Prop Thresholds", "mlb_pipeline.modeling.optimize_prop_thresholds",
         critical=False, post_output=False),
    Step("Monitor Prop Calibration", "mlb_pipeline.modeling.monitor_prop_calibration",
         critical=False, post_output=False),
]

# Steps run before day and evening slates to refresh injuries + odds before first pitch.
PRE_GAME_STEPS: list[Step] = [
    Step("Re-crawl Injuries (force)", "mlb_pipeline.crawler",
         args=("--force-meta",), critical=False, post_output=False, timeout_s=300),
    Step("Re-crawl Game Odds", "mlb_pipeline.crawler_oddsapi",
         args=("--skip-props",),
         critical=False, post_output=False, timeout_s=600),
    Step("Re-crawl Prop Odds", "mlb_pipeline.crawler_oddsapi",
         args=("--skip-live", "--force-props"),
         critical=True, post_output=False, timeout_s=600),
    Step("Re-parse Meta (injuries)",  "mlb_pipeline.parse_meta",
         critical=False, post_output=False, timeout_s=300),
    Step("Re-parse Game Odds + Morning-Lock Prop Close Snapshot", "mlb_pipeline.parse_oddsapi",
         args=("--prop-snapshot-role", "close"),
         critical=True,  post_output=False, timeout_s=300),
    Step("Rebuild Prop Offer Links", "mlb_pipeline.modeling.build_prop_offer_links_table",
         critical=True, post_output=False, timeout_s=300),
    Step("Player Props (pre-game)",   "mlb_pipeline.modeling.predict_player_props",
         critical=True,  post_output=True,  timeout_s=900),
    Step("Shadow-Lock Props (pre-game)", "mlb_pipeline.modeling.shadow_lock_prop_predictions",
         critical=True, post_output=False, timeout_s=120, fail_task_on_error=True),
    Step("Re-crawl Post-Lock Closing Prop Odds", "mlb_pipeline.crawler_oddsapi",
         args=("--skip-live", "--force-props"),
         critical=True, post_output=False, timeout_s=600, fail_task_on_error=True),
    Step("Capture Post-Lock Prop Close Snapshot", "mlb_pipeline.parse_oddsapi",
         args=("--prop-snapshot-role", "close"),
         critical=True, post_output=False, timeout_s=300, fail_task_on_error=True),
    Step("Refresh Prop Replay CLV", "mlb_pipeline.modeling.refresh_prop_replay_clv",
         critical=True, post_output=False, timeout_s=300, fail_task_on_error=True),
    Step("Build Prop Market Training Table", "mlb_pipeline.modeling.build_prop_market_training_table",
         args=("--lookback-days", "3", "--include-pending", "--no-replace"),
         critical=False, post_output=False, timeout_s=600),
    Step("Prop Walk-Forward Accuracy", "mlb_pipeline.modeling.prop_walk_forward_accuracy_report",
         args=("--no-refresh-clv",),
         critical=False, post_output=False, timeout_s=600),
    Step("Prop Shadow Selector", "mlb_pipeline.modeling.prop_shadow_selector",
         critical=False, post_output=False, timeout_s=300, fail_task_on_error=True),
    Step("Prop Miss Diagnostic", "mlb_pipeline.modeling.prop_miss_diagnostic_report",
         critical=False, post_output=False, timeout_s=180),
    Step("Prop Bucket Repair", "mlb_pipeline.modeling.prop_bucket_repair_report",
         critical=False, post_output=False, timeout_s=180),
    Step("Prop Target Quality", "mlb_pipeline.modeling.prop_target_quality_report",
         critical=False, post_output=False, timeout_s=180),
    Step("Prop Opportunity Feature Report", "mlb_pipeline.modeling.prop_opportunity_feature_report",
         args=("--lookback-days", "30"),
         critical=False, post_output=False, timeout_s=600),
    Step("Prop Snapshot Coverage", "mlb_pipeline.modeling.prop_snapshot_coverage_report",
         critical=False, post_output=False, timeout_s=120),
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

    rc, stdout, stderr, _secs = run_subprocess_tree(
        [sys.executable, "-m", mod, *args],
        timeout_s=timeout_s,
        cwd=str(_repo_root()),
        env=env,
        encoding="utf-8",
    )
    if rc == 124:
        log.error("[%s] timed out after %ds", mod, timeout_s)
    return rc, stdout, stderr


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
                    log.info("Discord post succeeded (%d)", r.status_code)
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
    parser.add_argument(
        "--pre-game", action="store_true",
        help="Re-crawl injuries+odds, rebuild offers, re-predict props, and post to Discord.",
    )
    parser.add_argument(
        "--lock-phase", default=None,
        help="Stable shadow-lock phase for a pre-game run, such as day_pregame or evening_pregame.",
    )
    parser.add_argument(
        "--fun-reopen-props", action="store_true",
        help="Use the research-only all-prop reopen policy for player props.",
    )
    args = parser.parse_args()
    if args.fun_reopen_props:
        os.environ["MLB_PROP_FUN_REOPEN"] = "1"

    from datetime import date as _date, datetime as _datetime
    from zoneinfo import ZoneInfo as _ZI
    now_et = _datetime.now(_ZI("America/New_York"))
    et_day = _date.fromisoformat(args.date) if args.date else now_et.date()
    os.environ["MLB_ET_DATE"] = et_day.isoformat()

    _CRAWL_MODULES  = {"mlb_pipeline.crawler_statsapi", "mlb_pipeline.crawler",
                       "mlb_pipeline.crawler_oddsapi", "mlb_pipeline.crawler_statcast",
                       "mlb_pipeline.crawler_statcast_extended"}
    _PARSE_MODULES  = {"mlb_pipeline.parse_all", "mlb_pipeline.compute_elo"}
    _TRAIN_MODULES  = {"mlb_pipeline.modeling.train_game_models",
                       "mlb_pipeline.modeling.train_player_prop_models",
                       "mlb_pipeline.modeling.train_binary_prop_models",
                       "mlb_pipeline.modeling.optimize_prop_thresholds",
                       "mlb_pipeline.modeling.monitor_prop_calibration",
                       "mlb_pipeline.modeling.build_prop_market_history_table",
                       "mlb_pipeline.modeling.train_prop_market_side_priors",
                       "mlb_pipeline.modeling.build_prop_market_training_table",
                       "mlb_pipeline.modeling.build_hitter_player_game_training_table",
                       "mlb_pipeline.modeling.train_hitter_player_game_outcome_models",
                       "mlb_pipeline.modeling.hitter_event_feature_ablation_report",
                       "mlb_pipeline.modeling.train_prop_side_recalibrators",
                        "mlb_pipeline.modeling.train_prop_betting_layer",
                        "mlb_pipeline.modeling.train_prop_direct_side_models",
                        "mlb_pipeline.modeling.train_prop_opportunity_models",
                        "mlb_pipeline.modeling.train_prop_bookability_model",
                        "mlb_pipeline.modeling.train_prop_market_residual_models",
                        "mlb_pipeline.modeling.train_prop_distribution_models",
                        "mlb_pipeline.modeling.compare_prop_probability_variants",
                        "mlb_pipeline.modeling.train_prop_bucket_reopen_policy"}
    _PREDICT_MODULES = {"mlb_pipeline.modeling.predict_today",
                        "mlb_pipeline.modeling.predict_player_props",
                        "mlb_pipeline.modeling.shadow_lock_prop_predictions",
                        "mlb_pipeline.modeling.offer_level_prop_audit_report",
                        "mlb_pipeline.modeling.prop_real_money_readiness_report",
                        "mlb_pipeline.modeling.prop_real_money_kill_switch",
                        "mlb_pipeline.modeling.prop_bucket_promotion_report",
                        "mlb_pipeline.modeling.prop_snapshot_coverage_report",
                        "mlb_pipeline.modeling.prop_slate_postmortem_report",
                        "mlb_pipeline.modeling.prop_shadow_selector",
                        "mlb_pipeline.modeling.prop_walk_forward_accuracy_report",
                        "mlb_pipeline.modeling.prop_miss_diagnostic_report",
                        "mlb_pipeline.modeling.prop_bucket_repair_report",
                        "mlb_pipeline.modeling.prop_target_quality_report",
                        "mlb_pipeline.modeling.prop_opportunity_feature_report"}

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
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.pre_game:
        lock_phase = args.lock_phase or ("day_pregame" if now_et.hour < 14 else "evening_pregame")
        active_steps = [
            Step(
                step.label,
                step.module,
                args=("--phase", lock_phase),
                critical=step.critical,
                post_output=step.post_output,
                timeout_s=step.timeout_s,
                fail_task_on_error=step.fail_task_on_error,
            )
            if step.module == "mlb_pipeline.modeling.shadow_lock_prop_predictions"
            else step
            for step in PRE_GAME_STEPS
        ]
    else:
        active_steps = STEPS

    wall_start = time.time()
    start_msg = (
        "⚾ **SuperNovaBets MLB** — pre-game update starting…"
        if args.pre_game
        else "⚾ **SuperNovaBets MLB** — daily pipeline starting…"
    )
    await _post(start_msg)

    results: list[tuple[str, bool, float]] = []
    halted = False
    task_failed = False

    for step in active_steps:
        if not args.pre_game and _should_skip(step):
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
            if step.fail_task_on_error:
                task_failed = True
                log.error(
                    "%s is required for scheduler success; final exit will be non-zero.",
                    step.label,
                )
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
    if halted or task_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
