from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Optional

from mlb_pipeline.subprocess_utils import kill_process_tree, run_subprocess_tree
from mlb_pipeline.atomic_io import atomic_write_text
from mlb_pipeline.modeling.prop_close_capture_schedule import decision_payload, load_due_close_targets

_ET = ZoneInfo("America/New_York")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


PROP_WALK_FORWARD_TIMEOUT_S = _env_int("MLB_PROP_WALK_FORWARD_TIMEOUT_S", 1800)
PROP_WALK_FORWARD_FRESH_MINUTES = _env_int("MLB_PROP_WALK_FORWARD_FRESH_MINUTES", 360)


# ---------- Optional Rich UI ----------
def _get_console():
    try:
        from rich.console import Console  # type: ignore
        return Console()
    except Exception:
        return None


def _p(console, msg: str) -> None:
    if console:
        console.print(msg)
    else:
        print(msg)


def _rule(console, title: str) -> None:
    if console:
        console.rule(title)
    else:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)


def _panel(console, msg: str, ok: bool) -> None:
    if console:
        try:
            from rich.panel import Panel  # type: ignore
            border = "green" if ok else "red"
            console.print(Panel(msg, border_style=border))
            return
        except Exception:
            pass
    prefix = "OK" if ok else "FAIL"
    print(f"[{prefix}] {msg}")


def _render_table(console, rows: list[tuple[str, str, str, str]]) -> None:
    if console:
        try:
            from rich.table import Table  # type: ignore
            t = Table(title="SuperNovaBets MLB Daily Run", show_lines=True)
            t.add_column("Step", style="bold")
            t.add_column("Status")
            t.add_column("Time")
            t.add_column("RC")
            for step, status, secs, rc in rows:
                t.add_row(step, status, secs, rc)
            console.print(t)
            return
        except Exception:
            pass

    # fallback
    print("\nSuperNovaBets MLB Daily Run")
    for step, status, secs, rc in rows:
        print(f"- {step}: {status} ({secs}, rc={rc})")


# ---------- Runner ----------
@dataclass(frozen=True)
class Step:
    name: str
    module: str
    args: tuple[str, ...] = ()
    timeout_s: int | None = None
    critical: bool = True
    parallel: bool = False   # if True, run concurrently with adjacent parallel steps
    fail_task_on_error: bool = False


@dataclass
class StepResult:
    name: str
    ok: bool
    rc: int
    secs: float
    stdout: str
    stderr: str


def _tail(s: str, n_lines: int = 60) -> str:
    if not s:
        return ""
    lines = s.splitlines()
    if len(lines) <= n_lines:
        return s
    return "\n".join(lines[-n_lines:])


def _run_step(step: Step, extra_env: Optional[dict[str, str]] = None) -> StepResult:
    cmd = [sys.executable, "-m", step.module, *step.args]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    rc, stdout, stderr, secs = run_subprocess_tree(
        cmd,
        timeout_s=step.timeout_s,
        env=env,
    )

    return StepResult(
        name=step.name,
        ok=(rc == 0),
        rc=rc,
        secs=secs,
        stdout=stdout.strip(),
        stderr=stderr.strip(),
    )


def _run_parallel_steps(
    parallel_steps: list[Step],
    extra_env: Optional[dict[str, str]] = None,
) -> list[StepResult]:
    """Launch all steps simultaneously with Popen, wait for all to finish."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    # Launch all processes at once
    launched: list[tuple[Step, subprocess.Popen, float]] = []
    for step in parallel_steps:
        cmd = [sys.executable, "-m", step.module, *step.args]
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        launched.append((step, proc, time.perf_counter()))

    # Collect results — communicate() blocks per-process but they all run in parallel
    results: list[StepResult] = []
    for step, proc, t_start in launched:
        try:
            stdout, stderr = proc.communicate(timeout=step.timeout_s)
            secs = time.perf_counter() - t_start
            results.append(StepResult(
                name=step.name,
                ok=(proc.returncode == 0),
                rc=proc.returncode,
                secs=secs,
                stdout=(stdout or "").strip(),
                stderr=(stderr or "").strip(),
            ))
        except subprocess.TimeoutExpired:
            kill_process_tree(proc)
            stdout, stderr = proc.communicate()
            results.append(StepResult(
                name=step.name,
                ok=False,
                rc=124,
                secs=float(step.timeout_s or 0),
                stdout=(stdout or "").strip(),
                stderr=f"Timeout after {step.timeout_s}s",
            ))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full SuperNovaBets MLB pipeline in order.")
    parser.add_argument("--date", type=str, default=None, help="ET date (YYYY-MM-DD). Default: today (ET).")
    parser.add_argument("--skip-crawl", action="store_true", help="Skip crawler steps.")
    parser.add_argument("--skip-parse", action="store_true", help="Skip parse/load step.")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training steps.")
    parser.add_argument("--skip-predict", action="store_true", help="Skip prediction steps.")
    parser.add_argument(
        "--full-prop-optuna",
        action="store_true",
        help="Run the expensive player-prop Optuna search. Default scheduled training uses --skip-optuna.",
    )
    parser.add_argument(
        "--close-only", action="store_true",
        help=(
            "Closing-line run near first pitch. "
            "Re-crawls live game and prop odds, captures immutable prop close snapshots, "
            "then grades outcomes and CLV. Skips train/predict."
        ),
    )
    parser.add_argument(
        "--prop-close-capture-only", action="store_true",
        help=(
            "Lightweight game-aware prop close capture. Runs only when an upcoming event "
            "is near a required or supplemental close target and that target has not already been captured."
        ),
    )
    parser.add_argument(
        "--pre-game", action="store_true",
        help=(
            "Pre-game update run. Re-crawls injuries and latest game odds, rebuilds prop offers, "
            "re-predicts player props, "
            "and auto-posts to Discord (DISCORD_FORMAT=1)."
        ),
    )
    parser.add_argument(
        "--lock-phase", default=None,
        help="Stable shadow-lock phase for a pre-game run, such as day_pregame or evening_pregame.",
    )
    args = parser.parse_args()

    console = _get_console()

    et_day: date
    if args.date:
        et_day = date.fromisoformat(args.date)
    else:
        et_day = datetime.now(tz=_ET).date()

    now_et = datetime.now(tz=_ET)
    forecast_phase = (
        args.lock_phase
        or ("day_pregame" if args.pre_game and now_et.hour < 14 else None)
        or ("evening_pregame" if args.pre_game else None)
        or ("targeted_close" if args.prop_close_capture_only else None)
        or ("close" if args.close_only else "daily")
    )
    extra_env = {
        "MLB_ET_DATE": et_day.isoformat(),
        "MLB_FORECAST_PHASE": forecast_phase,
        "MLB_FORECAST_RUN_ID": (
            f"{et_day.isoformat()}:{forecast_phase}:"
            f"{datetime.now(tz=ZoneInfo('UTC')).strftime('%Y%m%dT%H%M%SZ')}"
        ),
    }

    steps: list[Step] = []

    if args.prop_close_capture_only:
        force_capture = os.getenv("MLB_FORCE_TARGETED_CLOSE_CAPTURE", "").strip().lower() in {"1", "true", "yes"}
        try:
            due_targets = load_due_close_targets(slate_date=et_day)
        except Exception as exc:
            # Missing a close is worse than one redundant API call. A database
            # decision failure therefore fails open for capture, while the
            # crawler/parser steps still fail the task if they cannot run.
            due_targets = []
            force_capture = True
            _p(console, f"Targeted close due-check failed; capturing defensively: {exc}")
        if due_targets:
            _p(console, f"Targeted prop close windows due: {decision_payload(due_targets)}")
        if force_capture or due_targets:
            steps.extend([
                Step(
                    name="Targeted prop close capture with quality retry",
                    module="mlb_pipeline.modeling.targeted_prop_close_capture",
                    timeout_s=2700,
                    critical=True,
                    fail_task_on_error=True,
                ),
                Step(
                    name="Refresh prop replay CLV after targeted close",
                    module="mlb_pipeline.modeling.refresh_prop_replay_clv",
                    timeout_s=300,
                    critical=True,
                    fail_task_on_error=True,
                ),
            ])
        else:
            _p(console, "No uncaptured required or supplemental prop close window is due.")
    elif args.pre_game:
        # ── Pre-game update run (~4:30 PM ET) ────────────────────────────────
        # Re-fetches injuries + latest odds, records a close for the morning
        # lock, re-predicts and locks the refreshed props, then captures a
        # second close observation that can be valid for the refreshed lock.
        extra_env["DISCORD_FORMAT"] = "1"
        lock_phase = args.lock_phase or (
            "day_pregame" if datetime.now(tz=_ET).hour < 14 else "evening_pregame"
        )
        steps = [
            Step(
                name="Grade daily forecast ledger",
                module="mlb_pipeline.modeling.daily_forecast_ledger",
                args=("--ensure-schema", "--grade"),
                timeout_s=300,
                critical=False,
            ),
            Step(
                name="Re-crawl injuries (force-meta)",
                module="mlb_pipeline.crawler",
                args=("--force-meta",),
                timeout_s=120,
                critical=False,
            ),
            Step(
                name="Re-crawl same-day lineups",
                module="mlb_pipeline.crawler",
                args=(
                    "--force-lineups",
                    "--start-date", et_day.isoformat(),
                    "--end-date", et_day.isoformat(),
                ),
                timeout_s=300,
                critical=False,
            ),
            Step(
                name="Re-crawl game odds",
                module="mlb_pipeline.crawler_oddsapi",
                args=("--skip-props",),
                timeout_s=600,
                critical=False,
            ),
            Step(
                name="Re-crawl prop odds",
                module="mlb_pipeline.crawler_oddsapi",
                args=("--skip-live", "--force-props"),
                timeout_s=600,
                critical=True,
            ),
            Step(
                name="Re-parse meta (injuries)",
                module="mlb_pipeline.parse_meta",
                timeout_s=60,
                critical=False,
            ),
            Step(
                name="Re-parse same-day lineups",
                module="mlb_pipeline.parse_lineup",
                timeout_s=120,
                critical=False,
            ),
            Step(
                name="Re-parse game odds + morning-lock prop close snapshot",
                module="mlb_pipeline.parse_oddsapi",
                args=("--prop-snapshot-role", "close"),
                timeout_s=300,
                critical=True,
            ),
            Step(
                name="Rebuild prop offer links",
                module="mlb_pipeline.modeling.build_prop_offer_links_table",
                timeout_s=300,
                critical=True,
            ),
            Step(
                name="Daily forecast projection audit",
                module="mlb_pipeline.modeling.daily_forecast_projection_audit",
                timeout_s=300,
                critical=False,
            ),
            Step(
                name="Player-game bankroll proof",
                module="mlb_pipeline.modeling.prop_player_game_bankroll_model_proof",
                timeout_s=360,
                critical=False,
            ),
            Step(
                name="TB 1.5 line calibration",
                module="mlb_pipeline.modeling.prop_tb15_line_calibration",
                timeout_s=180,
                critical=False,
            ),
            Step(
                name="K-under repair report",
                module="mlb_pipeline.modeling.prop_k_under_repair_report",
                timeout_s=180,
                critical=False,
            ),
            Step(
                name="Exact-bucket CLV priors",
                module="mlb_pipeline.modeling.prop_exact_bucket_clv_priors",
                timeout_s=180,
                critical=False,
            ),
            Step(
                name="Prop micro promotion evaluation",
                module="mlb_pipeline.modeling.prop_micro_promotion_evaluation",
                timeout_s=60,
                critical=False,
            ),
            Step(
                name="Prop micro probability calibrator",
                module="mlb_pipeline.modeling.prop_micro_probability_calibrator",
                timeout_s=120,
                critical=False,
            ),
            Step(
                name="External pick fetch",
                module="mlb_pipeline.modeling.external_pick_fetcher",
                timeout_s=180,
                critical=False,
            ),
            Step(
                name="External pick import",
                module="mlb_pipeline.modeling.external_pick_ledger",
                timeout_s=120,
                critical=False,
            ),
            Step(
                name="Re-predict player props + post to Discord",
                module="mlb_pipeline.modeling.predict_player_props",
                timeout_s=300,
                critical=True,
            ),
            Step(
                name="External model comparison",
                module="mlb_pipeline.modeling.external_pick_ledger",
                args=("--skip-import",),
                timeout_s=120,
                critical=False,
            ),
            Step(
                name="Train AI bet selection model",
                module="mlb_pipeline.modeling.ai_bet_selection_model",
                args=("--lookback-days", "120", "--max-rows", "12000", "--max-market-families", "8"),
                timeout_s=300,
                critical=False,
            ),
            Step(
                name="AI pick engine",
                module="mlb_pipeline.modeling.ai_pick_engine",
                timeout_s=300,
                critical=False,
            ),
            Step(
                name="Shadow-lock prop predictions",
                module="mlb_pipeline.modeling.shadow_lock_prop_predictions",
                args=("--phase", lock_phase),
                timeout_s=120,
                critical=True,
                fail_task_on_error=True,
            ),
            Step(
                name="Re-crawl post-lock closing prop odds",
                module="mlb_pipeline.crawler_oddsapi",
                args=("--skip-live", "--force-props"),
                timeout_s=600,
                critical=True,
                fail_task_on_error=True,
            ),
            Step(
                name="Capture post-lock prop close snapshot",
                module="mlb_pipeline.parse_oddsapi",
                args=("--prop-snapshot-role", "close"),
                timeout_s=300,
                critical=True,
                fail_task_on_error=True,
            ),
            Step(
                name="Refresh prop replay CLV",
                module="mlb_pipeline.modeling.refresh_prop_replay_clv",
                timeout_s=300,
                critical=True,
                fail_task_on_error=True,
            ),
            Step(
                name="Build prop market training table",
                module="mlb_pipeline.modeling.build_prop_market_training_table",
                args=("--lookback-days", "3", "--include-pending", "--no-replace"),
                timeout_s=600,
                critical=False,
            ),
            Step(
                name="Prop walk-forward accuracy report",
                module="mlb_pipeline.modeling.prop_walk_forward_accuracy_report",
                args=(
                    "--no-refresh-clv",
                    "--lookback-days", "45",
                    "--skip-if-fresh-minutes", str(PROP_WALK_FORWARD_FRESH_MINUTES),
                ),
                timeout_s=PROP_WALK_FORWARD_TIMEOUT_S,
                critical=False,
            ),
            Step(
                name="Prop shadow selector report",
                module="mlb_pipeline.modeling.prop_shadow_selector",
                timeout_s=300,
                critical=False,
                fail_task_on_error=True,
            ),
            Step(
                name="Prop miss diagnostic report",
                module="mlb_pipeline.modeling.prop_miss_diagnostic_report",
                timeout_s=180,
                critical=False,
            ),
            Step(
                name="Prop bucket repair report",
                module="mlb_pipeline.modeling.prop_bucket_repair_report",
                timeout_s=180,
                critical=False,
            ),
            Step(
                name="TB prop repair report",
                module="mlb_pipeline.modeling.prop_tb_repair_report",
                timeout_s=180,
                critical=False,
            ),
            Step(
                name="Prop target quality report",
                module="mlb_pipeline.modeling.prop_target_quality_report",
                timeout_s=180,
                critical=False,
            ),
            Step(
                name="FanDuel one-sided diagnostic",
                module="mlb_pipeline.modeling.fanduel_one_sided_diagnostic",
                args=("--lookback-days", "30"),
                timeout_s=300,
                critical=False,
            ),
            Step(
                name="Prop snapshot coverage report",
                module="mlb_pipeline.modeling.prop_snapshot_coverage_report",
                timeout_s=120,
                critical=False,
            ),
            Step(
                name="Hitter live-vs-legacy forecast diff",
                module="mlb_pipeline.modeling.hitter_live_vs_legacy_forecast_diff_report",
                timeout_s=180,
                critical=False,
            ),
            Step(
                name="Prop micro gate sensitivity report",
                module="mlb_pipeline.modeling.prop_micro_gate_sensitivity_report",
                timeout_s=60,
                critical=False,
            ),
            Step(
                name="Prop micro bucket repair report",
                module="mlb_pipeline.modeling.prop_micro_bucket_repair_report",
                timeout_s=60,
                critical=False,
            ),
            Step(
                name="Prop trial candidate queue report",
                module="mlb_pipeline.modeling.prop_trial_candidate_queue_report",
                timeout_s=60,
                critical=False,
            ),
                Step(
                    name="Prop drift guard diagnostic",
                    module="mlb_pipeline.modeling.prop_drift_guard_diagnostic",
                    timeout_s=120,
                    critical=False,
                ),
                Step(
                    name="Prop bettable-now scan",
                    module="mlb_pipeline.modeling.prop_bettable_now_scan",
                    timeout_s=120,
                    critical=False,
                ),
                Step(
                    name="Lock micro projection ledger",
                    module="mlb_pipeline.modeling.lock_micro_projection_ledger",
                    timeout_s=120,
                    critical=False,
                ),
                Step(
                    name="Prop micro ledger report",
                    module="mlb_pipeline.modeling.prop_micro_ledger_report",
                    timeout_s=120,
                    critical=False,
                ),
                Step(
                    name="Prop micro loss diagnostic",
                    module="mlb_pipeline.modeling.prop_micro_loss_diagnostic",
                    timeout_s=120,
                    critical=False,
                ),
                Step(
                    name="Prop micro probability calibrator",
                    module="mlb_pipeline.modeling.prop_micro_probability_calibrator",
                    timeout_s=120,
                    critical=False,
                ),
                Step(
                    name="Prop post-gate candidate report",
                    module="mlb_pipeline.modeling.prop_post_gate_candidate_report",
                    timeout_s=120,
                    critical=False,
                ),
                Step(
                    name="Prop layer promotion control",
                    module="mlb_pipeline.modeling.prop_layer_promotion_report",
                timeout_s=60,
                critical=False,
            ),
            Step(
                name="Forecast repair error decomposition",
                module="mlb_pipeline.modeling.forecast_repair_error_report",
                timeout_s=600,
                critical=False,
            ),
        ]
    elif args.close_only:
        # ── Evening closing-line run ─────────────────────────────────────────
        extra_env["DISCORD_FORMAT"] = "1"
        close_results_hour_et = int(os.getenv("MLB_CLOSE_RESULTS_REFRESH_HOUR_ET", "18"))
        if now_et.hour >= close_results_hour_et:
            steps.append(Step(
                name="Refresh final game results (MLB Stats API)",
                module="mlb_pipeline.crawler_statsapi",
                args=(
                    "--season", f"{et_day.year}-regular",
                    "--start-date", et_day.isoformat(),
                    "--end-date", et_day.isoformat(),
                ),
                timeout_s=600,
                critical=False,
                fail_task_on_error=True,
            ))
        # Re-crawl live odds so game lines and prop lines get a late-day snapshot.
        # Prop CLV only accepts immutable close-role snapshots that were captured
        # after the prediction lock and within two hours of first pitch.
        steps.append(Step(
            name="Re-crawl closing game odds (Odds API)",
            module="mlb_pipeline.crawler_oddsapi",
            args=("--skip-props",),
            timeout_s=600,
            critical=False,
        ))
        steps.append(Step(
            name="Re-crawl closing prop odds (Odds API)",
            module="mlb_pipeline.crawler_oddsapi",
            args=("--skip-live", "--force-props"),
            timeout_s=600,
            critical=True,
            fail_task_on_error=True,
        ))
        steps.append(Step(
            name="Re-parse closing odds into odds.mlb_game_lines",
            module="mlb_pipeline.parse_oddsapi",
            args=("--prop-snapshot-role", "close"),
            timeout_s=300,
            critical=True,
            fail_task_on_error=True,
        ))
        steps.append(Step(
            name="Refresh prop replay CLV",
            module="mlb_pipeline.modeling.refresh_prop_replay_clv",
            timeout_s=300,
            critical=True,
            fail_task_on_error=True,
        ))
        steps.append(Step(
            name="Build prop market training table",
            module="mlb_pipeline.modeling.build_prop_market_training_table",
            args=("--lookback-days", "3", "--include-pending", "--no-replace"),
            timeout_s=600,
            critical=False,
        ))
        steps.append(Step(
            name="Train AI bet selection model",
            module="mlb_pipeline.modeling.ai_bet_selection_model",
            args=("--lookback-days", "120", "--max-rows", "12000", "--max-market-families", "8"),
            timeout_s=300,
            critical=False,
        ))
        steps.append(Step(
            name="Prop walk-forward accuracy report",
            module="mlb_pipeline.modeling.prop_walk_forward_accuracy_report",
            args=(
                "--no-refresh-clv",
                "--lookback-days", "45",
                "--skip-if-fresh-minutes", str(PROP_WALK_FORWARD_FRESH_MINUTES),
            ),
            timeout_s=PROP_WALK_FORWARD_TIMEOUT_S,
            critical=False,
        ))
        steps.append(Step(
            name="Prop shadow selector report",
            module="mlb_pipeline.modeling.prop_shadow_selector",
            timeout_s=300,
            critical=False,
            fail_task_on_error=True,
        ))
        steps.append(Step(
            name="Prop miss diagnostic report",
            module="mlb_pipeline.modeling.prop_miss_diagnostic_report",
            timeout_s=180,
            critical=False,
        ))
        steps.append(Step(
            name="Prop bucket repair report",
            module="mlb_pipeline.modeling.prop_bucket_repair_report",
            timeout_s=180,
            critical=False,
        ))
        steps.append(Step(
            name="TB prop repair report",
            module="mlb_pipeline.modeling.prop_tb_repair_report",
            timeout_s=180,
            critical=False,
        ))
        steps.append(Step(
            name="Prop target quality report",
            module="mlb_pipeline.modeling.prop_target_quality_report",
            timeout_s=180,
            critical=False,
        ))
        steps.append(Step(
            name="FanDuel one-sided diagnostic",
            module="mlb_pipeline.modeling.fanduel_one_sided_diagnostic",
            args=("--lookback-days", "30"),
            timeout_s=300,
            critical=False,
        ))
        steps.append(Step(
            name="Grade outcomes + ledgers",
            module="mlb_pipeline.modeling.update_outcomes",
            timeout_s=300,
            critical=False,
        ))
        steps.append(Step(
            name="Grade daily forecast ledger",
            module="mlb_pipeline.modeling.daily_forecast_ledger",
            args=("--ensure-schema", "--grade"),
            timeout_s=300,
            critical=False,
        ))
        steps.append(Step(
            name="Daily forecast projection audit",
            module="mlb_pipeline.modeling.daily_forecast_projection_audit",
            timeout_s=300,
            critical=False,
        ))
        steps.append(Step(
            name="Hitter live-vs-legacy forecast diff",
            module="mlb_pipeline.modeling.hitter_live_vs_legacy_forecast_diff_report",
            timeout_s=180,
            critical=False,
        ))
        steps.append(Step(
            name="Player-game bankroll proof",
            module="mlb_pipeline.modeling.prop_player_game_bankroll_model_proof",
            timeout_s=360,
            critical=False,
        ))
        steps.append(Step(
            name="TB 1.5 line calibration",
            module="mlb_pipeline.modeling.prop_tb15_line_calibration",
            timeout_s=180,
            critical=False,
        ))
        steps.append(Step(
            name="TB 1.5 close repair report",
            module="mlb_pipeline.modeling.prop_tb15_close_repair_report",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="K-under repair report",
            module="mlb_pipeline.modeling.prop_k_under_repair_report",
            timeout_s=180,
            critical=False,
        ))
        steps.append(Step(
            name="DK K 4.5-6.0 under repair diagnostic",
            module="mlb_pipeline.modeling.prop_k_under_46_repair_diagnostic",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Exact-bucket CLV priors",
            module="mlb_pipeline.modeling.prop_exact_bucket_clv_priors",
            timeout_s=180,
            critical=False,
        ))
        steps.append(Step(
            name="Prop micro promotion evaluation",
            module="mlb_pipeline.modeling.prop_micro_promotion_evaluation",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Prop micro gate sensitivity report",
            module="mlb_pipeline.modeling.prop_micro_gate_sensitivity_report",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Prop micro bucket repair report",
            module="mlb_pipeline.modeling.prop_micro_bucket_repair_report",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Prop trial candidate queue report",
            module="mlb_pipeline.modeling.prop_trial_candidate_queue_report",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Prop drift guard diagnostic",
            module="mlb_pipeline.modeling.prop_drift_guard_diagnostic",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop bettable-now scan",
            module="mlb_pipeline.modeling.prop_bettable_now_scan",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Lock micro projection ledger",
            module="mlb_pipeline.modeling.lock_micro_projection_ledger",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop micro ledger report",
            module="mlb_pipeline.modeling.prop_micro_ledger_report",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop micro loss diagnostic",
            module="mlb_pipeline.modeling.prop_micro_loss_diagnostic",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop micro probability calibrator",
            module="mlb_pipeline.modeling.prop_micro_probability_calibrator",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop post-gate candidate report",
            module="mlb_pipeline.modeling.prop_post_gate_candidate_report",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop layer promotion control",
            module="mlb_pipeline.modeling.prop_layer_promotion_report",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Forecast repair error decomposition",
            module="mlb_pipeline.modeling.forecast_repair_error_report",
            timeout_s=600,
            critical=False,
        ))
        steps.append(Step(
            name="TB tail repair challenger",
            module="mlb_pipeline.modeling.prop_tb_tail_repair_challenger",
            timeout_s=600,
            critical=False,
        ))
        steps.append(Step(
            name="Prop snapshot coverage report",
            module="mlb_pipeline.modeling.prop_snapshot_coverage_report",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Real-money operational prop reports",
            module="mlb_pipeline.modeling.prop_real_money_operational_reports",
            timeout_s=300,
            critical=False,
        ))
        steps.append(Step(
            name="Pitcher K-rate challenger diagnostic",
            module="mlb_pipeline.modeling.pitcher_k_rate_challenger_report",
            timeout_s=300,
            critical=False,
        ))
        steps.append(Step(
            name="End-of-slate prop close diagnostic",
            module="mlb_pipeline.modeling.prop_end_of_slate_close_diagnostic",
            args=("--date", et_day.isoformat()),
            timeout_s=180,
            critical=False,
            fail_task_on_error=True,
        ))
        steps.append(Step(
            name="Frozen-release five-date checkpoint",
            module="mlb_pipeline.modeling.prop_five_date_checkpoint",
            timeout_s=180,
            critical=False,
            fail_task_on_error=True,
        ))
        steps.append(Step(
            name="Post-checkpoint micro promotion evaluation",
            module="mlb_pipeline.modeling.prop_micro_promotion_evaluation",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Post-checkpoint micro gate sensitivity report",
            module="mlb_pipeline.modeling.prop_micro_gate_sensitivity_report",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Post-checkpoint micro bucket repair report",
            module="mlb_pipeline.modeling.prop_micro_bucket_repair_report",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Post-checkpoint prop trial candidate queue report",
            module="mlb_pipeline.modeling.prop_trial_candidate_queue_report",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Post-checkpoint prop drift guard diagnostic",
            module="mlb_pipeline.modeling.prop_drift_guard_diagnostic",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Post-checkpoint lock micro projection ledger",
            module="mlb_pipeline.modeling.lock_micro_projection_ledger",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Post-checkpoint prop micro ledger report",
            module="mlb_pipeline.modeling.prop_micro_ledger_report",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Post-checkpoint prop micro probability calibrator",
            module="mlb_pipeline.modeling.prop_micro_probability_calibrator",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Post-checkpoint prop layer promotion control",
            module="mlb_pipeline.modeling.prop_layer_promotion_report",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Post-checkpoint real-money operational prop reports",
            module="mlb_pipeline.modeling.prop_real_money_operational_reports",
            timeout_s=300,
            critical=False,
        ))
        steps.append(Step(
            name="Daily slate trust monitor",
            module="mlb_pipeline.modeling.prop_daily_slate_trust_monitor",
            args=("--date", et_day.isoformat()),
            timeout_s=600,
            critical=False,
            fail_task_on_error=True,
        ))
        steps.append(Step(
            name="Grade shadow prop replay",
            module="mlb_pipeline.modeling.grade_prop_prediction_replay",
            timeout_s=300,
            critical=False,
        ))
    else:
        # ── Normal full daily run ────────────────────────────────────────────
        steps.append(Step(
            name="Grade daily forecast ledger",
            module="mlb_pipeline.modeling.daily_forecast_ledger",
            args=("--ensure-schema", "--grade"),
            timeout_s=300,
            critical=False,
        ))
        if not args.skip_crawl:
            # All four crawlers hit different external APIs and use disjoint
            # (provider, endpoint, url) keys in raw.api_responses, so concurrent
            # ON CONFLICT DO UPDATE writes are safe.
            steps.append(Step(
                name="Crawl MLB Stats API (schedule/boxscores)",
                module="mlb_pipeline.crawler_statsapi",
                args=("--season", "2026-regular"),
                timeout_s=3600,
                critical=True,
                parallel=True,
            ))
            steps.append(Step(
                name="Crawl MSF (injuries/lineups)",
                module="mlb_pipeline.crawler",
                timeout_s=3600,
                critical=False,  # MSF returns 403 for MLB game data; non-critical
                parallel=True,
            ))
            steps.append(Step(
                name="Crawl odds (Odds API)",
                module="mlb_pipeline.crawler_oddsapi",
                args=("--force-props",),
                timeout_s=3600,
                critical=True,
                parallel=True,
            ))
            steps.append(Step(
                name="Crawl Statcast (Baseball Savant)",
                module="mlb_pipeline.crawler_statcast",
                timeout_s=300,
                critical=False,
                parallel=True,
            ))
            steps.append(Step(
                name="Crawl extended Statcast features",
                module="mlb_pipeline.crawler_statcast_extended",
                timeout_s=600,
                critical=False,
                parallel=True,
            ))
            steps.append(Step(
                name="Crawl weather (Open-Meteo)",
                module="mlb_pipeline.crawler_weather",
                timeout_s=300,
                critical=False,
                parallel=True,
            ))

        if not args.skip_parse:
            steps.append(Step(
                name="Parse + load (parse_all)",
                module="mlb_pipeline.parse_all",
                timeout_s=7200,
                critical=True,
            ))

        if not args.skip_parse:
            steps.append(Step(
                name="Compute Elo ratings",
                module="mlb_pipeline.compute_elo",
                timeout_s=600,
                critical=False,
            ))

        steps.append(Step(
            name="Grade shadow prop replay",
            module="mlb_pipeline.modeling.grade_prop_prediction_replay",
            timeout_s=300,
            critical=False,
        ))

        # Offer rows must be rebuilt after every odds parse, even when model
        # training is skipped.
        steps.append(Step(
            name="Build prop offer links",
            module="mlb_pipeline.modeling.build_prop_offer_links_table",
            timeout_s=300,
            critical=False,
        ))

        if not args.skip_predict:
            steps.append(Step(
                name="External pick fetch",
                module="mlb_pipeline.modeling.external_pick_fetcher",
                timeout_s=180,
                critical=False,
            ))
            steps.append(Step(
                name="External pick import",
                module="mlb_pipeline.modeling.external_pick_ledger",
                timeout_s=120,
                critical=False,
            ))
            # predict_today writes to bets.mlb_game_predictions,
            # predict_player_props writes to bets.mlb_prop_predictions — no overlap.
            steps.append(Step(
                name="Predict today",
                module="mlb_pipeline.modeling.predict_today",
                timeout_s=600,
                critical=False,
                parallel=True,
            ))
            steps.append(Step(
                name="Predict player props",
                module="mlb_pipeline.modeling.predict_player_props",
                timeout_s=900,
                critical=False,
                parallel=True,
            ))
            steps.append(Step(
                name="External model comparison",
                module="mlb_pipeline.modeling.external_pick_ledger",
                args=("--skip-import",),
                timeout_s=120,
                critical=False,
            ))
            steps.append(Step(
                name="Train AI bet selection model",
                module="mlb_pipeline.modeling.ai_bet_selection_model",
                args=("--lookback-days", "120", "--max-rows", "12000", "--max-market-families", "8"),
                timeout_s=300,
                critical=False,
            ))
            steps.append(Step(
                name="AI pick engine",
                module="mlb_pipeline.modeling.ai_pick_engine",
                timeout_s=300,
                critical=False,
            ))
            steps.append(Step(
                name="Shadow-lock prop predictions",
                module="mlb_pipeline.modeling.shadow_lock_prop_predictions",
                args=("--phase", "morning"),
                timeout_s=300,
                critical=False,
            ))

        # Train after predictions so a slow refresh cannot delay today's slate.
        # These artifacts are consumed by the next prediction run.
        if not args.skip_train:
            steps.append(Step(
                name="Train game models",
                module="mlb_pipeline.modeling.train_game_models",
                timeout_s=14400,
                critical=True,
                parallel=True,
            ))
            steps.append(Step(
                name="Train player prop models",
                module="mlb_pipeline.modeling.train_player_prop_models",
                args=() if args.full_prop_optuna else ("--skip-optuna",),
                timeout_s=21600,
                critical=False,
                parallel=True,
            ))
            steps.append(Step(
                name="Train binary prop classifiers",
                module="mlb_pipeline.modeling.train_binary_prop_models",
                timeout_s=10800,
                critical=False,
                parallel=True,
            ))
            steps.append(Step(
                name="Build prop market training table",
                module="mlb_pipeline.modeling.build_prop_market_training_table",
                args=("--lookback-days", "45", "--include-pending", "--ensure-schema", "--no-replace"),
                timeout_s=3600,
                critical=False,
                fail_task_on_error=True,
            ))
            steps.append(Step(
                name="Retrain AI bet selection model",
                module="mlb_pipeline.modeling.ai_bet_selection_model",
                args=("--lookback-days", "120", "--max-rows", "12000", "--max-market-families", "8"),
                timeout_s=300,
                critical=False,
            ))
            steps.append(Step(
                name="Build hitter player-game training table",
                module="mlb_pipeline.modeling.build_hitter_player_game_training_table",
                timeout_s=600,
                critical=False,
            ))
            steps.append(Step(
                name="Hitter event feature ablation report",
                module="mlb_pipeline.modeling.hitter_event_feature_ablation_report",
                timeout_s=900,
                critical=False,
            ))
            steps.append(Step(
                name="Train hitter player-game outcome models",
                module="mlb_pipeline.modeling.train_hitter_player_game_outcome_models",
                timeout_s=900,
                critical=False,
            ))
            steps.append(Step(
                name="Build prop market history table",
                module="mlb_pipeline.modeling.build_prop_market_history_table",
                timeout_s=600,
                critical=False,
            ))
            steps.append(Step(
                name="Train prop market side priors",
                module="mlb_pipeline.modeling.train_prop_market_side_priors",
                timeout_s=600,
                critical=False,
            ))
            steps.append(Step(
                name="Optimize prop thresholds",
                module="mlb_pipeline.modeling.optimize_prop_thresholds",
                timeout_s=300,
                critical=False,
            ))
            steps.append(Step(
                name="Monitor prop calibration",
                module="mlb_pipeline.modeling.monitor_prop_calibration",
                timeout_s=300,
                critical=False,
            ))
            steps.append(Step(
                name="Train prop side recalibrators",
                module="mlb_pipeline.modeling.train_prop_side_recalibrators",
                timeout_s=600,
                critical=False,
            ))
            steps.append(Step(
                name="Train prop betting layer",
                module="mlb_pipeline.modeling.train_prop_betting_layer",
                timeout_s=600,
                critical=False,
            ))
            steps.append(Step(
                name="Train prop direct side models",
                module="mlb_pipeline.modeling.train_prop_direct_side_models",
                timeout_s=600,
                critical=False,
            ))
            steps.append(Step(
                name="Train prop opportunity models",
                module="mlb_pipeline.modeling.train_prop_opportunity_models",
                timeout_s=600,
                critical=False,
            ))
            steps.append(Step(
                name="Train prop bookability model",
                module="mlb_pipeline.modeling.train_prop_bookability_model",
                timeout_s=600,
                critical=False,
            ))
            steps.append(Step(
                name="Train prop market-residual models",
                module="mlb_pipeline.modeling.train_prop_market_residual_models",
                timeout_s=600,
                critical=False,
            ))
            steps.append(Step(
                name="Refresh prop exact-bucket proof",
                module="mlb_pipeline.modeling.refresh_prop_exact_bucket_proof",
                timeout_s=3900,
                critical=False,
            ))
            steps.append(Step(
                name="Compare prop probability variants",
                module="mlb_pipeline.modeling.compare_prop_probability_variants",
                timeout_s=1800,
                critical=False,
            ))
            steps.append(Step(
                name="Prop opportunity feature report",
                module="mlb_pipeline.modeling.prop_opportunity_feature_report",
                args=("--lookback-days", "30"),
                timeout_s=1800,
                critical=False,
            ))
            steps.append(Step(
                name="Hitter player-rate challenger diagnostic",
                module="mlb_pipeline.modeling.hitter_player_rate_diagnostic",
                timeout_s=300,
                critical=False,
            ))
            steps.append(Step(
                name="Pitcher K-per-BF challenger diagnostic",
                module="mlb_pipeline.modeling.pitcher_k_rate_challenger_report",
                timeout_s=300,
                critical=False,
            ))
            steps.append(Step(
                name="Prospective prop opportunity audit",
                module="mlb_pipeline.modeling.prop_prospective_opportunity_audit",
                timeout_s=300,
                critical=False,
            ))
            steps.append(Step(
                name="Daily forecast projection audit",
                module="mlb_pipeline.modeling.daily_forecast_projection_audit",
                timeout_s=300,
                critical=False,
            ))
            steps.append(Step(
                name="Hitter live-vs-legacy forecast diff",
                module="mlb_pipeline.modeling.hitter_live_vs_legacy_forecast_diff_report",
                timeout_s=180,
                critical=False,
            ))
            steps.append(Step(
                name="Forecast repair error decomposition",
                module="mlb_pipeline.modeling.forecast_repair_error_report",
                timeout_s=600,
                critical=False,
            ))
            steps.append(Step(
                name="Train prop bucket reopen policy",
                module="mlb_pipeline.modeling.train_prop_bucket_reopen_policy",
                timeout_s=300,
                critical=False,
            ))

        steps.append(Step(
            name="Paper trading report",
            module="mlb_pipeline.modeling.paper_trading_report",
            args=("--days", "90"),
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Offer-level prop audit report",
            module="mlb_pipeline.modeling.offer_level_prop_audit_report",
            args=("--lookback-days", "90", "--min-bucket-rows", "5", "--top-n", "25"),
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop real-money readiness report",
            module="mlb_pipeline.modeling.prop_real_money_readiness_report",
            args=("--lookback-days", "90", "--top-n", "25"),
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop real-money kill switch",
            module="mlb_pipeline.modeling.prop_real_money_kill_switch",
            timeout_s=60,
            critical=False,
            fail_task_on_error=True,
        ))
        steps.append(Step(
            name="Prop bucket promotion report",
            module="mlb_pipeline.modeling.prop_bucket_promotion_report",
            args=("--lookback-days", "365", "--top-n", "25"),
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop micro promotion evaluation",
            module="mlb_pipeline.modeling.prop_micro_promotion_evaluation",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Prop micro gate sensitivity report",
            module="mlb_pipeline.modeling.prop_micro_gate_sensitivity_report",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Prop micro bucket repair report",
            module="mlb_pipeline.modeling.prop_micro_bucket_repair_report",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Prop trial candidate queue report",
            module="mlb_pipeline.modeling.prop_trial_candidate_queue_report",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Prop drift guard diagnostic",
            module="mlb_pipeline.modeling.prop_drift_guard_diagnostic",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop bettable-now scan",
            module="mlb_pipeline.modeling.prop_bettable_now_scan",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Lock micro projection ledger",
            module="mlb_pipeline.modeling.lock_micro_projection_ledger",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop micro ledger report",
            module="mlb_pipeline.modeling.prop_micro_ledger_report",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop micro loss diagnostic",
            module="mlb_pipeline.modeling.prop_micro_loss_diagnostic",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop micro probability calibrator",
            module="mlb_pipeline.modeling.prop_micro_probability_calibrator",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop post-gate candidate report",
            module="mlb_pipeline.modeling.prop_post_gate_candidate_report",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Prop layer promotion control",
            module="mlb_pipeline.modeling.prop_layer_promotion_report",
            timeout_s=60,
            critical=False,
        ))
        steps.append(Step(
            name="Prop walk-forward accuracy report",
            module="mlb_pipeline.modeling.prop_walk_forward_accuracy_report",
            args=(
                "--no-refresh-clv",
                "--lookback-days", "45",
                "--skip-if-fresh-minutes", str(PROP_WALK_FORWARD_FRESH_MINUTES),
            ),
            timeout_s=PROP_WALK_FORWARD_TIMEOUT_S,
            critical=False,
        ))
        steps.append(Step(
            name="Prop shadow selector report",
            module="mlb_pipeline.modeling.prop_shadow_selector",
            timeout_s=300,
            critical=False,
            fail_task_on_error=True,
        ))
        steps.append(Step(
            name="Prop miss diagnostic report",
            module="mlb_pipeline.modeling.prop_miss_diagnostic_report",
            timeout_s=180,
            critical=False,
        ))
        steps.append(Step(
            name="Prop bucket repair report",
            module="mlb_pipeline.modeling.prop_bucket_repair_report",
            timeout_s=180,
            critical=False,
        ))
        steps.append(Step(
            name="TB prop repair report",
            module="mlb_pipeline.modeling.prop_tb_repair_report",
            timeout_s=180,
            critical=False,
        ))
        steps.append(Step(
            name="TB tail repair challenger",
            module="mlb_pipeline.modeling.prop_tb_tail_repair_challenger",
            timeout_s=600,
            critical=False,
        ))
        steps.append(Step(
            name="Prop target quality report",
            module="mlb_pipeline.modeling.prop_target_quality_report",
            timeout_s=180,
            critical=False,
        ))
        steps.append(Step(
            name="FanDuel one-sided diagnostic",
            module="mlb_pipeline.modeling.fanduel_one_sided_diagnostic",
            args=("--lookback-days", "30"),
            timeout_s=300,
            critical=False,
        ))
        steps.append(Step(
            name="Prop snapshot coverage report",
            module="mlb_pipeline.modeling.prop_snapshot_coverage_report",
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="End-of-slate prop close diagnostic",
            module="mlb_pipeline.modeling.prop_end_of_slate_close_diagnostic",
            args=("--date", et_day.isoformat()),
            timeout_s=180,
            critical=False,
        ))
        steps.append(Step(
            name="Frozen-release five-date checkpoint",
            module="mlb_pipeline.modeling.prop_five_date_checkpoint",
            timeout_s=180,
            critical=False,
        ))
        steps.append(Step(
            name="Daily slate trust monitor",
            module="mlb_pipeline.modeling.prop_daily_slate_trust_monitor",
            args=("--date", et_day.isoformat()),
            timeout_s=600,
            critical=False,
        ))
        steps.append(Step(
            name="Prop slate post-mortem report",
            module="mlb_pipeline.modeling.prop_slate_postmortem_report",
            args=("--top-n", "25"),
            timeout_s=120,
            critical=False,
        ))
        steps.append(Step(
            name="Real-money gate audit report",
            module="mlb_pipeline.modeling.real_money_audit_report",
            args=("--days", "60"),
            timeout_s=120,
            critical=False,
        ))
    _suffix = (
        "_targeted_close"
        if args.prop_close_capture_only
        else "_close"
        if args.close_only
        else "_pregame"
        if args.pre_game
        else ""
    )
    report_path = Path("reports") / f"mlb_daily_{et_day.isoformat()}{_suffix}.md"

    _p(console, f"[bold]ET date:[/bold] {et_day.isoformat()}" if console else f"ET date: {et_day.isoformat()}")
    _p(console, f"[dim]Report:[/dim] {report_path}" if console else f"Report: {report_path}")

    results: list[StepResult] = []
    pipeline_failed = False
    task_failed = False

    # Group consecutive parallel steps so they run simultaneously
    for is_parallel, group_iter in itertools.groupby(steps, key=lambda s: s.parallel):
        if pipeline_failed:
            break
        group = list(group_iter)

        if is_parallel:
            _rule(console, "[bold]Training (parallel)[/bold]" if console else "Training (parallel)")
            names = ", ".join(s.name for s in group)
            _p(console, (f"[dim]Launching simultaneously: {names}[/dim]" if console
                         else f"Launching simultaneously: {names}"))
            group_results = _run_parallel_steps(group, extra_env=extra_env)
            results.extend(group_results)
            for r, step in zip(group_results, group):
                status_str = "OK" if r.ok else f"FAILED (rc={r.rc})"
                _panel(console, f"{r.name}: {status_str} ({r.secs:.0f}s)", ok=r.ok)
                if not r.ok and r.stderr:
                    _p(console, _tail(r.stderr, 20))
                if not r.ok and step.fail_task_on_error:
                    _p(
                        console,
                        "[red]Step is required for scheduler success; final exit will be non-zero.[/red]"
                        if console
                        else "Step is required for scheduler success; final exit will be non-zero.",
                    )
                    task_failed = True
                if not r.ok and step.critical:
                    _p(console, "[red]Stopping pipeline due to critical failure.[/red]" if console
                       else "Stopping pipeline (critical failure).")
                    pipeline_failed = True
        else:
            for step in group:
                if pipeline_failed:
                    break
                _rule(console, f"[bold]{step.name}[/bold]" if console else step.name)
                _p(console, (f"[dim]python -m {step.module} {' '.join(step.args)}[/dim]" if console
                             else f"python -m {step.module}"))

                try:
                    r = _run_step(step, extra_env=extra_env)
                except subprocess.TimeoutExpired:
                    r = StepResult(
                        name=step.name,
                        ok=False,
                        rc=124,
                        secs=float(step.timeout_s or 0),
                        stdout="",
                        stderr=f"Timeout after {step.timeout_s}s",
                    )

                results.append(r)

                if r.ok:
                    _panel(console, "OK", ok=True)
                else:
                    _panel(console, f"FAILED (rc={r.rc})", ok=False)
                    if r.stderr:
                        _p(console, _tail(r.stderr, 40))
                    if step.fail_task_on_error:
                        _p(
                            console,
                            "[red]Step is required for scheduler success; final exit will be non-zero.[/red]"
                            if console
                            else "Step is required for scheduler success; final exit will be non-zero.",
                        )
                        task_failed = True
                    if step.critical:
                        _p(console, "[red]Stopping pipeline due to critical failure.[/red]" if console
                           else "Stopping pipeline (critical failure).")
                        pipeline_failed = True

    # Summary table
    _rule(console, "[bold]Summary[/bold]" if console else "Summary")
    rows: list[tuple[str, str, str, str]] = []
    for r in results:
        status = "[green]OK[/green]" if (console and r.ok) else ("[red]FAIL[/red]" if console else ("OK" if r.ok else "FAIL"))
        rows.append((r.name, status, f"{r.secs:0.1f}s", str(r.rc)))
    _render_table(console, rows)

    # Write markdown report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    md: list[str] = []
    md.append(f"# SuperNovaBets MLB Daily Run ({et_day.isoformat()} ET)\n")
    md.append("## Summary\n")
    for r in results:
        md.append(f"- **{r.name}**: {'OK' if r.ok else 'FAIL'} (rc={r.rc}, {r.secs:0.1f}s)")
    md.append("\n## Outputs (tails)\n")
    for r in results:
        md.append(f"### {r.name}\n")
        md.append(f"- rc: {r.rc}\n")
        if r.stdout:
            md.append("**stdout (tail)**\n```")
            md.append(_tail(r.stdout, 120))
            md.append("```\n")
        if r.stderr:
            md.append("**stderr (tail)**\n```")
            md.append(_tail(r.stderr, 120))
            md.append("```\n")
    atomic_write_text(report_path, "\n".join(md))

    _p(console, f"\n[green]Saved report:[/green] {report_path}\n" if console else f"\nSaved report: {report_path}\n")
    if pipeline_failed or task_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
