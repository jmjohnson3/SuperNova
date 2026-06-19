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

_ET = ZoneInfo("America/New_York")


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
        "--close-only", action="store_true",
        help=(
            "Closing-line run near first pitch. "
            "Re-crawls live game and prop odds, captures immutable prop close snapshots, "
            "then grades outcomes and CLV. Skips train/predict."
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

    extra_env = {
        "MLB_ET_DATE": et_day.isoformat(),
    }

    steps: list[Step] = []

    if args.pre_game:
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
                name="Re-predict player props + post to Discord",
                module="mlb_pipeline.modeling.predict_player_props",
                timeout_s=300,
                critical=True,
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
                args=("--no-refresh-clv",),
                timeout_s=600,
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
                name="Prop snapshot coverage report",
                module="mlb_pipeline.modeling.prop_snapshot_coverage_report",
                timeout_s=120,
                critical=False,
            ),
        ]
    elif args.close_only:
        # ── Evening closing-line run ─────────────────────────────────────────
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
            name="Prop walk-forward accuracy report",
            module="mlb_pipeline.modeling.prop_walk_forward_accuracy_report",
            args=("--no-refresh-clv",),
            timeout_s=600,
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
            name="Grade outcomes + ledgers",
            module="mlb_pipeline.modeling.update_outcomes",
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
            name="Grade shadow prop replay",
            module="mlb_pipeline.modeling.grade_prop_prediction_replay",
            timeout_s=300,
            critical=False,
        ))
    else:
        # ── Normal full daily run ────────────────────────────────────────────
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
                timeout_s=300,
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
                name="Shadow-lock prop predictions",
                module="mlb_pipeline.modeling.shadow_lock_prop_predictions",
                args=("--phase", "morning"),
                timeout_s=120,
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
                timeout_s=10800,
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
                args=("--include-pending", "--ensure-schema"),
                timeout_s=1800,
                critical=False,
                fail_task_on_error=True,
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
                name="Train prop distribution models",
                module="mlb_pipeline.modeling.train_prop_distribution_models",
                timeout_s=600,
                critical=False,
            ))
            steps.append(Step(
                name="Compare prop probability variants",
                module="mlb_pipeline.modeling.compare_prop_probability_variants",
                timeout_s=900,
                critical=False,
            ))
            steps.append(Step(
                name="Prop opportunity feature report",
                module="mlb_pipeline.modeling.prop_opportunity_feature_report",
                args=("--lookback-days", "30"),
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
            name="Prop walk-forward accuracy report",
            module="mlb_pipeline.modeling.prop_walk_forward_accuracy_report",
            timeout_s=600,
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
            name="Prop opportunity feature report",
            module="mlb_pipeline.modeling.prop_opportunity_feature_report",
            args=("--lookback-days", "30"),
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
            name="Prop slate post-mortem report",
            module="mlb_pipeline.modeling.prop_slate_postmortem_report",
            args=("--top-n", "25"),
            timeout_s=120,
            critical=False,
        ))

    _suffix = "_close" if args.close_only else ("_pregame" if args.pre_game else "")
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
    report_path.write_text("\n".join(md), encoding="utf-8")

    _p(console, f"\n[green]Saved report:[/green] {report_path}\n" if console else f"\nSaved report: {report_path}\n")
    if pipeline_failed or task_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
