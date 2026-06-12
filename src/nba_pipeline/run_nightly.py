# src/nba_pipeline/run_nightly.py
"""
Post-game nightly run: crawl last night's MSF data, fetch closing odds,
parse everything into DB, then grade predictions.

Run after games finish (e.g., 11:30 PM - 1 AM ET):
    python -m nba_pipeline.run_nightly

Or for a specific date's data:
    python -m nba_pipeline.run_nightly --date 2026-03-22
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


# ---------- Optional Rich UI (mirrors run_daily.py) ----------
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
            t = Table(title="SuperNovaBets Nightly Run", show_lines=True)
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
    print("\nSuperNovaBets Nightly Run")
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
    t0 = time.perf_counter()

    cmd = [sys.executable, "-m", step.module, *step.args]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=step.timeout_s,
        check=False,
        env=env,
    )

    secs = time.perf_counter() - t0
    return StepResult(
        name=step.name,
        ok=(proc.returncode == 0),
        rc=proc.returncode,
        secs=secs,
        stdout=(proc.stdout or "").strip(),
        stderr=(proc.stderr or "").strip(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-game nightly run: crawl -> parse -> grade predictions."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="ET date of the games just played (YYYY-MM-DD). Default: today ET.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Force MSF crawler start date for backfill (YYYY-MM-DD). "
             "Overrides the default 2-day incremental lookback.",
    )
    parser.add_argument(
        "--skip-crawl-msf", action="store_true",
        help="Skip MySportsFeeds crawler (boxscores/gamelogs).",
    )
    parser.add_argument(
        "--skip-crawl-odds", action="store_true",
        help="Skip Odds API crawler (closing lines).",
    )
    parser.add_argument(
        "--skip-parse", action="store_true",
        help="Skip parse_all step.",
    )
    parser.add_argument(
        "--skip-outcomes", action="store_true",
        help="Skip update_outcomes (prediction grading).",
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="Write a markdown run-log to this path.",
    )
    args = parser.parse_args()

    console = _get_console()

    et_day: date
    if args.date:
        et_day = date.fromisoformat(args.date)
    else:
        et_day = datetime.now(tz=_ET).date()

    extra_env = {"NBA_ET_DATE": et_day.isoformat()}

    _p(console, f"[bold]Nightly run — ET date:[/bold] {et_day.isoformat()}" if console
       else f"Nightly run — ET date: {et_day.isoformat()}")

    steps: list[Step] = []

    # 1) MSF crawler: picks up last night's boxscores + player gamelogs.
    #    Defaults to 2-day lookback; use --start-date for a deeper backfill.
    if not args.skip_crawl_msf:
        msf_args: tuple[str, ...] = ()
        if args.start_date:
            msf_args = ("--start-date", args.start_date)
        steps.append(Step(
            name="Crawl MSF (boxscores / gamelogs)",
            module="nba_pipeline.crawler",
            args=msf_args,
            timeout_s=3600,
            critical=True,
        ))

    # 2) Odds API crawler: captures closing lines for CLV tracking.
    if not args.skip_crawl_odds:
        steps.append(Step(
            name="Crawl Odds API (closing lines)",
            module="nba_pipeline.crawler_oddsapi",
            timeout_s=300,
            critical=False,  # non-critical — CLV is a nice-to-have at night
        ))

    # 3) Parse everything into structured tables, then refresh Elo and features.
    if not args.skip_parse:
        steps.append(Step(
            name="Parse + load (parse_all)",
            module="nba_pipeline.parse_all",
            timeout_s=10800,
            critical=True,
        ))
        steps.append(Step(
            name="Compute Elo ratings",
            module="nba_pipeline.compute_elo",
            timeout_s=300,
            critical=True,
        ))
        steps.append(Step(
            name="Materialize features",
            module="nba_pipeline.materialize_features",
            timeout_s=1800,
            critical=True,
        ))

    # 4) Grade predictions: fill actuals, ATS record, CLV, prop over/under hits.
    if not args.skip_outcomes:
        steps.append(Step(
            name="Update outcomes (grade predictions)",
            module="nba_pipeline.modeling.update_outcomes",
            timeout_s=300,
            critical=False,
        ))

    results: list[StepResult] = []

    for step in steps:
        _rule(console, f"[bold]{step.name}[/bold]" if console else step.name)
        _p(
            console,
            (f"[dim]python -m {step.module} {' '.join(step.args)}[/dim]" if console
             else f"python -m {step.module} {' '.join(step.args)}").strip(),
        )

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
            if step.critical:
                _p(
                    console,
                    "[red]Stopping nightly run due to critical failure.[/red]"
                    if console else "Stopping nightly run (critical failure).",
                )
                break

    # Summary
    _rule(console, "[bold]Summary[/bold]" if console else "Summary")
    table_rows: list[tuple[str, str, str, str]] = []
    for r in results:
        if console:
            status = "[green]OK[/green]" if r.ok else "[red]FAIL[/red]"
        else:
            status = "OK" if r.ok else "FAIL"
        table_rows.append((r.name, status, f"{r.secs:.1f}s", str(r.rc)))
    _render_table(console, table_rows)

    # Optional markdown log
    report_path = (
        Path(args.report) if args.report
        else Path("reports") / f"nightly_{et_day.isoformat()}.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    md: list[str] = [f"# SuperNovaBets Nightly Run ({et_day.isoformat()} ET)\n\n## Summary\n"]
    for r in results:
        md.append(f"- **{r.name}**: {'OK' if r.ok else 'FAIL'} (rc={r.rc}, {r.secs:.1f}s)")
    md.append("\n## Outputs (tails)\n")
    for r in results:
        md.append(f"### {r.name}\n\n- rc: {r.rc}\n")
        if r.stdout:
            md.append("**stdout (tail)**\n```\n" + _tail(r.stdout, 120) + "\n```\n")
        if r.stderr:
            md.append("**stderr (tail)**\n```\n" + _tail(r.stderr, 120) + "\n```\n")
    report_path.write_text("\n".join(md), encoding="utf-8")
    _p(
        console,
        f"\n[green]Saved report:[/green] {report_path}\n" if console
        else f"\nSaved report: {report_path}\n",
    )


if __name__ == "__main__":
    main()
