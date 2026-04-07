from __future__ import annotations

import argparse
import os
import sys
import time
import subprocess
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Optional

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
    parser = argparse.ArgumentParser(description="Run the full SuperNovaBets MLB pipeline in order.")
    parser.add_argument("--date", type=str, default=None, help="ET date (YYYY-MM-DD). Default: today (ET).")
    parser.add_argument("--skip-crawl", action="store_true", help="Skip crawler steps.")
    parser.add_argument("--skip-parse", action="store_true", help="Skip parse/load step.")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training steps.")
    parser.add_argument("--skip-predict", action="store_true", help="Skip prediction steps.")
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

    if not args.skip_crawl:
        steps.append(Step(
            name="Crawl MLB Stats API (schedule/boxscores)",
            module="mlb_pipeline.crawler_statsapi",
            args=("--season", "2026-regular"),
            timeout_s=3600,
            critical=True,
        ))
        steps.append(Step(
            name="Crawl MSF (injuries/lineups)",
            module="mlb_pipeline.crawler",
            timeout_s=3600,
            critical=False,  # MSF returns 403 for MLB game data; non-critical
        ))
        steps.append(Step(
            name="Crawl odds (Odds API)",
            module="mlb_pipeline.crawler_oddsapi",
            timeout_s=3600,
            critical=True,
        ))
        steps.append(Step(
            name="Crawl Statcast (Baseball Savant)",
            module="mlb_pipeline.crawler_statcast",
            timeout_s=300,
            critical=False,
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

    if not args.skip_train:
        steps.append(Step(
            name="Train game models",
            module="mlb_pipeline.modeling.train_game_models",
            timeout_s=3600,
            critical=True,
        ))
        steps.append(Step(
            name="Train player prop models",
            module="mlb_pipeline.modeling.train_player_prop_models",
            timeout_s=1800,
            critical=False,
        ))

    if not args.skip_predict:
        steps.append(Step(
            name="Predict today",
            module="mlb_pipeline.modeling.predict_today",
            timeout_s=600,
            critical=False,
        ))
        steps.append(Step(
            name="Predict player props",
            module="mlb_pipeline.modeling.predict_player_props",
            timeout_s=300,
            critical=False,
        ))

    report_path = Path("reports") / f"mlb_daily_{et_day.isoformat()}.md"

    _p(console, f"[bold]ET date:[/bold] {et_day.isoformat()}" if console else f"ET date: {et_day.isoformat()}")
    _p(console, f"[dim]Report:[/dim] {report_path}" if console else f"Report: {report_path}")

    results: list[StepResult] = []

    for step in steps:
        _rule(console, f"[bold]{step.name}[/bold]" if console else step.name)
        _p(console, f"[dim]python -m {step.module} {' '.join(step.args)}[/dim]" if console else f"python -m {step.module}")

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
                _p(console, (_tail(r.stderr, 40)))
            if step.critical:
                _p(console, "[red]Stopping pipeline due to critical failure.[/red]" if console else "Stopping pipeline (critical failure).")
                break

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


if __name__ == "__main__":
    main()
