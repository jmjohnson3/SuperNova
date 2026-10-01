"""Run the offline NFL model refresh path for Task Scheduler."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from nfl_pipeline.integrity import nfl_season
from nfl_pipeline.subprocess_utils import run_subprocess
from nfl_pipeline.integrity import production_freeze

log = logging.getLogger("nfl_pipeline.run_training")


@dataclass(frozen=True)
class Step:
    label: str
    module: str
    args: tuple[str, ...] = ()
    timeout_s: int = 900


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(step: Step) -> tuple[int, str, str]:
    env = os.environ.copy()
    src_dir = str(_repo_root() / "src")
    env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    return run_subprocess(
        [sys.executable, "-m", step.module, *step.args],
        cwd=str(_repo_root()),
        env=env,
        timeout_s=step.timeout_s,
    )


def run_training(*, season: str | None = None, skip_context: bool = False) -> dict[str, Any]:
    season = season or str(nfl_season())
    context_args = ("--seasons", season)
    steps = [
        Step("NFL Schema", "nfl_pipeline.schema", timeout_s=180),
        Step("NFL Results/History", "nfl_pipeline.import_nflverse", args=("--seasons",f"{int(season)-1}-{season}"), timeout_s=1200),
        Step("NFL Context Import", "nfl_pipeline.import_context", args=context_args, timeout_s=600),
        Step("NFL Usage Context Import", "nfl_pipeline.import_usage_context", args=context_args, timeout_s=1800),
        Step("NFL Features", "nfl_pipeline.features", timeout_s=900),
        Step("NFL Game Features", "nfl_pipeline.game_features", timeout_s=600),
        Step("NFL Validated Release", "nfl_pipeline.modeling.train_validated_release", args=("--season",season, *(("--publish",) if not production_freeze() else ())), timeout_s=1800),
        Step("NFL Accuracy Challengers", "nfl_pipeline.modeling.train_accuracy_challengers", timeout_s=3600),
        Step("NFL Accuracy Components", "nfl_pipeline.modeling.train_accuracy_components", timeout_s=3600),
        Step("NFL Live Scoring Replay", "nfl_pipeline.modeling.live_scoring_replay", timeout_s=300),
        Step("NFL Exact-Line Prop Proof", "nfl_pipeline.modeling.prop_exact_line_proof", timeout_s=300),
        Step("NFL Train Exact-Line Prop Models", "nfl_pipeline.modeling.train_prop_exact_line_models", timeout_s=900),
        Step("NFL Active Release Report", "nfl_pipeline.modeling.model_holdout_report", timeout_s=120),
        Step("NFL Readiness Report", "nfl_pipeline.readiness_report", timeout_s=300),
        # Report-only refit of market anchoring on all settled weeks; installing is a manual --write.
        Step("NFL Market Calibration Refit Report", "nfl_pipeline.modeling.fit_market_calibration", timeout_s=1800),
    ]
    if skip_context:
        steps = [step for step in steps if step.module not in {"nfl_pipeline.import_context", "nfl_pipeline.import_usage_context"}]

    results: list[dict[str, Any]] = []
    status = "ok"
    for step in steps:
        log.info("Starting %s", step.label)
        rc, stdout, stderr = _run(step)
        record = {
            "label": step.label,
            "returncode": rc,
            "stdout_tail": stdout.strip()[-1200:],
            "stderr_tail": stderr.strip()[-1200:],
        }
        results.append(record)
        if rc != 0:
            status = "failed"
            break
        log.info("%s complete", step.label)
    return {"status": status, "season": season, "steps": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NFL offline model training and report refresh")
    parser.add_argument("--season", default=None, help="NFL season year, defaults to current year")
    parser.add_argument("--skip-context", action="store_true", help="Use already-imported roster/depth/usage context")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    result = run_training(season=args.season, skip_context=args.skip_context)
    print(json.dumps(result, indent=2, default=str))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
