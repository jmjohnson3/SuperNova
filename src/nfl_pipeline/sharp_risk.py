"""Validate readiness evidence and persist a loss stop independently of disposable reports."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.integrity import atomic_json


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def evaluate(*, now: datetime, report_path: Path, state_path: Path, experiment: dict | None,
             loss_limit: float, max_age_hours: float) -> dict:
    reasons = []
    loss_stop = state_path.with_suffix(".loss-stop.json")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("contract") != "nfl-sharp-risk-v1" or not isinstance(state.get("loss_paused"), bool):
            raise ValueError("invalid risk state")
    except FileNotFoundError:
        state = dict(contract="nfl-sharp-risk-v1", loss_paused=False)
    except (OSError, ValueError, AttributeError, TypeError):
        return dict(status="paused", loss_paused=True, realized_profit=None,
                    pause_reasons=["risk state unreadable; preserve it and reconcile the loss stop"])
    state["loss_paused"] = state["loss_paused"] or loss_stop.exists()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary = report["summary"]
        gate = summary["readiness"]
        profit = _number((summary.get("placed") or {}).get("realized_profit", 0))
        low = _number(((gate.get("gates") or {}).get("loss_pause") or {}).get("lowest_cumulative_profit"))
        # A past loss breach remains binding across strategy versions and report refreshes.
        if (profit is not None and profit <= loss_limit) or (low is not None and low <= loss_limit):
            state.update(loss_paused=True, loss_observed_at=now.isoformat())
        built = safe_time(report.get("built_at"))
        if built is None or not 0 <= (now - built).total_seconds() <= max_age_hours * 3600:
            reasons.append("readiness report is stale or has an invalid timestamp")
        if (not experiment or summary.get("experiment_id") != experiment.get("experiment_id")
                or not experiment.get("sha256") or summary.get("experiment_sha256") != experiment["sha256"]):
            reasons.append("readiness report does not match the active strategy registration")
        if gate.get("status") not in {"trial", "paused", "scale_review_ready"} or profit is None or low is None:
            reasons.append("readiness report is incomplete")
        if not reasons:
            state.update(last_valid_report_at=report["built_at"], realized_profit=profit)
        if gate.get("status") == "paused":
            pauses = gate.get("pause_reasons")
            reasons.extend(pauses if isinstance(pauses, list) and all(isinstance(p, str) for p in pauses)
                           and pauses else ["readiness pause"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        reasons.append("readiness report missing or unreadable")
        gate = {}
    if state["loss_paused"]:
        reasons.append("established loss pause; a new report or strategy cannot clear it")
        # A separate create-once latch cannot be overwritten by a concurrent unpaused reader.
        try:
            loss_stop.parent.mkdir(parents=True, exist_ok=True)
            with loss_stop.open("x", encoding="utf-8") as handle:
                json.dump(dict(loss_paused=True, observed_at=now.isoformat()), handle)
        except FileExistsError:
            pass
        except OSError:
            reasons.append("cannot persist loss stop")
    state["checked_at"] = now.isoformat()
    try:
        atomic_json(state_path, state)
    except OSError:
        reasons.append("cannot persist risk state")
    if loss_stop.exists():
        state["loss_paused"] = True
        reasons.append("established loss pause; a new report or strategy cannot clear it")
    return dict(status="paused" if reasons else gate["status"], pause_reasons=list(dict.fromkeys(reasons)),
                loss_paused=state["loss_paused"], realized_profit=state.get("realized_profit"))
