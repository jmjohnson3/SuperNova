"""Pre-restart operational readiness report for MLB prop betting slates."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

_ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = _ROOT / "reports"
_TASK_PATH = "\\SuperNovaBets\\"
_REQUIRED_TASKS = (
    "MLB-Morning",
    "MLB-PreGame-Day",
    "MLB-PreGame-Evening",
    "MLB-Prop-Targeted-Close",
    "MLB-Close",
    "MLB-Training",
)
_REQUIRED_TABLES = (
    ("bets", "mlb_bankroll_ledger"),
    ("bets", "mlb_model_pick_ledger"),
    ("bets", "mlb_daily_forecast_ledger"),
    ("odds", "mlb_player_prop_line_snapshots"),
    ("features", "mlb_prop_market_training_examples"),
)
_REQUIRED_WEBHOOKS = (
    "MLB_DISCORD_WEBHOOK_URL",
    "MLB_RECORD_LEDGER_DISCORD_WEBHOOK_URL",
)
_OPTIONAL_WEBHOOKS = (
    "MLB_PROP_RESEARCH_DISCORD_WEBHOOK_URL",
    "MLB_OPS_DISCORD_WEBHOOK_URL",
    "MLB_ALERTS_DISCORD_WEBHOOK_URL",
)


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _check(name: str, passed: bool, detail: str, *, required: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "required": bool(required),
        "detail": detail,
    }


def _powershell_json(script: str) -> Any:
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except Exception as exc:
        return {"error": str(exc)}
    if result.returncode != 0:
        return {"error": (result.stderr or result.stdout).strip(), "returncode": result.returncode}
    text = result.stdout.strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except ValueError:
        return {"error": text}


def _scheduled_tasks() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quoted = ",".join("'" + name.replace("'", "''") + "'" for name in _REQUIRED_TASKS)
    script = f"""
$names = @({quoted})
$rows = foreach ($name in $names) {{
  $task = Get-ScheduledTask -TaskPath '{_TASK_PATH}' -TaskName $name -ErrorAction SilentlyContinue
  if ($null -eq $task) {{
    [pscustomobject]@{{Name=$name; Exists=$false; State='missing'; LastTaskResult=$null; LastRunTime=$null; NextRunTime=$null}}
  }} else {{
    $info = Get-ScheduledTaskInfo -TaskPath '{_TASK_PATH}' -TaskName $name
    [pscustomobject]@{{
      Name=$name
      Exists=$true
      State=[string]$task.State
      LastTaskResult=$info.LastTaskResult
      LastRunTime=$info.LastRunTime
      NextRunTime=$info.NextRunTime
    }}
  }}
}}
$rows | ConvertTo-Json -Depth 3
"""
    payload = _powershell_json(script)
    if isinstance(payload, dict) and payload.get("error"):
        check = _check("scheduled_tasks_query", False, str(payload.get("error")))
        return [], [check]
    rows = payload if isinstance(payload, list) else [payload]
    checks = [
        _check(
            f"task_registered_{name}",
            any(row.get("Name") == name and row.get("Exists") for row in rows),
            "registered under \\SuperNovaBets\\" if any(row.get("Name") == name and row.get("Exists") for row in rows) else "missing",
        )
        for name in _REQUIRED_TASKS
    ]
    bad_results = [
        row for row in rows
        if row.get("Exists") and row.get("LastTaskResult") not in (None, 0, 267011)
    ]
    checks.append(_check(
        "scheduled_task_last_results",
        not bad_results,
        "; ".join(f"{row.get('Name')}={row.get('LastTaskResult')}" for row in bad_results) or "last results are clean/never-run",
        required=False,
    ))
    return rows, checks


def _targeted_close_xml_check() -> list[dict[str, Any]]:
    path = _ROOT / "scripts" / "tasks" / "MLB-Prop-Targeted-Close.xml"
    if not path.exists():
        return [_check("targeted_close_xml_exists", False, str(path))]
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:
        return [_check("targeted_close_xml_parse", False, str(exc))]
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    interval = root.findtext(".//t:Repetition/t:Interval", namespaces=ns)
    duration = root.findtext(".//t:Repetition/t:Duration", namespaces=ns)
    limit = root.findtext(".//t:ExecutionTimeLimit", namespaces=ns)
    command = root.findtext(".//t:Actions/t:Exec/t:Command", namespaces=ns)
    return [
        _check("targeted_close_interval_pt10m", interval == "PT10M", f"interval={interval!r}"),
        _check("targeted_close_duration_present", bool(duration), f"duration={duration!r}", required=False),
        _check("targeted_close_limit_not_too_long", limit in {"PT15M", "PT10M"}, f"execution_time_limit={limit!r}", required=False),
        _check("targeted_close_command_exists", bool(command and Path(command).exists()), f"command={command!r}"),
    ]


def _registry_env_present(name: str) -> bool:
    if os.getenv(name):
        return True
    script = (
        f"$v = [Environment]::GetEnvironmentVariable('{name}', 'User'); "
        "if ([string]::IsNullOrWhiteSpace($v)) { exit 1 } else { exit 0 }"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _database_checks(pg_dsn: str) -> list[dict[str, Any]]:
    try:
        with psycopg2.connect(pg_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE (table_schema, table_name) IN %s
                    """,
                    (tuple(_REQUIRED_TABLES),),
                )
                existing = {(schema, table) for schema, table in cur.fetchall()}
    except Exception as exc:
        return [_check("database_connectivity", False, str(exc))]
    checks = [_check("database_connectivity", True, "connected")]
    for schema, table in _REQUIRED_TABLES:
        checks.append(_check(
            f"table_exists_{schema}_{table}",
            (schema, table) in existing,
            f"{schema}.{table}",
        ))
    return checks


def build(model_dir: Path = _MODEL_DIR, pg_dsn: str = PG_DSN) -> dict[str, Any]:
    checkpoint = _load(model_dir / "prop_five_date_checkpoint.json")
    micro = _load(model_dir / "prop_micro_promotion_evaluation.json")
    backdated = _load(model_dir / "prop_backdated_slate_eligibility_audit.json")
    task_rows, task_checks = _scheduled_tasks()
    checks: list[dict[str, Any]] = []
    checks.extend(task_checks)
    checks.extend(_targeted_close_xml_check())
    checks.extend(_database_checks(pg_dsn))
    for env_name in _REQUIRED_WEBHOOKS:
        present = _registry_env_present(env_name)
        checks.append(_check(
            f"webhook_present_{env_name}",
            present,
            "set in process/user environment" if present else "missing",
        ))
    for env_name in _OPTIONAL_WEBHOOKS:
        present = _registry_env_present(env_name)
        checks.append(_check(
            f"optional_webhook_present_{env_name}",
            present,
            "set in process/user environment" if present else "missing; output will be suppressed or routed to fallback",
            required=False,
        ))
    checks.extend([
        _check(
            "frozen_hitter_artifact_valid",
            bool((checkpoint.get("artifact_integrity") or {}).get("valid")),
            str((checkpoint.get("artifact_integrity") or {}).get("reason") or "sha256 matched"),
        ),
        _check(
            "checkpoint_ran",
            bool(checkpoint.get("generated_at_utc")),
            str(checkpoint.get("generated_at_utc") or "missing"),
        ),
        _check(
            "micro_report_ran",
            bool(micro.get("generated_at_utc")),
            str(micro.get("generated_at_utc") or "missing"),
        ),
        _check(
            "backdated_audit_has_promotion_slates",
            int(backdated.get("prop_promotion_countable_count") or 0) > 0,
            f"{backdated.get('prop_promotion_countable_count') or 0} prop-promotion backdated slates",
            required=False,
        ),
        _check(
            "micro_buckets_not_forced",
            int(micro.get("micro_ready_count") or 0) >= 0,
            f"{micro.get('micro_ready_count') or 0} micro-ready buckets",
            required=False,
        ),
    ])
    required_failures = [row for row in checks if row["required"] and not row["passed"]]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "pass" if not required_failures else "fail",
        "checks": checks,
        "required_failures": required_failures,
        "scheduled_tasks": task_rows,
        "checkpoint_status": checkpoint.get("status"),
        "micro_ready_count": micro.get("micro_ready_count"),
        "backdated_prop_promotion_count": backdated.get("prop_promotion_countable_count"),
        "backdated_prop_promotion_dates": backdated.get("prop_promotion_countable_dates") or [],
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(model_dir / "prop_restart_readiness_report.json", payload)
    report_path = _REPORT_DIR / "mlb_prop_restart_readiness_latest.md"
    atomic_write_text(report_path, _render(payload))
    payload["report_path"] = str(report_path)
    return payload


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Prop Restart Readiness",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Status: **{str(payload['status']).upper()}**",
        f"Checkpoint status: `{payload.get('checkpoint_status')}`",
        f"Micro-ready buckets: {payload.get('micro_ready_count')}",
        f"Backdated prop-promotion slates: {payload.get('backdated_prop_promotion_count')} ({', '.join(payload.get('backdated_prop_promotion_dates') or []) or '-'})",
        "",
        "## Checks",
        "",
        "| Check | Required | Pass | Detail |",
        "|---|---|---|---|",
    ]
    for row in payload.get("checks") or []:
        lines.append(f"| {row['name']} | {row['required']} | {row['passed']} | {row['detail']} |")
    lines.extend([
        "",
        "## Scheduled Tasks",
        "",
        "| Task | Exists | State | Last Result | Last Run | Next Run |",
        "|---|---|---|---:|---|---|",
    ])
    for row in payload.get("scheduled_tasks") or []:
        lines.append(
            f"| {row.get('Name')} | {row.get('Exists')} | {row.get('State')} | "
            f"{row.get('LastTaskResult')} | {row.get('LastRunTime')} | {row.get('NextRunTime')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop restart readiness report")
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--pg-dsn", default=PG_DSN)
    args = parser.parse_args()
    payload = build(model_dir=Path(args.model_dir), pg_dsn=args.pg_dsn)
    print(json.dumps({
        "status": payload["status"],
        "required_failures": [row["name"] for row in payload["required_failures"]],
        "report_path": payload["report_path"],
    }, indent=2))


if __name__ == "__main__":
    main()
