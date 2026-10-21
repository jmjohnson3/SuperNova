"""Collect prospective NFL paper evidence without publishing or reserving wagers.

The default command is a read-only plan. ``--run`` enters the existing Windows
NFL operational mutex before any pipeline step. A run uses the currently active
immutable experiments; it never registers, trains, promotes, or rolls them over.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid

from nfl_pipeline.integrity import nfl_season

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
TOKEN = re.compile(r"[A-Za-z0-9_:-]{1,120}")


@dataclass(frozen=True)
class Step:
    name: str
    module: str
    args: tuple[str, ...]
    timeout_seconds: int


def steps(day):
    when = ("--date", str(day))
    season = nfl_season(day)
    return (
        Step("context", "nfl_pipeline.import_context", ("--seasons", str(season)), 600),
        Step("live_injuries", "nfl_pipeline.live_injuries", when, 180),
        Step("fresh_quotes", "nfl_pipeline.refresh_lock_quotes", when, 600),
        Step("game_forecasts", "nfl_pipeline.modeling.predict_today", when, 600),
        Step("team_context", "nfl_pipeline.modeling.score_accuracy_components", (*when, "--capture-team-context"), 300),
        Step("player_forecasts", "nfl_pipeline.modeling.predict_player_props", when, 1200),
        Step("paper_capture", "nfl_pipeline.modeling.receiving_paper_trial", ("capture", *when), 360),
    )


def plan(day, game_ids=()):
    return dict(status="plan_only", date=str(day), game_ids=list(game_ids),
        game_scope="requested_upcoming_games" if game_ids else "all_upcoming_games_on_date",
        steps=[dict(asdict(step), args=list(step.args)) for step in steps(day)],
        quote_refresh=dict(max_invocations=1, provider_configuration="existing_configured_provider_order",
            note="Existing provider market coverage, retries and quotas apply; this runner does not increase them."),
        execution="--run enters scripts/run_with_nfl_mutex.ps1; busy mutex exits without pipeline steps",
        guards="Active receiving, sharp, and cash freezes are verified before each step and at completion",
        paper_only=True, cash_authorized=False, wagers_placed=False,
        ledger_reservations=False, notifications_sent=False, retraining=False)


def guard_freezes():
    # Resolve active identifiers at call time so explicit successor registrations
    # work without changing this orchestration module or accepting old evidence.
    from nfl_pipeline import experiment_registry, cash_freeze, cash_trial_policy, sharp_watch
    from nfl_pipeline.modeling import receiving_paper_trial
    paper = experiment_registry.check_experiment(receiving_paper_trial.NAME)
    receiving_paper_trial._check_candidate(paper)
    sharp = experiment_registry.check_experiment(sharp_watch.EXPERIMENT_NAME)
    cash = cash_freeze.check_registration(cash_trial_policy.load_registration())
    release = json.loads((ROOT / "src/nfl_pipeline/modeling/models/active_release.json").read_text(encoding="utf-8"))["release_id"]
    if not isinstance(release, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", release):
        raise ValueError("Invalid active release identity")
    if paper["config"]["production_release"] != release:
        raise ValueError("Paper experiment production release differs")
    return dict(production_release=release, experiments={name: dict(
        experiment_id=doc["experiment_id"], sha256=doc["sha256"])
        for name, doc in (("receiving", paper), ("sharp", sharp), ("cash", cash))})


def upcoming_games(day, requested=()):
    import psycopg2
    from nfl_pipeline.db import PG_DSN
    with psycopg2.connect(PG_DSN, connect_timeout=10) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='30s'")
            cur.execute("""SELECT game_id,start_ts_utc FROM raw.nfl_games
                WHERE game_date_et=%s AND start_ts_utc>NOW() AND status!='final'
                AND (%s IS NULL OR game_id=ANY(%s)) ORDER BY start_ts_utc,game_id""",
                (day, list(requested) or None, list(requested) or None))
            return [dict(game_id=game, start_ts_utc=start.isoformat()) for game, start in cur.fetchall()]


def child_environment(game_ids, release):
    env = dict(os.environ)
    # These modules have no notification entry point; also prevent display-only
    # Discord formatting from replacing the structured JSON result.
    env.pop("DISCORD_WEBHOOK_URL", None)
    env["DISCORD_FORMAT"] = "0"
    env["NFL_RUN_GAME_IDS"] = json.dumps(list(game_ids))
    env["NFL_MODEL_RELEASE_ID"] = release
    return env


def execute_step(step, env):
    return subprocess.run([sys.executable, "-B", "-m", step.module, *step.args],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=step.timeout_seconds, check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def result_summary(stdout):
    """Whitelist aggregate results; provider URLs, credentials and raw logs stay out."""
    doc = json.loads(stdout)
    if not isinstance(doc, dict):
        raise ValueError("Expected a structured module result")
    source = doc.get("meta", doc)
    if not isinstance(source, dict):
        raise ValueError("Expected structured forecast metadata")
    result = {}
    for key in ("status", "provider", "provider_status"):
        value = source.get(key)
        if isinstance(value, str) and TOKEN.fullmatch(value):
            result[key] = value
    for key in ("saved", "predictions", "offers", "snapshot_players", "captured", "selected", "teams",
                "roster_rows", "roster_rows_upserted", "depth_rows_upserted", "injury_rows_upserted",
                "ready_for_scoring", "api_called", "betting_approved"):
        value = source.get(key)
        if isinstance(value, (int, float, bool)):
            result[key] = value
    for key in ("games", "unmatched"):
        value = source.get(key)
        if isinstance(value, list):
            result[key + "_count"] = len(value)
        elif isinstance(value, int):
            result[key + "_count"] = value
    exclusions = source.get("exclusions")
    if isinstance(exclusions, dict):
        result["exclusions"] = {k: v for k, v in exclusions.items()
            if isinstance(k, str) and TOKEN.fullmatch(k) and isinstance(v, int) and v >= 0}
    if isinstance(source.get("sources"), list):
        result["source_errors"] = sum(bool(row.get("error")) for row in source["sources"] if isinstance(row, dict))
    health = source.get("quote_health")
    if isinstance(health, dict):
        result["quote_health"] = {kind: {key: data[key] for key in ("observed_rows", "fresh_rows")
            if isinstance(data.get(key), int)} for kind, data in health.items()
            if kind in ("props", "games") and isinstance(data, dict)}
    return result


def step_problem(step, result):
    if step.name == "fresh_quotes" and (result.get("ready_for_scoring") is not True
            or result.get("quote_health", {}).get("props", {}).get("fresh_rows", 0) <= 0):
        return "no_fresh_prop_quotes"
    if step.name == "paper_capture" and (result.get("status") != "paper_only"
            or result.get("betting_approved") is not False):
        return "paper_capture_not_verified"
    if step.name == "context" and result.get("source_errors", 0):
        return "context_source_error"
    if result.get("status") in ("failed", "error", "not_registered"):
        return "module_reported_failure"
    return None


def _write_json(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        name = stream.name
        try:
            json.dump(doc, stream, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            stream.close(); Path(name).unlink(missing_ok=True)
            raise
    try:
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def save_receipt(doc, folder=REPORTS):
    folder = Path(folder)
    path = folder / f"nfl_paper_collection_{doc['date']}_{doc['run_id']}.json"
    # UUID run IDs preserve separate attempts, including failed and zero-capture runs.
    if path.exists():
        raise ValueError("Paper collection receipt already exists")
    _write_json(path, doc)
    for suffix in (doc["date"], "latest"):
        _write_json(folder / f"nfl_paper_collection_{suffix}.json", doc)


def collect(day, requested=(), *, guard=guard_freezes, games=upcoming_games,
            execute=execute_step, save=save_receipt, now=lambda: datetime.now(timezone.utc)):
    """Run inside the operational mutex; injectable functions keep tests offline."""
    doc = dict(plan(day, requested), run_id=uuid.uuid4().hex, status="running",
        started_at=now().isoformat(), ended_at=None, steps=[], capture_result=None,
        requested_game_ids=list(requested))
    try:
        frozen = guard()
        doc["freeze_verification"] = frozen
        selected = games(day, requested)
        doc["game_ids"] = [row["game_id"] for row in selected]
        doc["games"] = selected
        if not selected:
            doc["status"] = "no_upcoming_games"
        else:
            env = child_environment(doc["game_ids"], frozen["production_release"])
            for step in steps(day):
                if guard() != frozen:
                    raise ValueError("Active experiment identity changed during run")
                if all(datetime.fromisoformat(g["start_ts_utc"]) <= now() for g in selected):
                    doc["status"] = "no_upcoming_games_remaining"
                    break
                entry = dict(asdict(step), args=list(step.args), started_at=now().isoformat(), ended_at=None, status="running")
                doc["steps"].append(entry)
                try:
                    process = execute(step, env)
                    entry["exit_code"] = int(process.returncode)
                    if process.returncode:
                        entry.update(status="failed", reason="module_exit_nonzero")
                    else:
                        entry["result"] = result_summary(process.stdout)
                        problem = step_problem(step, entry["result"])
                        entry.update(status="failed" if problem else "completed")
                        if problem:
                            entry["reason"] = problem
                        if step.name == "paper_capture":
                            doc["capture_result"] = entry["result"]
                except subprocess.TimeoutExpired:
                    entry.update(status="failed", reason="module_timeout")
                except (OSError, ValueError, TypeError):
                    entry.update(status="failed", reason="module_result_unavailable")
                finally:
                    entry["ended_at"] = now().isoformat()
                if entry["status"] != "completed":
                    doc["status"] = "failed"
                    break
            else:
                if guard() != frozen:
                    raise ValueError("Active experiment identity changed during run")
                count = (doc["capture_result"] or {}).get("captured", 0)
                doc["status"] = "completed" if count else "completed_no_eligible_captures"
    except Exception as exc:
        # Do not put provider exceptions, connection strings or raw tracebacks in a report.
        doc.update(status="blocked", reason="preflight_or_dependency_check_failed", error_type=type(exc).__name__)
    finally:
        doc["ended_at"] = now().isoformat()
        save(doc)
    return doc


def mutex_command(day, game_ids):
    command = [sys.executable, "-B", "-m", "nfl_pipeline.paper_collection", "--date", str(day), "--_mutex-child"]
    for game in game_ids:
        command.extend(("--game-id", game))
    return ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", str(ROOT / "scripts/run_with_nfl_mutex.ps1"), "-WaitSeconds", "0",
        "-CommandText", subprocess.list2cmdline(command)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--game-id", action="append", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Execute once under the NFL operational mutex")
    mode.add_argument("--_mutex-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    from nfl_pipeline.game_scope import validate_ids
    selected = validate_ids(args.game_id) if args.game_id else []
    if args.run:
        if os.name != "nt":
            parser.error("Live collection requires the existing Windows operational mutex")
        return subprocess.run(mutex_command(args.date, selected), cwd=ROOT, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode
    doc = collect(args.date, selected) if args._mutex_child else plan(args.date, selected)
    print(json.dumps(doc, indent=2))
    return 1 if doc["status"] in ("failed", "blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
