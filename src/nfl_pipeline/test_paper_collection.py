"""Offline safety checks for the explicit prospective paper runner."""
from datetime import date, datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from nfl_pipeline import paper_collection as runner

DAY = date(2026, 10, 19)
NOW = datetime(2026, 10, 19, 12, tzinfo=timezone.utc)
FROZEN = dict(production_release="release", experiments={"receiving": {"experiment_id": "dynamic-successor"}})
GAMES = [dict(game_id="2026_06_SF_WAS", start_ts_utc=(NOW+timedelta(hours=8)).isoformat())]


def harness(*, guard=lambda: FROZEN, games=lambda *args: GAMES, execute=None):
    called = []; saved = []
    def run(step, env):
        called.append((step, env))
        if execute:
            return execute(step, env)
        value = dict(status="ok")
        if step.name == "fresh_quotes":
            value = dict(ready_for_scoring=True, quote_health={"props": {"fresh_rows": 2}})
        if step.name == "paper_capture":
            value = dict(status="paper_only", captured=0, selected=0,
                exclusions={"missing_quote_provider": 9}, betting_approved=False)
        return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr="secret provider URL")
    result = runner.collect(DAY, guard=guard, games=games, execute=run, save=saved.append, now=lambda: NOW)
    return result, called, saved


def test_plan_has_only_exact_paper_steps_and_never_runs(monkeypatch):
    monkeypatch.setattr(runner, "collect", lambda *args: pytest.fail("Plan must not collect"))
    monkeypatch.setattr(runner.sys, "argv", ["paper_collection", "--date", str(DAY)])
    assert runner.main() == 0
    plan = runner.plan(DAY)
    assert [s["name"] for s in plan["steps"]] == ["context", "live_injuries", "fresh_quotes", "game_forecasts",
        "team_context", "player_forecasts", "paper_capture"]
    assert sum(s["module"] == "nfl_pipeline.refresh_lock_quotes" for s in plan["steps"]) == 1
    assert not any(any(word in s["module"] for word in ("lock_ledger", "publish", "cash_", "train", "run_daily")) for s in plan["steps"])


def test_invalid_freeze_stops_before_provider_or_forecast_writes():
    def broken():
        raise ValueError("DSN=private_secret")
    result, called, saved = harness(guard=broken)
    assert not called and saved == [result]
    assert result["status"] == "blocked"
    assert "private_secret" not in json.dumps(result)


def test_no_upcoming_games_never_refreshes_quotes():
    result, called, _ = harness(games=lambda *args: [])
    assert not called and result["status"] == "no_upcoming_games"


def test_zero_capture_is_recorded_with_real_exclusions_and_scoped_env(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "private_hook")
    monkeypatch.setenv("DISCORD_FORMAT", "1")
    result, called, saved = harness()
    assert result["status"] == "completed_no_eligible_captures"
    assert result["capture_result"]["exclusions"] == {"missing_quote_provider": 9}
    assert len(called) == 7 and len(saved) == 1
    for _, env in called:
        assert json.loads(env["NFL_RUN_GAME_IDS"]) == [GAMES[0]["game_id"]]
        assert env["NFL_MODEL_RELEASE_ID"] == "release" and env["DISCORD_FORMAT"] == "0"
        assert "DISCORD_WEBHOOK_URL" not in env
    assert not result["cash_authorized"] and not result["notifications_sent"]
    assert "private_hook" not in json.dumps(result) and "secret provider" not in json.dumps(result)


def test_freeze_identity_cannot_change_between_steps():
    calls = [0]
    def guard():
        calls[0] += 1
        return FROZEN if calls[0] < 3 else dict(FROZEN, production_release="different")
    result, called, _ = harness(guard=guard)
    assert result["status"] == "blocked" and len(called) == 1


def test_no_fresh_props_prevents_predictions_even_if_game_quotes_fresh():
    def execute(step, env):
        return SimpleNamespace(returncode=0, stdout=json.dumps(dict(status="ok", ready_for_scoring=True,
            quote_health={"games": {"fresh_rows": 2}, "props": {"fresh_rows": 0}})))
    result, called, _ = harness(execute=execute)
    assert result["status"] == "failed" and len(called) == 3
    assert result["steps"][-1]["reason"] == "no_fresh_prop_quotes"


@pytest.mark.parametrize("kind", ["nonzero", "malformed", "timeout"])
def test_child_failures_stop_without_raw_output_leak(kind):
    def execute(step, env):
        if kind == "timeout":
            raise runner.subprocess.TimeoutExpired("secret command", 1, output="private output")
        return SimpleNamespace(returncode=1 if kind == "nonzero" else 0, stdout="private output")
    result, called, _ = harness(execute=execute)
    assert len(called) == 1 and result["status"] == "failed"
    assert "private" not in json.dumps(result) and "secret command" not in json.dumps(result)


def test_receipts_preserve_individual_attempts_and_dated_latest(tmp_path):
    first, _, _ = harness()
    runner.save_receipt(first, tmp_path)
    second = dict(first, run_id="second", status="blocked")
    runner.save_receipt(second, tmp_path)
    assert json.loads((tmp_path/f"nfl_paper_collection_{DAY}_{first['run_id']}.json").read_text()) == first
    assert json.loads((tmp_path/"nfl_paper_collection_latest.json").read_text()) == second
    with pytest.raises(ValueError, match="already exists"):
        runner.save_receipt(first, tmp_path)


def test_run_uses_existing_mutex_with_no_busy_wait(monkeypatch):
    called=[]
    monkeypatch.setattr(runner.os, "name", "nt")
    monkeypatch.setattr(runner.sys, "argv", ["paper_collection", "--date", str(DAY), "--run", "--game-id", GAMES[0]["game_id"]])
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: (called.append(command) or SimpleNamespace(returncode=75)))
    monkeypatch.setattr(runner, "collect", lambda *args: pytest.fail("Parent must enter mutex first"))
    assert runner.main() == 75
    command=called[0]
    assert command[command.index("-WaitSeconds")+1] == "0"
    assert "run_with_nfl_mutex.ps1" in command[command.index("-File")+1]
    assert "--_mutex-child" in command[-1] and GAMES[0]["game_id"] in command[-1]
