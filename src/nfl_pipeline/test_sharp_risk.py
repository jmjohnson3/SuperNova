import json
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from nfl_pipeline import sharp_bets, sharp_risk

NOW = datetime(2026, 10, 28, 15, tzinfo=timezone.utc)
EXPERIMENT = dict(experiment_id="test-v10", sha256="current-hash")


def document(profit=0, low=0, **changes):
    doc = dict(built_at=NOW.isoformat(), summary=dict(EXPERIMENT, placed=dict(realized_profit=profit),
               readiness=dict(status="trial", gates=dict(loss_pause=dict(lowest_cumulative_profit=low)))))
    doc["summary"]["experiment_sha256"] = doc["summary"].pop("sha256")
    doc.update(changes)
    return doc


def evaluate(tmp_path, doc):
    report = tmp_path / "report.json"
    if doc is not None:
        report.write_text(json.dumps(doc), encoding="utf-8")
    elif report.exists():
        report.unlink()
    return sharp_risk.evaluate(now=NOW, report_path=report, state_path=tmp_path / "risk.json",
                               experiment=EXPERIMENT, loss_limit=-30, max_age_hours=24)


def test_new_trial_requires_valid_current_report(tmp_path):
    assert evaluate(tmp_path, None)["status"] == "paused"
    assert evaluate(tmp_path, document())["status"] == "trial"


@pytest.mark.parametrize("change", ["stale", "future", "cohort", "hash", "invalid_profit", "missing_gate"])
def test_invalid_readiness_never_authorizes_a_stake(tmp_path, change):
    doc = document()
    if change in {"stale", "future"}:
        doc["built_at"] = (NOW + timedelta(hours=-25 if change == "stale" else 1)).isoformat()
    elif change == "cohort":
        doc["summary"]["experiment_id"] = "old-v7"
    elif change == "hash":
        doc["summary"]["experiment_sha256"] = "other-hash"
    elif change == "invalid_profit":
        doc["summary"]["placed"]["realized_profit"] = "nan"
    else:
        doc["summary"]["readiness"] = {}
    assert evaluate(tmp_path, doc)["status"] == "paused"


def test_loss_pause_survives_missing_corrupt_replaced_and_recovered_reports(tmp_path):
    assert evaluate(tmp_path, document(profit=-31, low=-31))["loss_paused"]
    assert evaluate(tmp_path, None)["loss_paused"]
    assert evaluate(tmp_path, {})["loss_paused"]
    assert evaluate(tmp_path, document(profit=10, low=0))["status"] == "paused"
    assert json.loads((tmp_path / "risk.json").read_text())["loss_paused"] is True
    (tmp_path / "risk.json").unlink()
    assert evaluate(tmp_path, document())["loss_paused"]


def test_historical_loss_breach_and_corrupt_risk_state_stay_paused(tmp_path):
    assert evaluate(tmp_path, document(profit=10, low=-31))["loss_paused"]
    (tmp_path / "risk.json").write_text("broken", encoding="utf-8")
    assert evaluate(tmp_path, document())["status"] == "paused"
    assert (tmp_path / "risk.json").read_text() == "broken"


def test_late_confirmation_forwards_and_persists_actual_stake(monkeypatch):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.side_effect = [[dict(alert_id=41, game_id="g")], [dict(alert_id=41, stake=10)]]
    result = sharp_bets.confirm_late(conn, [41], -115, stake=10)
    assert result["confirmed"][0]["stake"] == 10
    sql, params = cur.execute.call_args.args
    assert "stake = COALESCE(%(stake)s, stake)" in sql and params["stake"] == 10
    connect = MagicMock()
    monkeypatch.setattr(sharp_bets.psycopg2, "connect", connect)
    confirm = MagicMock(return_value={})
    monkeypatch.setattr(sharp_bets, "confirm_late", confirm)
    monkeypatch.setattr(sys, "argv", ["sharp_bets", "--confirm", "41", "--late", "--stake", "10"])
    sharp_bets.main()
    assert confirm.call_args.kwargs["stake"] == 10
