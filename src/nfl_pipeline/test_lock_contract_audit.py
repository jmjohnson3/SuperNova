from datetime import date, datetime, timezone

from nfl_pipeline import lock_contract_audit as audit


DAY = date(2026, 10, 15)


def _game():
    return {
        "game_id": "g1",
        "home_team_abbr": "DEN",
        "away_team_abbr": "SEA",
        "start_ts_utc": datetime(2026, 10, 16, 0, 15, tzinfo=timezone.utc),
        "status": "scheduled",
    }


def _verification(**changes):
    row = {
        "game_id": "g1",
        "current_fanduel_locks": 12,
        "receiving_locks": 5,
        "receiving_same_line_triplets": 5,
        "explicit_injury_context_locks": 12,
        "known_player_injury_context_locks": 3,
        "immutable_lock_violations": 0,
        "all_locks_missing_links": 0,
        "actionable_missing_links": 0,
    }
    row.update(changes)
    return row


def test_required_lock_contract_passes_only_complete_same_line_capture(monkeypatch):
    monkeypatch.setattr(audit, "scoped_game_ids", lambda: ["g1"])
    monkeypatch.setattr(audit, "_games", lambda day, scope: [_game()])
    monkeypatch.setattr(audit, "verify_locks", lambda day, ids: {"g1": _verification()})
    report = audit.build_report(DAY, required=True)
    assert report["status"] == "passed"
    assert report["summary"] == {"games": 1, "passed": 1, "waiting": 0, "failed": 0}


def test_missing_receiving_challenger_fails_required_pregame(monkeypatch):
    monkeypatch.setattr(audit, "scoped_game_ids", lambda: ["g1"])
    monkeypatch.setattr(audit, "_games", lambda day, scope: [_game()])
    monkeypatch.setattr(
        audit,
        "verify_locks",
        lambda day, ids: {"g1": _verification(receiving_same_line_triplets=4)},
    )
    assert audit.build_report(DAY, required=True)["status"] == "failed"


def test_morning_audit_waits_but_pregame_requires_locks(monkeypatch):
    monkeypatch.setattr(audit, "scoped_game_ids", lambda: None)
    monkeypatch.setattr(audit, "_games", lambda day, scope: [_game()])
    monkeypatch.setattr(audit, "verify_locks", lambda day, ids: {"g1": _verification(
        current_fanduel_locks=0,
        receiving_locks=0,
        receiving_same_line_triplets=0,
        explicit_injury_context_locks=0,
    )})
    assert audit.build_report(DAY, required=False)["status"] == "waiting"
    assert audit.build_report(DAY, required=True)["status"] == "failed"
