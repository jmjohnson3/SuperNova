from datetime import datetime, timedelta, timezone

from nfl_pipeline import pregame_checkpoints as checkpoints


NOW = datetime(2026, 10, 15, 20, 15, tzinfo=timezone.utc)


def game(minutes):
    return {
        "game_id": "g1",
        "start_ts_utc": NOW + timedelta(minutes=120),
        "minutes_to_start": minutes,
    }


def valid_verification():
    return {
        "game_id": "g1",
        "fresh_normalized_rows": 8,
        "fresh_true_pair_rows": 8,
        "fresh_true_pair_rows_with_links": 8,
        "current_fanduel_locks": 12,
        "receiving_locks": 5,
        "receiving_same_line_triplets": 5,
        "explicit_injury_context_locks": 12,
        "known_player_injury_context_locks": 3,
        "immutable_lock_violations": 0,
        "all_locks_missing_links": 0,
        "actionable_missing_links": 0,
        "verification_passed": True,
    }


def test_four_checkpoints_are_distinct_and_forward_only():
    state = {"contract": checkpoints.CONTRACT, "games": {}}
    first = checkpoints.plan([game(120)], state, NOW)
    assert first["due"] == {"g1": 120}
    checkpoints.record_attempt(state, [game(120)], first["due"], {"g1": valid_verification()}, NOW)

    assert checkpoints.plan([game(115)], state, NOW)["due"] == {}
    second = checkpoints.plan([game(60)], state, NOW)
    assert second["due"] == {"g1": 60}
    checkpoints.record_attempt(state, [game(60)], second["due"], {"g1": valid_verification()}, NOW)
    assert checkpoints.plan([game(20)], state, NOW)["due"] == {"g1": 20}
    checkpoints.record_attempt(state, [game(20)], {"g1": 20}, {"g1": valid_verification()}, NOW)
    assert checkpoints.plan([game(5)], state, NOW)["due"] == {"g1": 5}


def test_failed_verification_retries_only_inside_same_window():
    state = {"contract": checkpoints.CONTRACT, "games": {}}
    planned = checkpoints.plan([game(60)], state, NOW)
    bad = valid_verification()
    bad.update(fresh_true_pair_rows=7, verification_passed=False)
    checkpoints.record_attempt(state, [game(60)], planned["due"], {"g1": bad}, NOW)
    assert checkpoints.plan([game(55)], state, NOW)["due"] == {"g1": 60}
    row = checkpoints.plan([game(49)], state, NOW)["observations"][0]
    assert row["due_offset"] is None
    assert next(c for c in row["checkpoints"] if c["offset_minutes"] == 60)["status"] == "missed"


def test_checkpoint_verification_requires_pairs_context_and_immutable_locks():
    row = valid_verification()
    assert checkpoints.lock_verification_passed(row)
    assert checkpoints.verification_passed(row)
    for key, value in (
        ("fresh_true_pair_rows", 0),
        ("current_fanduel_locks", 0),
        ("explicit_injury_context_locks", 11),
        ("receiving_same_line_triplets", 4),
        ("immutable_lock_violations", 1),
        ("all_locks_missing_links", 1),
        ("actionable_missing_links", 1),
    ):
        broken = dict(row, **{key: value})
        assert not checkpoints.verification_passed(broken)


def test_checkpoint_close_evidence_does_not_require_provider_links():
    row = valid_verification()
    row["fresh_true_pair_rows_with_links"] = 7
    assert checkpoints.verification_passed(row)


def test_checkpoint_requires_every_refreshed_exact_line_to_be_true_paired():
    row = valid_verification()
    row["fresh_normalized_rows"] = 9
    assert not checkpoints.verification_passed(row)
