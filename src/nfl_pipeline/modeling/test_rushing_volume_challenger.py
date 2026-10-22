import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling import rushing_volume_challenger as rc

REGISTERED = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
KICK = datetime(2026, 10, 25, 17, tzinfo=timezone.utc)
REG = dict(registered_at=REGISTERED.isoformat(), production_release="rel")


class Booster:
    def predict(self, X):
        return np.full(len(X), 12.0)


MODEL = dict(booster=Booster(), features=["carries_avg_5"], positions=("RB", "QB"))


def record(fid=1, position="RB", created=REGISTERED + timedelta(hours=1), **payload):
    base = dict(game_id="g", player_id="p", position=position, season=2026, week=8, model_version="rel",
                forecast_outputs={"pricing": {"model_projection": 50.0}}, forecast_features={"carries_avg_5": 10.0},
                line=48.5, projection=48.5)
    return fid, dict(base, **payload), dict(start_ts_utc=KICK, created_at_utc=created)


def test_candidate_rescales_the_raw_model_projection_by_model_carries():
    rows, excluded = rc.score_records([record()], REG, MODEL, KICK - timedelta(hours=2))
    assert not excluded and rows[0]["reference"] == 50.0  # the model's number, not the anchored 48.5
    assert rows[0]["carries_model"] == 12.0 and rows[0]["candidate"] == pytest.approx(60.0)


@pytest.mark.parametrize("change,reason", [
    (dict(position="WR"), "position_outside_model"),
    (dict(created=REGISTERED - timedelta(hours=1)), "not_post_registration_pregame_forecast"),
    (dict(model_version="other"), "different_production_release"),
    (dict(forecast_outputs={}, line=48.5), "missing_raw_model_projection"),
    (dict(forecast_features={}), "missing_archived_features"),
    (dict(forecast_features={"carries_avg_5": 0.0}), "no_carry_history"),
])
def test_unusable_forecasts_are_counted_not_scored(change, reason):
    created = change.pop("created", REGISTERED + timedelta(hours=1))
    rows, excluded = rc.score_records([record(created=created, **change)], REG, MODEL, KICK - timedelta(hours=2))
    assert rows == [] and excluded == {reason: 1}


def test_an_unpriced_forecast_uses_its_projection_and_kickoff_ends_capture():
    unpriced = record(forecast_outputs={}, line=None, projection=40.0)
    assert rc.score_records([unpriced], REG, MODEL, KICK - timedelta(hours=2))[0][0]["reference"] == 40.0
    assert rc.score_records([record()], REG, MODEL, KICK)[1] == {"not_post_registration_pregame_forecast": 1}


def test_grading_keeps_the_last_capture_before_kickoff(tmp_path):
    registration = dict(REG, minimum_weeks=3)
    def write(captured, candidate):
        row = dict(game_id="g", player_id="p", season=2026, week=8, kickoff=KICK.isoformat(),
                   captured_at=captured.isoformat(), reference=50.0, candidate=candidate)
        path = tmp_path / "prospective" / "2026-10-25" / (captured.strftime("%H%M%S") + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(registration=registration, rows=[row])))
    write(KICK - timedelta(hours=5), 55.0)
    write(KICK - timedelta(hours=1), 58.0)
    write(KICK + timedelta(minutes=5), 99.0)  # after kickoff: never graded
    frame, excluded = rc.load_captures(tmp_path, registration)
    assert frame.candidate.tolist() == [58.0] and excluded["earlier_capture_of_same_player_game"] == 1
