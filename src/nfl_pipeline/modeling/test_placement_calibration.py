import json

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling import fit_placement_calibration as fit
from nfl_pipeline.modeling import predict_player_props as props

STATS = {"rushing_yards": {"QB": {"family": "scale", "k": 0.75}, "RB": {"family": "identity"}},
         "passing_yards": {"QB": {"family": "linear", "a": 9.6, "b": 1.0}}}


def test_scoring_applies_exactly_what_the_fitter_validated():
    metrics = {"placement_calibration": {"stats": STATS}}
    for stat, positions in STATS.items():
        for position, params in positions.items():
            for value in (0.0, 4.0, 31.5, 250.0):
                assert props._placed_projection(metrics, stat, {"position": position}, value) == \
                    pytest.approx(fit.apply(params, value))
    assert props._placed_projection(metrics, "rushing_yards", {"position": "QB"}, 20.0) == pytest.approx(15.0)
    assert props._placed_projection(metrics, "receiving_yards", {"position": "WR"}, 40.0) == 40.0  # not fitted
    assert props._placed_projection({}, "rushing_yards", {"position": "QB"}, 20.0) == 20.0          # no artifact


def test_missing_or_foreign_artifact_is_identity(tmp_path, monkeypatch):
    path = tmp_path / "placement_calibration.json"
    monkeypatch.setattr(props, "PLACEMENT_CALIBRATION_PATH", path)
    assert props._load_placement_calibration() == {}
    path.write_text(json.dumps({"contract": "something-else", "stats": STATS}))
    assert props._load_placement_calibration() == {}
    path.write_text(json.dumps({"contract": props.PLACEMENT_CALIBRATION_CONTRACT, "stats": STATS}))
    assert props._load_placement_calibration()["stats"] == STATS
    assert props.PLACEMENT_CALIBRATION_CONTRACT == fit.CONTRACT


def weeks(scale, n=40, start=2, count=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for week in range(start, start + count):
        p = rng.uniform(5, 60, n)
        rows += [dict(week=week, model_projection=v, actual=max(0.0, scale * v + rng.normal(0, 3))) for v in p]
    return pd.DataFrame(rows)


def test_a_projection_that_sits_too_high_every_week_is_scaled_down_on_held_out_weeks():
    result = fit.evaluate_group(weeks(0.75))
    assert result["status"] == "corrected" and result["heldout_weeks"] == [3, 4, 5, 6]
    assert result["heldout"][result["best_candidate"]]["mae"] < result["heldout"]["identity"]["mae"]
    assert abs(result["in_sample_after"]["above_share"] - 0.5) < 0.1


def test_a_well_placed_projection_is_left_alone():
    assert fit.evaluate_group(weeks(1.0))["chosen"] == {"family": "identity"}


def test_too_little_history_is_never_corrected():
    assert fit.evaluate_group(weeks(0.5, n=5, count=2))["chosen"] == {"family": "identity"}
