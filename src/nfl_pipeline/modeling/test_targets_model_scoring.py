import json

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling import predict_player_props as props
from nfl_pipeline.modeling import volume_model


class Booster:
    def predict(self, X):
        assert list(X.columns) == ["targets_avg_5", "is_home", "pos_WR", "pos_TE", "pos_RB"]
        return np.full(len(X), 9.0)


LOADED = dict(run_id="r", sha256="s", model=dict(booster=Booster(), features=["targets_avg_5", "is_home"],
                                                   positions=("WR", "TE", "RB")))


def snapshot():
    return pd.DataFrame(dict(position=["WR", "QB"], targets_avg_5=[5.0, 0.0], is_home=[True, False],
                             receptions_avg_3=[4.0, 0.0], receptions_avg_5=[4.0, 0.0], receptions_avg_10=[4.0, 0.0],
                             targets_avg_10=[5.0, 0.0], receiver_projected_targets_v3=[4.0, np.nan]))


def test_only_receivers_get_model_targets_and_booleans_are_numeric():
    out = props._attach_targets_model(snapshot(), LOADED)
    assert out._pred_targets_model.iloc[0] == 9.0 and pd.isna(out._pred_targets_model.iloc[1])
    assert "_pred_targets_model" not in snapshot()  # the input frame is not mutated


def test_receptions_use_model_targets_and_fall_back_to_v3():
    base = props._receptions_projection(snapshot())
    swapped = props._receptions_projection(props._attach_targets_model(snapshot(), LOADED))
    assert swapped[0] > base[0]  # 9 model targets instead of 4 from v3
    assert props._receptions_projection(props._attach_targets_model(snapshot(), None))[0] == base[0]


def test_pointer_must_match_its_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(props, "TARGETS_MODEL_POINTER", tmp_path / "targets_model.json")
    assert props._load_targets_model() is None
    (tmp_path / "targets_model.json").write_text(json.dumps(dict(contract=props.TARGETS_MODEL_CONTRACT,
                                                                 run_id="missing-run", sha256="x")))
    assert props._load_targets_model() is None  # artifact not on disk: fall back, never fail scoring
    assert props.TARGETS_MODEL_CONTRACT == volume_model.TARGETS_POINTER_CONTRACT
