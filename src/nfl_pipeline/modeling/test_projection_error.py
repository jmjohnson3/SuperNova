import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling import projection_error as pe


def frame(rows):
    base = dict(stat="receiving_yards", week=5, line=None, anchored_projection=None, position="WR",
                carries=0.0, pass_attempts=0.0, out_receptions=0.0, out_rushing_yards=0.0, out_passing_yards=0.0,
                receiver_projected_targets_v3=None, rb_projected_carries_v3=None, carries_avg_5=None,
                pass_attempts_avg_5=None, targets_avg_10=None, carries_avg_10=None)
    return pd.DataFrame([dict(base, **r) for r in rows])


def test_volume_and_efficiency_parts_add_up_to_the_error():
    df = pe.decompose(frame([
        dict(model_projection=40.0, targets_avg_5=5.0, targets=8.0, out_receiving_yards=50.0),   # more targets
        dict(model_projection=40.0, targets_avg_5=5.0, targets=5.0, out_receiving_yards=20.0),   # worse per target
        dict(model_projection=30.0, targets_avg_5=4.0, targets=0.0, out_receiving_yards=0.0),    # no volume at all
    ]), "receiving_yards")
    assert np.allclose(df.volume_part + df.efficiency_part, df.error)
    first, second, third = df.to_dict("records")
    assert first["volume_part"] == pytest.approx(24.0)        # 3 extra targets at the implied 8 yards each
    assert first["efficiency_part"] == pytest.approx(-14.0)   # 8 targets at 6.25 instead of 8
    assert second["volume_part"] == 0 and second["efficiency_part"] == pytest.approx(-20.0)
    assert second["knew_efficiency_error"] == pytest.approx(0.0)
    assert third["knew_volume_error"] == pytest.approx(0.0)   # knowing he saw no targets explains it all
    assert third["knew_efficiency_error"] == pytest.approx(-30.0)  # no rate to know: keeps the model's number


def test_rows_without_a_volume_estimate_are_scored_but_not_split():
    df = pe.decompose(frame([dict(model_projection=40.0, targets_avg_5=None, targets=5.0, out_receiving_yards=30.0),
                             dict(model_projection=20.0, targets_avg_5=4.0, targets=4.0, out_receiving_yards=30.0)]),
                      "receiving_yards")
    summary = pe.summarize_stat(df)
    assert summary["player_games"] == 2 and summary["decomposition"]["player_games"] == 1
    assert summary["bias"] == pytest.approx(0.0) and summary["above_share"] == pytest.approx(0.5)


def test_market_comparison_uses_only_priced_rows():
    df = pe.decompose(frame([dict(model_projection=40.0, targets_avg_5=5.0, targets=5.0, out_receiving_yards=30.0,
                                  line=35.5, anchored_projection=36.0),
                             dict(model_projection=20.0, targets_avg_5=4.0, targets=4.0, out_receiving_yards=30.0)]),
                      "receiving_yards")
    market = pe.summarize_stat(df)["vs_market"]
    assert market["player_games"] == 1
    assert market["model_mae"] == pytest.approx(10.0) and market["line_mae"] == pytest.approx(5.5)
    assert market["anchored_mae"] == pytest.approx(6.0)
