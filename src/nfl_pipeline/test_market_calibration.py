

def test_anchored_yardage_projection_is_the_median_not_the_mean():
    # Right-skewed empirical residuals (like rushing): pinned to the line as a mean, most mass falls under it.
    import numpy as np
    from nfl_pipeline.modeling import predict_player_props as props
    resid = np.random.default_rng(0).exponential(10.0, 4000) - 10.0
    dist = dict(kind="empirical_oof_residual", residual_sigma=10.0,
                residual_quantiles=[float(q) for q in np.quantile(resid, np.linspace(0, 1, 201))])
    as_mean = props._line_outcomes("rushing_yards", 20.5, 20.5, {}, dist, None)
    assert as_mean[0] < 0.45
    p50 = float(props._projection_distribution_summary("rushing_yards", 20.5, {}, dist, row=None)["projection_p50"])
    as_median = props._line_outcomes("rushing_yards", 20.5, 20.5, {}, dist, None, center=p50, shift=20.5 - p50)
    assert abs(as_median[0] / (as_median[0] + as_median[1]) - 0.5) < 0.02


def test_asymmetric_tail_scaling_changes_the_matching_side_only():
    from nfl_pipeline.modeling import predict_player_props as props
    dist = dict(kind="normal_residual", residual_sigma=10.0, residual_bias=0.0,
                accepted_distribution=True)
    lower_base = props._line_outcomes("receiving_yards", 50.0, 40.5, {}, dist, center=50.0)
    lower_wide = props._line_outcomes(
        "receiving_yards", 50.0, 40.5, {}, dist, center=50.0, lower_tail_scale=1.5
    )
    upper_base = props._line_outcomes("receiving_yards", 50.0, 60.5, {}, dist, center=50.0)
    upper_wide = props._line_outcomes(
        "receiving_yards", 50.0, 60.5, {}, dist, center=50.0, upper_tail_scale=1.5
    )
    assert lower_wide[0] < lower_base[0]
    assert upper_wide[0] > upper_base[0]


def test_asymmetric_summary_uses_separate_scales():
    from nfl_pipeline.modeling import predict_player_props as props
    summary = dict(projection_p10=10.0, projection_p50=20.0, projection_p90=40.0,
                   projection_confidence=0.5)
    widened = props._widen_summary(summary, 1.25, lower_tail_scale=1.2, upper_tail_scale=1.4)
    assert widened["projection_p10"] == 5.0
    assert widened["projection_p90"] == 55.0


def test_legacy_probability_trust_fails_closed(monkeypatch, tmp_path):
    import json
    from nfl_pipeline.modeling import predict_player_props as props
    path = tmp_path / "market.json"
    path.write_text(json.dumps({"contract": "legacy", "stats": {
        "rushing_yards": {"projection_trust": 0.0, "range_scale": 1.0, "probability_trust": 1.0}
    }}))
    monkeypatch.setattr(props, "MARKET_CALIBRATION_PATH", path)
    loaded = props._load_market_calibration()
    assert loaded["stats"]["rushing_yards"]["probability_trust"] == 0.0
    assert loaded["probability_compatibility"]["disabled_stats"] == ["rushing_yards"]


def test_interval_calibration_is_release_bound():
    from nfl_pipeline.modeling import predict_player_props as props
    metrics = {
        "live_model_version": "new-release",
        "interval_calibration": {
            "model_versions": ["old-release"],
            "stats": {"receiving_yards": {"lower_tail_scale": 1.2, "upper_tail_scale": 1.3}},
        },
    }
    params = props._market_calibration_params(metrics, "receiving_yards")
    assert params["lower_tail_scale"] == params["upper_tail_scale"] == 1.0
