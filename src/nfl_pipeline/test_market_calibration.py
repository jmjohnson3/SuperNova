

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
