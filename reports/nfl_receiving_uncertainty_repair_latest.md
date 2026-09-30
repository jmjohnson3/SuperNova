# Receiving Uncertainty Repair

Status: not_accepted
Offline only. Production and the pinned prospective trial were not changed.

| Variant | Brier | Calibration | 80% coverage | Interval score | Mean RMSE | Median MAE |
|---|---:|---:|---:|---:|---:|---:|
| conditional | 0.2283 | 0.93% | 79.6% | 81.94 | 24.97 | 17.09 |
| gated_tail | 0.2282 | 1.16% | 79.6% | 81.78 | 24.97 | 17.09 |
| reference | 0.2309 | 2.42% | 80.3% | 91.26 | 24.90 | 17.13 |
| tail_candidate | 0.2281 | 1.24% | 79.7% | 81.60 | 24.95 | 17.09 |

Gated tail vs conditional, week-grouped Brier gain: {"clusters": 46, "lower_95": -8.935467898551118e-05, "upper_95": 0.00015945132684910558, "mean_gain": 4.11193080092321e-05}

Historical refit uses lagged final player-games, not archived live lock-time features.
Proxy lines are not true offered-line or betting-edge proof.
Pinned receiving trial and production artifacts are untouched.
Raw tail_candidate uses tuning strength even if the separate selection gate rejected it.
