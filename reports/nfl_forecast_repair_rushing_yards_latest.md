# NFL Isolated Forecast Repair

rushing_yards

| Variant | Mean RMSE | Bias | Median MAE | Brier | Calibration | Below / above 80% interval |
|---|---:|---:|---:|---:|---:|---:|
| efficiency | 26.496 | +0.671 | 17.658 | 0.2305 | 2.07% | 11.3% / 11.2% |
| joint | 26.079 | -0.187 | 17.471 | 0.2316 | 3.63% | 9.0% / 11.3% |
| joint_calibrated | 26.079 | -0.187 | 17.471 | 0.2316 | 3.63% | 9.0% / 11.3% |
| reference | 26.687 | -0.009 | 17.734 | 0.2338 | 3.36% | 10.1% / 9.9% |
| workload | 26.413 | -2.882 | 17.567 | 0.2330 | 4.46% | 5.2% / 13.1% |

## Decisions

efficiency: {"expected": false, "typical": false, "probability": false}
joint: {"expected": false, "typical": true, "probability": false}
joint_calibrated: {"expected": false, "typical": true, "probability": false}
workload: {"expected": false, "typical": false, "probability": false}

Historical development only; these seasons have already been examined.
One player-game or game per outcome; whole-week chronology and clustered uncertainty.
Proxy lines are not real offered lines. Reference is a chronological recipe, not hindsight live production.
Receiving pinned policy remains unchanged. No odds, ledgers, locks or Discord posts are rewritten.
Models, rate tuning, residual tails, CDF fitting and CDF gating use separate chronological blocks.
No current injuries or actual routes are invented; unavailable inputs remain missing.
Deployment requires later captured same-offer validation, independently of cash approval.
