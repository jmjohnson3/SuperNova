# NFL Isolated Forecast Repair

passing_yards

| Variant | Mean RMSE | Bias | Median MAE | Brier | Calibration | Below / above 80% interval |
|---|---:|---:|---:|---:|---:|---:|
| efficiency | 89.626 | +0.532 | 71.167 | 0.2134 | 2.12% | 11.0% / 9.9% |
| joint | 85.394 | -1.148 | 67.738 | 0.2025 | 2.93% | 8.3% / 9.0% |
| joint_calibrated | 85.273 | -0.494 | 67.566 | 0.2020 | 2.80% | 8.6% / 9.0% |
| reference | 88.520 | +4.107 | 69.684 | 0.2075 | 2.11% | 12.5% / 10.0% |
| workload | 86.024 | -0.171 | 67.793 | 0.2025 | 2.47% | 7.4% / 9.2% |

## Decisions

efficiency: {"expected": false, "typical": false, "probability": false}
joint: {"expected": true, "typical": true, "probability": false}
joint_calibrated: {"expected": true, "typical": true, "probability": false}
workload: {"expected": true, "typical": false, "probability": false}

Historical development only; these seasons have already been examined.
One player-game or game per outcome; whole-week chronology and clustered uncertainty.
Proxy lines are not real offered lines. Reference is a chronological recipe, not hindsight live production.
Receiving pinned policy remains unchanged. No odds, ledgers, locks or Discord posts are rewritten.
Models, rate tuning, residual tails, CDF fitting and CDF gating use separate chronological blocks.
No current injuries or actual routes are invented; unavailable inputs remain missing.
Deployment requires later captured same-offer validation, independently of cash approval.
