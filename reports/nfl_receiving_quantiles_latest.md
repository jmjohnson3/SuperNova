# Receiving Quantile Forecasts

| Variant | Mean RMSE | Bias | Median MAE | Brier | Calibration | 80% coverage |
|---|---:|---:|---:|---:|---:|---:|
| blend | 23.847 | +0.932 | 16.185 | 0.2249 | 3.39% | 82.9% |
| calibrated_blend | 23.864 | +0.753 | 16.149 | 0.2244 | 2.13% | 82.9% |
| conditional | 24.342 | +0.735 | 16.470 | 0.2267 | 2.55% | 79.1% |
| quantile | 23.797 | +1.129 | 16.154 | 0.2281 | 4.28% | 68.3% |
| reference | 24.246 | +0.126 | 16.541 | 0.2296 | 3.49% | 80.4% |

| Variant | Expected yards | Typical yards | Probability |
|---|---|---|---|
| quantile | not_accepted | historical_screen_passed | not_accepted |
| blend | not_accepted | historical_screen_passed | not_accepted |
| calibrated_blend | not_accepted | historical_screen_passed | not_accepted |

Chronological refit reference, not immutable historical production forecasts.
Proxy-line Brier is not actual offered-line or selected-pick accuracy.
Historical development screen, not an untouched prospective acceptance test.
Calibration transforms one distribution, not each offered line independently.
No production pointer, betting limits, rankings or fixed trial selections changed.
