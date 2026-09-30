# Independent Target Components

| Variant | Target RMSE | Yard RMSE | Brier | Calibration | 80% coverage |
|---|---:|---:|---:|---:|---:|
| combined | 2.262 | 24.279 | 0.2307 | 2.85% | 82.1% |
| reference | 2.298 | 24.380 | 0.2299 | 3.02% | 79.8% |
| targets_only | 2.262 | 24.279 | 0.2303 | 4.60% | 80.2% |
| uncertainty_only | 2.298 | 24.380 | 0.2301 | 1.60% | 81.7% |

| Variant | Targets | Expected yards | Typical yards | Probability |
|---|---|---|---|---|
| targets_only | rejected | rejected | rejected | rejected |
| uncertainty_only | rejected | rejected | historical_screen_passed | rejected |
| combined | rejected | rejected | historical_screen_passed | rejected |

Independent output screens. No probability or cash requirement for point forecasts.

Historical proxy-line screen, not offered-line or prospective acceptance.
Week 3 is development evidence only.
Efficiency is shared; no new efficiency head was trained.
No component is installed automatically. Failed probabilities cannot veto a separate point screen.
