# Workload and Uncertainty Repair

Offline experiments only; no production or cash change.

| Stat / variant | MAE | Baseline MAE | Opportunity MAE | Brier | Calibration | 80% coverage |
|---|---:|---:|---:|---:|---:|---:|
| receiving_yards / existing_workload | 17.321 | 17.560 | 1.745 | 0.2419 | 7.86% | 78.1% |
| receiving_yards / conditional_efficiency | 17.594 | 17.560 | 1.794 | 0.2472 | 8.64% | 78.6% |
| receiving_yards / asymmetric_repair | 17.501 | 17.560 | 1.794 | 0.2434 | 9.06% | 84.9% |

receiving_yards paired week-grouped Brier gain: {"clusters": 24, "lower_95": -0.0033615367731042625, "upper_95": 0.0010327674121160036, "mean_gain": -0.001409983759767791}
Ready for separate prospective test: False
| rushing_yards / existing_workload | 18.199 | 18.506 | 3.107 | 0.2357 | 6.11% | 87.0% |
| rushing_yards / conditional_efficiency | 18.474 | 18.506 | 3.194 | 0.2373 | 6.56% | 88.1% |
| rushing_yards / asymmetric_repair | 18.573 | 18.506 | 3.194 | 0.2374 | 7.95% | 87.9% |

rushing_yards paired week-grouped Brier gain: {"clusters": 24, "lower_95": -0.004503777222117486, "upper_95": 0.0009992157235363588, "mean_gain": -0.0017409180708236975}
Ready for separate prospective test: False

Historical player-game refits use date-lagged finalized data, not recreated historical locks.
Proxy lines screen forecast quality, not betting profitability.
Actual future lock scoring and independent-week evidence remain necessary.
No refit after residual/calibration blocks: uncertainty and calibration match the tested model.
Missing routes/injuries remain unknown; no new data provider is fabricated.
