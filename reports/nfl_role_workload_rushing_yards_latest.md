# Role-Specific Workload Distribution

Stat: rushing_yards

| Variant | Opportunity RMSE | Yard RMSE | Bias | Median MAE | Brier | Calibration | 80% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| combined | 4.779 | 30.069 | -2.902 | 20.448 | 0.2287 | 3.13% | 82.4% |
| control | 4.910 | 30.256 | -1.949 | 20.496 | 0.2278 | 3.89% | 80.7% |
| reference | 4.936 | 30.269 | +0.844 | 20.612 | 0.2300 | 3.89% | 78.8% |
| tails_only | 4.903 | 30.307 | -2.717 | 20.496 | 0.2279 | 3.80% | 81.1% |
| workload_only | 4.781 | 30.143 | -2.983 | 20.489 | 0.2291 | 3.50% | 82.5% |

## Prior-Volume Roles

| Role / variant | Rows | Below | Above | Coverage |
|---|---:|---:|---:|---:|
| higher / combined | 730 | 6.6% | 13.0% | 80.4% |
| higher / control | 730 | 11.8% | 13.2% | 75.1% |
| higher / reference | 730 | 21.2% | 12.6% | 66.2% |
| higher / tails_only | 730 | 8.6% | 12.6% | 78.8% |
| higher / workload_only | 730 | 7.3% | 14.4% | 78.4% |
| lower / combined | 739 | 2.8% | 12.7% | 84.4% |
| lower / control | 739 | 2.3% | 11.5% | 86.2% |
| lower / reference | 739 | 1.4% | 7.3% | 91.3% |
| lower / tails_only | 739 | 3.8% | 12.7% | 83.5% |
| lower / workload_only | 739 | 2.3% | 11.1% | 86.6% |

## Output Decisions

control: opportunity=not_accepted, expected_yards=not_accepted, typical_yards=not_accepted, probability=not_accepted
workload_only: opportunity=historical_screen_passed, expected_yards=not_accepted, typical_yards=not_accepted, probability=not_accepted
tails_only: opportunity=not_accepted, expected_yards=not_accepted, typical_yards=not_accepted, probability=not_accepted
combined: opportunity=historical_screen_passed, expected_yards=not_accepted, typical_yards=not_accepted, probability=not_accepted

Historical development screen, not untouched prospective acceptance.
One row per player-game; expanding week tests and week-clustered uncertainty.
Efficiency point is identical between workload variants; role tails have zero mean.
Workload states are derived from one count curve, not contradictory independent classifiers.
Role means prior volume, not verified starter status. Unknown injury context stays unknown.
No normalization over offered players or hindsight participant pools.
Proxy-line scores do not prove actual offered-line or selected-pick improvement.
No production release, frozen trial, rankings, locks or betting approval changed.
