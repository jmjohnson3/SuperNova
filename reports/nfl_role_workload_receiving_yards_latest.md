# Role-Specific Workload Distribution

Stat: receiving_yards

| Variant | Opportunity RMSE | Yard RMSE | Bias | Median MAE | Brier | Calibration | 80% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| combined | 2.210 | 24.096 | -1.995 | 16.175 | 0.2242 | 2.42% | 84.3% |
| control | 2.281 | 24.251 | -1.423 | 16.398 | 0.2287 | 4.22% | 82.8% |
| reference | 2.286 | 24.199 | -0.065 | 16.524 | 0.2291 | 3.26% | 79.5% |
| tails_only | 2.271 | 24.292 | -1.919 | 16.397 | 0.2282 | 3.97% | 83.5% |
| workload_only | 2.214 | 24.121 | -2.035 | 16.206 | 0.2244 | 2.28% | 84.5% |

## Prior-Volume Roles

| Role / variant | Rows | Below | Above | Coverage |
|---|---:|---:|---:|---:|
| higher / combined | 1903 | 6.1% | 11.9% | 82.0% |
| higher / control | 1903 | 11.8% | 13.3% | 74.9% |
| higher / reference | 1903 | 24.6% | 15.8% | 59.6% |
| higher / tails_only | 1903 | 6.4% | 12.3% | 81.3% |
| higher / workload_only | 1903 | 6.8% | 12.9% | 80.2% |
| lower / combined | 3375 | 1.9% | 12.5% | 85.6% |
| lower / control | 3375 | 1.0% | 11.7% | 87.3% |
| lower / reference | 3375 | 1.5% | 7.7% | 90.8% |
| lower / tails_only | 3375 | 2.3% | 13.0% | 84.7% |
| lower / workload_only | 3375 | 1.7% | 11.3% | 86.9% |

## Output Decisions

control: opportunity=not_accepted, expected_yards=not_accepted, typical_yards=not_accepted, probability=not_accepted
workload_only: opportunity=historical_screen_passed, expected_yards=not_accepted, typical_yards=historical_screen_passed, probability=not_accepted
tails_only: opportunity=historical_screen_passed, expected_yards=not_accepted, typical_yards=not_accepted, probability=not_accepted
combined: opportunity=historical_screen_passed, expected_yards=not_accepted, typical_yards=historical_screen_passed, probability=not_accepted

Historical development screen, not untouched prospective acceptance.
One row per player-game; expanding week tests and week-clustered uncertainty.
Efficiency point is identical between workload variants; role tails have zero mean.
Workload states are derived from one count curve, not contradictory independent classifiers.
Role means prior volume, not verified starter status. Unknown injury context stays unknown.
No normalization over offered players or hindsight participant pools.
Proxy-line scores do not prove actual offered-line or selected-pick improvement.
No production release, frozen trial, rankings, locks or betting approval changed.
