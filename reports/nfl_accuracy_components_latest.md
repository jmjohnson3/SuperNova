# NFL Accuracy Components

Frozen production: nfl-20260918T143342Z

Mean, median, and probability approvals are independent. These are historical screens, not betting approvals.

| Component | Mean RMSE (candidate / reference) | Brier (candidate / reference) | 80% coverage | Prospective outputs |
|---|---:|---:|---:|---|
| receiving_conditional | 25.015 / 25.015 | 0.2289 / 0.2315 | 78.5% | probability |
| qb_tail | 83.434 / 85.780 | 0.2036 / 0.2123 | 84.8% | expected_mean, median |
| team_receiving_yards | 25.156 / 25.015 | 0.2364 / 0.2315 | 79.7% | none |
| team_rushing_yards | 26.891 / 26.533 | 0.2397 / 0.2390 | 79.6% | none |
| team_passing_yards | 94.973 / 85.780 | 0.2391 / 0.2123 | 73.2% | none |

## Latest Calibration Block

- qb_tail: disabled; coverage 91.8% -> 86.3%; interval score 303.481 -> 284.873; Brier 0.2051 -> 0.2051 (73 rows).

- Historical tests use proxy lines, not executable sportsbook prices.
- Previously inspected seasons are research evidence, not an untouched final test.
- Team pools use earlier team games; unknown newcomers keep unallocated volume and use the reference forecast.
- No true routes, first-read data, or unobserved historical injuries are fabricated.
- Historical component approval permits prospective scoring only, never bankroll promotion.
