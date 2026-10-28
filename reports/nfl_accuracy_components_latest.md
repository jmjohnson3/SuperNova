# NFL Accuracy Components

Frozen production: nfl-20260918T143342Z

Mean, median, and probability approvals are independent. These are historical screens, not betting approvals.

| Component | Mean RMSE (candidate / reference) | Brier (candidate / reference) | 80% coverage | Prospective outputs |
|---|---:|---:|---:|---|
| receiving_conditional | 25.012 / 25.012 | 0.2295 / 0.2320 | 78.5% | probability |
| qb_tail | 83.252 / 85.384 | 0.2033 / 0.2150 | 84.9% | none |
| team_receiving_yards | 23.995 / 24.574 | 0.2254 / 0.2319 | 84.6% | expected_mean, median |
| team_rushing_yards | 27.055 / 26.469 | 0.2315 / 0.2367 | 87.4% | none |
| team_passing_yards | 87.947 / 86.257 | 0.2113 / 0.2165 | 77.7% | none |

## Latest Calibration Block

- qb_tail: disabled; coverage 81.7% -> 86.7%; interval score 239.934 -> 246.779; Brier 0.1727 -> 0.1740 (60 rows).

- Historical tests use proxy lines, not executable sportsbook prices.
- Previously inspected seasons are research evidence, not an untouched final test.
- Team pools use earlier team games; unknown newcomers keep unallocated volume and use the reference forecast.
- No true routes, first-read data, or unobserved historical injuries are fabricated.
- Historical component approval permits prospective scoring only, never bankroll promotion.
