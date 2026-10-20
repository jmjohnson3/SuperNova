# NFL Accuracy Components

Frozen production: nfl-20260918T143342Z

Mean, median, and probability approvals are independent. These are historical screens, not betting approvals.

| Component | Mean RMSE (candidate / reference) | Brier (candidate / reference) | 80% coverage | Prospective outputs |
|---|---:|---:|---:|---|
| receiving_conditional | 25.101 / 25.101 | 0.2294 / 0.2320 | 78.5% | probability |
| qb_tail | 83.687 / 85.775 | 0.2041 / 0.2142 | 84.6% | none |
| team_receiving_yards | 24.105 / 24.680 | 0.2254 / 0.2319 | 84.2% | expected_mean, median |
| team_rushing_yards | 27.197 / 26.533 | 0.2313 / 0.2365 | 88.0% | none |
| team_passing_yards | 88.693 / 86.926 | 0.2127 / 0.2152 | 76.9% | none |

## Latest Calibration Block

- qb_tail: enabled; coverage 86.4% -> 83.3%; interval score 297.526 -> 295.212; Brier 0.2027 -> 0.2027 (66 rows).

- Historical tests use proxy lines, not executable sportsbook prices.
- Previously inspected seasons are research evidence, not an untouched final test.
- Team pools use earlier team games; unknown newcomers keep unallocated volume and use the reference forecast.
- No true routes, first-read data, or unobserved historical injuries are fabricated.
- Historical component approval permits prospective scoring only, never bankroll promotion.
