# NFL Accuracy Challengers

Production remains frozen: nfl-20260918T143342Z
Run: accuracy-20261027T092611729791Z

## Player Models

| Stat | Challenger MAE | Reference recipe MAE | Baseline MAE | Raw / calibrated Brier | Historical gate |
|---|---:|---:|---:|---:|---|
| receiving_yards | 17.845 | 17.223 | 18.116 | 0.2373 / 0.2321 | False |
| rushing_yards | 18.715 | 18.120 | 18.847 | 0.2403 / 0.2331 | False |
| passing_yards | 66.992 | 67.076 | 70.606 | 0.2090 / 0.2035 | False |

## Game Models

| Target | Challenger MAE | Statistical baseline | Retrospective market |
|---|---:|---:|---:|
| home_margin | 10.131 | 10.236 | 9.643 |
| total_points_actual | 10.620 | 10.770 | 10.171 |

## Interpretation

- Historical probabilities use explicitly unpriced proxy lines, never invented sportsbook odds.
- Reference is a chronological refit of the frozen residual architecture with audited lag inputs, not predictions from a model trained on future outcomes.
- Some legacy derived reference features are excluded because their lock-time dependencies cannot be reconstructed.
- Actual frozen-production comparison uses immutable prospective forecast records; historical recipe refits are not that proof.
- 2026 outcomes have been examined during previous research and are not an untouched final test.
- Inner time blocks separate model training, scale fitting, residual CDF fitting, calibration fitting, and calibration gating.
- Uncertainty intervals are evaluated on outer folds; confidence is learned error-within-tolerance probability, not chance a wager wins.
- Historical same-week context remains unknown unless captured before kickoff. True routes and first reads are not fabricated.
- Game market values are retrospective benchmarks only; they are never inputs to these challengers.
- Statistical gates use NFL-week clustered intervals. All artifacts remain challengers regardless of pass status.

Historical rows with captured as-of context: 1445
No challenger is wired into production probabilities, Discord ranking, or bankroll selection.
