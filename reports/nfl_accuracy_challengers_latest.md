# NFL Accuracy Challengers

Production remains frozen: nfl-20260918T143342Z
Run: accuracy-20261013T094056615708Z

## Player Models

| Stat | Challenger MAE | Reference recipe MAE | Baseline MAE | Raw / calibrated Brier | Historical gate |
|---|---:|---:|---:|---:|---|
| receiving_yards | 17.861 | 17.263 | 18.155 | 0.2350 / 0.2320 | False |
| rushing_yards | 18.802 | 18.138 | 18.869 | 0.2402 / 0.2340 | False |
| passing_yards | 67.452 | 67.365 | 70.895 | 0.2100 / 0.2043 | False |

## Game Models

| Target | Challenger MAE | Statistical baseline | Retrospective market |
|---|---:|---:|---:|
| home_margin | 10.181 | 10.293 | 9.698 |
| total_points_actual | 10.597 | 10.743 | 10.160 |

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

Historical rows with captured as-of context: 977
No challenger is wired into production probabilities, Discord ranking, or bankroll selection.
