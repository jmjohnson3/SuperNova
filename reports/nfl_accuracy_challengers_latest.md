# NFL Accuracy Challengers

Production remains frozen: nfl-20260918T143342Z
Run: accuracy-20260929T091204882511Z

## Player Models

| Stat | Challenger MAE | Reference recipe MAE | Baseline MAE | Raw / calibrated Brier | Historical gate |
|---|---:|---:|---:|---:|---|
| receiving_yards | 17.806 | 17.218 | 18.120 | 0.2343 / 0.2317 | False |
| rushing_yards | 18.869 | 18.201 | 18.941 | 0.2398 / 0.2333 | False |
| passing_yards | 67.240 | 67.360 | 70.895 | 0.2097 / 0.2037 | False |

## Game Models

| Target | Challenger MAE | Statistical baseline | Retrospective market |
|---|---:|---:|---:|
| home_margin | 10.269 | 10.383 | 9.743 |
| total_points_actual | 10.582 | 10.736 | 10.174 |

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

Historical rows with captured as-of context: 467
No challenger is wired into production probabilities, Discord ranking, or bankroll selection.
