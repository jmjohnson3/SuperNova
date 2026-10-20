# NFL Workload Uncertainty Diagnostic

Diagnostic only. No model deployment or betting approval.

| Stat / prior-volume role | Rows | Intervals | Below | Above | Coverage | Known opportunity |
|---|---:|---:|---:|---:|---:|---:|
| passing_yards / prior_pass_attempts_ge_25 | 20 | 20 | 0.0% | 0.0% | 100.0% | 0 |
| passing_yards / prior_pass_attempts_lt_25 | 4 | 4 | 0.0% | 0.0% | 100.0% | 0 |
| receiving_yards / prior_targets_ge_4 | 73 | 73 | 9.6% | 5.5% | 84.9% | 73 |
| receiving_yards / prior_targets_lt_4 | 90 | 90 | 15.6% | 5.6% | 78.9% | 90 |
| rushing_yards / prior_carries_ge_8 | 23 | 23 | 43.5% | 4.3% | 52.2% | 23 |
| rushing_yards / prior_carries_lt_8 | 47 | 47 | 6.4% | 6.4% | 87.2% | 47 |

- One earliest valid settled lock per release/player-game/stat; later revisions are not substituted.
- Opportunity is a saved lock-time feature with its exact source, not necessarily the driver of the yardage head.
- Implied efficiency is projection divided by saved opportunity, not a separately learned rate forecast.
- Opportunity error plus efficiency residual equals projection minus actual; this is descriptive, not causal attribution.
- Role proxies use lagged targets (4), carries (8), or attempts (25), not verified starter status. Missing injury data is unknown.
- No intervals, targets, or forecasts are reconstructed when absent from the original lock.

Full row details and breakdowns are in the adjacent JSON file.
