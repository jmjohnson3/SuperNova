# NFL Workload Uncertainty Diagnostic

Diagnostic only. No model deployment or betting approval.

| Stat / prior-volume role | Rows | Intervals | Below | Above | Coverage | Known opportunity |
|---|---:|---:|---:|---:|---:|---:|
| passing_yards / prior_pass_attempts_ge_25 | 28 | 28 | 3.6% | 7.1% | 89.3% | 0 |
| passing_yards / prior_pass_attempts_lt_25 | 6 | 6 | 0.0% | 33.3% | 66.7% | 0 |
| passing_yards / unknown | 28 | 28 | 7.1% | 10.7% | 82.1% | 0 |
| receiving_yards / prior_targets_ge_4 | 102 | 102 | 20.6% | 16.7% | 62.7% | 102 |
| receiving_yards / prior_targets_lt_4 | 134 | 134 | 7.5% | 11.2% | 81.3% | 134 |
| receiving_yards / unknown | 179 | 179 | 15.1% | 11.2% | 73.7% | 0 |
| rushing_yards / prior_carries_ge_8 | 45 | 45 | 22.2% | 22.2% | 55.6% | 45 |
| rushing_yards / prior_carries_lt_8 | 57 | 57 | 5.3% | 7.0% | 87.7% | 57 |
| rushing_yards / unknown | 89 | 89 | 19.1% | 7.9% | 73.0% | 0 |

- One earliest valid settled lock per release/player-game/stat; later revisions are not substituted.
- Opportunity is a saved lock-time feature with its exact source, not necessarily the driver of the yardage head.
- Implied efficiency is projection divided by saved opportunity, not a separately learned rate forecast.
- Opportunity error plus efficiency residual equals projection minus actual; this is descriptive, not causal attribution.
- Role proxies use lagged targets (4), carries (8), or attempts (25), not verified starter status. Missing injury data is unknown.
- No intervals, targets, or forecasts are reconstructed when absent from the original lock.

Full row details and breakdowns are in the adjacent JSON file.
