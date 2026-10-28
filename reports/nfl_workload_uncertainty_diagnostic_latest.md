# NFL Workload Uncertainty Diagnostic

Diagnostic only. No model deployment or betting approval.

| Stat / prior-volume role | Rows | Intervals | Below | Above | Coverage | Known opportunity |
|---|---:|---:|---:|---:|---:|---:|
| passing_yards / prior_pass_attempts_ge_25 | 22 | 22 | 4.5% | 0.0% | 95.5% | 0 |
| passing_yards / prior_pass_attempts_lt_25 | 2 | 2 | 0.0% | 0.0% | 100.0% | 0 |
| receiving_yards / prior_targets_ge_4 | 82 | 82 | 14.6% | 12.2% | 73.2% | 82 |
| receiving_yards / prior_targets_lt_4 | 78 | 78 | 11.5% | 0.0% | 88.5% | 78 |
| rushing_yards / prior_carries_ge_8 | 25 | 25 | 32.0% | 36.0% | 32.0% | 25 |
| rushing_yards / prior_carries_lt_8 | 42 | 42 | 7.1% | 11.9% | 81.0% | 42 |

- One earliest valid settled lock per release/player-game/stat; later revisions are not substituted.
- Opportunity is a saved lock-time feature with its exact source, not necessarily the driver of the yardage head.
- Implied efficiency is projection divided by saved opportunity, not a separately learned rate forecast.
- Opportunity error plus efficiency residual equals projection minus actual; this is descriptive, not causal attribution.
- Role proxies use lagged targets (4), carries (8), or attempts (25), not verified starter status. Missing injury data is unknown.
- No intervals, targets, or forecasts are reconstructed when absent from the original lock.

Full row details and breakdowns are in the adjacent JSON file.
