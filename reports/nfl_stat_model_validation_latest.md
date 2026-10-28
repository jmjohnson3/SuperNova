# NFL Per-Stat Model Validation

Status: evaluated
Eligible player-game shadow rows: 3124

| Stat | Component | Registered outputs | Player-games | Weeks | Candidate / production RMSE | Candidate / production Brier | Candidate / market Brier | Deployment |
|---|---|---|---:|---:|---:|---:|---:|---|
| receiving_yards | receiving-oe-v2-20261012T234225243219Z | probability | 316 | 2 | 26.941 / 26.995 | 0.2499 / 0.2499 | 0.2499 / 0.2499 | shadow |
| receiving_yards | components-20261015T203006649687Z-team_receiving_yards | expected_mean, median | 316 | 2 | 27.073 / 26.995 | 0.2499 / 0.2499 | 0.2499 / 0.2499 | shadow |
| passing_yards | components-20261010T222543607583Z-qb_tail | expected_mean, median | 83 | 3 | 67.907 / 63.655 | 0.2590 / 0.2500 | 0.2590 / 0.2500 | shadow |

## Exclusions

- awaiting_settlement: 1856
- not_current_registered_candidate: 733
- shadow_output_contract_mismatch: 73
- unverified_participation: 27

## Probability-Only Exclusions

- none

Forecast deployment remains separate from betting approval.
