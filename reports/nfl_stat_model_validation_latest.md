# NFL Per-Stat Model Validation

Status: evaluated
Eligible player-game shadow rows: 1696

| Stat | Component | Registered outputs | Player-games | Weeks | Candidate / production RMSE | Candidate / production Brier | Candidate / market Brier | Deployment |
|---|---|---|---:|---:|---:|---:|---:|---|
| receiving_yards | receiving-oe-v2-20261012T234225243219Z | probability | 159 | 1 | 26.677 / 26.053 | 0.2504 / 0.2504 | 0.2504 / 0.2504 | shadow |
| receiving_yards | components-20261015T203006649687Z-team_receiving_yards | expected_mean, median | 159 | 1 | 26.621 / 26.053 | 0.2504 / 0.2504 | 0.2504 / 0.2504 | shadow |
| passing_yards | components-20261010T222543607583Z-qb_tail | expected_mean, median | 56 | 2 | 69.343 / 63.597 | 0.2633 / 0.2500 | 0.2633 / 0.2500 | shadow |

## Exclusions

- awaiting_settlement: 1124
- not_current_registered_candidate: 733
- shadow_output_contract_mismatch: 73
- unverified_participation: 20

## Probability-Only Exclusions

- none

Forecast deployment remains separate from betting approval.
