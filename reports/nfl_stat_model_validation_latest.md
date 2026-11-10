# NFL Per-Stat Model Validation

Status: evaluated
Eligible player-game shadow rows: 4382

| Stat | Component | Registered outputs | Player-games | Weeks | Candidate / production RMSE | Candidate / production Brier | Candidate / market Brier | Deployment |
|---|---|---|---:|---:|---:|---:|---:|---|
| receiving_yards | receiving-oe-v2-20261012T234225243219Z | probability | 489 | 4 | 27.483 / 27.420 | 0.2498 / 0.2498 | 0.2498 / 0.2498 | shadow |
| receiving_yards | components-20261015T203006649687Z-team_receiving_yards | expected_mean, median | 489 | 4 | 27.599 / 27.420 | 0.2498 / 0.2498 | 0.2498 / 0.2498 | shadow |
| passing_yards | components-20261010T222543607583Z-qb_tail | expected_mean, median | 113 | 5 | 76.005 / 72.534 | 0.2566 / 0.2500 | 0.2566 / 0.2500 | shadow |

## Exclusions

- awaiting_settlement: 2581
- not_current_registered_candidate: 734
- shadow_output_contract_mismatch: 74
- unverified_participation: 26

## Probability-Only Exclusions

- none

Forecast deployment remains separate from betting approval.
