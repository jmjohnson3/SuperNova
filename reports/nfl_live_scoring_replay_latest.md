# NFL Live Scoring Replay

Status: waiting_for_settled_versioned_forecasts
Unique offer decisions: 0; settled: 0

## Replay Coverage


## Probability Stages

| Cohort | Stage | Rows | Brier | Calibration error |
|---|---|---:|---:|---:|

- Missing old scoring inputs are never replaced with current context or calibration.
- Stored-probability evaluation is separate from exact reproducibility; inspect replay_status.
- Multiple books/lines share one player-game outcome and receive inverse multiplicity weights.
- Intervals without locked bounds cannot establish interval coverage.
- Locked p10/p90 describe the stat forecast, not a distribution inferred from independently adjusted book probabilities.

## Prospective Challengers

Status: waiting_for_settled_prospective_challengers
Settled cohort rows: 0; unique pending player-games: 0
Different challenger runs are compared separately, not pooled as independent outcomes.
