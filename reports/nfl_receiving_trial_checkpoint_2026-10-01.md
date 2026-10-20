# NFL Receiving Trial Checkpoint

Date: 2026-10-01; status: evaluated
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 45; paired challenger captures: 45.
Current eligible locks: 13; quotes still fresh now: 0.
Fixed research selections: 2; missing current captures: [].
Settlement: {'settled_binary': 45}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 14 | 13 | 1 | 0.2436 / 0.0680 | 0.2436 / 0.0680 | 0.2472 | 0.7692 / 0.7692 |
| day_metrics / fixed_research | 2 | 2 | 1 | 0.2449 / 0.0419 | 0.2638 / 0.0704 | 0.2500 | 1.0000 / 1.0000 |
| cumulative_metrics / all_eligible | 14 | 13 | 1 | 0.2436 / 0.0680 | 0.2436 / 0.0680 | 0.2472 | 0.7692 / 0.7692 |
| cumulative_metrics / fixed_research | 7 | 7 | 2 | 0.2568 / 0.1292 | 0.2768 / 0.1540 | 0.2500 | 0.8571 / 0.8571 |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| all_eligible | 45 | 28 | 17 | 62.2% | False |
| fixed_research | 2 | 0 | 2 | 0.0% | False |

Exact locked/close lines, timestamps and unknown reasons are retained in the companion JSON.

## Forecast Deployment
Status: collecting_evidence; next action: keep_frozen_collect_evidence
- real_offers:insufficient_independent_weeks
- real_offers:brier_improvement_unconfirmed

Forecast deployment and betting permission are separate. Production was not changed.
Research selections are fixed; this report never reselects after results.
Freshness at original lock/capture is separate from whether a price is executable now.
Forecast review does not require ROI or CLV gates. Cash approval remains separate.
A positive checkpoint prepares a controlled release review; it never overwrites model artifacts.
