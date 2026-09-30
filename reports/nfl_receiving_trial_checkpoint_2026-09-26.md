# NFL Receiving Trial Checkpoint

Date: 2026-09-26; status: no_eligible_offers
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 0; paired challenger captures: 0.
Current eligible locks: 0; quotes still fresh now: 0.
Fixed research selections: 0; missing current captures: [].
Settlement: {}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 0 | - | - | pending | pending | pending | pending |
| day_metrics / fixed_research | 0 | - | - | pending | pending | pending | pending |
| cumulative_metrics / all_eligible | 16 | 10 | 1 | 0.2508 / 0.0272 | 0.2402 / 0.1611 | 0.2520 | 0.7000 / 0.8000 |
| cumulative_metrics / fixed_research | 2 | 2 | 1 | 0.2610 / 0.0565 | 0.2724 / 0.0719 | 0.2500 | 1.0000 / 1.0000 |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| all_eligible | 0 | 0 | 0 | pending kickoff | None |
| fixed_research | 0 | 0 | 0 | pending kickoff | None |

Exact locked/close lines, timestamps and unknown reasons are retained in the companion JSON.

## Forecast Deployment
Status: collecting_evidence; next action: keep_frozen_collect_evidence
- real_offers:insufficient_independent_weeks
- real_offers:brier_improvement_unconfirmed
- real_offers:calibration_not_improved

Forecast deployment and betting permission are separate. Production was not changed.
Research selections are fixed; this report never reselects after results.
Freshness at original lock/capture is separate from whether a price is executable now.
Forecast review does not require ROI or CLV gates. Cash approval remains separate.
A positive checkpoint prepares a controlled release review; it never overwrites model artifacts.
