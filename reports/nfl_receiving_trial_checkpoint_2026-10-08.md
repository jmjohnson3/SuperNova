# NFL Receiving Trial Checkpoint

Date: 2026-10-08; status: evaluated
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 36; paired challenger captures: 36.
Current eligible locks: 13; quotes still fresh now: 0.
Fixed research selections: 0; missing current captures: [].
Settlement: {'result_or_participation_unresolved': 36}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 0 | - | - | pending | pending | pending | pending |
| day_metrics / fixed_research | 0 | - | - | pending | pending | pending | pending |
| cumulative_metrics / all_eligible | 13 | 13 | 1 | 0.2510 / 0.0346 | 0.2510 / 0.0346 | 0.2511 | 0.7692 / 0.7692 |
| cumulative_metrics / fixed_research | 4 | 4 | 1 | 0.2725 / 0.2904 | 0.2725 / 0.2904 | 0.2500 | 0.2500 / 0.2500 |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| all_eligible | 36 | 22 | 14 | 61.1% | False |
| fixed_research | 0 | 0 | 0 | pending kickoff | None |

Exact locked/close lines, timestamps and unknown reasons are retained in the companion JSON.

## Forecast Deployment
Status: collecting_evidence; next action: keep_frozen_collect_evidence
- real_offers:insufficient_independent_weeks
- real_offers:brier_improvement_unconfirmed
- missing_prospective_capture_history

Forecast deployment and betting permission are separate. Production was not changed.
Research selections are fixed; this report never reselects after results.
Freshness at original lock/capture is separate from whether a price is executable now.
Forecast review does not require ROI or CLV gates. Cash approval remains separate.
A positive checkpoint prepares a controlled release review; it never overwrites model artifacts.
