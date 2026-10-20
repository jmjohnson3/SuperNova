# NFL Receiving Trial Checkpoint

Date: 2026-10-05; status: evaluated
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 21; paired challenger captures: 21.
Current eligible locks: 13; quotes still fresh now: 0.
Fixed research selections: 1; missing current captures: [].
Settlement: {'settled_binary': 20, 'void_nonparticipant': 1}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 13 | 13 | 1 | 0.2510 / 0.0346 | 0.2510 / 0.0346 | 0.2511 | 0.7692 / 0.7692 |
| day_metrics / fixed_research | 1 | 1 | 1 | 0.2854 / 0.5342 | 0.2854 / 0.5342 | 0.2500 | 1.0000 / 1.0000 |
| cumulative_metrics / all_eligible | 13 | 13 | 1 | 0.2510 / 0.0346 | 0.2510 / 0.0346 | 0.2511 | 0.7692 / 0.7692 |
| cumulative_metrics / fixed_research | 4 | 4 | 1 | 0.2725 / 0.2904 | 0.2725 / 0.2904 | 0.2500 | 0.2500 / 0.2500 |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| all_eligible | 21 | 17 | 4 | 81.0% | False |
| fixed_research | 1 | 0 | 1 | 0.0% | False |

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
