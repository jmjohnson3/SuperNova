# NFL Receiving Trial Checkpoint

Date: 2026-10-11; status: capture_incomplete
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 597; paired challenger captures: 596.
Current eligible locks: 138; quotes still fresh now: 0.
Fixed research selections: 3; missing current captures: [17350].
Settlement: {'settled_binary': 594, 'missing_validated_capture': 1, 'void_nonparticipant': 2}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 242 | 144 | 1 | 0.2498 / 0.0567 | 0.2498 / 0.0567 | 0.2497 | 0.7431 / 0.7708 |
| day_metrics / fixed_research | 3 | 3 | 1 | 0.2643 / 0.2025 | 0.2643 / 0.2025 | 0.2500 | 1.0000 / 1.0000 |
| cumulative_metrics / all_eligible | 242 | 144 | 1 | 0.2498 / 0.0567 | 0.2498 / 0.0567 | 0.2497 | 0.7431 / 0.7708 |
| cumulative_metrics / fixed_research | 3 | 3 | 1 | 0.2643 / 0.2025 | 0.2643 / 0.2025 | 0.2500 | 1.0000 / 1.0000 |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| all_eligible | 597 | 518 | 79 | 86.8% | False |
| fixed_research | 3 | 3 | 0 | 100.0% | True |

Exact locked/close lines, timestamps and unknown reasons are retained in the companion JSON.

## Forecast Deployment
Status: collecting_evidence; next action: keep_frozen_collect_evidence
- real_offers:insufficient_independent_weeks
- real_offers:brier_improvement_unconfirmed
- missing_current_challenger_captures
- missing_prospective_capture_history

Forecast deployment and betting permission are separate. Production was not changed.
Research selections are fixed; this report never reselects after results.
Freshness at original lock/capture is separate from whether a price is executable now.
Forecast review does not require ROI or CLV gates. Cash approval remains separate.
A positive checkpoint prepares a controlled release review; it never overwrites model artifacts.
