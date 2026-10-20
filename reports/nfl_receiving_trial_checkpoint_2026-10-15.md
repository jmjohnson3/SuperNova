# NFL Receiving Trial Checkpoint

Date: 2026-10-15; status: evaluated
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 85; paired challenger captures: 72.
Current eligible locks: 15; quotes still fresh now: 0.
Fixed research selections: 0; missing current captures: [].
Settlement: {'settled_binary': 72, 'missing_validated_capture': 13}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 15 | 15 | 1 | 0.2504 / 0.1035 | 0.2504 / 0.1035 | 0.2504 | 0.8000 / 0.7333 |
| day_metrics / fixed_research | 0 | - | - | pending | pending | pending | pending |
| cumulative_metrics / all_eligible | 15 | 15 | 1 | 0.2504 / 0.1035 | 0.2504 / 0.1035 | 0.2504 | 0.8000 / 0.7333 |
| cumulative_metrics / fixed_research | 0 | - | - | pending | pending | pending | pending |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| current_eligible | 15 | 15 | 0 | 100.0% | True |
| all_eligible | 85 | 61 | 24 | 71.8% | False |
| fixed_research | 0 | 0 | 0 | pending kickoff | None |

Exact locked/close lines, timestamps and unknown reasons are retained in the companion JSON.

## Forecast Deployment
Status: collecting_evidence; next action: keep_frozen_collect_evidence
- real_offers:insufficient_independent_weeks
- real_offers:brier_improvement_unconfirmed
- real_offers:distribution_coverage_not_confirmed
- missing_prospective_capture_history

Forecast deployment and betting permission are separate. Production was not changed.
Research selections are fixed; this report never reselects after results.
Freshness at original lock/capture is separate from whether a price is executable now.
Forecast review does not require ROI or CLV gates. Cash approval remains separate.
A positive checkpoint prepares a controlled release review; it never overwrites model artifacts.
