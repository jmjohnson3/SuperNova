# NFL Receiving Trial Checkpoint

Date: 2026-10-29; status: evaluated
Scope: this date only; cumulative deployment review requires an offline history report.
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 49; paired challenger captures: 49.
Current eligible locks: 16; quotes still fresh now: 0.
Fixed research selections: 0; missing current captures: [].
Settlement: {'settled_binary': 48, 'void_nonparticipant': 1}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 32 | 15 | 1 | 0.2500 / 0.0317 | 0.2500 / 0.0317 | 0.2500 | 0.9333 / 0.8667 |
| day_metrics / fixed_research | 0 | - | - | pending | pending | pending | pending |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| current_eligible | 16 | 12 | 4 | 75.0% | False |
| all_eligible | 49 | 30 | 19 | 61.2% | False |
| fixed_research | 0 | 0 | 0 | pending kickoff | None |

Exact locked/close lines, timestamps and unknown reasons are retained in the companion JSON.

## Forecast Deployment
Status: collecting_evidence; next action: keep_frozen_collect_evidence
- real_offers:insufficient_independent_weeks
- real_offers:brier_improvement_unconfirmed
- real_offers:distribution_coverage_not_confirmed
- date_partition_not_cumulative_review

Forecast deployment and betting permission are separate. Production was not changed.
Research selections are fixed; this report never reselects after results.
Freshness at original lock/capture is separate from whether a price is executable now.
Forecast review does not require ROI or CLV gates. Cash approval remains separate.
A positive checkpoint prepares a controlled release review; it never overwrites model artifacts.
