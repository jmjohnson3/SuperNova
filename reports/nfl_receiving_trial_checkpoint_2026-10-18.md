# NFL Receiving Trial Checkpoint

Date: 2026-10-18; status: capture_incomplete
Scope: this date only; cumulative deployment review requires an offline history report.
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 694; paired challenger captures: 457.
Current eligible locks: 117; quotes still fresh now: 0.
Fixed research selections: 0; missing current captures: [22882, 22883, 22885, 22886, 22887, 22888, 22890, 22892, 22893, 22894, 22895, 22896, 22897].
Settlement: {'settled_binary': 449, 'missing_validated_capture': 232, 'void_nonparticipant': 13}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 130 | 120 | 1 | 0.2506 / 0.0343 | 0.2506 / 0.0343 | 0.2506 | 0.7500 / 0.7333 |
| day_metrics / fixed_research | 0 | - | - | pending | pending | pending | pending |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| current_eligible | 117 | 107 | 10 | 91.5% | True |
| all_eligible | 694 | 582 | 112 | 83.9% | False |
| fixed_research | 0 | 0 | 0 | pending kickoff | None |

Exact locked/close lines, timestamps and unknown reasons are retained in the companion JSON.

## Forecast Deployment
Status: collecting_evidence; next action: keep_frozen_collect_evidence
- real_offers:insufficient_independent_weeks
- real_offers:brier_improvement_unconfirmed
- real_offers:distribution_coverage_not_confirmed
- missing_current_challenger_captures
- missing_prospective_capture_history
- date_partition_not_cumulative_review

Forecast deployment and betting permission are separate. Production was not changed.
Research selections are fixed; this report never reselects after results.
Freshness at original lock/capture is separate from whether a price is executable now.
Forecast review does not require ROI or CLV gates. Cash approval remains separate.
A positive checkpoint prepares a controlled release review; it never overwrites model artifacts.
