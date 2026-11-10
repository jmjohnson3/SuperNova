# NFL Receiving Trial Checkpoint

Date: 2026-11-08; status: capture_incomplete
Scope: this date only; cumulative deployment review requires an offline history report.
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 515; paired challenger captures: 470.
Current eligible locks: 126; quotes still fresh now: 0.
Fixed research selections: 0; missing current captures: [33359, 33360, 33361, 33363, 33364, 33365, 33367, 33368, 33369, 33372, 33373, 33374, 33380, 33381, 33382].
Settlement: {'settled_binary': 457, 'result_or_participation_unresolved': 7, 'missing_validated_capture': 42, 'void_nonparticipant': 9}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 189 | 133 | 1 | 0.2502 / 0.0340 | 0.2502 / 0.0340 | 0.2502 | 0.7519 / 0.7594 |
| day_metrics / fixed_research | 0 | - | - | pending | pending | pending | pending |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| current_eligible | 126 | 108 | 18 | 85.7% | False |
| all_eligible | 515 | 417 | 98 | 81.0% | False |
| fixed_research | 0 | 0 | 0 | pending kickoff | None |

Exact locked/close lines, timestamps and unknown reasons are retained in the companion JSON.

## Forecast Deployment
Status: collecting_evidence; next action: keep_frozen_collect_evidence
- real_offers:insufficient_independent_weeks
- real_offers:brier_improvement_unconfirmed
- missing_current_challenger_captures
- missing_prospective_capture_history
- date_partition_not_cumulative_review

Forecast deployment and betting permission are separate. Production was not changed.
Research selections are fixed; this report never reselects after results.
Freshness at original lock/capture is separate from whether a price is executable now.
Forecast review does not require ROI or CLV gates. Cash approval remains separate.
A positive checkpoint prepares a controlled release review; it never overwrites model artifacts.
