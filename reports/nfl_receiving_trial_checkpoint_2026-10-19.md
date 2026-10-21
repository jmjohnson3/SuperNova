# NFL Receiving Trial Checkpoint

Date: 2026-10-19; status: capture_incomplete
Scope: this date only; cumulative deployment review requires an offline history report.
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 9; paired challenger captures: 0.
Current eligible locks: 9; quotes still fresh now: 0.
Fixed research selections: 0; missing current captures: [22938, 22939, 22940, 22941, 22942, 22943, 22945, 22947, 22951].
Settlement: {'missing_validated_capture': 9}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 0 | - | - | pending | pending | pending | pending |
| day_metrics / fixed_research | 0 | - | - | pending | pending | pending | pending |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| current_eligible | 9 | 4 | 5 | 44.4% | False |
| all_eligible | 9 | 4 | 5 | 44.4% | False |
| fixed_research | 0 | 0 | 0 | pending kickoff | None |

Exact locked/close lines, timestamps and unknown reasons are retained in the companion JSON.

## Forecast Deployment
Status: collecting_evidence; next action: keep_frozen_collect_evidence
- real_offers:missing_matches
- missing_current_challenger_captures
- missing_prospective_capture_history
- date_partition_not_cumulative_review

Forecast deployment and betting permission are separate. Production was not changed.
Research selections are fixed; this report never reselects after results.
Freshness at original lock/capture is separate from whether a price is executable now.
Forecast review does not require ROI or CLV gates. Cash approval remains separate.
A positive checkpoint prepares a controlled release review; it never overwrites model artifacts.
