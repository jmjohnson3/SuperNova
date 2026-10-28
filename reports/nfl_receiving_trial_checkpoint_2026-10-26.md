# NFL Receiving Trial Checkpoint

Date: 2026-10-26; status: evaluated
Scope: this date only; cumulative deployment review requires an offline history report.
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 20; paired challenger captures: 20.
Current eligible locks: 11; quotes still fresh now: 0.
Fixed research selections: 0; missing current captures: [].
Settlement: {'settled_binary': 20}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 13 | 11 | 1 | 0.2500 / 0.1818 | 0.2500 / 0.1818 | 0.2500 | 0.8182 / 0.8182 |
| day_metrics / fixed_research | 0 | - | - | pending | pending | pending | pending |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| current_eligible | 11 | 9 | 2 | 81.8% | False |
| all_eligible | 20 | 17 | 3 | 85.0% | False |
| fixed_research | 0 | 0 | 0 | pending kickoff | None |

Exact locked/close lines, timestamps and unknown reasons are retained in the companion JSON.

## Forecast Deployment
Status: collecting_evidence; next action: keep_frozen_collect_evidence
- real_offers:insufficient_independent_weeks
- real_offers:brier_improvement_unconfirmed
- date_partition_not_cumulative_review

Forecast deployment and betting permission are separate. Production was not changed.
Research selections are fixed; this report never reselects after results.
Freshness at original lock/capture is separate from whether a price is executable now.
Forecast review does not require ROI or CLV gates. Cash approval remains separate.
A positive checkpoint prepares a controlled release review; it never overwrites model artifacts.
