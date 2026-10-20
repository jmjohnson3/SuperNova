# NFL Receiving Trial Checkpoint

Date: 2026-10-12; status: evaluated
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 47; paired challenger captures: 30.
Current eligible locks: 17; quotes still fresh now: 0.
Fixed research selections: 1; missing current captures: [].
Settlement: {'settled_binary': 30, 'missing_validated_capture': 17}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 29 | 17 | 1 | 0.2499 / 0.0351 | 0.2499 / 0.0351 | 0.2500 | 0.5294 / 0.5294 |
| day_metrics / fixed_research | 1 | 1 | 1 | 0.2069 / 0.4548 | 0.2069 / 0.4548 | 0.2500 | 0.0000 / 1.0000 |
| cumulative_metrics / all_eligible | 271 | 161 | 1 | 0.2498 / 0.0470 | 0.2498 / 0.0470 | 0.2498 | 0.7205 / 0.7453 |
| cumulative_metrics / fixed_research | 4 | 4 | 1 | 0.2500 / 0.0382 | 0.2500 / 0.0382 | 0.2500 | 0.7500 / 1.0000 |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| all_eligible | 47 | 42 | 5 | 89.4% | False |
| fixed_research | 1 | 1 | 0 | 100.0% | True |

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
