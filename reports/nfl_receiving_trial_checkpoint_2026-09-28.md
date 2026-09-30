# NFL Receiving Trial Checkpoint

Date: 2026-09-28; status: evaluated
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 21; paired challenger captures: 21.
Current eligible locks: 11; quotes still fresh now: 0.
Fixed research selections: 2; missing current captures: [].
Settlement: {'settled_binary': 21}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 17 | 12 | 1 | 0.2533 / 0.0341 | 0.2405 / 0.0392 | 0.2448 | 0.8333 / 0.7500 |
| day_metrics / fixed_research | 2 | 2 | 1 | 0.1842 / 0.4290 | 0.1774 / 0.4212 | 0.2500 | 1.0000 / 1.0000 |
| cumulative_metrics / all_eligible | 254 | 169 | 1 | 0.2510 / 0.0449 | 0.2505 / 0.0264 | 0.2500 | 0.7219 / 0.7751 |
| cumulative_metrics / fixed_research | 5 | 5 | 1 | 0.2616 / 0.1642 | 0.2820 / 0.1874 | 0.2500 | 0.8000 / 0.8000 |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| all_eligible | 21 | 17 | 4 | 81.0% | False |
| fixed_research | 2 | 0 | 2 | 0.0% | False |

Exact locked/close lines, timestamps and unknown reasons are retained in the companion JSON.

## Forecast Deployment
Status: collecting_evidence; next action: keep_frozen_collect_evidence
- real_offers:insufficient_independent_weeks
- real_offers:brier_improvement_unconfirmed

Forecast deployment and betting permission are separate. Production was not changed.
Research selections are fixed; this report never reselects after results.
Freshness at original lock/capture is separate from whether a price is executable now.
Forecast review does not require ROI or CLV gates. Cash approval remains separate.
A positive checkpoint prepares a controlled release review; it never overwrites model artifacts.
