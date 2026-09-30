# NFL Receiving Trial Checkpoint

Date: 2026-09-27; status: evaluated
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 529; paired challenger captures: 529.
Current eligible locks: 138; quotes still fresh now: 0.
Fixed research selections: 3; missing current captures: [].
Settlement: {'settled_binary': 524, 'void_nonparticipant': 5}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 221 | 147 | 1 | 0.2508 / 0.0473 | 0.2521 / 0.0381 | 0.2503 | 0.7143 / 0.7755 |
| day_metrics / fixed_research | 3 | 3 | 1 | 0.3132 / 0.5596 | 0.3518 / 0.5931 | 0.2500 | 0.6667 / 0.6667 |
| cumulative_metrics / all_eligible | 237 | 157 | 1 | 0.2508 / 0.0457 | 0.2513 / 0.0254 | 0.2504 | 0.7134 / 0.7771 |
| cumulative_metrics / fixed_research | 3 | 3 | 1 | 0.3132 / 0.5596 | 0.3518 / 0.5931 | 0.2500 | 0.6667 / 0.6667 |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| all_eligible | 529 | 420 | 109 | 79.4% | False |
| fixed_research | 3 | 3 | 0 | 100.0% | True |

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
