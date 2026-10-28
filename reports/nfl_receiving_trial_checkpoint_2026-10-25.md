# NFL Receiving Trial Checkpoint

Date: 2026-10-25; status: capture_incomplete
Scope: this date only; cumulative deployment review requires an offline history report.
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 589; paired challenger captures: 385.
Current eligible locks: 115; quotes still fresh now: 0.
Fixed research selections: 0; missing current captures: [26369, 26370, 26372, 26373, 26374, 26375, 26376, 26377, 26380, 26382, 26383, 26385, 26386, 26388, 26389, 26390, 26391, 26392, 26394, 26395, 26396, 26397, 26398, 26399, 26400, 26401, 26403, 26404, 26406, 26407, 26408, 26409, 26410, 26411, 26412, 26414, 26417, 26418, 26419, 26420, 26421, 26424, 26425, 26426, 26427, 26428, 26429, 26431, 26433, 26438, 26439, 26440, 26441, 26442, 26443, 26444, 26446, 26447, 26448, 26449, 26453, 26455, 26457, 26458, 26460, 26461, 26463, 26464].
Settlement: {'settled_binary': 381, 'void_nonparticipant': 7, 'missing_validated_capture': 201}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 184 | 130 | 1 | 0.2493 / 0.0480 | 0.2493 / 0.0480 | 0.2493 | 0.7538 / 0.7308 |
| day_metrics / fixed_research | 0 | - | - | pending | pending | pending | pending |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| current_eligible | 115 | 111 | 4 | 96.5% | True |
| all_eligible | 589 | 496 | 93 | 84.2% | False |
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
