# NFL Receiving Trial Checkpoint

Date: 2026-10-04; status: capture_incomplete
Pinned variant: stable_ensemble_calibrated (benchmark-20260921T230305171980Z)

Eligible original locks: 904; paired challenger captures: 474.
Current eligible locks: 159; quotes still fresh now: 0.
Fixed research selections: 3; missing current captures: [13421, 13422, 13423, 13424, 13425, 13426, 13427, 13428, 13429, 13430, 13431, 13433, 13434, 13435, 13436, 13437, 13438, 13439, 13440, 13441, 13442, 13443, 13444, 13446, 13447, 13448, 13449, 13450, 13451, 13452, 13453, 13454, 13455, 13456, 13457, 13458, 13460, 13462, 13464, 13465, 13466, 13468, 13469, 13471, 13472, 13473, 13474, 13475, 13476, 13478, 13479, 13480, 13481, 13482, 13483, 13484, 13485, 13486, 13487, 13489, 13490, 13491, 13492, 13493, 13494, 13496, 13497, 13498, 13500, 13501, 13505, 13506, 13507, 13508, 13509, 13510, 13511, 13512, 13513, 13514, 13515, 13516, 13517, 13518, 13519, 13520, 13521, 13523, 13524, 13525, 13526, 13527, 13528, 13529, 13530, 13531, 13532, 13533, 13534, 13535, 13536, 14347, 14348, 14350, 14351, 14352, 14353, 14354, 14355, 14357, 14358, 14359, 14360, 14361, 14362, 14363, 14364, 14365, 14367, 14368, 14369, 14371, 14372, 14373, 14374, 14375, 14376, 14377, 14378, 14379, 14380, 14382, 14384, 14385, 14386, 14387, 14388, 14607, 14608, 14609, 14610, 14611, 14612, 14614, 14615, 14616, 14617, 14621, 14622, 14623].
Settlement: {'settled_binary': 444, 'missing_validated_capture': 412, 'result_or_participation_unresolved': 48}

| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |
|---|---:|---:|---:|---|---|---|---|
| day_metrics / all_eligible | 300 | 155 | 1 | 0.2505 / 0.0056 | 0.2505 / 0.0056 | 0.2501 | 0.7161 / 0.7742 |
| day_metrics / fixed_research | 3 | 3 | 1 | 0.2682 / 0.2092 | 0.2682 / 0.2092 | 0.2500 | 0.0000 / 0.0000 |
| cumulative_metrics / all_eligible | 314 | 168 | 1 | 0.2500 / 0.0052 | 0.2500 / 0.0052 | 0.2499 | 0.7202 / 0.7738 |
| cumulative_metrics / fixed_research | 3 | 3 | 1 | 0.2682 / 0.2092 | 0.2682 / 0.2092 | 0.2500 | 0.0000 / 0.0000 |

## Exact Closes

| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |
|---|---:|---:|---:|---|---|
| all_eligible | 904 | 716 | 188 | 79.2% | False |
| fixed_research | 3 | 2 | 1 | 66.7% | False |

Exact locked/close lines, timestamps and unknown reasons are retained in the companion JSON.

## Forecast Deployment
Status: collecting_evidence; next action: keep_frozen_collect_evidence
- real_offers:insufficient_independent_weeks
- real_offers:brier_improvement_unconfirmed
- missing_current_challenger_captures
- missing_prospective_capture_history

Forecast deployment and betting permission are separate. Production was not changed.
Research selections are fixed; this report never reselects after results.
Freshness at original lock/capture is separate from whether a price is executable now.
Forecast review does not require ROI or CLV gates. Cash approval remains separate.
A positive checkpoint prepares a controlled release review; it never overwrites model artifacts.
