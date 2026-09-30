# Target-Volume-Only Challenger

rejected

| Variant | Target RMSE | Yard RMSE | Brier | Calibration | 80% coverage |
|---|---:|---:|---:|---:|---:|
| challenger | 2.252 | 24.140 | 0.2287 | 4.61% | 80.7% |
| control | 2.282 | 24.300 | 0.2272 | 1.46% | 80.1% |
| long_role | 2.254 | 24.205 | 0.2267 | 2.96% | 80.4% |
| recent_share | 2.335 | 24.500 | 0.2299 | 4.40% | 79.9% |
| reference | 2.286 | 24.232 | 0.2296 | 3.27% | 80.1% |

## Context
{"rows": 30541, "verified_locks": 156, "status": {"missing_verified_lock": 30385, "no_matching_pregame_observations": 156}, "fields": {"depth_rank": {"nonmissing": 0, "distinct": 0}, "expected_starter": {"nonmissing": 0, "distinct": 0}, "depth_movement": {"nonmissing": 0, "distinct": 0}, "injury_status": {"nonmissing": 0, "distinct": 0}, "practice_status": {"nonmissing": 0, "distinct": 0}, "roster_status": {"nonmissing": 0, "distinct": 0}, "teammate_absences": {"nonmissing": 0, "distinct": 0}, "teammate_limited": {"nonmissing": 0, "distinct": 0}}}

Efficiency recipe, efficiency residuals, and ranking are not tuned between target variants.
The control and target variants share one conditional yardage mixture; the reference is tested separately.
Historical proxy lines are not real offered-line or profitable-bet evidence.
No missing historical lock is reconstructed. Unlearnable role fields are excluded and reported.
Week 3 is diagnostic only; post-development locked evidence must pass before deployment.

No production deployment or cash approval.
