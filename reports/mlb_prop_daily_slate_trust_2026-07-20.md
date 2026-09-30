# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-20T20:08:47Z
Slate: 2026-07-20
Status: **PROVISIONAL**
Decision: `wait_for_final_results_or_repair_failed_checks`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | False | 0/15 games final |
| valid_close_coverage | False | 0.0% valid exact closes; required >= 90% |
| stale_close_rate | True | 1.6% stale closes; required <= 2% |
| targeted_close_captures | False | 0/15 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | False | 0 graded, 292 pending, 0 void |
| pitcher_anchor_settled_or_voided | False | 0 graded, 24 pending, 0 void |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 2432
- Valid exact closes: 0 (0.0%)
- Stale close rate: 1.6%

| Close Reason | Rows |
|---|---:|
| close_outside_two_hour_window | 2390 |
| stale_close_before_lock | 40 |
| no_valid_close_snapshot | 2 |

## Frozen Release Checkpoint

- Checkpoint status: `evaluation_ready_no_micro_buckets`
- Hitter dates: 11 completed, 0 remaining
- Pitcher dates: 13 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
