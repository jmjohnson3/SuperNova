# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-17T03:13:55Z
Slate: 2026-07-16
Status: **FAIL**
Decision: `do_not_use_slate_for_clean_promotion_evidence`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | True | 1/1 games final |
| valid_close_coverage | False | 0.0% valid exact closes; required >= 90% |
| stale_close_rate | True | 0.0% stale closes; required <= 2% |
| targeted_close_captures | False | 0/1 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | True | 15 graded, 0 pending, 5 void |
| pitcher_anchor_settled_or_voided | True | 1 graded, 0 pending, 0 void |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 178
- Valid exact closes: 0 (0.0%)
- Stale close rate: 0.0%

| Close Reason | Rows |
|---|---:|
| close_outside_two_hour_window | 176 |
| no_valid_close_snapshot | 2 |

## Frozen Release Checkpoint

- Checkpoint status: `evaluation_ready_no_micro_buckets`
- Hitter dates: 8 completed, 0 remaining
- Pitcher dates: 10 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
