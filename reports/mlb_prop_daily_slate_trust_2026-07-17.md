# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-18T04:01:46Z
Slate: 2026-07-17
Status: **PROVISIONAL**
Decision: `wait_for_final_results_or_repair_failed_checks`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | False | 10/15 games final |
| valid_close_coverage | False | 83.3% valid exact closes; required >= 90% |
| stale_close_rate | True | 0.3% stale closes; required <= 2% |
| targeted_close_captures | False | 13/15 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | False | 162 graded, 80 pending, 59 void |
| pitcher_anchor_settled_or_voided | False | 17 graded, 6 pending, 1 void |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 2777
- Valid exact closes: 2314 (83.3%)
- Stale close rate: 0.3%

| Close Reason | Rows |
|---|---:|
| valid_close | 2314 |
| close_outside_two_hour_window | 218 |
| player_prop_unavailable_at_close | 166 |
| line_disappeared_at_close | 52 |
| stale_close_before_lock | 9 |
| player_market_unavailable_at_close | 8 |
| fallback_other_book_only | 8 |
| no_valid_close_snapshot | 2 |

## Frozen Release Checkpoint

- Checkpoint status: `evaluation_ready_no_micro_buckets`
- Hitter dates: 8 completed, 0 remaining
- Pitcher dates: 10 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
