# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-19T04:06:49Z
Slate: 2026-07-18
Status: **PROVISIONAL**
Decision: `wait_for_final_results_or_repair_failed_checks`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | False | 13/16 games final |
| valid_close_coverage | False | 82.4% valid exact closes; required >= 90% |
| stale_close_rate | True | 0.9% stale closes; required <= 2% |
| targeted_close_captures | False | 7/16 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | False | 193 graded, 42 pending, 81 void |
| pitcher_anchor_settled_or_voided | False | 18 graded, 2 pending, 2 void |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 2657
- Valid exact closes: 2190 (82.4%)
- Stale close rate: 0.9%

| Close Reason | Rows |
|---|---:|
| valid_close | 2190 |
| player_prop_unavailable_at_close | 246 |
| close_outside_two_hour_window | 152 |
| line_disappeared_at_close | 34 |
| stale_close_before_lock | 25 |
| player_market_unavailable_at_close | 5 |
| fallback_other_book_only | 5 |

## Frozen Release Checkpoint

- Checkpoint status: `evaluation_ready_no_micro_buckets`
- Hitter dates: 9 completed, 0 remaining
- Pitcher dates: 11 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
