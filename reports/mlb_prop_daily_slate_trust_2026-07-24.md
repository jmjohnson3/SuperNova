# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-24T23:23:31Z
Slate: 2026-07-24
Status: **PROVISIONAL**
Decision: `wait_for_final_results_or_repair_failed_checks`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | False | 0/15 games final |
| valid_close_coverage | False | 83.3% valid exact closes; required >= 90% |
| stale_close_rate | True | 0.1% stale closes; required <= 2% |
| targeted_close_captures | False | 4/15 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | False | 0 graded, 298 pending, 0 void |
| pitcher_anchor_settled_or_voided | False | 0 graded, 20 pending, 0 void |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 2891
- Valid exact closes: 2409 (83.3%)
- Stale close rate: 0.1%

| Close Reason | Rows |
|---|---:|
| valid_close | 2409 |
| player_prop_unavailable_at_close | 272 |
| close_outside_two_hour_window | 142 |
| line_disappeared_at_close | 38 |
| player_market_unavailable_at_close | 14 |
| fallback_other_book_only | 14 |
| stale_close_before_lock | 2 |

## Frozen Release Checkpoint

- Checkpoint status: `evaluation_ready_no_micro_buckets`
- Hitter dates: 15 completed, 0 remaining
- Pitcher dates: 17 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
