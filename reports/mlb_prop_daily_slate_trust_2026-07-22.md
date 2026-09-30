# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-23T03:11:13Z
Slate: 2026-07-22
Status: **PROVISIONAL**
Decision: `wait_for_final_results_or_repair_failed_checks`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | False | 16/17 games final |
| valid_close_coverage | False | 85.9% valid exact closes; required >= 90% |
| stale_close_rate | True | 0.1% stale closes; required <= 2% |
| targeted_close_captures | False | 5/17 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | False | 241 graded, 21 pending, 66 void |
| pitcher_anchor_settled_or_voided | False | 26 graded, 2 pending, 0 void |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 3058
- Valid exact closes: 2626 (85.9%)
- Stale close rate: 0.1%

| Close Reason | Rows |
|---|---:|
| valid_close | 2626 |
| player_prop_unavailable_at_close | 358 |
| line_disappeared_at_close | 48 |
| player_market_unavailable_at_close | 12 |
| fallback_other_book_only | 12 |
| stale_close_before_lock | 2 |

## Frozen Release Checkpoint

- Checkpoint status: `evaluation_ready_no_micro_buckets`
- Hitter dates: 13 completed, 0 remaining
- Pitcher dates: 15 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
