# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-20T04:09:51Z
Slate: 2026-07-19
Status: **FAIL**
Decision: `do_not_use_slate_for_clean_promotion_evidence`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | True | 16/16 games final |
| valid_close_coverage | False | 86.2% valid exact closes; required >= 90% |
| stale_close_rate | True | 0.1% stale closes; required <= 2% |
| targeted_close_captures | False | 14/16 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | True | 239 graded, 0 pending, 74 void |
| pitcher_anchor_settled_or_voided | True | 25 graded, 0 pending, 0 void |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 2705
- Valid exact closes: 2332 (86.2%)
- Stale close rate: 0.1%

| Close Reason | Rows |
|---|---:|
| valid_close | 2332 |
| player_prop_unavailable_at_close | 310 |
| line_disappeared_at_close | 43 |
| player_market_unavailable_at_close | 9 |
| fallback_other_book_only | 9 |
| stale_close_before_lock | 2 |

## Frozen Release Checkpoint

- Checkpoint status: `evaluation_ready_no_micro_buckets`
- Hitter dates: 11 completed, 0 remaining
- Pitcher dates: 13 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
