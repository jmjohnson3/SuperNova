# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-24T04:12:18Z
Slate: 2026-07-23
Status: **FAIL**
Decision: `do_not_use_slate_for_clean_promotion_evidence`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | True | 5/5 games final |
| valid_close_coverage | False | 86.2% valid exact closes; required >= 90% |
| stale_close_rate | True | 0.0% stale closes; required <= 2% |
| targeted_close_captures | True | 5/5 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | True | 60 graded, 0 pending, 23 void |
| pitcher_anchor_settled_or_voided | True | 6 graded, 0 pending, 0 void |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 751
- Valid exact closes: 647 (86.2%)
- Stale close rate: 0.0%

| Close Reason | Rows |
|---|---:|
| valid_close | 647 |
| player_prop_unavailable_at_close | 90 |
| line_disappeared_at_close | 8 |
| fallback_other_book_only | 3 |
| player_market_unavailable_at_close | 3 |

## Frozen Release Checkpoint

- Checkpoint status: `evaluation_ready_no_micro_buckets`
- Hitter dates: 15 completed, 0 remaining
- Pitcher dates: 17 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
