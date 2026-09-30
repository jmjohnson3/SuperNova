# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-09T20:17:13Z
Slate: 2026-07-08
Status: **FAIL**
Decision: `do_not_use_slate_for_clean_promotion_evidence`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | True | 15/15 games final |
| valid_close_coverage | False | 86.7% valid exact closes; required >= 90% |
| stale_close_rate | True | 0.0% stale closes; required <= 2% |
| targeted_close_captures | False | 9/16 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | True | 227 graded, 0 pending, 81 void |
| pitcher_anchor_settled_or_voided | True | 24 graded, 0 pending, 0 void |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 2957
- Valid exact closes: 2565 (86.7%)
- Stale close rate: 0.0%

| Close Reason | Rows |
|---|---:|
| valid_close | 2565 |
| player_prop_unavailable_at_close | 330 |
| line_disappeared_at_close | 42 |
| player_market_unavailable_at_close | 10 |
| fallback_other_book_only | 10 |

## Frozen Release Checkpoint

- Checkpoint status: `collecting`
- Hitter dates: 3 completed, 2 remaining
- Pitcher dates: 5 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
