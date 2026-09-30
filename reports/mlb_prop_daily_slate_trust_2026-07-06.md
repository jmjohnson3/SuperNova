# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-09T15:26:13Z
Slate: 2026-07-06
Status: **FAIL**
Decision: `do_not_use_slate_for_clean_promotion_evidence`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | True | 8/8 games final |
| valid_close_coverage | False | 87.7% valid exact closes; required >= 90% |
| stale_close_rate | True | 0.9% stale closes; required <= 2% |
| targeted_close_captures | False | 5/8 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | True | 130 graded, 0 pending, 32 void |
| pitcher_anchor_settled_or_voided | True | 14 graded, 0 pending, 0 void |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 1532
- Valid exact closes: 1344 (87.7%)
- Stale close rate: 0.9%

| Close Reason | Rows |
|---|---:|
| valid_close | 1344 |
| close_outside_two_hour_window | 143 |
| line_disappeared_at_close | 26 |
| stale_close_before_lock | 14 |
| fallback_other_book_only | 5 |

## Frozen Release Checkpoint

- Checkpoint status: `collecting`
- Hitter dates: 2 completed, 3 remaining
- Pitcher dates: 5 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
