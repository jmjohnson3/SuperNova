# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-10T04:05:50Z
Slate: 2026-07-09
Status: **PROVISIONAL**
Decision: `wait_for_final_results_or_repair_failed_checks`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | False | 11/13 games final |
| valid_close_coverage | False | 89.6% valid exact closes; required >= 90% |
| stale_close_rate | True | 0.0% stale closes; required <= 2% |
| targeted_close_captures | False | 11/13 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | False | 180 graded, 91 pending, 0 void |
| pitcher_anchor_settled_or_voided | False | 19 graded, 3 pending, 0 void |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 2575
- Valid exact closes: 2307 (89.6%)
- Stale close rate: 0.0%

| Close Reason | Rows |
|---|---:|
| valid_close | 2307 |
| player_prop_unavailable_at_close | 215 |
| line_disappeared_at_close | 28 |
| fallback_other_book_only | 13 |
| player_market_unavailable_at_close | 12 |

## Frozen Release Checkpoint

- Checkpoint status: `collecting`
- Hitter dates: 3 completed, 2 remaining
- Pitcher dates: 5 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
