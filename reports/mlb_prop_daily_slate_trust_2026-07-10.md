# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-11T04:08:40Z
Slate: 2026-07-10
Status: **PROVISIONAL**
Decision: `wait_for_final_results_or_repair_failed_checks`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | False | 9/15 games final |
| valid_close_coverage | False | 87.6% valid exact closes; required >= 90% |
| stale_close_rate | True | 0.0% stale closes; required <= 2% |
| targeted_close_captures | False | 14/16 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | False | 136 graded, 169 pending, 0 void |
| pitcher_anchor_settled_or_voided | False | 14 graded, 11 pending, 0 void |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 2946
- Valid exact closes: 2580 (87.6%)
- Stale close rate: 0.0%

| Close Reason | Rows |
|---|---:|
| valid_close | 2580 |
| player_prop_unavailable_at_close | 298 |
| line_disappeared_at_close | 46 |
| player_market_unavailable_at_close | 13 |
| fallback_other_book_only | 9 |

## Frozen Release Checkpoint

- Checkpoint status: `collecting`
- Hitter dates: 4 completed, 1 remaining
- Pitcher dates: 6 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
