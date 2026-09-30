# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-12T04:01:44Z
Slate: 2026-07-11
Status: **PROVISIONAL**
Decision: `wait_for_final_results_or_repair_failed_checks`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | False | 14/16 games final |
| valid_close_coverage | False | 89.2% valid exact closes; required >= 90% |
| stale_close_rate | True | 0.0% stale closes; required <= 2% |
| targeted_close_captures | True | 16/16 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | False | 224 graded, 104 pending, 0 void |
| pitcher_anchor_settled_or_voided | False | 20 graded, 4 pending, 0 void |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 2874
- Valid exact closes: 2563 (89.2%)
- Stale close rate: 0.0%

| Close Reason | Rows |
|---|---:|
| valid_close | 2563 |
| player_prop_unavailable_at_close | 236 |
| line_disappeared_at_close | 49 |
| player_market_unavailable_at_close | 13 |
| fallback_other_book_only | 13 |

## Frozen Release Checkpoint

- Checkpoint status: `collecting`
- Hitter dates: 4 completed, 1 remaining
- Pitcher dates: 6 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
