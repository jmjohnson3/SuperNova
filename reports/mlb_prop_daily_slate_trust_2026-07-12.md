# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-13T01:08:31Z
Slate: 2026-07-12
Status: **PASS**
Decision: `slate_trusted_for_evaluation`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | True | 15/15 games final |
| valid_close_coverage | True | 91.0% valid exact closes; required >= 90% |
| stale_close_rate | True | 0.0% stale closes; required <= 2% |
| targeted_close_captures | True | 15/15 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | True | 234 graded, 0 pending, 72 void |
| pitcher_anchor_settled_or_voided | True | 25 graded, 0 pending, 0 void |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 2601
- Valid exact closes: 2366 (91.0%)
- Stale close rate: 0.0%

| Close Reason | Rows |
|---|---:|
| valid_close | 2366 |
| player_prop_unavailable_at_close | 200 |
| line_disappeared_at_close | 29 |
| player_market_unavailable_at_close | 3 |
| fallback_other_book_only | 3 |

## Frozen Release Checkpoint

- Checkpoint status: `evaluation_ready_no_micro_buckets`
- Hitter dates: 7 completed, 0 remaining
- Pitcher dates: 9 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
