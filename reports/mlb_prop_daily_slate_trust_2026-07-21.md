# MLB Prop Daily Slate Trust Monitor

Generated UTC: 2026-07-21T11:19:31Z
Slate: 2026-07-21
Status: **PROVISIONAL**
Decision: `wait_for_final_results_or_repair_failed_checks`

## Required Checks

| Check | Pass | Detail |
|---|---|---|
| all_games_finalized | False | 0/15 games final |
| valid_close_coverage | False | 0.0% valid exact closes; required >= 90% |
| stale_close_rate | False | 0.0% stale closes; required <= 2% |
| targeted_close_captures | False | 0/13 events have T-120/T-60/T-20 captures |
| hitter_anchor_settled_or_voided | False | missing hitter anchor row |
| pitcher_anchor_settled_or_voided | False | missing pitcher anchor row |
| frozen_hitter_artifact_valid | True | sha256 matched |

## Close Quality

- Locked executable offers: 0
- Valid exact closes: 0 (-)
- Stale close rate: -

| Close Reason | Rows |
|---|---:|

## Frozen Release Checkpoint

- Checkpoint status: `evaluation_ready_no_micro_buckets`
- Hitter dates: 11 completed, 0 remaining
- Pitcher dates: 13 completed, 0 remaining
- Hitter $1 micro buckets: 0
- Pitcher $1 micro buckets: 0
