# MLB Prop Five-Date Checkpoint

Generated UTC: 2026-07-09T23:21:57Z
Status: **collecting**
Canonical phase: `day_pregame`
Frozen hitter artifact valid: **True**

A date counts only after every game is final and the release's anchor forecasts have zero pending rows.

## Hitter Release

Release: `hitter-pg-2026-07-06-r2`
Status: **collecting**
Completed dates: 3 / 5
Dates remaining: 2
Pending anchor rows: 170
Verified nonparticipant voids: 182
Strict clean completed dates: 0
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 613 | 3 | 0.785 | 0.744 | -0.040 | -0.431 | False |
| batter_total_bases | 613 | 3 | 1.295 | 1.410 | 0.115 | -0.222 | True |
| batter_home_runs | 613 | 3 | 0.239 | 0.224 | -0.015 | 0.038 | False |
| hitter_plate_appearances | 613 | 3 | 0.775 | 0.775 | 0.000 | -0.009 | False |
| hitter_plate_appearances_challenger | 580 | 3 | 0.779 | 0.773 | -0.006 | -0.038 | False |

| Date | Graded | Pending | Voids | Games Final | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-06 | 130 | 0 | 32 | True | True |
| 2026-07-07 | 256 | 0 | 69 | True | True |
| 2026-07-08 | 227 | 0 | 81 | True | True |
| 2026-07-09 | 101 | 170 | 0 | False | False |

## Hitter Hits/HR Shadow

Release: `hitter-rate-shadow-2026-07-09-r1`
Status: **collecting**
Completed dates: 0 / 5
Dates remaining: 5
Pending anchor rows: 0
Verified nonparticipant voids: 0
Strict clean completed dates: 0
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 0 | 0 | - | - | - | - | False |
| batter_home_runs | 0 | 0 | - | - | - | - | False |

| Date | Graded | Pending | Voids | Games Final | Complete |
|---|---:|---:|---:|---|---|

## Pitcher Release

Release: `pitcher-k-2026-07-03-r1`
Status: **evaluation_ready**
Completed dates: 5 / 5
Dates remaining: 0
Pending anchor rows: 12
Verified nonparticipant voids: 2
Strict clean completed dates: 0
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| pitcher_strikeouts | 105 | 5 | 2.007 | 1.976 | -0.031 | 0.172 | False |
| pitcher_batters_faced | 100 | 5 | 2.699 | 2.699 | 0.000 | 0.729 | False |
| pitcher_pitch_count | 100 | 5 | 11.902 | 17.761 | 5.859 | 1.486 | True |
| pitcher_innings | 101 | 5 | 0.925 | 1.168 | 0.243 | 0.183 | True |

| Date | Graded | Pending | Voids | Games Final | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-04 | 22 | 0 | 2 | True | True |
| 2026-07-05 | 25 | 0 | 0 | True | True |
| 2026-07-06 | 14 | 0 | 0 | True | True |
| 2026-07-07 | 20 | 0 | 0 | True | True |
| 2026-07-08 | 24 | 0 | 0 | True | True |
| 2026-07-09 | 10 | 12 | 0 | False | False |

## Decision

Keep production frozen and continue prospective collection. No $1 micro promotion is allowed yet.
