# MLB Prop Five-Date Checkpoint

Generated UTC: 2026-07-11T23:02:48Z
Status: **collecting**
Canonical phase: `day_pregame`
Frozen hitter artifact valid: **True**

A date counts only after every game is final and the release's anchor forecasts have zero pending rows.

## Hitter Release

Release: `hitter-pg-2026-07-06-r2`
Status: **collecting**
Completed dates: 4 / 5
Dates remaining: 1
Pending anchor rows: 358
Verified nonparticipant voids: 243
Strict clean completed dates: 0
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 823 | 4 | 0.755 | 0.731 | -0.024 | -0.390 | False |
| batter_total_bases | 823 | 4 | 1.285 | 1.404 | 0.118 | -0.186 | True |
| batter_home_runs | 823 | 4 | 0.244 | 0.226 | -0.018 | 0.032 | False |
| hitter_plate_appearances | 823 | 4 | 0.757 | 0.757 | 0.000 | 0.036 | False |
| hitter_plate_appearances_challenger | 790 | 4 | 0.767 | 0.755 | -0.013 | 0.005 | False |

| Date | Graded | Pending | Voids | Games Final | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-06 | 130 | 0 | 32 | True | True |
| 2026-07-07 | 256 | 0 | 69 | True | True |
| 2026-07-08 | 227 | 0 | 81 | True | True |
| 2026-07-09 | 210 | 0 | 61 | True | True |
| 2026-07-10 | 210 | 95 | 0 | False | False |
| 2026-07-11 | 65 | 263 | 0 | False | False |

## Hitter Hits/HR Shadow

Release: `hitter-rate-shadow-2026-07-09-r1`
Status: **collecting**
Completed dates: 0 / 5
Dates remaining: 5
Pending anchor rows: 358
Verified nonparticipant voids: 0
Strict clean completed dates: 0
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 0 | 0 | - | - | - | - | False |
| batter_home_runs | 0 | 0 | - | - | - | - | False |

| Date | Graded | Pending | Voids | Games Final | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-10 | 210 | 95 | 0 | False | False |
| 2026-07-11 | 65 | 263 | 0 | False | False |

## Pitcher Release

Release: `pitcher-k-2026-07-03-r1`
Status: **evaluation_ready**
Completed dates: 6 / 5
Dates remaining: 0
Pending anchor rows: 23
Verified nonparticipant voids: 2
Strict clean completed dates: 0
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| pitcher_strikeouts | 127 | 6 | 2.005 | 2.007 | 0.002 | 0.102 | True |
| pitcher_batters_faced | 122 | 6 | 2.914 | 2.914 | 0.000 | 0.854 | False |
| pitcher_pitch_count | 122 | 6 | 11.919 | 17.449 | 5.530 | 2.424 | True |
| pitcher_innings | 123 | 6 | 1.009 | 1.236 | 0.227 | 0.211 | True |

| Date | Graded | Pending | Voids | Games Final | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-04 | 22 | 0 | 2 | True | True |
| 2026-07-05 | 25 | 0 | 0 | True | True |
| 2026-07-06 | 14 | 0 | 0 | True | True |
| 2026-07-07 | 20 | 0 | 0 | True | True |
| 2026-07-08 | 24 | 0 | 0 | True | True |
| 2026-07-09 | 22 | 0 | 0 | True | True |
| 2026-07-10 | 22 | 3 | 0 | False | False |
| 2026-07-11 | 4 | 20 | 0 | False | False |

## Decision

Keep production frozen and continue prospective collection. No $1 micro promotion is allowed yet.
