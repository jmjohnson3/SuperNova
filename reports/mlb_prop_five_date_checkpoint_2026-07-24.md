# MLB Prop Five-Date Checkpoint

Generated UTC: 2026-07-24T23:23:30Z
Status: **evaluation_ready_no_micro_buckets**
Canonical phase: `day_pregame`
Frozen hitter artifact valid: **True**

A date counts only after every game is final or terminal non-played, and the release's anchor forecasts have zero pending rows.

## Hitter Release

Release: `hitter-pg-2026-07-06-r2`
Status: **evaluation_ready**
Completed dates: 15 / 5
Dates remaining: 0
Pending anchor rows: 298
Verified nonparticipant voids: 982
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 2970 | 15 | 0.725 | 0.722 | -0.002 | -0.042 | False |
| batter_total_bases | 2970 | 15 | 1.391 | 1.405 | 0.014 | 0.113 | True |
| batter_home_runs | 2970 | 15 | 0.261 | 0.229 | -0.032 | 0.053 | False |
| hitter_plate_appearances | 2970 | 15 | 0.741 | 0.753 | 0.012 | 0.079 | True |
| hitter_plate_appearances_challenger | 2831 | 15 | 0.733 | 0.757 | 0.024 | 0.050 | True |

| Date | Graded | Pending | Voids | Games Settled | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-06 | 130 | 0 | 32 | True | True |
| 2026-07-07 | 256 | 0 | 69 | True | True |
| 2026-07-08 | 227 | 0 | 81 | True | True |
| 2026-07-09 | 210 | 0 | 61 | True | True |
| 2026-07-10 | 210 | 0 | 95 | True | True |
| 2026-07-11 | 253 | 0 | 75 | True | True |
| 2026-07-12 | 234 | 0 | 72 | True | True |
| 2026-07-16 | 15 | 0 | 5 | True | True |
| 2026-07-17 | 224 | 0 | 77 | True | True |
| 2026-07-18 | 223 | 0 | 93 | True | True |
| 2026-07-19 | 239 | 0 | 74 | True | True |
| 2026-07-20 | 234 | 0 | 58 | True | True |
| 2026-07-21 | 199 | 0 | 95 | True | True |
| 2026-07-22 | 256 | 0 | 72 | True | True |
| 2026-07-23 | 60 | 0 | 23 | True | True |
| 2026-07-24 | 0 | 298 | 0 | False | False |

## Hitter Hits/HR Shadow

Release: `hitter-rate-shadow-2026-07-09-r1`
Status: **evaluation_ready**
Completed dates: 11 / 5
Dates remaining: 0
Pending anchor rows: 298
Verified nonparticipant voids: 739
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 2147 | 11 | 0.731 | 0.713 | -0.018 | -0.320 | False |
| batter_home_runs | 2147 | 11 | 0.249 | 0.312 | 0.063 | 0.035 | True |

| Date | Graded | Pending | Voids | Games Settled | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-10 | 210 | 0 | 95 | True | True |
| 2026-07-11 | 253 | 0 | 75 | True | True |
| 2026-07-12 | 234 | 0 | 72 | True | True |
| 2026-07-16 | 15 | 0 | 5 | True | True |
| 2026-07-17 | 224 | 0 | 77 | True | True |
| 2026-07-18 | 223 | 0 | 93 | True | True |
| 2026-07-19 | 239 | 0 | 74 | True | True |
| 2026-07-20 | 234 | 0 | 58 | True | True |
| 2026-07-21 | 199 | 0 | 95 | True | True |
| 2026-07-22 | 256 | 0 | 72 | True | True |
| 2026-07-23 | 60 | 0 | 23 | True | True |
| 2026-07-24 | 0 | 298 | 0 | False | False |

## Pitcher Release

Release: `pitcher-k-2026-07-03-r1`
Status: **evaluation_ready**
Completed dates: 17 / 5
Dates remaining: 0
Pending anchor rows: 20
Verified nonparticipant voids: 11
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| pitcher_strikeouts | 349 | 17 | 1.836 | 1.876 | 0.041 | 0.042 | True |
| pitcher_batters_faced | 334 | 17 | 3.179 | 3.179 | 0.000 | 0.659 | False |
| pitcher_pitch_count | 331 | 17 | 12.321 | 17.475 | 5.154 | 2.300 | True |
| pitcher_innings | 339 | 17 | 0.997 | 1.210 | 0.213 | 0.130 | True |

| Date | Graded | Pending | Voids | Games Settled | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-04 | 22 | 0 | 2 | True | True |
| 2026-07-05 | 25 | 0 | 0 | True | True |
| 2026-07-06 | 14 | 0 | 0 | True | True |
| 2026-07-07 | 20 | 0 | 0 | True | True |
| 2026-07-08 | 24 | 0 | 0 | True | True |
| 2026-07-09 | 22 | 0 | 0 | True | True |
| 2026-07-10 | 22 | 0 | 3 | True | True |
| 2026-07-11 | 24 | 0 | 0 | True | True |
| 2026-07-12 | 25 | 0 | 0 | True | True |
| 2026-07-16 | 1 | 0 | 0 | True | True |
| 2026-07-17 | 23 | 0 | 1 | True | True |
| 2026-07-18 | 20 | 0 | 2 | True | True |
| 2026-07-19 | 25 | 0 | 0 | True | True |
| 2026-07-20 | 24 | 0 | 0 | True | True |
| 2026-07-21 | 24 | 0 | 3 | True | True |
| 2026-07-22 | 28 | 0 | 0 | True | True |
| 2026-07-23 | 6 | 0 | 0 | True | True |
| 2026-07-24 | 0 | 20 | 0 | False | False |

## Decision

The five-date evaluation is available, but no exact bucket currently passes every $1 micro gate.
