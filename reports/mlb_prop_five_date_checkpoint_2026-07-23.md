# MLB Prop Five-Date Checkpoint

Generated UTC: 2026-07-23T22:13:17Z
Status: **evaluation_ready_no_micro_buckets**
Canonical phase: `day_pregame`
Frozen hitter artifact valid: **True**

A date counts only after every game is final or terminal non-played, and the release's anchor forecasts have zero pending rows.

## Hitter Release

Release: `hitter-pg-2026-07-06-r2`
Status: **evaluation_ready**
Completed dates: 14 / 5
Dates remaining: 0
Pending anchor rows: 83
Verified nonparticipant voids: 959
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 2910 | 14 | 0.727 | 0.722 | -0.004 | -0.046 | False |
| batter_total_bases | 2910 | 14 | 1.391 | 1.406 | 0.015 | 0.103 | True |
| batter_home_runs | 2910 | 14 | 0.261 | 0.229 | -0.032 | 0.054 | False |
| hitter_plate_appearances | 2910 | 14 | 0.742 | 0.752 | 0.011 | 0.082 | True |
| hitter_plate_appearances_challenger | 2771 | 14 | 0.734 | 0.756 | 0.023 | 0.052 | True |

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
| 2026-07-23 | 0 | 83 | 0 | False | False |

## Hitter Hits/HR Shadow

Release: `hitter-rate-shadow-2026-07-09-r1`
Status: **evaluation_ready**
Completed dates: 10 / 5
Dates remaining: 0
Pending anchor rows: 83
Verified nonparticipant voids: 716
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 2087 | 10 | 0.734 | 0.716 | -0.018 | -0.322 | False |
| batter_home_runs | 2087 | 10 | 0.249 | 0.312 | 0.063 | 0.035 | True |

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
| 2026-07-23 | 0 | 83 | 0 | False | False |

## Pitcher Release

Release: `pitcher-k-2026-07-03-r1`
Status: **evaluation_ready**
Completed dates: 16 / 5
Dates remaining: 0
Pending anchor rows: 6
Verified nonparticipant voids: 11
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| pitcher_strikeouts | 343 | 16 | 1.831 | 1.870 | 0.039 | 0.064 | True |
| pitcher_batters_faced | 328 | 16 | 3.179 | 3.179 | 0.000 | 0.667 | False |
| pitcher_pitch_count | 326 | 16 | 12.335 | 17.377 | 5.043 | 2.359 | True |
| pitcher_innings | 334 | 16 | 0.992 | 1.204 | 0.213 | 0.135 | True |

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
| 2026-07-23 | 0 | 6 | 0 | False | False |

## Decision

The five-date evaluation is available, but no exact bucket currently passes every $1 micro gate.
