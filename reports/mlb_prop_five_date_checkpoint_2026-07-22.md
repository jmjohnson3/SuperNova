# MLB Prop Five-Date Checkpoint

Generated UTC: 2026-07-22T23:20:48Z
Status: **evaluation_ready_no_micro_buckets**
Canonical phase: `day_pregame`
Frozen hitter artifact valid: **True**

A date counts only after every game is final or terminal non-played, and the release's anchor forecasts have zero pending rows.

## Hitter Release

Release: `hitter-pg-2026-07-06-r2`
Status: **evaluation_ready**
Completed dates: 13 / 5
Dates remaining: 0
Pending anchor rows: 174
Verified nonparticipant voids: 923
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 2654 | 13 | 0.728 | 0.725 | -0.003 | -0.061 | False |
| batter_total_bases | 2654 | 13 | 1.383 | 1.413 | 0.030 | 0.065 | True |
| batter_home_runs | 2654 | 13 | 0.263 | 0.231 | -0.032 | 0.053 | False |
| hitter_plate_appearances | 2654 | 13 | 0.740 | 0.750 | 0.011 | 0.080 | True |
| hitter_plate_appearances_challenger | 2542 | 13 | 0.730 | 0.753 | 0.024 | 0.051 | True |

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
| 2026-07-22 | 118 | 174 | 36 | False | False |

## Hitter Hits/HR Shadow

Release: `hitter-rate-shadow-2026-07-09-r1`
Status: **evaluation_ready**
Completed dates: 9 / 5
Dates remaining: 0
Pending anchor rows: 174
Verified nonparticipant voids: 680
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 1831 | 9 | 0.735 | 0.716 | -0.018 | -0.319 | False |
| batter_home_runs | 1831 | 9 | 0.251 | 0.312 | 0.061 | 0.032 | True |

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
| 2026-07-22 | 118 | 174 | 36 | False | False |

## Pitcher Release

Release: `pitcher-k-2026-07-03-r1`
Status: **evaluation_ready**
Completed dates: 15 / 5
Dates remaining: 0
Pending anchor rows: 15
Verified nonparticipant voids: 11
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| pitcher_strikeouts | 315 | 15 | 1.827 | 1.894 | 0.067 | 0.068 | True |
| pitcher_batters_faced | 302 | 15 | 3.208 | 3.208 | 0.000 | 0.738 | False |
| pitcher_pitch_count | 300 | 15 | 12.713 | 17.605 | 4.893 | 2.340 | True |
| pitcher_innings | 306 | 15 | 1.016 | 1.231 | 0.215 | 0.161 | True |

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
| 2026-07-22 | 13 | 15 | 0 | False | False |

## Decision

The five-date evaluation is available, but no exact bucket currently passes every $1 micro gate.
