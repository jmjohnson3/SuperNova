# MLB Prop Five-Date Checkpoint

Generated UTC: 2026-08-20T08:38:05Z
Status: **evaluation_ready_no_micro_buckets**
Canonical phase: `day_pregame`
Frozen hitter artifact valid: **True**

A date counts only after every game is final or terminal non-played, and the release's anchor forecasts have zero pending rows.

## Hitter Release

Release: `hitter-pg-2026-07-06-r2`
Status: **evaluation_ready**
Completed dates: 21 / 5
Dates remaining: 0
Pending anchor rows: 0
Verified nonparticipant voids: 1547
Strict clean completed dates: 2
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 4299 | 21 | 0.720 | 0.728 | 0.007 | -0.005 | True |
| batter_total_bases | 4299 | 21 | 1.398 | 1.385 | -0.013 | 0.234 | False |
| batter_home_runs | 4299 | 21 | 0.249 | 0.217 | -0.032 | 0.060 | False |
| hitter_plate_appearances | 4299 | 21 | 0.753 | 0.757 | 0.004 | 0.082 | True |
| hitter_plate_appearances_challenger | 4136 | 21 | 0.749 | 0.760 | 0.011 | 0.049 | True |

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
| 2026-07-24 | 235 | 0 | 63 | True | True |
| 2026-07-25 | 232 | 0 | 64 | True | True |
| 2026-07-26 | 230 | 0 | 57 | True | True |
| 2026-07-27 | 161 | 0 | 61 | True | True |
| 2026-07-28 | 230 | 0 | 78 | True | True |
| 2026-07-29 | 241 | 0 | 59 | True | True |
| 2026-07-30 | 0 | 0 | 183 | True | False |

## Hitter Hits/HR Shadow

Release: `hitter-rate-shadow-2026-07-09-r1`
Status: **evaluation_ready**
Completed dates: 17 / 5
Dates remaining: 0
Pending anchor rows: 0
Verified nonparticipant voids: 1304
Strict clean completed dates: 2
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 3476 | 17 | 0.730 | 0.714 | -0.016 | -0.310 | False |
| batter_home_runs | 3476 | 17 | 0.238 | 0.306 | 0.067 | 0.050 | True |

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
| 2026-07-24 | 235 | 0 | 63 | True | True |
| 2026-07-25 | 232 | 0 | 64 | True | True |
| 2026-07-26 | 230 | 0 | 57 | True | True |
| 2026-07-27 | 161 | 0 | 61 | True | True |
| 2026-07-28 | 230 | 0 | 78 | True | True |
| 2026-07-29 | 241 | 0 | 59 | True | True |
| 2026-07-30 | 0 | 0 | 183 | True | False |

## Pitcher Release

Release: `pitcher-k-2026-07-03-r1`
Status: **evaluation_ready**
Completed dates: 23 / 5
Dates remaining: 0
Pending anchor rows: 0
Verified nonparticipant voids: 31
Strict clean completed dates: 2
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| pitcher_strikeouts | 476 | 23 | 1.831 | 1.918 | 0.087 | 0.128 | True |
| pitcher_batters_faced | 456 | 23 | 3.085 | 3.085 | 0.000 | 0.526 | False |
| pitcher_pitch_count | 452 | 23 | 12.335 | 17.366 | 5.031 | 1.591 | True |
| pitcher_innings | 464 | 23 | 0.989 | 1.197 | 0.207 | 0.076 | True |

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
| 2026-07-24 | 19 | 0 | 1 | True | True |
| 2026-07-25 | 24 | 0 | 0 | True | True |
| 2026-07-26 | 25 | 0 | 0 | True | True |
| 2026-07-27 | 17 | 0 | 2 | True | True |
| 2026-07-28 | 22 | 0 | 1 | True | True |
| 2026-07-29 | 20 | 0 | 1 | True | True |
| 2026-07-30 | 0 | 0 | 15 | True | False |

## Decision

The five-date evaluation is available, but no exact bucket currently passes every $1 micro gate.
