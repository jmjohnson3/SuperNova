# MLB Prop Five-Date Checkpoint

Generated UTC: 2026-07-20T20:08:46Z
Status: **evaluation_ready_no_micro_buckets**
Canonical phase: `day_pregame`
Frozen hitter artifact valid: **True**

A date counts only after every game is final or terminal non-played, and the release's anchor forecasts have zero pending rows.

## Hitter Release

Release: `hitter-pg-2026-07-06-r2`
Status: **evaluation_ready**
Completed dates: 11 / 5
Dates remaining: 0
Pending anchor rows: 292
Verified nonparticipant voids: 734
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 2221 | 11 | 0.726 | 0.725 | -0.001 | -0.070 | False |
| batter_total_bases | 2221 | 11 | 1.352 | 1.405 | 0.054 | 0.028 | True |
| batter_home_runs | 2221 | 11 | 0.263 | 0.228 | -0.034 | 0.059 | False |
| hitter_plate_appearances | 2221 | 11 | 0.735 | 0.742 | 0.007 | 0.075 | True |
| hitter_plate_appearances_challenger | 2109 | 11 | 0.723 | 0.745 | 0.022 | 0.040 | True |

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
| 2026-07-20 | 0 | 292 | 0 | False | False |

## Hitter Hits/HR Shadow

Release: `hitter-rate-shadow-2026-07-09-r1`
Status: **evaluation_ready**
Completed dates: 7 / 5
Dates remaining: 0
Pending anchor rows: 292
Verified nonparticipant voids: 491
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 1398 | 7 | 0.728 | 0.709 | -0.020 | -0.337 | False |
| batter_home_runs | 1398 | 7 | 0.246 | 0.306 | 0.060 | 0.036 | True |

| Date | Graded | Pending | Voids | Games Settled | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-10 | 210 | 0 | 95 | True | True |
| 2026-07-11 | 253 | 0 | 75 | True | True |
| 2026-07-12 | 234 | 0 | 72 | True | True |
| 2026-07-16 | 15 | 0 | 5 | True | True |
| 2026-07-17 | 224 | 0 | 77 | True | True |
| 2026-07-18 | 223 | 0 | 93 | True | True |
| 2026-07-19 | 239 | 0 | 74 | True | True |
| 2026-07-20 | 0 | 292 | 0 | False | False |

## Pitcher Release

Release: `pitcher-k-2026-07-03-r1`
Status: **evaluation_ready**
Completed dates: 13 / 5
Dates remaining: 0
Pending anchor rows: 24
Verified nonparticipant voids: 8
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| pitcher_strikeouts | 267 | 13 | 1.885 | 1.912 | 0.028 | 0.024 | True |
| pitcher_batters_faced | 257 | 13 | 3.199 | 3.199 | 0.000 | 0.858 | False |
| pitcher_pitch_count | 255 | 13 | 12.313 | 17.602 | 5.289 | 2.847 | True |
| pitcher_innings | 259 | 13 | 1.017 | 1.246 | 0.228 | 0.195 | True |

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
| 2026-07-20 | 0 | 24 | 0 | False | False |

## Decision

The five-date evaluation is available, but no exact bucket currently passes every $1 micro gate.
