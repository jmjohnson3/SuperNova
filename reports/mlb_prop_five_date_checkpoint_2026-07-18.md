# MLB Prop Five-Date Checkpoint

Generated UTC: 2026-07-18T23:07:12Z
Status: **evaluation_ready_no_micro_buckets**
Canonical phase: `day_pregame`
Frozen hitter artifact valid: **True**

A date counts only after every game is final or terminal non-played, and the release's anchor forecasts have zero pending rows.

## Hitter Release

Release: `hitter-pg-2026-07-06-r2`
Status: **evaluation_ready**
Completed dates: 9 / 5
Dates remaining: 0
Pending anchor rows: 178
Verified nonparticipant voids: 619
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 1759 | 9 | 0.729 | 0.723 | -0.007 | -0.111 | False |
| batter_total_bases | 1759 | 9 | 1.317 | 1.406 | 0.089 | -0.077 | True |
| batter_home_runs | 1759 | 9 | 0.267 | 0.228 | -0.039 | 0.067 | False |
| hitter_plate_appearances | 1759 | 9 | 0.744 | 0.742 | -0.002 | 0.087 | False |
| hitter_plate_appearances_challenger | 1678 | 9 | 0.728 | 0.745 | 0.017 | 0.048 | True |

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
| 2026-07-18 | 86 | 178 | 52 | False | False |

## Hitter Hits/HR Shadow

Release: `hitter-rate-shadow-2026-07-09-r1`
Status: **evaluation_ready**
Completed dates: 5 / 5
Dates remaining: 0
Pending anchor rows: 178
Verified nonparticipant voids: 376
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 936 | 5 | 0.719 | 0.707 | -0.013 | -0.322 | False |
| batter_home_runs | 936 | 5 | 0.246 | 0.304 | 0.058 | 0.038 | True |

| Date | Graded | Pending | Voids | Games Settled | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-10 | 210 | 0 | 95 | True | True |
| 2026-07-11 | 253 | 0 | 75 | True | True |
| 2026-07-12 | 234 | 0 | 72 | True | True |
| 2026-07-16 | 15 | 0 | 5 | True | True |
| 2026-07-17 | 224 | 0 | 77 | True | True |
| 2026-07-18 | 86 | 178 | 52 | False | False |

## Pitcher Release

Release: `pitcher-k-2026-07-03-r1`
Status: **evaluation_ready**
Completed dates: 11 / 5
Dates remaining: 0
Pending anchor rows: 13
Verified nonparticipant voids: 8
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| pitcher_strikeouts | 222 | 11 | 1.942 | 1.943 | 0.001 | 0.102 | True |
| pitcher_batters_faced | 214 | 11 | 3.138 | 3.138 | 0.000 | 0.948 | False |
| pitcher_pitch_count | 212 | 11 | 12.383 | 17.857 | 5.474 | 2.753 | True |
| pitcher_innings | 216 | 11 | 0.997 | 1.228 | 0.232 | 0.174 | True |

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
| 2026-07-18 | 7 | 13 | 2 | False | False |

## Decision

The five-date evaluation is available, but no exact bucket currently passes every $1 micro gate.
