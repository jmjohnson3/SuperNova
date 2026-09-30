# MLB Prop Five-Date Checkpoint

Generated UTC: 2026-07-12T23:09:45Z
Status: **evaluation_ready_no_micro_buckets**
Canonical phase: `day_pregame`
Frozen hitter artifact valid: **True**

A date counts only after every game is final or terminal non-played, and the release's anchor forecasts have zero pending rows.

## Hitter Release

Release: `hitter-pg-2026-07-06-r2`
Status: **evaluation_ready**
Completed dates: 6 / 5
Dates remaining: 0
Pending anchor rows: 21
Verified nonparticipant voids: 479
Strict clean completed dates: 0
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 1286 | 6 | 0.742 | 0.734 | -0.007 | -0.204 | False |
| batter_total_bases | 1286 | 6 | 1.309 | 1.435 | 0.126 | -0.187 | True |
| batter_home_runs | 1286 | 6 | 0.268 | 0.229 | -0.038 | 0.058 | False |
| hitter_plate_appearances | 1286 | 6 | 0.752 | 0.752 | 0.000 | 0.095 | False |
| hitter_plate_appearances_challenger | 1238 | 6 | 0.739 | 0.751 | 0.012 | 0.048 | True |

| Date | Graded | Pending | Voids | Games Settled | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-06 | 130 | 0 | 32 | True | True |
| 2026-07-07 | 256 | 0 | 69 | True | True |
| 2026-07-08 | 227 | 0 | 81 | True | True |
| 2026-07-09 | 210 | 0 | 61 | True | True |
| 2026-07-10 | 210 | 0 | 95 | True | True |
| 2026-07-11 | 253 | 0 | 75 | True | True |
| 2026-07-12 | 219 | 21 | 66 | False | False |

## Hitter Hits/HR Shadow

Release: `hitter-rate-shadow-2026-07-09-r1`
Status: **collecting**
Completed dates: 2 / 5
Dates remaining: 3
Pending anchor rows: 21
Verified nonparticipant voids: 236
Strict clean completed dates: 0
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 463 | 2 | 0.732 | 0.718 | -0.013 | -0.331 | False |
| batter_home_runs | 463 | 2 | 0.257 | 0.311 | 0.054 | 0.024 | True |

| Date | Graded | Pending | Voids | Games Settled | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-10 | 210 | 0 | 95 | True | True |
| 2026-07-11 | 253 | 0 | 75 | True | True |
| 2026-07-12 | 219 | 21 | 66 | False | False |

## Pitcher Release

Release: `pitcher-k-2026-07-03-r1`
Status: **evaluation_ready**
Completed dates: 8 / 5
Dates remaining: 0
Pending anchor rows: 2
Verified nonparticipant voids: 5
Strict clean completed dates: 0
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| pitcher_strikeouts | 173 | 8 | 1.960 | 1.947 | -0.013 | 0.219 | False |
| pitcher_batters_faced | 165 | 8 | 3.019 | 3.019 | 0.000 | 0.741 | False |
| pitcher_pitch_count | 165 | 8 | 12.160 | 17.504 | 5.344 | 2.675 | True |
| pitcher_innings | 169 | 8 | 0.991 | 1.215 | 0.223 | 0.175 | True |

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
| 2026-07-12 | 23 | 2 | 0 | False | False |

## Decision

The five-date evaluation is available, but no exact bucket currently passes every $1 micro gate.
