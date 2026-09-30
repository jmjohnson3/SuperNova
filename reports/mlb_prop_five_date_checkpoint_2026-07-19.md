# MLB Prop Five-Date Checkpoint

Generated UTC: 2026-07-19T23:09:40Z
Status: **evaluation_ready_no_micro_buckets**
Canonical phase: `day_pregame`
Frozen hitter artifact valid: **True**

A date counts only after every game is final or terminal non-played, and the release's anchor forecasts have zero pending rows.

## Hitter Release

Release: `hitter-pg-2026-07-06-r2`
Status: **evaluation_ready**
Completed dates: 10 / 5
Dates remaining: 0
Pending anchor rows: 94
Verified nonparticipant voids: 714
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 1982 | 10 | 0.729 | 0.723 | -0.006 | -0.084 | False |
| batter_total_bases | 1982 | 10 | 1.336 | 1.397 | 0.062 | -0.007 | True |
| batter_home_runs | 1982 | 10 | 0.263 | 0.227 | -0.037 | 0.066 | False |
| hitter_plate_appearances | 1982 | 10 | 0.737 | 0.739 | 0.003 | 0.086 | True |
| hitter_plate_appearances_challenger | 1884 | 10 | 0.721 | 0.740 | 0.019 | 0.048 | True |

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
| 2026-07-19 | 165 | 94 | 54 | False | False |

## Hitter Hits/HR Shadow

Release: `hitter-rate-shadow-2026-07-09-r1`
Status: **evaluation_ready**
Completed dates: 6 / 5
Dates remaining: 0
Pending anchor rows: 94
Verified nonparticipant voids: 471
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 1159 | 6 | 0.724 | 0.710 | -0.014 | -0.324 | False |
| batter_home_runs | 1159 | 6 | 0.244 | 0.305 | 0.061 | 0.042 | True |

| Date | Graded | Pending | Voids | Games Settled | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-10 | 210 | 0 | 95 | True | True |
| 2026-07-11 | 253 | 0 | 75 | True | True |
| 2026-07-12 | 234 | 0 | 72 | True | True |
| 2026-07-16 | 15 | 0 | 5 | True | True |
| 2026-07-17 | 224 | 0 | 77 | True | True |
| 2026-07-18 | 223 | 0 | 93 | True | True |
| 2026-07-19 | 165 | 94 | 54 | False | False |

## Pitcher Release

Release: `pitcher-k-2026-07-03-r1`
Status: **evaluation_ready**
Completed dates: 12 / 5
Dates remaining: 0
Pending anchor rows: 7
Verified nonparticipant voids: 8
Strict clean completed dates: 1
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| pitcher_strikeouts | 242 | 12 | 1.912 | 1.918 | 0.007 | 0.017 | True |
| pitcher_batters_faced | 232 | 12 | 3.175 | 3.175 | 0.000 | 0.866 | False |
| pitcher_pitch_count | 230 | 12 | 12.254 | 17.761 | 5.507 | 2.699 | True |
| pitcher_innings | 234 | 12 | 0.996 | 1.233 | 0.237 | 0.174 | True |

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
| 2026-07-19 | 18 | 7 | 0 | False | False |

## Decision

The five-date evaluation is available, but no exact bucket currently passes every $1 micro gate.
