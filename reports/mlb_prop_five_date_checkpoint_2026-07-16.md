# MLB Prop Five-Date Checkpoint

Generated UTC: 2026-07-16T23:14:01Z
Status: **evaluation_ready_no_micro_buckets**
Canonical phase: `day_pregame`
Frozen hitter artifact valid: **True**

A date counts only after every game is final or terminal non-played, and the release's anchor forecasts have zero pending rows.

## Hitter Release

Release: `hitter-pg-2026-07-06-r2`
Status: **evaluation_ready**
Completed dates: 7 / 5
Dates remaining: 0
Pending anchor rows: 20
Verified nonparticipant voids: 485
Strict clean completed dates: 0
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 1520 | 7 | 0.735 | 0.731 | -0.003 | -0.144 | False |
| batter_total_bases | 1520 | 7 | 1.291 | 1.414 | 0.123 | -0.159 | True |
| batter_home_runs | 1520 | 7 | 0.271 | 0.227 | -0.044 | 0.072 | False |
| hitter_plate_appearances | 1520 | 7 | 0.741 | 0.741 | 0.000 | 0.101 | False |
| hitter_plate_appearances_challenger | 1472 | 7 | 0.718 | 0.739 | 0.021 | 0.058 | True |

| Date | Graded | Pending | Voids | Games Settled | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-06 | 130 | 0 | 32 | True | True |
| 2026-07-07 | 256 | 0 | 69 | True | True |
| 2026-07-08 | 227 | 0 | 81 | True | True |
| 2026-07-09 | 210 | 0 | 61 | True | True |
| 2026-07-10 | 210 | 0 | 95 | True | True |
| 2026-07-11 | 253 | 0 | 75 | True | True |
| 2026-07-12 | 234 | 0 | 72 | True | True |
| 2026-07-16 | 0 | 20 | 0 | False | False |

## Hitter Hits/HR Shadow

Release: `hitter-rate-shadow-2026-07-09-r1`
Status: **collecting**
Completed dates: 3 / 5
Dates remaining: 2
Pending anchor rows: 20
Verified nonparticipant voids: 242
Strict clean completed dates: 0
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 697 | 3 | 0.720 | 0.711 | -0.009 | -0.313 | False |
| batter_home_runs | 697 | 3 | 0.248 | 0.304 | 0.056 | 0.040 | True |

| Date | Graded | Pending | Voids | Games Settled | Complete |
|---|---:|---:|---:|---|---|
| 2026-07-10 | 210 | 0 | 95 | True | True |
| 2026-07-11 | 253 | 0 | 75 | True | True |
| 2026-07-12 | 234 | 0 | 72 | True | True |
| 2026-07-16 | 0 | 20 | 0 | False | False |

## Pitcher Release

Release: `pitcher-k-2026-07-03-r1`
Status: **evaluation_ready**
Completed dates: 9 / 5
Dates remaining: 0
Pending anchor rows: 1
Verified nonparticipant voids: 5
Strict clean completed dates: 0
Checkpoint-ready $1 micro buckets: 0

| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| pitcher_strikeouts | 198 | 9 | 1.988 | 1.970 | -0.018 | 0.241 | False |
| pitcher_batters_faced | 190 | 9 | 3.211 | 3.211 | 0.000 | 1.039 | False |
| pitcher_pitch_count | 189 | 9 | 12.558 | 17.790 | 5.232 | 3.333 | True |
| pitcher_innings | 193 | 9 | 1.027 | 1.238 | 0.211 | 0.220 | True |

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
| 2026-07-16 | 0 | 1 | 0 | False | False |

## Decision

The five-date evaluation is available, but no exact bucket currently passes every $1 micro gate.
