# NFL Receiver Spike Miss Diagnostic

This audits receiving-yards misses on one row per player-game, before exact prop lines, so duplicated offers cannot fake evidence.

- Status: ready
- Training rows: 29073
- Model artifact status: ready
- Opportunity artifact status: ready
- Built at: 2026-09-16T02:17:31Z

## Summary

| Holdout Rows | Model MAE | Baseline MAE | Gain | Bias | Big Miss Rows | Missed Spike Rows | Target MAE | Target Bias | Spike Model |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 194 | 18.500 | 21.426 | 2.926 | -3.267 | 54 | 31 | 2.053 | 0.672 | yes |

## Big-Miss Causes

| Cause | Rows | Model MAE | Baseline MAE | Gain | Bias | Target Err | Route Err | Air Err | YPT Err | Spike P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| target_projection_over | 13 | 44.568 | 40.240 | -4.328 | 44.568 | -4.812 | -0.175 | -57.523 | -8.976 | 0.238 |
| target_projection_under | 11 | 68.345 | 58.268 | -10.077 | -68.345 | 4.290 | 0.096 | 48.436 | 17.409 | 0.032 |
| air_yards_role_under | 6 | 42.898 | 34.600 | -8.299 | -42.898 | 0.355 | -0.157 | 53.878 | 23.683 | 0.076 |
| yards_per_target_efficiency_under | 5 | 50.800 | 46.910 | -3.890 | -50.800 | -1.263 | -0.244 | 4.880 | 44.040 | 0.237 |
| route_snap_projection_over | 5 | 31.142 | 43.200 | 12.059 | 31.142 | -1.887 | -0.279 | -40.960 | -23.498 | 0.223 |
| player_rate_or_noise | 5 | 28.938 | 22.663 | -6.275 | -28.938 | 1.726 | -0.088 | -0.480 | 2.318 | 0.024 |
| game_script_pass_volume_over | 5 | 27.227 | 20.035 | -7.192 | 27.227 | 0.056 | -0.067 | -4.640 | -39.107 | 0.262 |
| route_snap_projection_under | 4 | 55.997 | 41.818 | -14.179 | -55.997 | 0.046 | 0.210 | 31.500 | 53.151 | 0.068 |

## Missed Spike Rows

| Week | Player | Team | Actual | Projection | Baseline | Error | Cause | Tgt Act/Pred | Route Act/Prior | Air Err | YPT Err | Spike P |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 2026-1 | C.Olave | NO | 182.0 | 29.8 | 65.6 | -152.2 | target_projection_under | 13.0/6.2 | 0.86/0.60 | 176.2 | 85.8 | 0.00 |
| 2026-1 | C.Watson | GB | 147.0 | 31.2 | 30.2 | -115.8 | target_projection_under | 8.0/5.3 | 0.76/0.67 | 62.0 | 47.2 | 0.04 |
| 2026-1 | D.Kincaid | BUF | 130.0 | 15.9 | 22.4 | -114.1 | route_snap_projection_under | 6.0/3.6 | 0.72/0.57 | 67.6 | 93.3 | 0.00 |
| 2026-1 | J.Coker | CAR | 138.0 | 35.6 | 50.6 | -102.4 | target_projection_under | 9.0/5.4 | 0.84/0.90 | 71.2 | 55.7 | 0.25 |
| 2026-1 | Z.Flowers | BAL | 150.0 | 55.3 | 65.6 | -94.7 | yards_per_target_efficiency_under | 6.0/7.4 | 0.29/0.77 | 12.6 | 94.1 | 0.44 |
| 2026-1 | Bi.Robinson | ATL | 90.0 | 13.8 | 15.6 | -76.2 | target_projection_under | 10.0/3.8 | 0.77/0.90 | -40.4 | 65.6 | 0.00 |
| 2026-1 | K.Raymond | CHI | 84.0 | 19.1 | 30.2 | -64.9 | target_projection_under | 9.0/3.5 | 0.60/0.32 | 54.6 | -17.2 | 0.00 |
| 2026-1 | J.Smith-Njigba | SEA | 122.0 | 63.6 | 65.6 | -58.4 | yards_per_target_efficiency_under | 11.0/10.0 | 0.90/0.97 | 13.8 | 39.2 | 0.55 |
| 2026-1 | K.Bourne | ARI | 75.0 | 17.6 | 30.2 | -57.4 | target_projection_under | 8.0/3.7 | 0.67/0.62 | 13.6 | -2.1 | 0.00 |
| 2026-1 | P.Washington | JAX | 83.0 | 28.7 | 30.2 | -54.3 | air_yards_role_under | 6.0/5.4 | 0.61/0.96 | 57.2 | 34.7 | 0.06 |
| 2026-1 | I.Likely | NYG | 78.0 | 25.9 | 22.4 | -52.1 | target_projection_under | 8.0/4.2 | 0.49/0.76 | 24.6 | -11.8 | 0.00 |
| 2026-1 | M.Gesicki | CIN | 78.0 | 26.3 | 33.8 | -51.7 | air_yards_role_under | 7.0/4.9 | 0.33/0.55 | 29.6 | 26.3 | 0.00 |
| 2026-1 | D.Vele | NO | 69.0 | 19.2 | 23.2 | -49.8 | target_projection_under | 9.0/4.3 | 0.91/0.62 | 41.6 | -4.5 | 0.00 |
| 2026-1 | A.Mitchell | NYJ | 60.0 | 12.5 | 30.2 | -47.5 | route_snap_projection_under | 3.0/3.0 | 0.69/0.45 | 21.8 | 45.5 | 0.00 |
| 2026-1 | J.Brooks | CAR | 48.0 | 1.6 | 9.6 | -46.4 | air_yards_role_under | 2.0/1.9 | 0.19/0.10 | 32.7 | 32.7 | 0.00 |

## Biggest Absolute Misses

| Week | Player | Team | Actual | Projection | Baseline | Error | Cause | Tgt Act/Pred | Air Err | YPT Err | Vacancy |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 2026-1 | C.Olave | NO | 182.0 | 29.8 | 65.6 | -152.2 | target_projection_under | 13.0/6.2 | 176.2 | 85.8 | 0.21 |
| 2026-1 | C.Watson | GB | 147.0 | 31.2 | 30.2 | -115.8 | target_projection_under | 8.0/5.3 | 62.0 | 47.2 | 0.00 |
| 2026-1 | D.Kincaid | BUF | 130.0 | 15.9 | 22.4 | -114.1 | route_snap_projection_under | 6.0/3.6 | 67.6 | 93.3 | 0.00 |
| 2026-1 | J.Coker | CAR | 138.0 | 35.6 | 50.6 | -102.4 | target_projection_under | 9.0/5.4 | 71.2 | 55.7 | 0.00 |
| 2026-1 | Z.Flowers | BAL | 150.0 | 55.3 | 65.6 | -94.7 | yards_per_target_efficiency_under | 6.0/7.4 | 12.6 | 94.1 | 0.09 |
| 2026-1 | Bi.Robinson | ATL | 90.0 | 13.8 | 15.6 | -76.2 | target_projection_under | 10.0/3.8 | -40.4 | 65.6 | 0.00 |
| 2026-1 | J.Chase | CIN | 12.0 | 82.8 | 65.6 | 70.8 | target_projection_over | 4.0/10.0 | -65.6 | -22.3 | 0.00 |
| 2026-1 | R.Shaheed | SEA | 4.0 | 70.0 | 50.6 | 66.0 | target_projection_over | 3.0/5.7 | -105.8 | -19.0 | 0.10 |
| 2026-1 | K.Raymond | CHI | 84.0 | 19.1 | 30.2 | -64.9 | target_projection_under | 9.0/3.5 | 54.6 | -17.2 | 0.06 |
| 2026-1 | J.Smith-Njigba | SEA | 122.0 | 63.6 | 65.6 | -58.4 | yards_per_target_efficiency_under | 11.0/10.0 | 13.8 | 39.2 | 0.11 |
| 2026-1 | K.Bourne | ARI | 75.0 | 17.6 | 30.2 | -57.4 | target_projection_under | 8.0/3.7 | 13.6 | -2.1 | 0.00 |
| 2026-1 | P.Washington | JAX | 83.0 | 28.7 | 30.2 | -54.3 | air_yards_role_under | 6.0/5.4 | 57.2 | 34.7 | 0.06 |
| 2026-1 | I.Likely | NYG | 78.0 | 25.9 | 22.4 | -52.1 | target_projection_under | 8.0/4.2 | 24.6 | -11.8 | 0.09 |
| 2026-1 | M.Gesicki | CIN | 78.0 | 26.3 | 33.8 | -51.7 | air_yards_role_under | 7.0/4.9 | 29.6 | 26.3 | 0.11 |
| 2026-1 | R.Rice | KC | 19.0 | 69.9 | 65.6 | 50.9 | target_projection_over | 2.0/8.9 | -34.6 | 2.8 | 0.00 |