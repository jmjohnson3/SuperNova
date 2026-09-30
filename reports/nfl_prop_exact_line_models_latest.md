# NFL Exact-Line Prop Models

These models train only after real NFL player prop offers have been locked, graded, and matched to true paired prices.

- Status: insufficient_independent_history
- Training rows: 861
- True-paired rows: 861
- Trained at: 2026-09-30T18:47:32Z
- Deployment: offline challenger only; no bankroll or micro approval
- Evidence audit: {"scanned_rows": 6523, "eligible_rows": 861, "exclusions": {"legacy_lock_unverified": 449, "projection_without_offer": 3580, "missing_true_pair": 32, "participation_unverified": 44, "later_revision_same_decision": 1539, "game_not_final": 18}, "unique_player_games": 403, "independent_weeks": 2}

## Stat-Side Models

| Stat | Side | Status | Rows | Train | Holdout | Brier | Model Brier | Market Brier | AUC | Accepted | CLV Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| passing_tds | over | insufficient_rows | 28 | 0 | 0 | - | - | - | - | no | - |
| passing_tds | under | insufficient_rows | 41 | 0 | 0 | - | - | - | - | no | - |
| passing_yards | over | insufficient_rows | 28 | 0 | 0 | - | - | - | - | no | - |
| passing_yards | under | insufficient_rows | 73 | 0 | 0 | - | - | - | - | no | - |
| receiving_yards | over | insufficient_rows | 187 | 0 | 0 | - | - | - | - | no | - |
| receiving_yards | under | insufficient_split_rows | 0 | 0 | 0 | - | - | - | - | no | - |
| rushing_yards | over | insufficient_rows | 96 | 0 | 0 | - | - | - | - | no | - |
| rushing_yards | under | insufficient_rows | 129 | 0 | 0 | - | - | - | - | no | - |

## Exact Buckets

| Stat | Side | Book | Line Bucket | Price Bucket | Rows | Win% | ROI | Model Brier | Market Brier | CLV Rows | CLV Beat | Avg CLV |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| receiving_yards | under | fanduel | yards_30_39 | fair_lay_100_149 | 40 | 47.5% | -0.107 | 0.253 | 0.250 | 27 | 0.0% | -0.001 |
| receiving_yards | under | fanduel | yards_10_19 | fair_lay_100_149 | 34 | 61.8% | +0.161 | 0.250 | 0.252 | 22 | 0.0% | -0.002 |
| receiving_yards | over | fanduel | yards_10_19 | fair_lay_100_149 | 34 | 55.9% | +0.050 | 0.250 | 0.250 | 26 | 0.0% | -0.004 |
| receiving_yards | under | fanduel | yards_20_29 | fair_lay_100_149 | 33 | 54.5% | +0.024 | 0.252 | 0.250 | 21 | 0.0% | -0.002 |
| receiving_yards | under | draftkings | yards_10_19 | fair_lay_100_149 | 23 | 43.5% | -0.175 | 0.275 | 0.249 | 12 | 16.7% | -0.003 |
| passing_yards | under | fanduel | pass_yards_200_224 | fair_lay_100_149 | 23 | 39.1% | -0.264 | 0.255 | 0.250 | 14 | 0.0% | -0.001 |
| receiving_yards | under | fanduel | yards_40_49 | fair_lay_100_149 | 22 | 40.9% | -0.231 | 0.278 | 0.250 | 12 | 0.0% | -0.001 |
| receiving_yards | over | fanduel | yards_0_9 | fair_lay_100_149 | 21 | 38.1% | -0.276 | 0.264 | 0.253 | 16 | 18.8% | +0.003 |
| receiving_yards | under | fanduel | yards_50_59 | fair_lay_100_149 | 19 | 47.4% | -0.110 | 0.257 | 0.250 | 11 | 0.0% | -0.001 |
| receiving_yards | over | fanduel | yards_30_39 | fair_lay_100_149 | 18 | 50.0% | -0.060 | 0.249 | 0.250 | 15 | 0.0% | -0.001 |
| rushing_yards | under | fanduel | yards_10_19 | fair_lay_100_149 | 18 | 33.3% | -0.373 | 0.277 | 0.250 | 11 | 0.0% | -0.001 |
| rushing_yards | over | fanduel | yards_10_19 | fair_lay_100_149 | 18 | 22.2% | -0.582 | 0.269 | 0.250 | 13 | 0.0% | -0.001 |
| receiving_yards | over | fanduel | yards_20_29 | fair_lay_100_149 | 17 | 82.4% | +0.548 | 0.210 | 0.250 | 10 | 10.0% | +0.001 |
| receiving_yards | under | draftkings | yards_30_39 | fair_lay_100_149 | 17 | 41.2% | -0.219 | 0.273 | 0.251 | 8 | 0.0% | -0.004 |
| passing_yards | under | draftkings | pass_yards_200_224 | fair_lay_100_149 | 16 | 50.0% | -0.052 | 0.249 | 0.250 | 7 | 28.6% | -0.001 |
| receiving_yards | over | draftkings | yards_10_19 | fair_lay_100_149 | 15 | 53.3% | +0.011 | 0.251 | 0.250 | 8 | 25.0% | -0.001 |
| receiving_yards | under | fanduel | yards_0_9 | fair_lay_100_149 | 14 | 64.3% | +0.206 | 0.248 | 0.249 | 12 | 0.0% | -0.001 |
| rushing_yards | under | fanduel | yards_20_29 | fair_lay_100_149 | 14 | 50.0% | -0.060 | 0.259 | 0.250 | 10 | 0.0% | -0.002 |
| receiving_yards | under | fanduel | yards_70_79 | fair_lay_100_149 | 14 | 35.7% | -0.327 | 0.252 | 0.250 | 6 | 0.0% | +0.000 |
| receiving_yards | over | fanduel | yards_40_49 | fair_lay_100_149 | 14 | 35.7% | -0.328 | 0.254 | 0.250 | 10 | 0.0% | -0.002 |
| receiving_yards | under | draftkings | yards_20_29 | fair_lay_100_149 | 13 | 69.2% | +0.315 | 0.251 | 0.253 | 7 | 28.6% | -0.003 |
| receiving_yards | over | draftkings | yards_30_39 | fair_lay_100_149 | 13 | 61.5% | +0.161 | 0.248 | 0.248 | 8 | 37.5% | -0.001 |
| rushing_yards | under | fanduel | yards_50_59 | fair_lay_100_149 | 13 | 38.5% | -0.276 | 0.272 | 0.250 | 9 | 0.0% | -0.001 |
| rushing_yards | over | draftkings | yards_10_19 | fair_lay_100_149 | 13 | 15.4% | -0.705 | 0.292 | 0.250 | 9 | 33.3% | -0.000 |
| receiving_yards | under | draftkings | yards_40_49 | fair_lay_100_149 | 12 | 33.3% | -0.371 | 0.303 | 0.252 | 4 | 0.0% | -0.003 |
| receiving_yards | under | fanduel | yards_60_69 | fair_lay_100_149 | 12 | 25.0% | -0.530 | 0.272 | 0.250 | 4 | 0.0% | -0.002 |
| rushing_yards | under | fanduel | yards_30_39 | fair_lay_100_149 | 11 | 63.6% | +0.197 | 0.227 | 0.250 | 6 | 0.0% | -0.001 |
| passing_yards | under | fanduel | pass_yards_175_199 | fair_lay_100_149 | 11 | 63.6% | +0.196 | 0.237 | 0.250 | 5 | 0.0% | -0.000 |
| receiving_yards | over | draftkings | yards_40_49 | fair_lay_100_149 | 11 | 54.5% | +0.035 | 0.266 | 0.249 | 4 | 25.0% | +0.000 |
| rushing_yards | under | draftkings | yards_20_29 | fair_lay_100_149 | 11 | 45.5% | -0.136 | 0.251 | 0.249 | 8 | 12.5% | -0.001 |
| receiving_yards | over | draftkings | yards_50_59 | fair_lay_100_149 | 10 | 80.0% | +0.512 | 0.199 | 0.249 | 5 | 60.0% | +0.001 |
| receiving_yards | over | draftkings | yards_20_29 | fair_lay_100_149 | 10 | 70.0% | +0.333 | 0.220 | 0.251 | 4 | 100.0% | +0.004 |
| rushing_yards | under | fanduel | yards_0_9 | fair_lay_100_149 | 9 | 88.9% | +0.670 | 0.223 | 0.250 | 6 | 0.0% | -0.001 |
| passing_yards | over | fanduel | pass_yards_225_249 | fair_lay_100_149 | 9 | 66.7% | +0.254 | 0.249 | 0.250 | 6 | 0.0% | -0.001 |
| receiving_yards | under | draftkings | yards_50_59 | fair_lay_100_149 | 8 | 75.0% | +0.424 | 0.225 | 0.251 | 3 | 66.7% | +0.002 |
| rushing_yards | under | draftkings | yards_30_39 | fair_lay_100_149 | 8 | 62.5% | +0.185 | 0.249 | 0.251 | 4 | 50.0% | -0.001 |
| rushing_yards | under | draftkings | yards_60_69 | fair_lay_100_149 | 8 | 62.5% | +0.183 | 0.252 | 0.250 | 3 | 33.3% | +0.002 |
| rushing_yards | over | fanduel | yards_30_39 | fair_lay_100_149 | 8 | 0.0% | -1.000 | 0.268 | 0.250 | 7 | 0.0% | -0.003 |
| passing_tds | under | draftkings | td_alt_1_5_plus | fair_lay_100_149 | 7 | 85.7% | +0.546 | 0.195 | 0.229 | 7 | 71.4% | +0.003 |
| passing_tds | under | fanduel | td_alt_1_5_plus | lay_150_199 | 7 | 71.4% | +0.112 | 0.207 | 0.218 | 7 | 28.6% | -0.011 |
| receiving_yards | under | draftkings | yards_60_69 | fair_lay_100_149 | 7 | 57.1% | +0.084 | 0.251 | 0.250 | 2 | 0.0% | -0.002 |
| receiving_yards | over | fanduel | yards_50_59 | fair_lay_100_149 | 7 | 42.9% | -0.194 | 0.258 | 0.250 | 4 | 0.0% | -0.001 |
| passing_yards | under | fanduel | pass_yards_225_249 | fair_lay_100_149 | 7 | 28.6% | -0.464 | 0.276 | 0.250 | 4 | 0.0% | -0.002 |
| rushing_yards | over | fanduel | yards_50_59 | fair_lay_100_149 | 7 | 28.6% | -0.464 | 0.258 | 0.250 | 5 | 0.0% | -0.002 |
| passing_tds | under | fanduel | td_0_5 | plus_150_249 | 6 | 50.0% | +0.323 | 0.197 | 0.275 | 5 | 60.0% | +0.010 |
| rushing_yards | under | fanduel | yards_40_49 | fair_lay_100_149 | 6 | 50.0% | -0.061 | 0.275 | 0.250 | 4 | 0.0% | -0.001 |
| rushing_yards | under | fanduel | yards_60_69 | fair_lay_100_149 | 6 | 50.0% | -0.061 | 0.263 | 0.250 | 3 | 0.0% | -0.001 |
| passing_yards | over | draftkings | pass_yards_200_224 | fair_lay_100_149 | 6 | 16.7% | -0.685 | 0.269 | 0.251 | 3 | 33.3% | +0.000 |
| receiving_yards | under | draftkings | yards_0_9 | fair_lay_100_149 | 5 | 80.0% | +0.517 | 0.223 | 0.248 | 3 | 66.7% | +0.009 |
| passing_tds | over | fanduel | td_alt_1_5_plus | fair_lay_100_149 | 5 | 80.0% | +0.510 | 0.225 | 0.264 | 5 | 60.0% | -0.002 |
| receiving_yards | over | draftkings | yards_0_9 | fair_lay_100_149 | 5 | 40.0% | -0.224 | 0.273 | 0.252 | 5 | 60.0% | +0.006 |
| rushing_yards | over | draftkings | yards_60_69 | fair_lay_100_149 | 5 | 40.0% | -0.238 | 0.270 | 0.253 | 4 | 50.0% | -0.003 |
| rushing_yards | over | draftkings | yards_50_59 | fair_lay_100_149 | 5 | 40.0% | -0.243 | 0.250 | 0.251 | 1 | 0.0% | -0.009 |
| passing_tds | over | draftkings | td_alt_1_5_plus | plus_150_249 | 5 | 20.0% | -0.392 | 0.179 | 0.180 | 5 | 40.0% | -0.004 |
| rushing_yards | over | fanduel | yards_20_29 | fair_lay_100_149 | 5 | 20.0% | -0.625 | 0.265 | 0.250 | 4 | 0.0% | -0.002 |
| passing_yards | under | fanduel | pass_yards_150_174 | fair_lay_100_149 | 5 | 0.0% | -1.000 | 0.266 | 0.250 | 4 | 0.0% | -0.001 |
| passing_yards | under | draftkings | pass_yards_175_199 | fair_lay_100_149 | 4 | 75.0% | +0.424 | 0.215 | 0.251 | 2 | 50.0% | +0.002 |
| rushing_yards | under | draftkings | yards_40_49 | fair_lay_100_149 | 4 | 75.0% | +0.418 | 0.229 | 0.248 | 2 | 50.0% | +0.002 |
| passing_yards | over | draftkings | pass_yards_225_249 | fair_lay_100_149 | 4 | 75.0% | +0.416 | 0.243 | 0.249 | 1 | 100.0% | +0.009 |
| rushing_yards | over | fanduel | yards_0_9 | fair_lay_100_149 | 4 | 75.0% | +0.410 | 0.232 | 0.244 | 3 | 0.0% | -0.001 |
| passing_tds | over | draftkings | td_alt_1_5_plus | fair_lay_100_149 | 4 | 75.0% | +0.357 | 0.231 | 0.242 | 4 | 25.0% | -0.006 |
| passing_tds | over | draftkings | td_alt_1_5_plus | plus_100_149 | 4 | 50.0% | +0.148 | 0.267 | 0.261 | 4 | 75.0% | +0.006 |
| passing_tds | over | fanduel | td_alt_1_5_plus | plus_100_149 | 4 | 50.0% | +0.130 | 0.255 | 0.260 | 4 | 0.0% | -0.010 |
| rushing_yards | under | draftkings | yards_10_19 | fair_lay_100_149 | 4 | 50.0% | -0.045 | 0.240 | 0.255 | 3 | 66.7% | +0.002 |
| receiving_yards | over | fanduel | yards_70_79 | fair_lay_100_149 | 4 | 50.0% | -0.059 | 0.262 | 0.250 | 3 | 0.0% | -0.001 |
| rushing_yards | over | draftkings | yards_30_39 | fair_lay_100_149 | 4 | 25.0% | -0.529 | 0.281 | 0.249 | 2 | 50.0% | +0.001 |
| passing_yards | under | fanduel | pass_yards_250_274 | fair_lay_100_149 | 4 | 25.0% | -0.531 | 0.280 | 0.250 | 3 | 0.0% | -0.001 |
| passing_yards | over | fanduel | pass_yards_200_224 | fair_lay_100_149 | 4 | 25.0% | -0.531 | 0.266 | 0.250 | 1 | 0.0% | -0.002 |
| rushing_yards | over | draftkings | yards_20_29 | fair_lay_100_149 | 4 | 0.0% | -1.000 | 0.409 | 0.255 | 3 | 0.0% | -0.006 |
| rushing_yards | under | fanduel | yards_70_79 | fair_lay_100_149 | 4 | 0.0% | -1.000 | 0.344 | 0.250 | 3 | 0.0% | +0.000 |
| rushing_yards | over | draftkings | yards_0_9 | fair_lay_100_149 | 3 | 66.7% | +0.287 | 0.225 | 0.259 | 3 | 33.3% | +0.004 |
| rushing_yards | under | draftkings | yards_0_9 | fair_lay_100_149 | 3 | 66.7% | +0.273 | 0.242 | 0.254 | 3 | 33.3% | -0.003 |
| receiving_yards | under | draftkings | yards_70_79 | fair_lay_100_149 | 3 | 66.7% | +0.270 | 0.249 | 0.251 | 1 | 100.0% | +0.002 |
| rushing_yards | under | draftkings | yards_50_59 | fair_lay_100_149 | 3 | 66.7% | +0.259 | 0.223 | 0.248 | 0 | - | - |
| rushing_yards | over | fanduel | yards_70_79 | fair_lay_100_149 | 3 | 66.7% | +0.254 | 0.278 | 0.250 | 2 | 0.0% | -0.001 |
| rushing_yards | over | draftkings | yards_40_49 | fair_lay_100_149 | 3 | 66.7% | +0.252 | 0.232 | 0.249 | 1 | 0.0% | -0.007 |
| passing_tds | under | draftkings | td_alt_1_5_plus | lay_200_plus | 3 | 66.7% | -0.020 | 0.262 | 0.220 | 3 | 33.3% | -0.010 |
| passing_tds | under | fanduel | td_alt_1_5_plus | plus_100_149 | 3 | 33.3% | -0.287 | 0.278 | 0.226 | 3 | 33.3% | -0.002 |
| receiving_yards | over | fanduel | yards_0_9 | plus_100_149 | 3 | 33.3% | -0.307 | 0.269 | 0.227 | 3 | 0.0% | +0.000 |
| rushing_yards | over | fanduel | yards_60_69 | fair_lay_100_149 | 3 | 33.3% | -0.372 | 0.254 | 0.250 | 3 | 0.0% | -0.001 |
