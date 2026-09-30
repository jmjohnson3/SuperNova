# NFL Player Projection Audit

This audits one player-game forecast per stat before odds selection, so offer duplication cannot inflate evidence.

- Status: ready
- Training rows: 29073
- Model artifact status: ready
- Opportunity artifact status: ready
- Built at: 2026-09-15T23:33:01Z

## Stat Summary

| Stat | Holdout Rows | Model MAE | Baseline MAE | Gain | Bias | Model Used | Workload Adj | Pass |
|---|---:|---:|---:|---:|---:|---|---|---|
| passing_yards | 33 | 63.207 | 63.207 | 0.000 | 3.416 | baseline | no | no |
| rushing_yards | 86 | 15.803 | 18.613 | 2.810 | -1.629 | yes | yes | yes |
| passing_tds | 33 | 0.899 | 0.921 | 0.022 | -0.239 | yes | yes | yes |
| rushing_tds | 53 | 0.436 | 0.445 | 0.010 | -0.189 | yes | yes | yes |
| receiving_yards | 194 | 18.500 | 21.426 | 2.926 | -3.267 | yes | yes | yes |
| receiving_tds | 141 | 0.351 | 0.382 | 0.032 | -0.040 | yes | yes | yes |

## Error Decomposition

### passing_yards

| Error Type | Rows | Model MAE | Baseline MAE | Gain | Bias |
|---|---:|---:|---:|---:|---:|
| player_rate_or_efficiency | 20 | 48.716 | 48.716 | 0.000 | 6.789 |
| opportunity_high | 8 | 59.713 | 59.713 | 0.000 | -44.479 |
| opportunity_low | 4 | 149.323 | 149.323 | 0.000 | 74.073 |
| projected_full_workload_failed | 1 | 36.501 | 36.501 | 0.000 | 36.501 |

### rushing_yards

| Error Type | Rows | Model MAE | Baseline MAE | Gain | Bias |
|---|---:|---:|---:|---:|---:|
| player_rate_or_efficiency | 52 | 9.607 | 11.426 | 1.820 | -2.597 |
| opportunity_high | 15 | 29.414 | 29.540 | 0.126 | -25.608 |
| opportunity_low | 15 | 23.641 | 31.489 | 7.848 | 23.043 |
| projected_full_workload_failed | 4 | 15.915 | 22.775 | 6.860 | 8.352 |

### passing_tds

| Error Type | Rows | Model MAE | Baseline MAE | Gain | Bias |
|---|---:|---:|---:|---:|---:|
| td_rate_or_variance | 15 | 0.710 | 0.760 | 0.050 | -0.037 |
| opportunity_low | 9 | 0.975 | 1.148 | 0.173 | 0.222 |
| opportunity_high | 7 | 1.307 | 1.120 | -0.187 | -1.211 |
| td_role_opportunity | 2 | 0.549 | 0.413 | -0.136 | -0.431 |

### rushing_tds

| Error Type | Rows | Model MAE | Baseline MAE | Gain | Bias |
|---|---:|---:|---:|---:|---:|
| td_rate_or_variance | 31 | 0.231 | 0.231 | -0.000 | -0.095 |
| opportunity_high | 8 | 1.243 | 1.119 | -0.124 | -1.192 |
| opportunity_low | 7 | 0.375 | 0.578 | 0.203 | 0.375 |
| td_role_opportunity | 7 | 0.482 | 0.493 | 0.012 | -0.026 |

### receiving_yards

| Error Type | Rows | Model MAE | Baseline MAE | Gain | Bias |
|---|---:|---:|---:|---:|---:|
| player_rate_or_efficiency | 133 | 13.251 | 16.786 | 3.536 | -4.272 |
| opportunity_low | 30 | 25.142 | 30.845 | 5.702 | 23.845 |
| opportunity_high | 21 | 44.752 | 37.881 | -6.872 | -40.786 |
| projected_full_workload_failed | 9 | 14.264 | 21.419 | 7.155 | 8.860 |
| model_flagged_limited_usage | 1 | 4.200 | 10.430 | 6.230 | -4.200 |

### receiving_tds

| Error Type | Rows | Model MAE | Baseline MAE | Gain | Bias |
|---|---:|---:|---:|---:|---:|
| td_rate_or_variance | 125 | 0.321 | 0.327 | 0.006 | -0.011 |
| opportunity_high | 6 | 0.735 | 0.658 | -0.077 | -0.682 |
| td_role_opportunity | 6 | 0.592 | 0.831 | 0.239 | -0.227 |
| opportunity_low | 4 | 0.323 | 1.021 | 0.698 | 0.323 |

## Biggest Misses

### passing_yards

| Week | Player | Team | Pos | Depth | Actual | Projection | Baseline | Error Type |
|---|---|---|---|---:|---:|---:|---:|---|
| 2026-1 | S.Darnold | SEA | QB | 1.0 | 13.000 | 220.646 | 220.646 | opportunity_low |
| 2026-1 | K.Murray | MIN | QB | 1.0 | 18.000 | 220.646 | 220.646 | opportunity_low |
| 2026-1 | J.Love | GB | QB | 1.0 | 387.000 | 220.646 | 220.646 | opportunity_high |
| 2026-1 | D.Lock | SEA | QB | 2.0 | 187.000 | 36.501 | 36.501 | opportunity_low |
| 2026-1 | B.Young | CAR | QB | 1.0 | 361.000 | 220.646 | 220.646 | opportunity_high |
| 2026-1 | J.Allen | BUF | QB | 1.0 | 334.000 | 220.646 | 220.646 | player_rate_or_efficiency |
| 2026-1 | L.Jackson | BAL | QB | 1.0 | 324.000 | 220.646 | 220.646 | player_rate_or_efficiency |
| 2026-1 | C.Wentz | MIN | QB | 2.0 | 133.000 | 36.501 | 36.501 | player_rate_or_efficiency |
| 2026-1 | B.Nix | DEN | QB | 1.0 | 131.000 | 220.646 | 220.646 | player_rate_or_efficiency |
| 2026-1 | C.Rush | ATL | QB | 3.0 | 143.000 | 220.646 | 220.646 | player_rate_or_efficiency |
| 2026-1 | M.Stafford | LA | QB | 1.0 | 155.000 | 220.646 | 220.646 | player_rate_or_efficiency |
| 2026-1 | K.Cousins | LV | QB | 1.0 | 160.000 | 220.646 | 220.646 | player_rate_or_efficiency |

### rushing_yards

| Week | Player | Team | Pos | Depth | Actual | Projection | Baseline | Error Type |
|---|---|---|---|---:|---:|---:|---:|---|
| 2026-1 | K.Walker | KC | RB | 1.0 | 173.000 | 59.724 | 47.000 | opportunity_high |
| 2026-1 | R.Dowdle | PIT | RB | 2.0 | 15.000 | 85.440 | 83.300 | opportunity_low |
| 2026-1 | D.Henry | BAL | RB | 1.0 | 144.000 | 89.195 | 113.900 | player_rate_or_efficiency |
| 2026-1 | K.Mitchell | LAC | RB | 2.0 | 9.000 | 61.407 | 47.333 | opportunity_low |
| 2026-1 | J.Mason | MIN | RB | 2.0 | 59.000 | 7.072 | 54.200 | opportunity_high |
| 2026-1 | J.Gibbs | DET | RB | 1.0 | 156.000 | 107.053 | 86.100 | opportunity_high |
| 2026-1 | B.Corum | LA | RB | 2.0 | 54.000 | 6.632 | 13.400 | opportunity_high |
| 2026-1 | B.Hall | NYJ | RB | 1.0 | 102.000 | 60.276 | 56.600 | opportunity_high |
| 2026-1 | C.Williams | CHI | QB | 1.0 | 65.000 | 23.893 | 27.300 | opportunity_high |
| 2026-1 | D.Swift | CHI | RB | 1.0 | 124.000 | 85.154 | 50.500 | player_rate_or_efficiency |
| 2026-1 | K.Williams | LA | RB | 1.0 | 41.000 | 78.891 | 87.900 | opportunity_low |
| 2026-1 | D.Achane | MIA | RB | 1.0 | 36.000 | 70.809 | 55.000 | player_rate_or_efficiency |

### passing_tds

| Week | Player | Team | Pos | Depth | Actual | Projection | Baseline | Error Type |
|---|---|---|---|---:|---:|---:|---:|---|
| 2026-1 | T.Lawrence | JAX | QB | 1.0 | 4.000 | 1.091 | 1.467 | opportunity_low |
| 2026-1 | C.Wentz | MIN | QB | 2.0 | 3.000 | 0.328 | 0.292 | opportunity_high |
| 2026-1 | K.Cousins | LV | QB | 1.0 | 3.000 | 0.832 | 1.467 | opportunity_high |
| 2026-1 | J.Hurts | PHI | QB | 1.0 | 3.000 | 0.978 | 1.467 | td_rate_or_variance |
| 2026-1 | B.Purdy | SF | QB | 1.0 | 3.000 | 1.167 | 1.467 | opportunity_high |
| 2026-1 | B.Young | CAR | QB | 1.0 | 3.000 | 1.250 | 1.467 | td_rate_or_variance |
| 2026-1 | M.Willis | MIA | QB | 1.0 | 0.000 | 1.479 | 1.467 | td_rate_or_variance |
| 2026-1 | B.Mayfield | TB | QB | 1.0 | 0.000 | 1.305 | 1.467 | td_rate_or_variance |
| 2026-1 | M.Stafford | LA | QB | 1.0 | 0.000 | 1.274 | 1.467 | opportunity_low |
| 2026-1 | S.Darnold | SEA | QB | 1.0 | 0.000 | 1.267 | 1.467 | opportunity_low |
| 2026-1 | K.Murray | MIN | QB | 1.0 | 0.000 | 1.191 | 1.467 | opportunity_low |
| 2026-1 | G.Smith | NYJ | QB | 1.0 | 0.000 | 1.031 | 1.467 | opportunity_low |

### rushing_tds

| Week | Player | Team | Pos | Depth | Actual | Projection | Baseline | Error Type |
|---|---|---|---|---:|---:|---:|---:|---|
| 2026-1 | D.Henry | BAL | RB | 1.0 | 3.000 | 0.340 | 1.210 | opportunity_high |
| 2026-1 | D.Swift | CHI | RB | 1.0 | 3.000 | 0.361 | 0.197 | opportunity_high |
| 2026-1 | D.Montgomery | HOU | RB | 1.0 | 2.000 | 0.223 | 0.408 | td_role_opportunity |
| 2026-1 | J.Gibbs | DET | RB | 1.0 | 2.000 | 0.481 | 1.507 | opportunity_high |
| 2026-1 | J.Taylor | IND | RB | 1.0 | 2.000 | 0.861 | 1.370 | td_rate_or_variance |
| 2026-1 | J.Mason | MIN | RB | 2.0 | 1.000 | 0.023 | 0.000 | opportunity_high |
| 2026-1 | K.Miller | NO | RB | 2.0 | 1.000 | 0.218 | 0.250 | td_rate_or_variance |
| 2026-1 | J.Williams | DAL | RB | 1.0 | 1.000 | 0.225 | 0.000 | opportunity_high |
| 2026-1 | C.Brown | CIN | RB | 1.0 | 1.000 | 0.251 | 0.417 | td_rate_or_variance |
| 2026-1 | K.Williams | LA | RB | 1.0 | 1.000 | 0.279 | 0.390 | td_rate_or_variance |
| 2026-1 | A.Jones | MIN | RB | 1.0 | 1.000 | 0.327 | 0.200 | opportunity_high |
| 2026-1 | K.Walker | KC | RB | 1.0 | 1.000 | 0.346 | 0.182 | td_rate_or_variance |

### receiving_yards

| Week | Player | Team | Pos | Depth | Actual | Projection | Baseline | Error Type |
|---|---|---|---|---:|---:|---:|---:|---|
| 2026-1 | C.Olave | NO | WR | 1.0 | 182.000 | 29.809 | 65.630 | opportunity_high |
| 2026-1 | C.Watson | GB | WR | 1.0 | 147.000 | 31.156 | 30.249 | opportunity_high |
| 2026-1 | D.Kincaid | BUF | TE | 1.0 | 130.000 | 15.878 | 22.430 | player_rate_or_efficiency |
| 2026-1 | J.Coker | CAR | WR | 2.0 | 138.000 | 35.622 | 50.604 | opportunity_high |
| 2026-1 | Z.Flowers | BAL | WR | 1.0 | 150.000 | 55.291 | 65.630 | player_rate_or_efficiency |
| 2026-1 | Bi.Robinson | ATL | RB | 1.0 | 90.000 | 13.831 | 15.560 | opportunity_high |
| 2026-1 | J.Chase | CIN | WR | 1.0 | 12.000 | 82.785 | 65.630 | opportunity_low |
| 2026-1 | R.Shaheed | SEA | WR | 1.0 | 4.000 | 69.967 | 50.604 | opportunity_low |
| 2026-1 | K.Raymond | CHI | WR | 1.0 | 84.000 | 19.072 | 30.249 | opportunity_high |
| 2026-1 | J.Smith-Njigba | SEA | WR | 1.0 | 122.000 | 63.603 | 65.630 | opportunity_high |
| 2026-1 | K.Bourne | ARI | WR | 3.0 | 75.000 | 17.587 | 30.249 | opportunity_high |
| 2026-1 | P.Washington | JAX | WR | 1.0 | 83.000 | 28.664 | 30.249 | player_rate_or_efficiency |

### receiving_tds

| Week | Player | Team | Pos | Depth | Actual | Projection | Baseline | Error Type |
|---|---|---|---|---:|---:|---:|---:|---|
| 2026-1 | D.Goedert | PHI | TE | 1.0 | 2.000 | 0.271 | 0.047 | td_role_opportunity |
| 2026-1 | I.Likely | NYG | TE | 1.0 | 2.000 | 0.273 | 0.256 | opportunity_high |
| 2026-1 | J.Coker | CAR | WR | 2.0 | 2.000 | 0.314 | 0.383 | td_rate_or_variance |
| 2026-1 | C.Watson | GB | WR | 1.0 | 2.000 | 0.325 | 0.182 | td_rate_or_variance |
| 2026-1 | A.St. Brown | DET | WR | 1.0 | 2.000 | 0.510 | 1.075 | td_rate_or_variance |
| 2026-1 | J.Jefferson | MIN | WR | 1.0 | 2.000 | 0.541 | 0.683 | td_rate_or_variance |
| 2026-1 | J.Palmer | BUF | WR | 4.0 | 1.000 | 0.105 | 0.326 | td_rate_or_variance |
| 2026-1 | M.Gesicki | CIN | TE | 1.0 | 1.000 | 0.159 | 0.297 | td_rate_or_variance |
| 2026-1 | S.Diggs | WAS | WR | 2.0 | 1.000 | 0.163 | 0.125 | td_rate_or_variance |
| 2026-1 | N.Fant | NO | TE | 2.0 | 1.000 | 0.173 | 0.214 | td_rate_or_variance |
| 2026-1 | B.Strange | JAX | TE | 1.0 | 1.000 | 0.173 | 0.081 | td_rate_or_variance |
| 2026-1 | D.Robinson | SF | WR | 4.0 | 1.000 | 0.203 | 0.387 | td_rate_or_variance |
