# MLB Prop Opportunity Models

Generated UTC: 2026-06-19T06:52:56Z
Rows: 76157
Date range: 2026-05-31 to 2026-06-17
Status: ready

## Regression Holdouts

| Model | Rows | Base MAE | Model MAE | Base RMSE | Model RMSE | Model Bias | R2 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| hitter_pa | 207 | 0.712 | 0.626 | 0.926 | 0.821 | -0.135 | 0.107 | model_helped |
| pitcher_bf | 27 | 3.111 | - | 4.227 | - | - | - | insufficient_rows |
| pitcher_ip | 27 | 0.939 | - | 1.305 | - | - | - | insufficient_rows |
| pitcher_pitch_count_proxy | 27 | 11.978 | - | 16.275 | - | - | - | insufficient_rows |

## Pitcher Joint Opportunity Rebuild

Status: trained | Live gate: False | Improved targets: 0/3

| Target | Base MAE | Joint MAE | MAE Gain | 10-90 Coverage | 25-75 Coverage |
|---|---:|---:|---:|---:|---:|
| bf | 3.214 | 3.281 | -0.066 | 59.5% | 21.4% |
| pitch_count | 12.375 | 12.630 | -0.255 | 59.5% | 21.4% |
| innings | 0.961 | 1.054 | -0.093 | 66.7% | 31.0% |

K-line holdout rows: 104 | baseline Brier: 0.268 | joint Brier: 0.266 | gain: +0.002.

The joint model remains shadow-only unless at least two workload targets and K-line Brier improve on the same date holdout.

## Low-PA / Removal Risk

| Rows | Actual | Avg Prob | Brier | Log Loss | AUC | Status |
|---:|---:|---:|---:|---:|---:|---|
| 207 | 5.3% | 6.6% | 0.049 | 0.186 | 0.774 | trained |

## Lineup Slot Impact

| Slot | Rows | Avg Actual PA | Low-PA Rate |
|---:|---:|---:|---:|
| 1 | 422 | 4.517 | 2.4% |
| 2 | 433 | 4.386 | 3.5% |
| 3 | 441 | 4.286 | 2.5% |
| 4 | 405 | 4.185 | 2.7% |
| 5 | 352 | 4.014 | 5.1% |
| 6 | 363 | 3.882 | 8.5% |
| 7 | 341 | 3.710 | 9.1% |
| 8 | 290 | 3.507 | 12.4% |
| 9 | 328 | 3.250 | 21.6% |

## Largest Coefficients

### hitter_pa

| Feature | Coef |
|---|---:|
| projected_pa_x_slot_prior | +0.936 |
| opponent_abbr=PIT | +0.713 |
| opponent_abbr=COL | +0.594 |
| opponent_abbr=ATH | +0.566 |
| opponent_abbr=WAS | -0.487 |
| team_abbr=SF | +0.485 |
| team_abbr=NYY | +0.387 |
| opponent_abbr=CHC | -0.373 |
| opponent_abbr=MIN | +0.371 |
| team_abbr=MIL | +0.364 |
| team_abbr=ATL | -0.355 |
| team_abbr=CHC | -0.349 |

### hitter_low_pa

| Feature | Coef |
|---|---:|
| projected_pa_x_slot_prior | -1.527 |
| opponent_abbr=WAS | +1.292 |
| lineup_slot_low_pa_prior | +1.042 |
| confirmed_batting_order | -0.803 |
| opponent_abbr=NYY | +0.790 |
| opponent_abbr=TOR | +0.784 |
| opponent_abbr=ARI | -0.775 |
| opponent_abbr=ATH | -0.774 |
| opponent_abbr=PIT | -0.735 |
| opponent_abbr=SEA | -0.702 |
| opponent_abbr=MIA | +0.667 |
| opponent_abbr=BAL | -0.661 |

### pitcher_bf

| Feature | Coef |
|---|---:|

### pitcher_ip

| Feature | Coef |
|---|---:|

### pitcher_pitch_count_proxy

| Feature | Coef |
|---|---:|

### pitcher_joint_opportunity

| Feature | Coef |
|---|---:|

