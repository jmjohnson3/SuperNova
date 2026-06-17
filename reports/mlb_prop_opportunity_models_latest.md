# MLB Prop Opportunity Models

Generated UTC: 2026-06-17T08:51:05Z
Rows: 65072
Date range: 2026-05-31 to 2026-06-15
Status: ready

## Regression Holdouts

| Model | Rows | Base MAE | Model MAE | Base RMSE | Model RMSE | Model Bias | R2 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| hitter_pa | 158 | 0.594 | 0.526 | 0.783 | 0.729 | +0.015 | 0.211 | model_helped |
| pitcher_bf | 15 | 3.400 | - | 3.980 | - | - | - | insufficient_rows |
| pitcher_ip | 15 | 1.001 | - | 1.262 | - | - | - | insufficient_rows |
| pitcher_pitch_count_proxy | 15 | 13.090 | - | 15.322 | - | - | - | insufficient_rows |

## Low-PA / Removal Risk

| Rows | Actual | Avg Prob | Brier | Log Loss | AUC | Status |
|---:|---:|---:|---:|---:|---:|---|
| 158 | 8.2% | 8.2% | 0.067 | 0.243 | 0.777 | trained |

## Lineup Slot Impact

| Slot | Rows | Avg Actual PA | Low-PA Rate |
|---:|---:|---:|---:|
| 1 | 366 | 4.505 | 2.5% |
| 2 | 377 | 4.403 | 3.2% |
| 3 | 386 | 4.280 | 2.8% |
| 4 | 355 | 4.211 | 2.3% |
| 5 | 309 | 4.000 | 5.8% |
| 6 | 315 | 3.898 | 8.9% |
| 7 | 298 | 3.725 | 9.1% |
| 8 | 252 | 3.480 | 13.1% |
| 9 | 286 | 3.227 | 22.4% |

## Largest Coefficients

### hitter_pa

| Feature | Coef |
|---|---:|
| projected_pa_x_slot_prior | +1.018 |
| opponent_abbr=COL | +0.629 |
| opponent_abbr=PIT | +0.617 |
| opponent_abbr=ATH | +0.606 |
| opponent_abbr=WAS | -0.553 |
| opponent_abbr=MIN | +0.544 |
| team_abbr=SF | +0.513 |
| team_abbr=MIL | +0.482 |
| team_abbr=CHC | -0.405 |
| opponent_abbr=SF | +0.384 |
| favorite_flag | +0.379 |
| confirmed_batting_order | +0.360 |

### hitter_low_pa

| Feature | Coef |
|---|---:|
| projected_pa_x_slot_prior | -1.391 |
| opponent_abbr=WAS | +1.236 |
| lineup_slot_low_pa_prior | +1.075 |
| confirmed_batting_order | -0.902 |
| opponent_abbr=ATH | -0.758 |
| team_abbr=MIN | +0.741 |
| opponent_abbr=CIN | +0.695 |
| opponent_abbr=BAL | -0.674 |
| team_abbr=HOU | +0.666 |
| opponent_abbr=PIT | -0.661 |
| team_abbr=DET | +0.654 |
| team_abbr=TEX | +0.648 |

### pitcher_bf

| Feature | Coef |
|---|---:|

### pitcher_ip

| Feature | Coef |
|---|---:|

### pitcher_pitch_count_proxy

| Feature | Coef |
|---|---:|

