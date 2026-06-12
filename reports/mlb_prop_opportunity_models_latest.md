# MLB Prop Opportunity Models

Generated UTC: 2026-06-12T08:54:55Z
Rows: 38710
Date range: 2026-05-31 to 2026-06-10
Status: ready

## Regression Holdouts

| Model | Rows | Base MAE | Model MAE | Base RMSE | Model RMSE | Model Bias | R2 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| hitter_pa | 228 | 0.698 | 0.635 | 0.894 | 0.831 | -0.095 | 0.152 | model_helped |
| pitcher_bf | 24 | 3.008 | - | 3.953 | - | - | - | insufficient_rows |
| pitcher_ip | 24 | 0.924 | - | 1.120 | - | - | - | insufficient_rows |
| pitcher_pitch_count_proxy | 24 | 11.582 | - | 15.218 | - | - | - | insufficient_rows |

## Low-PA / Removal Risk

| Rows | Actual | Avg Prob | Brier | Log Loss | AUC | Status |
|---:|---:|---:|---:|---:|---:|---|
| 228 | 6.1% | 7.9% | 0.061 | 0.219 | 0.750 | trained |

## Lineup Slot Impact

| Slot | Rows | Avg Actual PA | Low-PA Rate |
|---:|---:|---:|---:|
| 1 | 251 | 4.510 | 2.8% |
| 2 | 259 | 4.432 | 3.1% |
| 3 | 265 | 4.325 | 2.3% |
| 4 | 241 | 4.245 | 2.5% |
| 5 | 211 | 3.995 | 6.6% |
| 6 | 210 | 3.971 | 6.7% |
| 7 | 206 | 3.772 | 8.7% |
| 8 | 166 | 3.470 | 13.9% |
| 9 | 195 | 3.200 | 26.2% |

## Largest Coefficients

### hitter_pa

| Feature | Coef |
|---|---:|
| projected_pa_x_slot_prior | +0.783 |
| opponent_abbr=PIT | +0.770 |
| opponent_abbr=WAS | -0.735 |
| opponent_abbr=COL | +0.700 |
| opponent_abbr=ATH | +0.606 |
| team_abbr=MIN | -0.564 |
| opponent_abbr=ATL | -0.560 |
| opponent_abbr=NYM | -0.555 |
| opponent_abbr=LAD | -0.516 |
| opponent_abbr=MIN | +0.486 |
| team_abbr=SF | +0.480 |
| opponent_abbr=MIL | -0.469 |

### hitter_low_pa

| Feature | Coef |
|---|---:|
| lineup_slot_low_pa_prior | +1.405 |
| opponent_abbr=WAS | +1.179 |
| team_abbr=MIN | +0.783 |
| opponent_abbr=PIT | -0.679 |
| projected_pa_x_slot_prior | -0.663 |
| opponent_abbr=ATH | -0.637 |
| confirmed_batting_order | -0.607 |
| opponent_abbr=TOR | +0.587 |
| opponent_abbr=CIN | +0.575 |
| team_abbr=BAL | +0.552 |
| bullpen_quality_risk | -0.543 |
| team_abbr=TOR | -0.541 |

### pitcher_bf

| Feature | Coef |
|---|---:|

### pitcher_ip

| Feature | Coef |
|---|---:|

### pitcher_pitch_count_proxy

| Feature | Coef |
|---|---:|

