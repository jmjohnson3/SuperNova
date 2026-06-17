# MLB Prop Opportunity Models

Generated UTC: 2026-06-13T20:13:20Z
Rows: 48398
Date range: 2026-05-31 to 2026-06-12
Status: ready

## Regression Holdouts

| Model | Rows | Base MAE | Model MAE | Base RMSE | Model RMSE | Model Bias | R2 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| hitter_pa | 234 | 0.673 | 0.619 | 0.827 | 0.783 | -0.141 | 0.103 | model_helped |
| pitcher_bf | 26 | 4.521 | - | 5.890 | - | - | - | insufficient_rows |
| pitcher_ip | 26 | 1.522 | - | 1.936 | - | - | - | insufficient_rows |
| pitcher_pitch_count_proxy | 26 | 17.404 | - | 22.675 | - | - | - | insufficient_rows |

## Low-PA / Removal Risk

| Rows | Actual | Avg Prob | Brier | Log Loss | AUC | Status |
|---:|---:|---:|---:|---:|---:|---|
| 234 | 3.8% | 10.2% | 0.048 | 0.183 | 0.762 | trained |

## Lineup Slot Impact

| Slot | Rows | Avg Actual PA | Low-PA Rate |
|---:|---:|---:|---:|
| 1 | 292 | 4.497 | 2.4% |
| 2 | 301 | 4.429 | 3.0% |
| 3 | 309 | 4.320 | 2.3% |
| 4 | 282 | 4.234 | 2.8% |
| 5 | 244 | 3.996 | 6.1% |
| 6 | 245 | 3.935 | 7.3% |
| 7 | 244 | 3.758 | 8.2% |
| 8 | 197 | 3.497 | 12.2% |
| 9 | 232 | 3.259 | 22.8% |

## Largest Coefficients

### hitter_pa

| Feature | Coef |
|---|---:|
| projected_pa_x_slot_prior | +0.866 |
| opponent_abbr=PIT | +0.859 |
| opponent_abbr=WAS | -0.670 |
| opponent_abbr=MIL | -0.653 |
| opponent_abbr=COL | +0.612 |
| opponent_abbr=ATH | +0.552 |
| opponent_abbr=ATL | -0.523 |
| team_abbr=SF | +0.522 |
| opponent_abbr=MIN | +0.504 |
| opponent_abbr=SF | +0.458 |
| opponent_abbr=NYM | -0.416 |
| team_abbr=MIN | -0.415 |

### hitter_low_pa

| Feature | Coef |
|---|---:|
| opponent_abbr=WAS | +1.317 |
| lineup_slot_low_pa_prior | +1.245 |
| projected_pa_x_slot_prior | -1.136 |
| opponent_abbr=PIT | -1.049 |
| opponent_abbr=TOR | +0.942 |
| opponent_abbr=CIN | +0.769 |
| confirmed_batting_order | -0.714 |
| opponent_abbr=ATH | -0.654 |
| opponent_abbr=BAL | -0.624 |
| blowout_risk | -0.620 |
| team_abbr=KC | -0.608 |
| bullpen_quality_risk | -0.571 |

### pitcher_bf

| Feature | Coef |
|---|---:|

### pitcher_ip

| Feature | Coef |
|---|---:|

### pitcher_pitch_count_proxy

| Feature | Coef |
|---|---:|

