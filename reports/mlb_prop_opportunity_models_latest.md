# MLB Prop Opportunity Models

Generated UTC: 2026-06-24T08:14:20Z
Rows: 102734
Date range: 2026-05-31 to 2026-06-22
Status: ready

## Regression Holdouts

| Model | Rows | Base MAE | Model MAE | Base RMSE | Model RMSE | Model Bias | R2 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| hitter_pa | 192 | 0.629 | 0.625 | 0.879 | 0.930 | +0.319 | -0.068 | keep_baseline |
| pitcher_bf | 19 | 2.484 | - | 3.381 | - | - | - | insufficient_rows |
| pitcher_ip | 19 | 1.009 | - | 1.219 | - | - | - | insufficient_rows |
| pitcher_pitch_count_proxy | 19 | 9.564 | - | 13.017 | - | - | - | insufficient_rows |

## Pitcher Joint Opportunity Rebuild

Status: trained | Live gate: False | Improved targets: 3/3

| Target | Base MAE | Joint MAE | MAE Gain | 10-90 Coverage | 25-75 Coverage |
|---|---:|---:|---:|---:|---:|
| bf | 3.000 | 2.811 | +0.189 | 71.4% | 33.3% |
| pitch_count | 11.550 | 10.822 | +0.728 | 71.4% | 33.3% |
| innings | 1.042 | 0.968 | +0.074 | 64.3% | 38.1% |

K-line holdout rows: 114 | baseline Brier: 0.246 | joint Brier: 0.250 | gain: -0.004.

The joint model remains shadow-only unless at least two workload targets and K-line Brier improve on the same date holdout.

## Low-PA / Removal Risk

| Rows | Actual | Avg Prob | Brier | Log Loss | AUC | Status |
|---:|---:|---:|---:|---:|---:|---|
| 192 | 9.9% | 8.1% | 0.085 | 0.307 | 0.701 | trained |

## Lineup Slot Impact

| Slot | Rows | Avg Actual PA | Low-PA Rate |
|---:|---:|---:|---:|
| 1 | 542 | 4.469 | 2.8% |
| 2 | 542 | 4.354 | 3.5% |
| 3 | 555 | 4.276 | 2.5% |
| 4 | 517 | 4.178 | 2.7% |
| 5 | 452 | 4.004 | 5.1% |
| 6 | 461 | 3.863 | 8.7% |
| 7 | 438 | 3.653 | 11.0% |
| 8 | 385 | 3.475 | 14.0% |
| 9 | 423 | 3.258 | 20.3% |

## Largest Coefficients

### hitter_pa

| Feature | Coef |
|---|---:|
| projected_pa_x_slot_prior | +1.101 |
| opponent_abbr=ATH | +0.484 |
| opponent_abbr=PIT | +0.399 |
| team_abbr=SF | +0.393 |
| opponent_abbr=CHC | -0.386 |
| opponent_abbr=MIN | +0.382 |
| team_abbr=ATL | -0.359 |
| team_abbr=MIL | +0.355 |
| opponent_abbr=TB | -0.346 |
| confirmed_batting_order | +0.341 |
| vs_hand_tb_per_pa | +0.332 |
| batter_vs_hand_tb_avg_10 | -0.312 |

### hitter_low_pa

| Feature | Coef |
|---|---:|
| projected_pa_x_slot_prior | -2.058 |
| opponent_abbr=LAD | -1.088 |
| confirmed_batting_order | -0.991 |
| opponent_abbr=WAS | +0.974 |
| team_abbr=SF | -0.888 |
| opponent_abbr=TB | +0.840 |
| opponent_abbr=CIN | +0.749 |
| lineup_slot_low_pa_prior | +0.708 |
| opponent_abbr=ARI | -0.706 |
| opponent_abbr=BAL | -0.702 |
| team_abbr=CIN | -0.690 |
| opponent_abbr=CHC | +0.640 |

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

