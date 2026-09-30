# MLB Prop Opportunity Models

Generated UTC: 2026-09-01T10:22:42Z
Rows: 295467
Date range: 2026-06-01 to 2026-07-30
Status: ready

## Regression Holdouts

| Model | Rows | Base MAE | Model MAE | Base RMSE | Model RMSE | Model Bias | R2 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| hitter_pa | 4777 | 0.681 | 0.662 | 0.932 | 0.899 | +0.005 | 0.130 | model_helped |
| pitcher_bf | 213 | 2.916 | 2.791 | 3.900 | 3.718 | -0.030 | -0.052 | model_helped |
| pitcher_ip | 213 | 0.968 | 0.964 | 1.253 | 1.249 | -0.175 | -0.119 | model_helped |
| pitcher_pitch_count_proxy | 213 | 12.308 | 9.666 | 16.047 | 13.164 | -0.343 | -0.066 | model_helped |

## Pitcher Joint Opportunity Rebuild

Status: trained | Live gate: True | Improved targets: 3/3

| Target | Base MAE | Joint MAE | MAE Gain | 10-90 Coverage | 25-75 Coverage |
|---|---:|---:|---:|---:|---:|
| bf | 2.916 | 2.686 | +0.231 | 65.6% | 39.2% |
| pitch_count | 12.308 | 9.157 | +3.151 | 67.5% | 39.2% |
| innings | 0.968 | 0.885 | +0.083 | 70.8% | 42.9% |

K opportunity adjustment alpha: 1.0
K-line holdout rows: 300 | baseline Brier: 0.249 | joint Brier: 0.248 | gain: +0.001.

The joint model remains shadow-only unless at least two workload targets and K-line Brier improve on the same date holdout.

## Low-PA / Removal Risk

| Rows | Actual | Avg Prob | Brier | Log Loss | AUC | Status |
|---:|---:|---:|---:|---:|---:|---|
| 4777 | 7.9% | 8.0% | 0.070 | 0.258 | 0.715 | trained |

## Lineup Slot Impact

| Slot | Rows | Avg Actual PA | Low-PA Rate |
|---:|---:|---:|---:|
| 1 | 1251 | 4.452 | 2.4% |
| 2 | 1256 | 4.363 | 3.4% |
| 3 | 1312 | 4.279 | 2.7% |
| 4 | 1271 | 4.147 | 4.1% |
| 5 | 1115 | 3.964 | 6.4% |
| 6 | 1202 | 3.863 | 8.3% |
| 7 | 1135 | 3.643 | 12.2% |
| 8 | 1083 | 3.463 | 15.3% |
| 9 | 1123 | 3.389 | 16.7% |

## Largest Coefficients

### hitter_pa

| Feature | Coef |
|---|---:|
| projected_pa_x_slot_prior | +0.509 |
| confirmed_batting_order | -0.360 |
| opponent_abbr=TB | -0.309 |
| team_abbr=CHC | +0.304 |
| slot_prior_x_home_ninth_penalty | -0.283 |
| opponent_abbr=PIT | +0.269 |
| opponent_abbr=CHC | -0.253 |
| opponent_abbr=MIN | +0.251 |
| opponent_abbr=KC | +0.248 |
| pa_underprojection_risk_score | -0.246 |
| team_abbr=ATL | -0.233 |
| lineup_slot_pa_prior | -0.227 |

### hitter_low_pa

| Feature | Coef |
|---|---:|
| confirmed_lineup_source=actual | -1.070 |
| team_abbr=KC | -0.918 |
| opponent_abbr=LAD | -0.836 |
| opponent_abbr=CHC | +0.804 |
| team_abbr=CLE | +0.750 |
| team_abbr=CIN | -0.666 |
| team_abbr=SF | -0.655 |
| blowout_risk | +0.605 |
| opponent_abbr=WAS | +0.597 |
| pa_underprojection_risk_score | +0.574 |
| team_abbr=MIA | +0.572 |
| opponent_abbr=ARI | -0.539 |

### pitcher_bf

| Feature | Coef |
|---|---:|
| opponent_abbr=ATH | +2.062 |
| team_abbr=LAA | -1.885 |
| opponent_abbr=DET | -1.775 |
| team_abbr=MIN | +1.555 |
| pitcher_favorite_leash | +1.507 |
| pitcher_bf_x_bullpen_fatigue | +1.429 |
| opponent_abbr=HOU | +1.421 |
| opponent_abbr=CHC | -1.328 |
| team_abbr=BAL | +1.248 |
| team_abbr=COL | +1.238 |
| team_abbr=PHI | +1.218 |
| team_abbr=SD | -1.210 |

### pitcher_ip

| Feature | Coef |
|---|---:|
| team_abbr=ATH | -0.541 |
| opponent_abbr=CHC | -0.534 |
| team_abbr=SEA | +0.528 |
| team_abbr=HOU | -0.521 |
| opponent_abbr=TEX | +0.501 |
| team_abbr=LAA | -0.491 |
| team_abbr=LAD | +0.486 |
| team_abbr=SD | -0.433 |
| opponent_abbr=ATH | +0.430 |
| pitcher_bf_x_bullpen_fatigue | +0.424 |
| opponent_abbr=DET | -0.421 |
| opponent_abbr=HOU | +0.366 |

### pitcher_pitch_count_proxy

| Feature | Coef |
|---|---:|
| team_abbr=TB | -7.524 |
| opponent_abbr=DET | -7.433 |
| pitcher_bf_x_bullpen_fatigue | +7.122 |
| team_abbr=CIN | +6.649 |
| pitcher_favorite_leash | +6.380 |
| team_abbr=NYM | +6.199 |
| opponent_abbr=TB | -5.758 |
| team_abbr=ATL | -5.740 |
| team_abbr=SD | -5.724 |
| team_abbr=PHI | +5.019 |
| opponent_abbr=ATH | +4.554 |
| team_abbr=PIT | +4.476 |

### pitcher_joint_opportunity

| Feature | Coef |
|---|---:|

