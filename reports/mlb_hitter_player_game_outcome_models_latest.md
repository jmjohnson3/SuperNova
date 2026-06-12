# MLB Hitter Player-Game Outcome Models

Generated: 2026-06-12T15:12:15.443692+00:00
Rows: 51249 | Train: 43355 | Holdout: 7894
Holdout: 2026-05-15 to 2026-06-11
Status: ok

## Recommendation

- Production status: diagnostic_only
- Passes basic gate: False
- PA MAE gain vs slot prior: 0.430
- Hits MAE gain vs slot-rate prior: -0.003
- TB MAE gain vs slot-rate prior: -0.021
- HR MAE gain vs slot-rate prior: 0.008
- Direct event TB MAE gain vs independent rates: -0.007
- HR-any Brier gain vs prior: 0.00307

## Feature Coverage

| Feature | Coverage |
|---|---:|
| park_run_factor | 80.5% |
| park_hr_factor | 80.5% |
| park_babip_factor | 100.0% |
| own_lineup_xwoba_avg | 100.0% |
| own_lineup_barrel_avg | 100.0% |
| lineup_confirmed_flag | 100.0% |
| confirmed_team_lineup_slots | 100.0% |
| team_lineup_confirmed_flag | 100.0% |
| batter_sc_barrel_rate | 88.2% |
| batter_sc_xwoba | 88.2% |
| batter_sc_xslg | 88.2% |
| batter_sprint_speed | 86.9% |
| batter_disc_whiff_pct | 85.5% |
| opp_sp_sc_barrel_rate | 86.4% |
| opp_sp_sc_xwoba | 86.4% |
| opp_sp_fb_pct | 67.5% |
| opp_sp_fb_xwoba | 67.5% |
| opp_sp_sl_pct | 35.4% |
| opp_sp_ch_pct | 36.5% |
| opp_sp_fastball_family_pct | 78.1% |
| opp_sp_pitch_diversity | 78.9% |

## Opportunity

| Model | Rows | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| PA model | 7894 | 0.521 | 0.659 | -0.015 |
| Slot prior | 7894 | 0.951 | 1.303 | -0.028 |
| Existing projected PA | 7818 | 0.891 | 1.180 | -0.019 |

## Structured Counts

| Target | Model Rows | Model MAE | Prior MAE | Model Bias |
|---|---:|---:|---:|---:|
| Hits | 7587 | 0.639 | 0.637 | 0.019 |
| Total bases | 7587 | 1.232 | 1.211 | 0.066 |
| Home runs | 7587 | 0.191 | 0.200 | 0.014 |

## Direct Per-PA Event Model

- Active event curve: linear_multinomial
- Train event rows: 80284
- Holdout player-games: 7587
- Weighted event log loss: 0.99885
- Classes: double, hr, out, single, triple, walk

| Target | Rows | Direct Event MAE | Independent Rate MAE | Direct Bias |
|---|---:|---:|---:|---:|
| Hits | 7587 | 0.640 | 0.639 | 0.006 |
| Total bases | 7587 | 1.239 | 1.232 | 0.042 |
| Home runs | 7587 | 0.194 | 0.191 | 0.011 |

## Event Model Candidates

| Candidate | Log Loss | Hits MAE | TB MAE | HR MAE |
|---|---:|---:|---:|---:|
| linear_multinomial | 0.99885 | 0.640 | 1.239 | 0.194 |
| boosted_binary_calibrated | 1.00848 | 0.639 | 1.256 | 0.203 |

| Event | Actual / PA | Predicted Prob | Bias / PA |
|---|---:|---:|---:|
| out | 0.6950 | 0.6924 | 0.0027 |
| walk | 0.0866 | 0.0941 | -0.0076 |
| single | 0.1410 | 0.1405 | 0.0005 |
| double | 0.0418 | 0.0419 | -0.0002 |
| triple | 0.0038 | 0.0033 | 0.0004 |
| hr | 0.0319 | 0.0277 | 0.0041 |

## HR Rare Event

- Rows: 7587
- Model Brier: 0.09571
- Prior Brier: 0.09877
- AUC: 0.670

## Existing Prop Projection Holdout

| Target | Rows | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Hits | 2062 | 0.692 | 0.880 | 0.025 |
| Total bases | 1902 | 1.363 | 1.805 | 0.137 |
| Home runs | 1732 | 0.243 | 0.368 | -0.006 |
