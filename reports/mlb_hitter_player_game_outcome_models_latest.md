# MLB Hitter Player-Game Outcome Models

Generated: 2026-07-02T07:12:57.972247+00:00
Rows: 51366 | Train: 43446 | Holdout: 7920
Holdout: 2026-06-03 to 2026-06-30
Status: ok

## Recommendation

- Production status: diagnostic_only
- Passes basic gate: False
- PA MAE gain vs slot prior: 0.265
- Hits MAE gain vs slot-rate prior: 0.005
- TB MAE gain vs slot-rate prior: -0.020
- HR MAE gain vs slot-rate prior: 0.001
- Direct event hits MAE gain vs slot prior: 0.006
- Direct event TB MAE gain vs slot prior: -0.030
- Direct event TB MAE gain vs independent rates: -0.009
- Direct TB count repair enabled: True
- Direct TB count repair alpha: 0.500
- Direct TB count validation MAE gain: 0.011
- HR-any Brier gain vs prior: 0.00272

## Feature Coverage

| Feature | Coverage |
|---|---:|
| park_run_factor | 80.3% |
| park_hr_factor | 80.3% |
| park_babip_factor | 100.0% |
| own_lineup_xwoba_avg | 100.0% |
| own_lineup_barrel_avg | 100.0% |
| lineup_confirmed_flag | 100.0% |
| confirmed_team_lineup_slots | 100.0% |
| team_lineup_confirmed_flag | 100.0% |
| lineup_slot_pa_prior | 0.0% |
| home_favorite_ninth_penalty | 0.0% |
| blowout_risk | 0.0% |
| catcher_low_pa_risk | 0.0% |
| platoon_advantage_flag | 0.0% |
| batter_sc_barrel_rate | 88.2% |
| batter_sc_xwoba | 88.2% |
| batter_sc_xslg | 88.2% |
| batter_sprint_speed | 86.9% |
| batter_disc_whiff_pct | 85.5% |
| opp_sp_sc_barrel_rate | 87.0% |
| opp_sp_sc_xwoba | 87.0% |
| opp_sp_fb_pct | 68.7% |
| opp_sp_fb_xwoba | 68.7% |
| opp_sp_sl_pct | 36.5% |
| opp_sp_ch_pct | 36.4% |
| opp_sp_fastball_family_pct | 78.7% |
| opp_sp_pitch_diversity | 79.4% |

## Opportunity

| Model | Rows | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Selected PA model | 7920 | 0.702 | 1.041 | -0.102 |
| Single-mean PA | 7920 | 0.702 | 1.041 | -0.102 |
| Two-part PA | 7920 | 0.760 | 1.003 | 0.009 |
| Slot prior | 7920 | 0.968 | 1.315 | -0.029 |
| Existing projected PA | 7838 | 0.897 | 1.208 | -0.052 |

- Two-part PA enabled: False
- Low-PA distribution enabled: True
- Low-PA Brier: 0.09492 vs baseline 0.15502
- Immutable lock-context low-PA rows: 1971
- Temporal lineup/proxy validation rows: 7920
- Temporal distribution Brier: 0.09492 vs baseline 0.15502
- Activation reason: distribution_only_gain
- Conditional normal-play PA MAE: 0.460

## Structured Counts

| Target | Model Rows | Model MAE | Prior MAE | Model Bias |
|---|---:|---:|---:|---:|
| Hits | 7583 | 0.675 | 0.681 | 0.032 |
| Total bases | 7583 | 1.293 | 1.272 | 0.071 |
| Home runs | 7583 | 0.209 | 0.209 | 0.011 |

## Direct Per-PA Event Model

- Active event curve: hierarchical_conditional_lgbm
- Train event rows: 155384
- Holdout player-games: 7583
- Weighted event Brier: 0.48630
- Weighted event log loss: 1.00020
- Classes: out, walk, single, double, triple, hr
- TB-state residual enabled: False
- TB-state blend alpha: 0.000
- TB-state validation Brier gain: 0.00000

| Target | Rows | Direct Event MAE | Independent Rate MAE | Direct Bias |
|---|---:|---:|---:|---:|
| Hits | 7583 | 0.675 | 0.675 | 0.010 |
| Total bases | 7583 | 1.302 | 1.293 | 0.046 |
| Home runs | 7583 | 0.208 | 0.209 | 0.012 |

## Event Model Candidates

- Selected: hierarchical_conditional_lgbm

| Candidate | Brier | Log Loss | Composite | Hits MAE | TB MAE | HR MAE |
|---|---:|---:|---:|---:|---:|---:|
| hierarchical_conditional_lgbm | 0.48630 | 1.00020 | 2.28331 | 0.675 | 1.302 | 0.208 |

| Event | Actual / PA | Predicted Prob | Bias / PA |
|---|---:|---:|---:|
| out | 0.6924 | 0.6968 | -0.0044 |
| walk | 0.0848 | 0.0897 | -0.0049 |
| single | 0.1433 | 0.1390 | 0.0043 |
| double | 0.0415 | 0.0415 | -0.0001 |
| triple | 0.0039 | 0.0032 | 0.0007 |
| hr | 0.0342 | 0.0298 | 0.0044 |

## Conditional XBH Calibration

- Enabled: False
- Method: temporal_empirical_bayes_conditional_logit_offsets
- Validation rows: 3855
- TB MAE before / after: 1.286 / 1.293
- Event Brier before / after: 0.48329 / 0.48330
- Offsets: {"hr_given_xbh": 0.12313895414241031, "triple_given_non_hr_xbh": 0.2275764771298685, "xbh_given_hit": 0.022625442643038594}

## HR Rare Event

- Rows: 7583
- Model Brier: 0.10148
- Prior Brier: 0.10421
- AUC: 0.644

## Existing Prop Projection Holdout

| Target | Rows | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Hits | 5579 | 0.705 | 0.885 | -0.009 |
| Total bases | 5532 | 1.388 | 1.807 | -0.030 |
| Home runs | 5522 | 0.264 | 0.374 | -0.045 |
