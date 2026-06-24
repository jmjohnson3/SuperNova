# MLB Hitter Player-Game Outcome Models

Generated: 2026-06-24T08:05:33.150921+00:00
Rows: 51119 | Train: 43360 | Holdout: 7759
Holdout: 2026-05-26 to 2026-06-22
Status: ok

## Recommendation

- Production status: diagnostic_only
- Passes basic gate: False
- PA MAE gain vs slot prior: 0.258
- Hits MAE gain vs slot-rate prior: 0.010
- TB MAE gain vs slot-rate prior: -0.006
- HR MAE gain vs slot-rate prior: 0.005
- Direct event hits MAE gain vs slot prior: 0.006
- Direct event TB MAE gain vs slot prior: -0.026
- Direct event TB MAE gain vs independent rates: -0.020
- HR-any Brier gain vs prior: 0.00350

## Feature Coverage

| Feature | Coverage |
|---|---:|
| park_run_factor | 80.7% |
| park_hr_factor | 80.7% |
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
| opp_sp_sc_barrel_rate | 86.8% |
| opp_sp_sc_xwoba | 86.8% |
| opp_sp_fb_pct | 68.3% |
| opp_sp_fb_xwoba | 68.3% |
| opp_sp_sl_pct | 36.0% |
| opp_sp_ch_pct | 36.4% |
| opp_sp_fastball_family_pct | 78.4% |
| opp_sp_pitch_diversity | 79.2% |

## Opportunity

| Model | Rows | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Selected PA model | 7759 | 0.703 | 1.022 | -0.083 |
| Single-mean PA | 7759 | 0.703 | 1.022 | -0.083 |
| Two-part PA | 7759 | 0.763 | 0.999 | 0.025 |
| Slot prior | 7759 | 0.961 | 1.307 | -0.030 |
| Existing projected PA | 7686 | 0.900 | 1.201 | -0.046 |

- Two-part PA enabled: False
- Low-PA Brier: 0.09302 vs baseline 0.15573
- Leakage-safe pregame low-PA rows: 0
- Activation reason: insufficient_pregame_low_pa_rows
- Conditional normal-play PA MAE: 0.455

## Structured Counts

| Target | Model Rows | Model MAE | Prior MAE | Model Bias |
|---|---:|---:|---:|---:|
| Hits | 7442 | 0.674 | 0.684 | 0.060 |
| Total bases | 7442 | 1.288 | 1.282 | 0.133 |
| Home runs | 7442 | 0.206 | 0.210 | 0.021 |

## Direct Per-PA Event Model

- Active event curve: hierarchical_conditional_lgbm
- Train event rows: 155242
- Holdout player-games: 7442
- Weighted event Brier: 0.48853
- Weighted event log loss: 1.00542
- Classes: out, walk, single, double, triple, hr
- TB-state residual enabled: False
- TB-state blend alpha: 0.000
- TB-state validation Brier gain: 0.00000

| Target | Rows | Direct Event MAE | Independent Rate MAE | Direct Bias |
|---|---:|---:|---:|---:|
| Hits | 7442 | 0.678 | 0.674 | 0.028 |
| Total bases | 7442 | 1.308 | 1.288 | 0.092 |
| Home runs | 7442 | 0.208 | 0.206 | 0.020 |

## Event Model Candidates

- Selected: hierarchical_conditional_lgbm

| Candidate | Brier | Log Loss | Composite | Hits MAE | TB MAE | HR MAE |
|---|---:|---:|---:|---:|---:|---:|
| hierarchical_conditional_lgbm | 0.48853 | 1.00542 | 2.29324 | 0.678 | 1.308 | 0.208 |

| Event | Actual / PA | Predicted Prob | Bias / PA |
|---|---:|---:|---:|
| out | 0.6903 | 0.6989 | -0.0086 |
| walk | 0.0847 | 0.0890 | -0.0043 |
| single | 0.1442 | 0.1392 | 0.0051 |
| double | 0.0418 | 0.0413 | 0.0005 |
| triple | 0.0040 | 0.0029 | 0.0010 |
| hr | 0.0350 | 0.0287 | 0.0064 |

## HR Rare Event

- Rows: 7442
- Model Brier: 0.10344
- Prior Brier: 0.10694
- AUC: 0.667

## Existing Prop Projection Holdout

| Target | Rows | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Hits | 4305 | 0.695 | 0.882 | 0.016 |
| Total bases | 4130 | 1.386 | 1.830 | 0.067 |
| Home runs | 3959 | 0.255 | 0.376 | -0.020 |
