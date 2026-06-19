# MLB Hitter Player-Game Outcome Models

Generated: 2026-06-19T16:03:19.672390+00:00
Rows: 51213 | Train: 43473 | Holdout: 7740
Holdout: 2026-05-21 to 2026-06-17
Status: ok

## Recommendation

- Production status: diagnostic_only
- Passes basic gate: False
- PA MAE gain vs slot prior: 0.256
- Hits MAE gain vs slot-rate prior: 0.009
- TB MAE gain vs slot-rate prior: -0.005
- HR MAE gain vs slot-rate prior: 0.007
- Direct event hits MAE gain vs slot prior: 0.004
- Direct event TB MAE gain vs slot prior: -0.025
- Direct event TB MAE gain vs independent rates: -0.020
- HR-any Brier gain vs prior: 0.00328

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
| opp_sp_sc_barrel_rate | 86.5% |
| opp_sp_sc_xwoba | 86.5% |
| opp_sp_fb_pct | 68.0% |
| opp_sp_fb_xwoba | 68.0% |
| opp_sp_sl_pct | 35.8% |
| opp_sp_ch_pct | 36.3% |
| opp_sp_fastball_family_pct | 78.2% |
| opp_sp_pitch_diversity | 79.0% |

## Opportunity

| Model | Rows | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Selected PA model | 7740 | 0.688 | 1.007 | -0.077 |
| Single-mean PA | 7740 | 0.688 | 1.007 | -0.077 |
| Two-part PA | 7740 | 0.745 | 0.979 | 0.042 |
| Slot prior | 7740 | 0.945 | 1.291 | -0.022 |
| Existing projected PA | 7663 | 0.889 | 1.182 | -0.030 |

- Two-part PA enabled: False
- Low-PA Brier: 0.08604 vs baseline 0.15230
- Leakage-safe pregame low-PA rows: 0
- Activation reason: insufficient_pregame_low_pa_rows
- Conditional normal-play PA MAE: 0.451

## Structured Counts

| Target | Model Rows | Model MAE | Prior MAE | Model Bias |
|---|---:|---:|---:|---:|
| Hits | 7437 | 0.672 | 0.680 | 0.042 |
| Total bases | 7437 | 1.278 | 1.273 | 0.101 |
| Home runs | 7437 | 0.201 | 0.208 | 0.017 |

## Direct Per-PA Event Model

- Active event curve: hierarchical_conditional_lgbm
- Train event rows: 155952
- Holdout player-games: 7437
- Weighted event Brier: 0.48563
- Weighted event log loss: 0.99947
- Classes: out, walk, single, double, triple, hr
- TB-state residual enabled: True
- TB-state blend alpha: 0.250
- TB-state validation Brier gain: 0.00010

| Target | Rows | Direct Event MAE | Independent Rate MAE | Direct Bias |
|---|---:|---:|---:|---:|
| Hits | 7437 | 0.676 | 0.672 | 0.011 |
| Total bases | 7437 | 1.299 | 1.278 | 0.060 |
| Home runs | 7437 | 0.204 | 0.201 | 0.016 |

## Event Model Candidates

- Selected: hierarchical_conditional_lgbm

| Candidate | Brier | Log Loss | Composite | Hits MAE | TB MAE | HR MAE |
|---|---:|---:|---:|---:|---:|---:|
| hierarchical_conditional_lgbm | 0.48563 | 0.99947 | 2.27896 | 0.676 | 1.299 | 0.204 |

| Event | Actual / PA | Predicted Prob | Bias / PA |
|---|---:|---:|---:|
| out | 0.6930 | 0.6976 | -0.0045 |
| walk | 0.0862 | 0.0901 | -0.0039 |
| single | 0.1419 | 0.1393 | 0.0027 |
| double | 0.0411 | 0.0416 | -0.0004 |
| triple | 0.0039 | 0.0031 | 0.0008 |
| hr | 0.0338 | 0.0284 | 0.0054 |

## HR Rare Event

- Rows: 7437
- Model Brier: 0.10063
- Prior Brier: 0.10391
- AUC: 0.667

## Existing Prop Projection Holdout

| Target | Rows | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Hits | 3346 | 0.693 | 0.882 | 0.035 |
| Total bases | 3176 | 1.381 | 1.841 | 0.133 |
| Home runs | 3006 | 0.253 | 0.380 | -0.007 |
