# MLB Hitter Player-Game Outcome Models

Generated: 2026-06-17T08:38:37.498932+00:00
Rows: 51146 | Train: 43365 | Holdout: 7781
Holdout: 2026-05-19 to 2026-06-15
Status: ok

## Recommendation

- Production status: diagnostic_only
- Passes basic gate: False
- PA MAE gain vs slot prior: 0.433
- Hits MAE gain vs slot-rate prior: -0.002
- TB MAE gain vs slot-rate prior: -0.017
- HR MAE gain vs slot-rate prior: 0.007
- Direct event TB MAE gain vs independent rates: -0.007
- HR-any Brier gain vs prior: 0.00310

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
| opp_sp_sc_barrel_rate | 86.5% |
| opp_sp_sc_xwoba | 86.5% |
| opp_sp_fb_pct | 67.9% |
| opp_sp_fb_xwoba | 67.9% |
| opp_sp_sl_pct | 35.7% |
| opp_sp_ch_pct | 36.4% |
| opp_sp_fastball_family_pct | 78.3% |
| opp_sp_pitch_diversity | 79.0% |

## Opportunity

| Model | Rows | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| PA model | 7781 | 0.511 | 0.647 | -0.022 |
| Slot prior | 7781 | 0.944 | 1.292 | -0.033 |
| Existing projected PA | 7704 | 0.890 | 1.185 | -0.036 |

## Structured Counts

| Target | Model Rows | Model MAE | Prior MAE | Model Bias |
|---|---:|---:|---:|---:|
| Hits | 7471 | 0.641 | 0.639 | 0.030 |
| Total bases | 7471 | 1.241 | 1.224 | 0.089 |
| Home runs | 7471 | 0.197 | 0.203 | 0.017 |

## Direct Per-PA Event Model

- Active event curve: linear_multinomial
- Train event rows: 80335
- Holdout player-games: 7471
- Weighted event Brier: 0.48582
- Weighted event log loss: 1.00636
- Classes: double, hr, out, single, triple, walk

| Target | Rows | Direct Event MAE | Independent Rate MAE | Direct Bias |
|---|---:|---:|---:|---:|
| Hits | 7471 | 0.642 | 0.641 | 0.014 |
| Total bases | 7471 | 1.247 | 1.241 | 0.063 |
| Home runs | 7471 | 0.199 | 0.197 | 0.014 |

## Event Model Candidates

- Selected: linear_multinomial

| Candidate | Brier | Log Loss | Composite | Hits MAE | TB MAE | HR MAE |
|---|---:|---:|---:|---:|---:|---:|
| linear_multinomial | 0.48582 | 1.00636 | 2.26584 | 0.642 | 1.247 | 0.199 |
| boosted_binary_calibrated | 0.48841 | 1.01053 | 2.28055 | 0.643 | 1.264 | 0.206 |

| Event | Actual / PA | Predicted Prob | Bias / PA |
|---|---:|---:|---:|
| out | 0.6947 | 0.6938 | 0.0009 |
| walk | 0.0855 | 0.0939 | -0.0084 |
| single | 0.1418 | 0.1408 | 0.0010 |
| double | 0.0409 | 0.0402 | 0.0007 |
| triple | 0.0039 | 0.0032 | 0.0007 |
| hr | 0.0332 | 0.0282 | 0.0050 |

## HR Rare Event

- Rows: 7471
- Model Brier: 0.09892
- Prior Brier: 0.10201
- AUC: 0.666

## Existing Prop Projection Holdout

| Target | Rows | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Hits | 2909 | 0.694 | 0.882 | 0.033 |
| Total bases | 2741 | 1.377 | 1.839 | 0.134 |
| Home runs | 2571 | 0.249 | 0.378 | -0.005 |
