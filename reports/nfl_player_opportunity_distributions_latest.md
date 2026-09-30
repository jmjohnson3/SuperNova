# NFL Player Opportunity Distributions

These curves describe opportunity uncertainty on one row per player/game before bet-offer pricing.

- Status: ready
- Training rows: 29073
- Trained at: 2026-09-15T09:10:06Z

| Opportunity | Rows | MAE | Baseline MAE | Gain | Sigma | Bias | Brier | Base Brier | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qb_pass_attempts | 33 | 6.713 | 8.055 | +1.342 | 9.185 | -1.696 | 0.214 | 0.257 | yes |
| qb_rush_carries | 33 | 1.967 | 2.291 | +0.323 | 2.348 | +0.781 | 0.233 | 0.273 | yes |
| rb_carries | 53 | 4.198 | 4.904 | +0.706 | 5.393 | -0.839 | 0.268 | 0.264 | no |
| rb_targets | 53 | 1.658 | 1.658 | +0.000 | 2.178 | +0.118 | 0.237 | 0.237 | no |
| receiver_targets | 141 | 2.139 | 2.509 | +0.371 | 2.520 | -0.768 | 0.204 | 0.233 | yes |
| receiver_receptions | 141 | 1.643 | 1.816 | +0.172 | 2.076 | -0.496 | 0.201 | 0.227 | yes |
| qb_passing_td_role | 33 | 0.915 | 1.133 | +0.219 | 1.155 | +0.149 | 0.198 | 0.270 | yes |
| skill_td_role | 194 | 0.377 | 0.429 | +0.052 | 1.000 | +0.157 | 0.184 | 0.186 | yes |
| route_participation | 0 | 0.000 | 0.000 | +0.000 | 0.000 | +0.000 | 0.000 | 0.000 | no |
| pass_route_opportunity_share | 1029 | 0.125 | 0.136 | +0.011 | 0.350 | -0.003 | 0.235 | 0.250 | yes |
| snap_share | 182 | 0.153 | 0.171 | +0.018 | 0.350 | -0.021 | 0.228 | 0.250 | yes |
| full_workload_probability | 182 | 0.313 | 0.360 | +0.047 | 0.440 | -0.016 | 0.194 | 0.187 | no |
| limited_usage_risk | 182 | 0.229 | 0.268 | +0.038 | 0.357 | +0.025 | 0.128 | 0.144 | yes |
| high_pass_attempt_probability | 33 | 0.182 | 0.562 | +0.381 | 0.328 | +0.080 | 0.114 | 0.359 | yes |
| high_carry_probability | 86 | 0.098 | 0.463 | +0.365 | 0.260 | +0.026 | 0.068 | 0.272 | yes |
| high_target_probability | 194 | 0.504 | 0.504 | +0.000 | 0.388 | -0.415 | 0.322 | 0.322 | no |
| receiver_spike_volume_probability | 194 | 0.266 | 0.266 | +0.000 | 0.344 | -0.093 | 0.127 | 0.127 | no |
| receiver_target_route_spike_probability | 194 | 0.164 | 0.217 | +0.053 | 0.269 | -0.047 | 0.074 | 0.089 | yes |
| receiver_target_spike_v2_probability | 194 | 0.126 | 0.291 | +0.165 | 0.275 | -0.004 | 0.076 | 0.139 | yes |
| receiver_spike_under_correction_v4_probability | 194 | 0.178 | 0.251 | +0.073 | 0.274 | -0.065 | 0.080 | 0.109 | yes |
| rb_carry_spike_v2_probability | 53 | 0.147 | 0.435 | +0.289 | 0.288 | +0.040 | 0.085 | 0.266 | yes |
| rb_carry_under_correction_v4_probability | 53 | 0.172 | 0.460 | +0.289 | 0.325 | +0.042 | 0.107 | 0.279 | yes |
| spike_snap_share_probability | 182 | 0.227 | 0.351 | +0.124 | 0.433 | -0.035 | 0.188 | 0.204 | yes |
