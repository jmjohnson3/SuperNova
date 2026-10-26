# NFL Player Opportunity Models

These models estimate opportunity before pricing player props: pass attempts, carries, targets, receptions, TD role, and full-workload/rest risk.
Route and snap-share targets are included, but they require a source with non-null route/snap data.

- Status: ready
- Training rows: 36840
- Trained at: 2026-10-26T22:52:22Z

| Model | Holdout Rows | MAE | Baseline MAE | Gain | Bias | Accepted | Upside Signal | Status |
|---|---:|---:|---:|---:|---:|---|---|---|
| qb_pass_attempts | 124 | 8.460 | 8.385 | -0.074 | -1.920 | no | no | trained |
| qb_rush_carries | 124 | 1.522 | 1.596 | +0.074 | -0.062 | yes | no | trained |
| rb_carries | 291 | 3.428 | 3.680 | +0.252 | -0.875 | yes | no | trained |
| rb_targets | 291 | 1.401 | 1.439 | +0.038 | -0.401 | yes | no | trained |
| receiver_targets | 765 | 1.835 | 1.890 | +0.055 | -0.354 | yes | no | trained |
| receiver_receptions | 765 | 1.366 | 1.392 | +0.026 | -0.218 | yes | no | trained |
| qb_passing_td_role | 124 | 0.742 | 0.778 | +0.036 | -0.031 | yes | no | trained |
| skill_td_role | 1056 | 0.287 | 0.337 | +0.050 | -0.109 | yes | no | trained |
| route_participation | 0 | 0.000 | 0.000 | +0.000 | +0.000 | no | no | insufficient_rows |
| pass_route_opportunity_share | 1132 | 0.138 | 0.143 | +0.005 | -0.002 | yes | no | trained |
| receiver_target_share | 1044 | 0.048 | 0.050 | +0.001 | -0.008 | yes | no | trained |
| receiver_air_yards_share | 1044 | 0.056 | 0.063 | +0.006 | -0.017 | yes | no | trained |
| receiver_air_yards | 1044 | 16.324 | 17.499 | +1.176 | -4.668 | yes | no | trained |
| snap_share | 1180 | 0.125 | 0.130 | +0.004 | +0.016 | yes | no | trained |
| full_workload_probability | 1180 | 0.259 | 0.337 | +0.078 | +0.010 | yes | no | trained |
| limited_usage_risk | 1180 | 0.227 | 0.311 | +0.085 | -0.069 | yes | no | trained |
| high_pass_attempt_probability | 124 | 0.381 | 0.373 | -0.008 | -0.039 | yes | yes | trained |
| high_carry_probability | 415 | 0.207 | 0.187 | -0.020 | +0.033 | yes | yes | trained |
| high_target_probability | 1056 | 0.162 | 0.152 | -0.010 | +0.012 | no | yes | trained |
| receiver_spike_volume_probability | 1056 | 0.189 | 0.214 | +0.026 | -0.008 | yes | yes | trained |
| receiver_target_route_spike_probability | 1056 | 0.148 | 0.188 | +0.040 | -0.010 | yes | yes | trained |
| receiver_target_spike_v2_probability | 1056 | 0.184 | 0.212 | +0.028 | +0.041 | yes | yes | trained |
| receiver_air_yards_spike_probability | 1044 | 0.189 | 0.186 | -0.003 | +0.021 | yes | yes | trained |
| receiver_spike_under_correction_v4_probability | 1056 | 0.172 | 0.212 | +0.040 | +0.025 | yes | yes | trained |
| rb_carry_spike_v2_probability | 291 | 0.242 | 0.260 | +0.018 | +0.015 | yes | yes | trained |
| rb_carry_under_correction_v4_probability | 291 | 0.246 | 0.260 | +0.014 | +0.015 | yes | yes | trained |
| spike_snap_share_probability | 1180 | 0.201 | 0.253 | +0.052 | -0.012 | yes | yes | trained |

## Data Coverage

| Field | Non-null Rows | Coverage |
|---|---:|---:|
| pass_attempts | 36840 | 100.0% |
| carries | 36840 | 100.0% |
| targets | 36840 | 100.0% |
| receptions | 36840 | 100.0% |
| route_participation | 0 | 0.0% |
| pass_route_opportunities | 29968 | 81.3% |
| pass_route_opportunity_share | 29968 | 81.3% |
| snap_share | 36485 | 99.0% |
| offense_snap_share | 36485 | 99.0% |
| target_share | 31570 | 85.7% |
| air_yards_share | 31570 | 85.7% |
| receiving_air_yards | 36823 | 100.0% |
| wopr | 31570 | 85.7% |
| targets_per_route_run | 0 | 0.0% |
| yards_per_route_run | 0 | 0.0% |
| first_read_targets | 0 | 0.0% |
| first_read_target_share | 0 | 0.0% |
| end_zone_targets | 1662 | 4.5% |
| end_zone_target_share | 0 | 0.0% |
| red_zone_carries | 36840 | 100.0% |
| red_zone_targets | 36840 | 100.0% |
| red_zone_pass_attempts | 36840 | 100.0% |
| red_zone_touches | 36840 | 100.0% |
| goal_line_carries | 36840 | 100.0% |
| goal_line_targets | 36840 | 100.0% |
| team_implied_points | 36840 | 100.0% |
| depth_pos_rank | 2092 | 5.7% |
| starter_confidence | 36840 | 100.0% |
| rest_risk_score | 36840 | 100.0% |
| full_workload_score | 36840 | 100.0% |
| limited_workload_risk_score | 36840 | 100.0% |
| high_usage_fragility_score | 36840 | 100.0% |
| usage_volatility_score | 36840 | 100.0% |
| backup_role_score | 36840 | 100.0% |
| starter_role_stability_score | 36840 | 100.0% |
| recent_snap_drop_score | 36840 | 100.0% |
| recent_snap_rise_score | 36840 | 100.0% |
| recent_target_spike_score | 36840 | 100.0% |
| recent_carry_spike_score | 36840 | 100.0% |
| recent_route_spike_score | 36840 | 100.0% |
| actual_full_workload | 36485 | 99.0% |
| actual_limited_usage | 36485 | 99.0% |
| high_pass_attempt_score | 36840 | 100.0% |
| high_carry_score | 36840 | 100.0% |
| high_target_score | 36840 | 100.0% |
| receiver_target_eruption_score | 36840 | 100.0% |
| air_yards_spike_path_score | 36840 | 100.0% |
| receiver_air_yards_eruption_score | 36840 | 100.0% |
| receiver_explosive_spike_score | 36840 | 100.0% |
| receiver_spike_volume_score | 36840 | 100.0% |
| receiver_spike_volume_anchor_yards | 36840 | 100.0% |
| receiver_target_command_score | 36840 | 100.0% |
| receiver_route_spike_readiness_score | 36840 | 100.0% |
| receiver_target_route_spike_score | 36840 | 100.0% |
| receiver_contextual_spike_score | 36840 | 100.0% |
| receiver_target_spike_v2_score | 36840 | 100.0% |
| receiver_air_yards_spike_v2_score | 36840 | 100.0% |
| receiver_ypt_efficiency_spike_score | 36840 | 100.0% |
| receiver_spike_yards_anchor_v2 | 36840 | 100.0% |
| receiver_projected_targets_v2 | 36840 | 100.0% |
| receiver_high_value_target_score | 36840 | 100.0% |
| rb_carry_spike_v2_score | 36840 | 100.0% |
| rb_projected_carries_v2 | 36840 | 100.0% |
| rb_rush_yards_anchor_v2 | 36840 | 100.0% |
| same_week_usage_confidence_v3_score | 36840 | 100.0% |
| receiver_live_spike_v3_score | 36840 | 100.0% |
| receiver_projected_targets_v3 | 36840 | 100.0% |
| receiver_spike_yards_anchor_v3 | 36840 | 100.0% |
| rb_live_carry_v3_score | 36840 | 100.0% |
| rb_projected_carries_v3 | 36840 | 100.0% |
| rb_rush_yards_anchor_v3 | 36840 | 100.0% |
| yardage_projection_volatility_v3_score | 36840 | 100.0% |
| receiving_usage_history_quality_score | 36840 | 100.0% |
| rb_usage_history_quality_score | 36840 | 100.0% |
| td_usage_history_quality_score | 36840 | 100.0% |
| live_usage_context_quality_v4_score | 36840 | 100.0% |
| receiver_spike_under_correction_v4_score | 36840 | 100.0% |
| receiver_spike_yards_anchor_v4 | 36840 | 100.0% |
| rb_carry_under_correction_v4_score | 36840 | 100.0% |
| rb_rush_yards_anchor_v4 | 36840 | 100.0% |
| yardage_projection_volatility_v4_score | 36840 | 100.0% |
| workload_downside_v2_score | 36840 | 100.0% |
| workload_upside_v2_score | 36840 | 100.0% |
| receiver_teammate_vacancy_score | 36840 | 100.0% |
| teammate_receiver_injury_pressure_score | 36840 | 100.0% |
| same_week_teammate_skill_injury_count | 2325 | 6.3% |
| same_week_teammate_skill_injury_score | 2061 | 5.6% |
| same_week_teammate_receiver_injury_score | 1865 | 5.1% |
| same_week_teammate_receiver_out_count | 2325 | 6.3% |
| receiver_target_eruption_anchor_targets | 36840 | 100.0% |
| receiver_air_yards_spike_anchor_yards | 36840 | 100.0% |
| receiver_target_route_spike_anchor_targets | 36840 | 100.0% |
| receiver_target_route_spike_anchor_yards | 36840 | 100.0% |
| spike_snap_share_score | 36840 | 100.0% |
| spike_target_opportunity_score | 36840 | 100.0% |
| spike_carry_opportunity_score | 36840 | 100.0% |
| spike_pass_attempt_opportunity_score | 36840 | 100.0% |
| target_spike_path_score | 36840 | 100.0% |
| carry_spike_path_score | 36840 | 100.0% |
| pass_spike_path_score | 36840 | 100.0% |
| normal_usage_path_score | 36840 | 100.0% |
| weird_usage_risk_score | 36840 | 100.0% |
| role_continuity_score | 36840 | 100.0% |
| projected_starter_score | 36840 | 100.0% |
| actual_high_pass_attempts | 36840 | 100.0% |
| actual_high_carries | 36840 | 100.0% |
| actual_high_targets | 36840 | 100.0% |
| actual_receiver_spike_volume | 32380 | 87.9% |
| actual_receiver_target_route_spike | 32380 | 87.9% |
| actual_receiver_air_yards_spike | 32363 | 87.8% |
| actual_spike_snap_share | 36485 | 99.0% |
