# NFL Player Opportunity Models

These models estimate opportunity before pricing player props: pass attempts, carries, targets, receptions, TD role, and full-workload/rest risk.
Route and snap-share targets are included, but they require a source with non-null route/snap data.

- Status: ready
- Training rows: 29073
- Trained at: 2026-09-15T15:16:00Z

| Model | Holdout Rows | MAE | Baseline MAE | Gain | Bias | Accepted | Upside Signal | Status |
|---|---:|---:|---:|---:|---:|---|---|---|
| qb_pass_attempts | 33 | 6.713 | 7.934 | +1.221 | +1.696 | yes | no | trained |
| qb_rush_carries | 33 | 1.967 | 2.082 | +0.114 | -0.781 | yes | no | trained |
| rb_carries | 53 | 4.198 | 4.531 | +0.334 | +0.839 | yes | no | trained |
| rb_targets | 53 | 1.566 | 1.549 | -0.017 | -0.142 | no | no | trained |
| receiver_targets | 141 | 2.139 | 2.426 | +0.287 | +0.768 | yes | no | trained |
| receiver_receptions | 141 | 1.643 | 1.778 | +0.135 | +0.496 | yes | no | trained |
| qb_passing_td_role | 33 | 0.915 | 0.921 | +0.006 | -0.149 | yes | no | trained |
| skill_td_role | 194 | 0.377 | 0.424 | +0.048 | -0.157 | yes | no | trained |
| route_participation | 0 | 0.000 | 0.000 | +0.000 | +0.000 | no | no | insufficient_rows |
| pass_route_opportunity_share | 1029 | 0.125 | 0.136 | +0.011 | +0.003 | yes | no | trained |
| receiver_target_share | 194 | 0.059 | 0.064 | +0.006 | +0.023 | yes | no | trained |
| receiver_air_yards_share | 194 | 0.078 | 0.085 | +0.007 | -0.008 | yes | no | trained |
| receiver_air_yards | 194 | 19.195 | 21.492 | +2.297 | -2.040 | yes | no | trained |
| snap_share | 194 | 0.150 | 0.165 | +0.016 | +0.019 | yes | no | trained |
| full_workload_probability | 194 | 0.341 | 0.358 | +0.016 | -0.006 | yes | no | trained |
| limited_usage_risk | 194 | 0.239 | 0.262 | +0.023 | -0.021 | yes | no | trained |
| high_pass_attempt_probability | 33 | 0.182 | 0.313 | +0.131 | -0.080 | yes | yes | trained |
| high_carry_probability | 86 | 0.121 | 0.165 | +0.044 | -0.014 | yes | yes | trained |
| high_target_probability | 194 | 0.144 | 0.157 | +0.013 | -0.016 | no | yes | trained |
| receiver_spike_volume_probability | 194 | 0.257 | 0.256 | -0.001 | +0.077 | no | no | trained |
| receiver_target_route_spike_probability | 194 | 0.180 | 0.197 | +0.016 | +0.060 | yes | yes | trained |
| receiver_target_spike_v2_probability | 194 | 0.161 | 0.197 | +0.035 | +0.038 | yes | yes | trained |
| receiver_air_yards_spike_probability | 194 | 0.216 | 0.202 | -0.014 | +0.009 | yes | yes | trained |
| receiver_spike_under_correction_v4_probability | 194 | 0.190 | 0.197 | +0.006 | +0.070 | yes | yes | trained |
| rb_carry_spike_v2_probability | 53 | 0.147 | 0.257 | +0.110 | -0.040 | yes | yes | trained |
| rb_carry_under_correction_v4_probability | 53 | 0.189 | 0.257 | +0.068 | -0.033 | yes | yes | trained |
| spike_snap_share_probability | 194 | 0.272 | 0.355 | +0.083 | +0.102 | yes | yes | trained |

## Data Coverage

| Field | Non-null Rows | Coverage |
|---|---:|---:|
| pass_attempts | 29073 | 100.0% |
| carries | 29073 | 100.0% |
| targets | 29073 | 100.0% |
| receptions | 29073 | 100.0% |
| route_participation | 0 | 0.0% |
| pass_route_opportunities | 25106 | 86.4% |
| pass_route_opportunity_share | 25106 | 86.4% |
| snap_share | 28948 | 99.6% |
| offense_snap_share | 28948 | 99.6% |
| target_share | 23820 | 81.9% |
| air_yards_share | 23820 | 81.9% |
| receiving_air_yards | 29073 | 100.0% |
| wopr | 23820 | 81.9% |
| targets_per_route_run | 0 | 0.0% |
| yards_per_route_run | 0 | 0.0% |
| first_read_targets | 0 | 0.0% |
| first_read_target_share | 0 | 0.0% |
| end_zone_targets | 194 | 0.7% |
| end_zone_target_share | 0 | 0.0% |
| red_zone_carries | 29073 | 100.0% |
| red_zone_targets | 29073 | 100.0% |
| red_zone_pass_attempts | 29073 | 100.0% |
| red_zone_touches | 29073 | 100.0% |
| goal_line_carries | 29073 | 100.0% |
| goal_line_targets | 29073 | 100.0% |
| team_implied_points | 29073 | 100.0% |
| depth_pos_rank | 28480 | 98.0% |
| starter_confidence | 29073 | 100.0% |
| rest_risk_score | 29073 | 100.0% |
| full_workload_score | 29073 | 100.0% |
| limited_workload_risk_score | 29073 | 100.0% |
| high_usage_fragility_score | 29073 | 100.0% |
| usage_volatility_score | 29073 | 100.0% |
| backup_role_score | 29073 | 100.0% |
| starter_role_stability_score | 29073 | 100.0% |
| recent_snap_drop_score | 29073 | 100.0% |
| recent_snap_rise_score | 29073 | 100.0% |
| recent_target_spike_score | 29073 | 100.0% |
| recent_carry_spike_score | 29073 | 100.0% |
| recent_route_spike_score | 29073 | 100.0% |
| actual_full_workload | 28948 | 99.6% |
| actual_limited_usage | 28948 | 99.6% |
| high_pass_attempt_score | 29073 | 100.0% |
| high_carry_score | 29073 | 100.0% |
| high_target_score | 29073 | 100.0% |
| receiver_target_eruption_score | 29073 | 100.0% |
| air_yards_spike_path_score | 29073 | 100.0% |
| receiver_air_yards_eruption_score | 29073 | 100.0% |
| receiver_explosive_spike_score | 29073 | 100.0% |
| receiver_spike_volume_score | 29073 | 100.0% |
| receiver_spike_volume_anchor_yards | 29073 | 100.0% |
| receiver_target_command_score | 29073 | 100.0% |
| receiver_route_spike_readiness_score | 29073 | 100.0% |
| receiver_target_route_spike_score | 29073 | 100.0% |
| receiver_contextual_spike_score | 29073 | 100.0% |
| receiver_target_spike_v2_score | 29073 | 100.0% |
| receiver_air_yards_spike_v2_score | 29073 | 100.0% |
| receiver_ypt_efficiency_spike_score | 29073 | 100.0% |
| receiver_spike_yards_anchor_v2 | 29073 | 100.0% |
| receiver_projected_targets_v2 | 29073 | 100.0% |
| receiver_high_value_target_score | 29073 | 100.0% |
| rb_carry_spike_v2_score | 29073 | 100.0% |
| rb_projected_carries_v2 | 29073 | 100.0% |
| rb_rush_yards_anchor_v2 | 29073 | 100.0% |
| same_week_usage_confidence_v3_score | 29073 | 100.0% |
| receiver_live_spike_v3_score | 29073 | 100.0% |
| receiver_projected_targets_v3 | 29073 | 100.0% |
| receiver_spike_yards_anchor_v3 | 29073 | 100.0% |
| rb_live_carry_v3_score | 29073 | 100.0% |
| rb_projected_carries_v3 | 29073 | 100.0% |
| rb_rush_yards_anchor_v3 | 29073 | 100.0% |
| yardage_projection_volatility_v3_score | 29073 | 100.0% |
| receiving_usage_history_quality_score | 29073 | 100.0% |
| rb_usage_history_quality_score | 29073 | 100.0% |
| td_usage_history_quality_score | 29073 | 100.0% |
| live_usage_context_quality_v4_score | 29073 | 100.0% |
| receiver_spike_under_correction_v4_score | 29073 | 100.0% |
| receiver_spike_yards_anchor_v4 | 29073 | 100.0% |
| rb_carry_under_correction_v4_score | 29073 | 100.0% |
| rb_rush_yards_anchor_v4 | 29073 | 100.0% |
| yardage_projection_volatility_v4_score | 29073 | 100.0% |
| workload_downside_v2_score | 29073 | 100.0% |
| workload_upside_v2_score | 29073 | 100.0% |
| receiver_teammate_vacancy_score | 29073 | 100.0% |
| teammate_receiver_injury_pressure_score | 29073 | 100.0% |
| same_week_teammate_skill_injury_count | 29073 | 100.0% |
| same_week_teammate_skill_injury_score | 29073 | 100.0% |
| same_week_teammate_receiver_injury_score | 29073 | 100.0% |
| same_week_teammate_receiver_out_count | 29073 | 100.0% |
| receiver_target_eruption_anchor_targets | 29073 | 100.0% |
| receiver_air_yards_spike_anchor_yards | 29073 | 100.0% |
| receiver_target_route_spike_anchor_targets | 29073 | 100.0% |
| receiver_target_route_spike_anchor_yards | 29073 | 100.0% |
| spike_snap_share_score | 29073 | 100.0% |
| spike_target_opportunity_score | 29073 | 100.0% |
| spike_carry_opportunity_score | 29073 | 100.0% |
| spike_pass_attempt_opportunity_score | 29073 | 100.0% |
| target_spike_path_score | 29073 | 100.0% |
| carry_spike_path_score | 29073 | 100.0% |
| pass_spike_path_score | 29073 | 100.0% |
| normal_usage_path_score | 29073 | 100.0% |
| weird_usage_risk_score | 29073 | 100.0% |
| role_continuity_score | 29073 | 100.0% |
| projected_starter_score | 29073 | 100.0% |
| actual_high_pass_attempts | 29073 | 100.0% |
| actual_high_carries | 29073 | 100.0% |
| actual_high_targets | 29073 | 100.0% |
| actual_receiver_spike_volume | 25463 | 87.6% |
| actual_receiver_target_route_spike | 25463 | 87.6% |
| actual_receiver_air_yards_spike | 25463 | 87.6% |
| actual_spike_snap_share | 28948 | 99.6% |
