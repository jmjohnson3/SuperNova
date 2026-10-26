# Opportunity Models: Production vs Parity Challenger

Holdout: 2026 wk 4, 2026 wk 5, 2026 wk 6, 2026 wk 7 (out of sample for both). Challenger: `src/nfl_pipeline/modeling/models/opportunity_parity/opportunity-parity-20261026T225152Z`. Both scored with the live scoring function on the corrected training table. Lower MAE is better.

| Model | Rows | Production MAE | Challenger MAE | Baseline MAE | Change |
|---|---:|---:|---:|---:|---:|
| full_workload_probability | 1180 | 0.3035 | 0.2590 | 0.3372 | -14.7% |
| limited_usage_risk | 1180 | 0.2621 | 0.2261 | 0.3111 | -13.7% |
| receiver_spike_under_correction_v4_probability | 1056 | 0.2016 | 0.1751 | 0.2277 | -13.1% |
| receiver_spike_volume_probability | 1056 | 0.2143 | 0.1886 | 0.2143 | -12.0% |
| receiver_target_route_spike_probability | 1056 | 0.1622 | 0.1479 | 0.1879 | -8.8% |
| qb_passing_td_role | 124 | 0.8055 | 0.7422 | 0.8281 | -7.9% |
| receiver_target_share | 1044 | 0.0521 | 0.0482 | 0.0495 | -7.4% |
| receiver_targets | 765 | 1.9337 | 1.8347 | 1.8897 | -5.1% |
| receiver_receptions | 765 | 1.4340 | 1.3664 | 1.3923 | -4.7% |
| receiver_air_yards_share | 1044 | 0.0645 | 0.0615 | 0.0656 | -4.6% |
| rb_targets | 291 | 1.4596 | 1.4011 | 1.4596 | -4.0% |
| qb_pass_attempts | 124 | 9.3056 | 8.9401 | 8.9401 | -3.9% |
| rb_carries | 291 | 3.5486 | 3.4277 | 3.6798 | -3.4% |
| qb_rush_carries | 124 | 1.5742 | 1.5223 | 1.6914 | -3.3% |
| receiver_air_yards | 1044 | 17.5795 | 17.2538 | 18.1846 | -1.9% |
| receiver_air_yards_spike_probability | 1044 | 0.2426 | 0.2428 | 0.2937 | +0.1% |
| snap_share | 1180 | 0.1234 | 0.1253 | 0.1296 | +1.6% |
| skill_td_role | 1056 | 0.2811 | 0.2868 | 0.3366 | +2.0% |
| high_target_probability | 1056 | 0.3198 | 0.3432 | 0.3781 | +7.3% |
| receiver_target_spike_v2_probability | 1056 | 0.1760 | 0.1940 | 0.2409 | +10.3% |
| spike_snap_share_probability | 1180 | 0.1812 | 0.2013 | 0.2534 | +11.1% |
| rb_carry_under_correction_v4_probability | 291 | 0.1998 | 0.2675 | 0.3691 | +33.9% |
| high_pass_attempt_probability | 124 | 0.2948 | 0.4346 | 0.5271 | +47.4% |
| high_carry_probability | 415 | 0.2101 | 0.3124 | 0.3981 | +48.7% |
| rb_carry_spike_v2_probability | 291 | 0.1550 | 0.2599 | 0.3493 | +67.7% |

Challenger better on 15 models, worse on 9, within 0.5% on 1.
