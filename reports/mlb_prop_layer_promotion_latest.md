# MLB Prop Layer Promotion Control

Generated UTC: 2026-09-30T22:52:20Z
Status: **ready**
Automatic integration mode: `safe_next_rung`
Kill switch: `None`

## Layers

| Layer | Current | Target | Live Scoring | Auto Integrate | Key Blockers |
|---|---|---|---|---|---|
| Hitter Hits/HR Shadow | shadow | production_scoring | True | True | - |
| Hitter PA v3 | challenger | production_scoring | False | False | completed_dates<5, hitter_plate_appearances_challenger_projection_gate_failed, hitter_plate_appearances_challenger_projection_not_better_than_baseline, hitter_plate_appearances_challenger_rows<50, projection_audit_artifact_missing |
| Total Bases Projection | production_tracking | production_scoring_verified | False | False | batter_total_bases_projection_gate_failed, batter_total_bases_projection_not_better_than_baseline, batter_total_bases_rows<100, completed_dates<5, projection_audit_artifact_missing |
| Pitcher K Projection | production_tracking | production_scoring_verified | False | False | completed_dates<5, pitcher_strikeouts_projection_gate_failed, pitcher_strikeouts_projection_not_better_than_baseline, pitcher_strikeouts_rows<100, projection_audit_artifact_missing |
| CLV Direction Model | support | support_enabled | False | False | clv_auc<0.55, clv_brier_missing, clv_target_not_ready, market_residual_artifact_missing |
| Bookability / Close Capture | support | support_enabled | False | False | bookability_artifact_missing, bookability_artifact_not_ready, bookability_holdout_missing, valid_close_capture_target_not_ready |
| Distribution / Exact-Line Selector | shadow_support | support_enabled | False | False | distribution_artifact_missing, distribution_artifact_not_ready, no_true_pair_bucket_model_selection |
| Exact Bucket $1 Micro | watch | micro | False | False | kill_switch_artifact_missing, micro_promotion_artifact_missing, micro_ready_exact_buckets=0 |

## Auto Integrations

| Integration | Enabled | Target | Blockers |
|---|---|---|---|
| hitter_rate_challenger_production | True | production_scoring | - |
| hitter_pa_v3_production | False | production_scoring | completed_dates<5, hitter_plate_appearances_challenger_projection_gate_failed, hitter_plate_appearances_challenger_projection_not_better_than_baseline, hitter_plate_appearances_challenger_rows<50, projection_audit_artifact_missing |
| exact_bucket_micro_ladder | False | micro | kill_switch_artifact_missing, micro_promotion_artifact_missing, micro_ready_exact_buckets=0 |

## Ladder

| Tier | Buckets |
|---|---:|

