# MLB Prop Layer Promotion Control

Generated UTC: 2026-09-01T11:25:51Z
Status: **ready**
Automatic integration mode: `safe_next_rung`
Kill switch: `disabled`

## Layers

| Layer | Current | Target | Live Scoring | Auto Integrate | Key Blockers |
|---|---|---|---|---|---|
| Hitter Hits/HR Shadow | shadow | production_scoring | False | False | batter_hits_projection_not_better_than_baseline, five_date_checkpoint_artifact_stale>36h |
| Hitter PA v3 | challenger | production_scoring | False | False | five_date_checkpoint_artifact_stale>36h |
| Total Bases Projection | production_tracking | production_scoring_verified | False | False | batter_total_bases_projection_gate_failed, batter_total_bases_projection_not_better_than_baseline, five_date_checkpoint_artifact_stale>36h |
| Pitcher K Projection | production_tracking | production_scoring_verified | False | False | five_date_checkpoint_artifact_stale>36h |
| CLV Direction Model | support | support_enabled | True | False | - |
| Bookability / Close Capture | support | support_enabled | True | False | - |
| Distribution / Exact-Line Selector | shadow_support | support_enabled | True | False | - |
| Exact Bucket $1 Micro | watch | micro | False | False | micro_ready_exact_buckets=0, real_money_kill_switch_active |

## Auto Integrations

| Integration | Enabled | Target | Blockers |
|---|---|---|---|
| hitter_rate_challenger_production | False | production_scoring | batter_hits_projection_not_better_than_baseline, five_date_checkpoint_artifact_stale>36h |
| hitter_pa_v3_production | False | production_scoring | five_date_checkpoint_artifact_stale>36h |
| exact_bucket_micro_ladder | False | micro | micro_ready_exact_buckets=0, real_money_kill_switch_active |

## Ladder

| Tier | Buckets |
|---|---:|
| watch | 132 |

