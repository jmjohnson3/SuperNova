# MLB Shadow Challenger Promotion Report

Generated UTC: 2026-07-30T16:01:06Z
Status: **ready**
Layer control: `ready`
Minimum completed dates: 5

| Layer | Mode | Dates | Rows | MAE | Baseline MAE | Brier | Baseline Brier | Promotion | Blockers |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Hitter opportunity / rates | shadow -> production | 0 | 51553 | - | - | - | - | False | challenger_not_accepted |
| TB conditional/XBH distribution | shadow -> production | 0 | 51553 | - | - | - | - | False | challenger_not_accepted |
| TB 4+ tail / line calibration challenger | shadow -> production | 48 | 8986 | 1.365 | 1.398 | 0.245 | 0.263 | True | - |
| Pitcher opportunity | shadow -> production | 0 | 34120 | - | - | - | - | False | challenger_not_accepted |
| Pitcher K-rate conservative challenger | shadow -> production | 53 | 1078 | 1.818 | 1.818 | 0.252 | 0.252 | False | challenger_not_accepted |
| Hitter Hits/HR Shadow | shadow -> production_scoring | 11 | 0 | - | - | - | - | False | batter_hits_projection_not_better_than_baseline, five_date_checkpoint_artifact_stale>36h |
| Hitter PA v3 | challenger -> production_scoring | 15 | 0 | - | - | - | - | False | five_date_checkpoint_artifact_stale>36h |
| Total Bases Projection | production_tracking -> production_scoring_verified | 15 | 0 | - | - | - | - | False | batter_total_bases_projection_gate_failed, five_date_checkpoint_artifact_stale>36h |
| Pitcher K Projection | production_tracking -> production_scoring_verified | 17 | 0 | - | - | - | - | False | five_date_checkpoint_artifact_stale>36h |
| CLV Direction Model | support -> support_enabled | 0 | 0 | - | - | - | - | False | market_residual_artifact_stale>36h |
| Bookability / Close Capture | support -> support_enabled | 0 | 0 | - | - | - | - | False | bookability_artifact_stale>36h |
| Distribution / Exact-Line Selector | shadow_support -> support_enabled | 0 | 0 | - | - | - | - | False | distribution_artifact_stale>36h |
| Exact Bucket $1 Micro | watch -> micro | 0 | 0 | - | - | - | - | False | micro_ready_exact_buckets=0, real_money_kill_switch_active |
