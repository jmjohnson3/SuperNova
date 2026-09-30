# NFL Player Feature Ablation

This report removes feature groups one at a time on the holdout split. Positive loss means the group helped the full model; negative loss means the model was better without that group.

- Status: ready
- Rows: 29073
- Built at: 2026-09-15T09:13:47Z

## qb_yards

- Status: ready
- Holdout rows: 33
- Full MAE: 72.536 | Baseline MAE: 63.207 | Gain: -9.329

| Feature Group | Removed Cols | MAE Loss Removed | Brier Loss Removed | Helpful | Hurts Full |
|---|---:|---:|---:|---|---|
| route_target_quality | 148 | 1.360 | 0.000 | yes | no |
| game_market_env | 13 | -0.017 | 0.000 | no | yes |
| depth_injury | 32 | -0.027 | 0.000 | no | yes |
| workload_rest | 33 | -0.681 | 0.000 | no | yes |
| td_red_zone | 101 | -1.511 | 0.000 | no | yes |
| rb_carry_role | 51 | -1.976 | 0.000 | no | yes |

## rb_rush_yards

- Status: ready
- Holdout rows: 53
- Full MAE: 20.558 | Baseline MAE: 23.214 | Gain: 2.656

| Feature Group | Removed Cols | MAE Loss Removed | Brier Loss Removed | Helpful | Hurts Full |
|---|---:|---:|---:|---|---|
| depth_injury | 39 | 0.483 | 0.000 | yes | no |
| route_target_quality | 148 | 0.293 | 0.000 | yes | no |
| game_market_env | 13 | 0.193 | 0.000 | yes | no |
| rb_carry_role | 51 | 0.080 | 0.000 | yes | no |
| td_red_zone | 101 | -0.038 | 0.000 | no | yes |
| workload_rest | 33 | -0.122 | 0.000 | no | yes |

## receiving_yards

- Status: ready
- Holdout rows: 194
- Full MAE: 18.476 | Baseline MAE: 21.426 | Gain: 2.950

| Feature Group | Removed Cols | MAE Loss Removed | Brier Loss Removed | Helpful | Hurts Full |
|---|---:|---:|---:|---|---|
| workload_rest | 33 | 0.002 | 0.000 | yes | no |
| depth_injury | 55 | -0.006 | 0.000 | no | yes |
| rb_carry_role | 51 | -0.074 | 0.000 | no | yes |
| game_market_env | 13 | -0.089 | 0.000 | no | yes |
| td_red_zone | 101 | -0.121 | 0.000 | no | yes |
| route_target_quality | 148 | -0.281 | 0.000 | no | yes |

## rush_td

- Status: ready
- Holdout rows: 53
- Full MAE: 0.415 | Baseline MAE: 0.445 | Gain: 0.030

| Feature Group | Removed Cols | MAE Loss Removed | Brier Loss Removed | Helpful | Hurts Full |
|---|---:|---:|---:|---|---|
| workload_rest | 33 | 0.000 | 0.000 | no | no |
| depth_injury | 39 | 0.000 | 0.000 | no | no |
| route_target_quality | 148 | 0.000 | 0.000 | no | no |
| rb_carry_role | 51 | 0.000 | 0.000 | no | no |
| td_red_zone | 101 | 0.000 | 0.000 | no | no |
| game_market_env | 13 | 0.000 | 0.000 | no | no |

## receiving_td_any

- Status: ready
- Holdout rows: 141
- Full MAE: 0.360 | Baseline MAE: 0.332 | Gain: -0.028

| Feature Group | Removed Cols | MAE Loss Removed | Brier Loss Removed | Helpful | Hurts Full |
|---|---:|---:|---:|---|---|
| td_red_zone | 101 | 0.002 | -0.001 | yes | yes |
| game_market_env | 13 | 0.002 | 0.003 | yes | no |
| workload_rest | 33 | 0.002 | 0.001 | yes | no |
| depth_injury | 50 | 0.001 | 0.002 | yes | no |
| route_target_quality | 148 | 0.001 | 0.000 | yes | no |
| rb_carry_role | 51 | -0.002 | -0.000 | no | yes |

## limited_usage_risk

- Status: ready
- Holdout rows: 182
- Full MAE: 0.276 | Baseline MAE: 0.336 | Gain: 0.060

| Feature Group | Removed Cols | MAE Loss Removed | Brier Loss Removed | Helpful | Hurts Full |
|---|---:|---:|---:|---|---|
| rb_carry_role | 51 | 0.008 | 0.003 | yes | no |
| depth_injury | 55 | 0.003 | 0.001 | yes | no |
| route_target_quality | 148 | -0.000 | 0.003 | yes | no |
| td_red_zone | 101 | -0.000 | -0.000 | no | no |
| game_market_env | 13 | -0.003 | -0.002 | no | yes |
| workload_rest | 33 | -0.005 | -0.003 | no | yes |
