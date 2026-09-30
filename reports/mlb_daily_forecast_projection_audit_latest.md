# MLB Daily Forecast Projection Audit

Generated UTC: 2026-09-01T11:14:02Z
Canonical evidence phase: day_pregame
Forecast ledger rows: 45717
Pending forecasts: 28
Canonical graded forecasts: 34120
Date range: 2026-07-02 to 2026-07-29
Status: ready

## Projection Skill

| Stat | Type | Rows | Dates | MAE | Baseline MAE | Gain | Bias | Eligible | Blockers |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| batter_hits | player | 5139 | 25 | 0.724 | 0.726 | 0.002 | -0.070 | True | - |
| batter_home_runs | player | 5139 | 25 | 0.251 | 0.220 | -0.031 | 0.054 | False | model_not_better_than_simple_baseline |
| batter_total_bases | player | 5139 | 25 | 1.384 | 1.392 | 0.008 | 0.160 | True | - |
| hitter_plate_appearances | opportunity | 5139 | 25 | 0.757 | 0.763 | 0.006 | 0.072 | True | - |
| hitter_plate_appearances_challenger | opportunity | 4617 | 23 | 0.749 | 0.765 | 0.016 | 0.058 | True | - |
| pitcher_batters_faced | opportunity | 492 | 25 | 3.096 | 3.096 | 0.000 | 0.584 | True | - |
| pitcher_innings | opportunity | 501 | 25 | 1.001 | 1.219 | 0.218 | 0.110 | True | - |
| pitcher_pitch_count | opportunity | 488 | 25 | 12.524 | 17.600 | 5.076 | 1.862 | True | - |
| pitcher_strikeouts | player | 514 | 25 | 1.845 | 1.937 | 0.092 | 0.182 | True | - |

## Model-Version Cohorts

Each repaired model starts a new prospective evidence clock. Minimum proof is five dates; the target is ten.

| Stat | Family | Version | Rows | Dates | MAE | Baseline | Eligible | Need 5 / 10 |
|---|---|---|---:|---:|---:|---:|---|---:|
| batter_hits | hitter_rate_shadow_direct_player_game_blend | hitter-rate-shadow-2026-07-09-r1 | 3476 | 17 | 0.730 | 0.714 | False | 0 / 0 |
| batter_home_runs | hitter_rate_shadow_direct_player_game_blend | hitter-rate-shadow-2026-07-09-r1 | 3476 | 17 | 0.238 | 0.306 | True | 0 / 0 |
| batter_hits | hitter_count_regression | hitter-pg-2026-07-06-r2 | 3045 | 15 | 0.712 | 0.727 | True | 0 / 0 |
| batter_home_runs | hitter_hr_direct_player_game_blend | hitter-pg-2026-07-06-r2 | 3602 | 18 | 0.238 | 0.214 | False | 0 / 0 |
| batter_total_bases | hitter_tb_component_rebuild_from_rate_challengers | hitter-pg-2026-07-06-r2 | 2779 | 14 | 1.457 | 1.369 | False | 0 / 0 |
| hitter_plate_appearances | hitter_pa_baseline | hitter-pg-2026-07-06-r2 | 4299 | 21 | 0.753 | 0.757 | True | 0 / 0 |
| hitter_plate_appearances_challenger | hitter_pa_v3_challenger | hitter-pg-2026-07-06-r2 | 4136 | 21 | 0.749 | 0.760 | True | 0 / 0 |
| pitcher_batters_faced | pitcher_opportunity_v3 | pitcher-k-2026-07-03-r1 | 456 | 23 | 3.085 | 3.085 | True | 0 / 0 |
| pitcher_innings | pitcher_opportunity_v3 | pitcher-k-2026-07-03-r1 | 464 | 23 | 0.989 | 1.197 | True | 0 / 0 |
| pitcher_pitch_count | pitcher_opportunity_v3 | pitcher-k-2026-07-03-r1 | 452 | 23 | 12.335 | 17.366 | True | 0 / 0 |
| pitcher_strikeouts | pitcher_k_regression | pitcher-k-2026-07-03-r1 | 476 | 23 | 1.831 | 1.918 | True | 0 / 0 |
| batter_hits | hitter_hits_direct_player_game_blend | hitter-pg-2026-07-06-r2 | 1254 | 6 | 0.740 | 0.730 | False | 0 / 4 |
| batter_home_runs | hitter_count_regression | hitter-pg-2026-07-06-r2 | 697 | 3 | 0.304 | 0.229 | False | 2 / 7 |
| batter_total_bases | hitter_tb_direct_player_game_blend | hitter-pg-2026-07-06-r2 | 1520 | 7 | 1.291 | 1.414 | True | 0 / 3 |
| batter_hits | hitter_hits_direct_player_game_blend | hitter-pg-2026-07-03-r1 | 481 | 2 | 0.721 | 0.708 | False | 3 / 8 |
| batter_home_runs | hitter_hr_direct_player_game_blend | hitter-pg-2026-07-03-r1 | 481 | 2 | 0.256 | 0.236 | False | 3 / 8 |
| batter_total_bases | hitter_tb_direct_player_game_blend | hitter-pg-2026-07-03-r1 | 481 | 2 | 1.257 | 1.426 | False | 3 / 8 |
| hitter_plate_appearances | hitter_pa_baseline | hitter-pg-2026-07-03-r1 | 481 | 2 | 0.808 | 0.808 | False | 3 / 8 |
| hitter_plate_appearances_challenger | hitter_pa_v3_challenger | hitter-pg-2026-07-03-r1 | 481 | 2 | 0.748 | 0.808 | False | 3 / 8 |
| batter_hits | hitter_hits_direct_player_game_blend | hitter_pg:2026-07-03T08:22:56.064748+00:00 | 210 | 1 | 0.781 | 0.757 | False | 4 / 9 |
| batter_home_runs | hitter_hr_direct_player_game_blend | hitter_pg:2026-07-03T08:22:56.064748+00:00 | 210 | 1 | 0.273 | 0.236 | False | 4 / 9 |
| batter_total_bases | hitter_tb_direct_player_game_blend | hitter_pg:2026-07-03T08:22:56.064748+00:00 | 210 | 1 | 1.412 | 1.479 | False | 4 / 9 |
| hitter_plate_appearances | hitter_pa_v3 | hitter_pg:2026-07-03T08:22:56.064748+00:00 | 210 | 1 | 0.724 | 0.785 | False | 4 / 9 |
| pitcher_batters_faced | pitcher_opportunity_v3 | current | 36 | 2 | 3.236 | 3.236 | False | 3 / 8 |
| pitcher_innings | pitcher_opportunity_v3 | current | 37 | 2 | 1.148 | 1.504 | False | 3 / 8 |
| pitcher_pitch_count | pitcher_opportunity_v3 | current | 36 | 2 | 14.898 | 20.540 | False | 3 / 8 |
| pitcher_strikeouts | pitcher_k_regression | current | 38 | 2 | 2.021 | 2.174 | False | 3 / 8 |
| batter_hits | hitter_hits_direct_player_game_blend | hitter_pg:2026-07-02T14:31:24.102972+00:00 | 149 | 1 | 0.777 | 0.691 | False | 4 / 9 |
| batter_home_runs | hitter_hr_direct_player_game_blend | hitter_pg:2026-07-02T14:31:24.102972+00:00 | 149 | 1 | 0.258 | 0.223 | False | 4 / 9 |
| batter_total_bases | hitter_tb_direct_player_game_blend | hitter_pg:2026-07-02T14:31:24.102972+00:00 | 149 | 1 | 1.326 | 1.360 | False | 4 / 9 |
| hitter_plate_appearances | hitter_pa_v3 | hitter_pg:2026-07-02T14:31:24.102972+00:00 | 149 | 1 | 0.762 | 0.757 | False | 4 / 9 |

## Active Prospective Collection

| Stat | Family | Version | Pending | Graded | Graded Dates | Need 5 / 10 |
|---|---|---|---:|---:|---:|---:|
| batter_hits | hitter_count_regression | hitter-pg-2026-07-06-r2 | 0 | 3045 | 15 | 0 / 0 |
| batter_home_runs | hitter_hr_direct_player_game_blend | hitter-pg-2026-07-06-r2 | 0 | 3602 | 18 | 0 / 0 |
| batter_total_bases | hitter_tb_component_rebuild_from_rate_challengers | hitter-pg-2026-07-06-r2 | 0 | 2779 | 14 | 0 / 0 |
| hitter_plate_appearances | hitter_pa_baseline | hitter-pg-2026-07-06-r2 | 0 | 4299 | 21 | 0 / 0 |
| hitter_plate_appearances_challenger | hitter_pa_v3_challenger | hitter-pg-2026-07-06-r2 | 0 | 4136 | 21 | 0 / 0 |
| pitcher_batters_faced | pitcher_opportunity_v3 | pitcher-k-2026-07-03-r1 | 14 | 456 | 23 | 0 / 0 |
| pitcher_innings | pitcher_opportunity_v3 | pitcher-k-2026-07-03-r1 | 0 | 464 | 23 | 0 / 0 |
| pitcher_pitch_count | pitcher_opportunity_v3 | pitcher-k-2026-07-03-r1 | 12 | 452 | 23 | 0 / 0 |
| pitcher_strikeouts | pitcher_k_regression | pitcher-k-2026-07-03-r1 | 0 | 476 | 23 | 0 / 0 |

## Offer Translation Skill

Only true, non-synthetic paired offers are used for market comparison.

| Market | Rows | True Pair | Pair Rate | Model Brier | Paired Model | Market Brier | Gain vs Market |
|---|---:|---:|---:|---:|---:|---:|---:|
| batter_hits | 52147 | 36815 | 70.6% | 0.185 | 0.239 | 0.229 | -0.011 |
| batter_home_runs | 20713 | 0 | 0.0% | 0.059 | - | - | - |
| batter_total_bases | 52574 | 19819 | 37.7% | 0.169 | 0.247 | 0.242 | -0.005 |
| pitcher_strikeouts | 5041 | 5041 | 100.0% | 0.247 | 0.247 | 0.245 | -0.002 |

## Walk-Forward Windows

Complete seven-day OOF windows: 2
