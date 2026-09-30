# MLB Player-Game Projection Ledger v2

Generated UTC: 2026-07-30T16:00:47Z
Status: **ready**
Lookback days: 45
Rows: 136242; graded: 101723

## Projection Skill by Version

| Type | Stat | Model | Version | Rows | Graded | MAE | Baseline MAE | MAE-Baseline | Bias |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| opportunity | hitter_plate_appearances | hitter_pa_baseline | hitter-pg-2026-07-06-r2 | 17476 | 12862 | 0.756 | 0.762 | -0.005 | 0.072 |
| opportunity | hitter_plate_appearances | hitter_pa_baseline | hitter-pg-2026-07-03-r1 | 1807 | 1376 | 0.803 | 0.803 | 0.000 | 0.041 |
| opportunity | hitter_plate_appearances | hitter_pa_v3 | hitter_pg:2026-07-03T08:22:56.064748+00:00 | 540 | 420 | 0.724 | 0.785 | -0.061 | -0.118 |
| opportunity | hitter_plate_appearances | hitter_pa_v3 | hitter_pg:2026-07-02T14:31:24.102972+00:00 | 510 | 396 | 0.751 | 0.739 | 0.012 | 0.003 |
| opportunity | hitter_plate_appearances | hitter_pa_v3 | current | 228 | 176 | 0.751 | 0.748 | 0.003 | -0.040 |
| opportunity | hitter_plate_appearances_challenger | hitter_pa_v3_challenger | hitter-pg-2026-07-06-r2 | 16843 | 12378 | 0.754 | 0.764 | -0.010 | 0.034 |
| opportunity | hitter_plate_appearances_challenger | hitter_pa_v3_challenger | hitter-pg-2026-07-03-r1 | 1807 | 1376 | 0.738 | 0.803 | -0.066 | 0.095 |
| opportunity | pitcher_batters_faced | pitcher_opportunity_v3 | pitcher-k-2026-07-03-r1 | 1504 | 1367 | 2.991 | 2.991 | 0.000 | 0.386 |
| opportunity | pitcher_batters_faced | pitcher_opportunity_v3 | current | 101 | 99 | 3.390 | 3.390 | 0.000 | 1.495 |
| opportunity | pitcher_innings | pitcher_opportunity_v3 | pitcher-k-2026-07-03-r1 | 1489 | 1400 | 0.984 | 1.176 | -0.192 | 0.040 |
| opportunity | pitcher_innings | pitcher_opportunity_v3 | current | 101 | 101 | 1.271 | 1.660 | -0.389 | 0.609 |
| opportunity | pitcher_pitch_count | pitcher_opportunity_v3 | pitcher-k-2026-07-03-r1 | 1489 | 1357 | 12.175 | 16.913 | -4.739 | 0.809 |
| opportunity | pitcher_pitch_count | pitcher_opportunity_v3 | current | 101 | 99 | 16.123 | 21.460 | -5.337 | 5.730 |
| player | batter_hits | hitter_rate_shadow_direct_player_game_blend | hitter-rate-shadow-2026-07-09-r1 | 14467 | 10544 | 0.731 | 0.717 | 0.014 | -0.291 |
| player | batter_hits | hitter_count_regression | hitter-pg-2026-07-06-r2 | 11338 | 8312 | 0.717 | 0.733 | -0.016 | 0.100 |
| player | batter_hits | hitter_hits_direct_player_game_blend | hitter-pg-2026-07-06-r2 | 6138 | 4550 | 0.730 | 0.722 | 0.009 | -0.226 |
| player | batter_hits | hitter_hits_direct_player_game_blend | hitter-pg-2026-07-03-r1 | 1807 | 1376 | 0.727 | 0.716 | 0.011 | -0.386 |
| player | batter_hits | hitter_hits_direct_player_game_blend | hitter_pg:2026-07-03T08:22:56.064748+00:00 | 540 | 420 | 0.784 | 0.757 | 0.027 | -0.431 |
| player | batter_hits | hitter_hits_direct_player_game_blend | hitter_pg:2026-07-02T14:31:24.102972+00:00 | 510 | 396 | 0.787 | 0.695 | 0.092 | -0.454 |
| player | batter_hits | hitter_count_regression | current | 228 | 176 | 0.693 | 0.690 | 0.003 | 0.061 |
| player | batter_home_runs | hitter_hr_direct_player_game_blend | hitter-pg-2026-07-06-r2 | 15148 | 11147 | 0.238 | 0.213 | 0.025 | 0.052 |
| player | batter_home_runs | hitter_rate_shadow_direct_player_game_blend | hitter-rate-shadow-2026-07-09-r1 | 14467 | 10544 | 0.239 | 0.308 | -0.069 | 0.053 |
| player | batter_home_runs | hitter_count_regression | hitter-pg-2026-07-06-r2 | 2328 | 1715 | 0.308 | 0.233 | 0.075 | 0.113 |
| player | batter_home_runs | hitter_hr_direct_player_game_blend | hitter-pg-2026-07-03-r1 | 1807 | 1376 | 0.258 | 0.236 | 0.022 | 0.019 |
| player | batter_home_runs | hitter_hr_direct_player_game_blend | hitter_pg:2026-07-03T08:22:56.064748+00:00 | 540 | 420 | 0.271 | 0.236 | 0.035 | 0.003 |
| player | batter_home_runs | hitter_hr_direct_player_game_blend | hitter_pg:2026-07-02T14:31:24.102972+00:00 | 510 | 396 | 0.260 | 0.226 | 0.033 | 0.042 |
| player | batter_home_runs | hitter_count_regression | current | 228 | 176 | 0.326 | 0.227 | 0.099 | 0.125 |
| player | batter_total_bases | hitter_tb_component_rebuild_from_rate_challengers | hitter-pg-2026-07-06-r2 | 12139 | 8829 | 1.455 | 1.366 | 0.089 | 0.448 |
| player | batter_total_bases | hitter_tb_direct_player_game_blend | hitter-pg-2026-07-06-r2 | 5337 | 4033 | 1.294 | 1.419 | -0.125 | -0.166 |
| player | batter_total_bases | hitter_tb_direct_player_game_blend | hitter-pg-2026-07-03-r1 | 1807 | 1376 | 1.281 | 1.437 | -0.156 | -0.206 |
| player | batter_total_bases | hitter_tb_direct_player_game_blend | hitter_pg:2026-07-03T08:22:56.064748+00:00 | 540 | 420 | 1.412 | 1.479 | -0.067 | -0.304 |
| player | batter_total_bases | hitter_tb_direct_player_game_blend | hitter_pg:2026-07-02T14:31:24.102972+00:00 | 510 | 396 | 1.338 | 1.378 | -0.041 | -0.228 |
| player | batter_total_bases | hitter_tb_direct_player_game_blend | current | 191 | 149 | 1.327 | 1.360 | -0.033 | -0.195 |
| player | batter_total_bases | hitter_count_regression | current | 37 | 27 | 1.629 | 1.305 | 0.324 | 0.557 |
| player | pitcher_strikeouts | pitcher_k_regression | pitcher-k-2026-07-03-r1 | 1525 | 1434 | 1.799 | 1.918 | -0.125 | 0.116 |
| player | pitcher_strikeouts | pitcher_k_regression | current | 104 | 104 | 2.067 | 2.318 | -0.200 | 0.927 |

## Recent Locked Player Forecasts

| Date | Player | Stat | Projection | Baseline | Actual | Error | Version | Status |
|---|---|---|---:|---:|---:|---:|---|---|
| 2026-07-30 | A.J. Ewing | batter_hits | 0.482 | 0.890 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Alec Burleson | batter_hits | 0.718 | 1.359 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Alex Bregman | batter_hits | 0.541 | 0.980 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Alex Freeland | batter_hits | 0.225 | 0.450 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Alex Jackson | batter_hits | 0.263 | 0.526 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Ali Sánchez | batter_hits | 0.185 | 0.370 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Andrés Chaparro | batter_hits | 0.231 | 0.461 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Andrew Benintendi | batter_hits | 0.254 | 0.508 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Andrew Velazquez | batter_hits | 0.255 | 0.510 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Andruw Monasterio | batter_hits | 0.494 | 0.988 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Andy Pages | batter_hits | 0.687 | 1.272 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Anthony Seigler | batter_hits | 0.646 | 1.292 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Anthony Volpe | batter_hits | 0.242 | 0.485 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Austin Riley | batter_hits | 0.566 | 1.132 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Austin Wynns | batter_hits | 0.300 | 0.600 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Ben Rice | batter_hits | 0.718 | 1.360 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Ben Williamson | batter_hits | 0.498 | 0.995 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Blaze Jordan | batter_hits | 0.471 | 0.942 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Bo Bichette | batter_hits | 0.570 | 1.038 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Braden Montgomery | batter_hits | 0.371 | 0.741 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Brady House | batter_hits | 0.365 | 0.655 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Brandon Lowe | batter_hits | 0.790 | 1.504 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Brandon Nimmo | batter_hits | 0.540 | 0.977 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Brett Baty | batter_hits | 0.345 | 0.691 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Brewer Hicklen | batter_hits | 0.339 | 0.678 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Brooks Lee | batter_hits | 0.524 | 1.048 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Bryan Reynolds | batter_hits | 0.754 | 1.431 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Bryce Eldridge | batter_hits | 0.592 | 1.082 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Caleb Durbin | batter_hits | 0.681 | 1.363 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Carlos Cortes | batter_hits | 0.472 | 0.944 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Carlos Narváez | batter_hits | 0.235 | 0.470 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Carson Benge | batter_hits | 0.450 | 0.824 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Carson Kelly | batter_hits | 0.369 | 0.737 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Carter Jensen | batter_hits | 0.653 | 1.226 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Ceddanne Rafaela | batter_hits | 0.764 | 1.421 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Cedric Mullins | batter_hits | 0.466 | 0.855 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Chandler Simpson | batter_hits | 0.780 | 1.561 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Chase Meidroth | batter_hits | 0.495 | 0.990 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | CJ Abrams | batter_hits | 0.595 | 1.113 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
| 2026-07-30 | Cole Young | batter_hits | 0.571 | 1.063 | - | - | hitter-rate-shadow-2026-07-09-r1 | pending |
