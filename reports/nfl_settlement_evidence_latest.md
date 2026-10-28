# NFL Post-Settlement Evidence

Date: 2026-10-25. FanDuel offers only.

| Stat / scope | Rows | MAE | Bias | Brier model / FanDuel | Calibration model / FanDuel | 80% coverage | ROI | Valid close | Avg CLV |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| passing_tds / all_offers | 114 | 0.843 | 0.043 | 0.216 / 0.216 | 16.8% / 16.8% | 95.7% | -23.9% | 100.0% | -0.1% |
| passing_tds / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| passing_yards / all_offers | 111 | 53.587 | 2.848 | 0.250 / 0.250 | 5.8% / 5.8% | 95.7% | 1.7% | 77.2% | -0.1% |
| passing_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| receiving_yards / all_offers | 593 | 20.958 | -2.943 | 0.249 / 0.249 | 5.2% / 4.8% | 74.8% | -17.1% | 83.5% | -0.1% |
| receiving_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| receptions / all_offers | 628 | 1.377 | 0.128 | 0.240 / 0.240 | 8.1% / 8.1% | 88.1% | 1.3% | 97.6% | -0.2% |
| receptions / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| rushing_yards / all_offers | 261 | 22.037 | -8.278 | 0.250 / 0.250 | 1.3% / 1.3% | 64.8% | -5.1% | 80.6% | 0.2% |
| rushing_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |

## Independent Policy Decisions

- passing_yards | forecast_repair:2259ac0e8a035747d3ef41b78dc91bfc13cba8244a09a2a7f4153e90affc2487:joint_calibrated | blocked: needs_three_independent_weeks, needs_60_settled_player_games, calibration_not_improved, weekly_gain_uncertain
- passing_yards | stat_component:components-20261010T222543607583Z-qb_tail | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- receiving_yards | raw_distribution | blocked: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- receiving_yards | receiving_quantile:d1f896b4284f98b8151c4e6b71747fcfb5b446b0ed7f2b6abc7ec66d490c112d | blocked: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
- receiving_yards | role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined | blocked: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
- receiving_yards | stat_component:components-20261015T203006649687Z-team_receiving_yards | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- receiving_yards | stat_component:receiving-oe-v2-20261012T234225243219Z | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- rushing_yards | forecast_repair:a426cb68d3dad523560c81c234b61857d0dfc34970b916872e33f56799105c3c:joint_calibrated | blocked: needs_three_independent_weeks, needs_60_settled_player_games, calibration_not_improved, weekly_gain_uncertain
- total | raw_distribution | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- total | forecast_repair:cc8914500edbe8abe243f67561bbdc3ab446e7e7f2c6518da2577b65f313de96:small_total | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven

No policy in this report authorizes bankroll betting. $1 micro remains a separate execution contract.
