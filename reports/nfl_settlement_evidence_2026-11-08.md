# NFL Post-Settlement Evidence

Date: 2026-11-08. FanDuel offers only.

| Stat / scope | Rows | MAE | Bias | Brier model / FanDuel | Calibration model / FanDuel | 80% coverage | ROI | Valid close | Avg CLV |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| passing_tds / all_offers | 96 | 0.878 | -0.148 | 0.226 / 0.226 | 9.5% / 9.5% | 91.3% | 4.3% | 98.9% | -0.3% |
| passing_tds / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| passing_yards / all_offers | 96 | 52.413 | -11.978 | 0.250 / 0.250 | 2.2% / 2.2% | 91.3% | -5.9% | 73.1% | -0.1% |
| passing_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| receiving_yards / all_offers | 515 | 20.680 | -7.966 | 0.250 / 0.250 | 3.0% / 3.0% | 74.4% | 0.3% | 79.4% | -0.0% |
| receiving_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| receptions / all_offers | 522 | 1.384 | -0.148 | 0.258 / 0.258 | 10.6% / 10.6% | 87.4% | -9.3% | 97.2% | 0.2% |
| receptions / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| rushing_yards / all_offers | 236 | 17.445 | -1.427 | 0.250 / 0.250 | 0.9% / 0.9% | 76.4% | -6.3% | 77.5% | -0.1% |
| rushing_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |

## Independent Policy Decisions

- passing_yards | forecast_repair:2259ac0e8a035747d3ef41b78dc91bfc13cba8244a09a2a7f4153e90affc2487:joint_calibrated | blocked: calibration_not_improved, weekly_gain_uncertain
- passing_yards | stat_component:components-20261010T222543607583Z-qb_tail | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- passing_yards | raw_distribution | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- receiving_yards | raw_distribution | blocked: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- receiving_yards | receiving_quantile:d1f896b4284f98b8151c4e6b71747fcfb5b446b0ed7f2b6abc7ec66d490c112d | blocked: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
- receiving_yards | role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined | blocked: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
- receiving_yards | stat_component:components-20261015T203006649687Z-team_receiving_yards | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- receiving_yards | stat_component:receiving-oe-v2-20261012T234225243219Z | blocked: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- rushing_yards | forecast_repair:a426cb68d3dad523560c81c234b61857d0dfc34970b916872e33f56799105c3c:joint_calibrated | blocked: weekly_gain_uncertain
- rushing_yards | raw_distribution | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- total | raw_distribution | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, weekly_gain_uncertain, coverage_not_proven, selected_probabilities_worse
- total | forecast_repair:cc8914500edbe8abe243f67561bbdc3ab446e7e7f2c6518da2577b65f313de96:small_total | blocked: needs_three_independent_weeks, needs_60_settled_player_games, weekly_gain_uncertain, coverage_not_proven

No policy in this report authorizes bankroll betting. $1 micro remains a separate execution contract.
