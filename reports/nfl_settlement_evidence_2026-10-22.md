# NFL Post-Settlement Evidence

Date: 2026-10-22. FanDuel offers only.

| Stat / scope | Rows | MAE | Bias | Brier model / FanDuel | Calibration model / FanDuel | 80% coverage | ROI | Valid close | Avg CLV |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| passing_tds / all_offers | 8 | - | - | - / - | - / - | - | - | 100.0% | -0.2% |
| passing_tds / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| passing_yards / all_offers | 7 | - | - | - / - | - / - | - | - | 12.5% | 0.0% |
| passing_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| receiving_yards / all_offers | 54 | - | - | - / - | - / - | - | - | 62.2% | -0.3% |
| receiving_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| receptions / all_offers | 57 | - | - | - / - | - / - | - | - | 98.3% | -0.7% |
| receptions / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| rushing_yards / all_offers | 22 | - | - | - / - | - / - | - | - | 79.8% | -1.0% |
| rushing_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |

## Independent Policy Decisions

- passing_yards | forecast_repair:2259ac0e8a035747d3ef41b78dc91bfc13cba8244a09a2a7f4153e90affc2487:joint_calibrated | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- passing_yards | stat_component:components-20261010T222543607583Z-qb_tail | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- receiving_yards | raw_distribution | blocked: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- receiving_yards | receiving_quantile:d1f896b4284f98b8151c4e6b71747fcfb5b446b0ed7f2b6abc7ec66d490c112d | blocked: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
- receiving_yards | role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined | blocked: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
- rushing_yards | forecast_repair:a426cb68d3dad523560c81c234b61857d0dfc34970b916872e33f56799105c3c:joint_calibrated | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven

No policy in this report authorizes bankroll betting. $1 micro remains a separate execution contract.
