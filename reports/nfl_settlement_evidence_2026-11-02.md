# NFL Post-Settlement Evidence

Date: 2026-11-02. FanDuel offers only.

| Stat / scope | Rows | MAE | Bias | Brier model / FanDuel | Calibration model / FanDuel | 80% coverage | ROI | Valid close | Avg CLV |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| passing_tds / all_offers | 2 | 0.400 | 0.000 | 0.205 / 0.205 | 45.0% / 45.0% | 100.0% | -5.8% | 0.0% | - |
| passing_tds / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| passing_yards / all_offers | 2 | 35.000 | 15.500 | 0.250 / 0.250 | 0.0% / 0.0% | 100.0% | -5.8% | 0.0% | - |
| passing_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| receiving_yards / all_offers | 11 | 12.955 | -5.591 | 0.250 / 0.250 | 4.5% / 4.5% | 100.0% | 2.8% | 0.0% | - |
| receiving_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| receptions / all_offers | 11 | 1.429 | -0.078 | 0.245 / 0.245 | 16.7% / 16.7% | 100.0% | -23.6% | 0.0% | - |
| receptions / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| rushing_yards / all_offers | 6 | 12.100 | 0.300 | 0.250 / 0.250 | 10.0% / 10.0% | 80.0% | 13.1% | 0.0% | - |
| rushing_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |

## Independent Policy Decisions

- passing_yards | forecast_repair:2259ac0e8a035747d3ef41b78dc91bfc13cba8244a09a2a7f4153e90affc2487:joint_calibrated | blocked: needs_three_independent_weeks, needs_60_settled_player_games, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- passing_yards | stat_component:components-20261010T222543607583Z-qb_tail | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- receiving_yards | raw_distribution | blocked: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- receiving_yards | receiving_quantile:d1f896b4284f98b8151c4e6b71747fcfb5b446b0ed7f2b6abc7ec66d490c112d | blocked: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
- receiving_yards | role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined | blocked: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
- receiving_yards | stat_component:components-20261015T203006649687Z-team_receiving_yards | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- receiving_yards | stat_component:receiving-oe-v2-20261012T234225243219Z | blocked: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- rushing_yards | forecast_repair:a426cb68d3dad523560c81c234b61857d0dfc34970b916872e33f56799105c3c:joint_calibrated | blocked: needs_three_independent_weeks, final_brier_not_improved, weekly_gain_uncertain
- total | raw_distribution | blocked: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, weekly_gain_uncertain, selected_probabilities_worse
- total | forecast_repair:cc8914500edbe8abe243f67561bbdc3ab446e7e7f2c6518da2577b65f313de96:small_total | blocked: needs_three_independent_weeks, needs_60_settled_player_games, weekly_gain_uncertain

No policy in this report authorizes bankroll betting. $1 micro remains a separate execution contract.
