# NFL Post-Settlement Evidence

Date: 2026-10-12. FanDuel offers only.

| Stat / scope | Rows | MAE | Bias | Brier model / FanDuel | Calibration model / FanDuel | 80% coverage | ROI | Valid close | Avg CLV |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| passing_tds / all_offers | 6 | 1.000 | -0.200 | 0.288 / 0.288 | 53.5% / 53.5% | 100.0% | -100.0% | 100.0% | 0.2% |
| passing_tds / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| passing_yards / all_offers | 6 | 70.000 | -35.500 | 0.250 / 0.250 | 0.0% / 0.0% | 100.0% | -5.8% | 100.0% | 0.0% |
| passing_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| receiving_yards / all_offers | 47 | 26.967 | -8.700 | 0.250 / 0.252 | 5.3% / 5.7% | 60.0% | 6.0% | 89.2% | 0.1% |
| receiving_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| receptions / all_offers | 46 | 1.911 | -0.581 | 0.276 / 0.277 | 24.1% / 24.5% | 80.0% | 56.2% | 100.0% | 0.2% |
| receptions / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| rushing_yards / all_offers | 16 | 26.500 | -13.167 | 0.244 / 0.245 | 10.7% / 10.6% | 50.0% | -29.3% | 90.3% | 0.0% |
| rushing_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |

## Independent Policy Decisions

- receiving_yards | raw_distribution | blocked: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- receiving_yards | receiving_quantile:d1f896b4284f98b8151c4e6b71747fcfb5b446b0ed7f2b6abc7ec66d490c112d | blocked: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
- receiving_yards | role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined | blocked: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain

No policy in this report authorizes bankroll betting. $1 micro remains a separate execution contract.
