# NFL Post-Settlement Evidence

Date: 2026-10-15. FanDuel offers only.

| Stat / scope | Rows | MAE | Bias | Brier model / FanDuel | Calibration model / FanDuel | 80% coverage | ROI | Valid close | Avg CLV |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| passing_tds / all_offers | 12 | 1.400 | 1.400 | 0.153 / 0.153 | 38.9% / 38.9% | 100.0% | -8.6% | 100.0% | 0.9% |
| passing_tds / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| passing_yards / all_offers | 11 | 71.000 | 19.500 | 0.250 / 0.250 | 1.7% / 1.7% | 100.0% | -14.5% | 63.3% | 0.0% |
| passing_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| receiving_yards / all_offers | 85 | 15.500 | -2.700 | 0.250 / 0.250 | 5.4% / 5.4% | 80.0% | -0.1% | 72.3% | -0.2% |
| receiving_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| receptions / all_offers | 86 | 1.312 | -0.420 | 0.278 / 0.278 | 16.8% / 16.8% | 80.0% | 3.1% | 100.0% | 0.6% |
| receptions / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |
| rushing_yards / all_offers | 29 | 9.900 | 3.500 | 0.250 / 0.250 | 6.0% / 6.0% | 100.0% | -15.6% | 85.3% | -0.2% |
| rushing_yards / selected_trials | 0 | - | - | - / - | - / - | - | - | - | - |

## Independent Policy Decisions

- receiving_yards | raw_distribution | blocked: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
- receiving_yards | receiving_quantile:d1f896b4284f98b8151c4e6b71747fcfb5b446b0ed7f2b6abc7ec66d490c112d | blocked: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
- receiving_yards | role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined | blocked: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain

No policy in this report authorizes bankroll betting. $1 micro remains a separate execution contract.
