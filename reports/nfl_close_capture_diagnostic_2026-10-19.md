# NFL Close Capture Diagnostic

Date: 2026-10-19; completed-window valid closes: 15/28
Observed alternative lines do not establish availability or price CLV at the locked line. Unknown CLV stays null.

Future games are not counted as failed closes. Counts are locked forecast rows, not independent bets.

| Scope | Completed rows | Valid | Coverage | Meets 90% |
|---|---:|---:|---:|---|
| all_locked | 28 | 15 | 53.6% | False |
| current_fanduel | 28 | 15 | 53.6% | False |
| selected_real_tier | 0 | 0 | pending | None |
| fanduel | 28 | 15 | 53.6% | False |

| Prediction | Player | Book | Locked line | Status / Phase | Observed lines |
|---|---|---|---:|---|---|
| 22924 | J.Daniels | fanduel | 210.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('124.5'), Decimal('149.5'), Decimal('174.5'), Decimal('199.5'), Decimal('209.5'), Decimal('224.5'), Decimal('249.5'), Decimal('274.5'), Decimal('299.5'), Decimal('324.5'), Decimal('349.5')] |
| 22926 | A.Ekeler | fanduel | 18.5 | player_market_not_observed_in_capture / completed_window | None |
| 22927 | R.White | fanduel | 45.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('44.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5')] |
| 22928 | B.Purdy | fanduel | 17.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('21.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5')] |
| 22929 | J.Daniels | fanduel | 37.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('38.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5')] |
| 22930 | K.Black | fanduel | 24.5 | player_market_not_observed_in_capture / completed_window | None |
| 22938 | M.Evans | fanduel | 71.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('64.5'), Decimal('66.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 22942 | A.Ekeler | fanduel | 7.5 | player_market_not_observed_in_capture / completed_window | None |
| 22945 | R.White | fanduel | 13.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5')] |
| 22947 | C.Okonkwo | fanduel | 26.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5')] |
| 22951 | A.Williams | fanduel | 23.5 | player_market_not_observed_in_capture / completed_window | None |
| 22957 | A.Ekeler | fanduel | 1.5 | player_market_not_observed_in_capture / completed_window | None |
| 22966 | A.Williams | fanduel | 2.5 | player_market_not_observed_in_capture / completed_window | None |
