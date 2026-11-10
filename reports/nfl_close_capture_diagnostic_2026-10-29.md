# NFL Close Capture Diagnostic

Date: 2026-10-29; completed-window valid closes: 98/126
Observed alternative lines do not establish availability or price CLV at the locked line. Unknown CLV stays null.

Future games are not counted as failed closes. Counts are locked forecast rows, not independent bets.

| Scope | Completed rows | Valid | Coverage | Meets 90% |
|---|---:|---:|---:|---|
| all_locked | 126 | 98 | 77.8% | False |
| current_fanduel | 40 | 35 | 87.5% | False |
| selected_real_tier | 0 | 0 | pending | None |
| fanduel | 126 | 98 | 77.8% | False |

| Prediction | Player | Book | Locked line | Status / Phase | Observed lines |
|---|---|---|---:|---|---|
| 26998 | J.Love | fanduel | 260.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('174.5'), Decimal('199.5'), Decimal('224.5'), Decimal('249.5'), Decimal('264.5'), Decimal('274.5'), Decimal('299.5'), Decimal('324.5'), Decimal('349.5'), Decimal('374.5'), Decimal('399.5')] |
| 27001 | J.Jacobs | fanduel | 58.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('60.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5')] |
| 27004 | M.Lloyd | fanduel | 18.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('16.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5')] |
| 27015 | C.Hubbard | fanduel | 13.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('16.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5')] |
| 27017 | J.Metchie | fanduel | 15.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('16.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5')] |
| 27021 | T.Kraft | fanduel | 41.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('43.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5')] |
| 27023 | M.Lloyd | fanduel | 5.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5')] |
| 27026 | M.Golden | fanduel | 64.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('57.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 27131 | C.Hubbard | fanduel | 15.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('16.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5')] |
| 27133 | B.Melton | fanduel | 6.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
| 27134 | J.Metchie | fanduel | 15.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('16.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5')] |
| 27136 | C.Watson | fanduel | 71.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('70.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 27055 | J.Love | fanduel | 260.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('174.5'), Decimal('199.5'), Decimal('224.5'), Decimal('249.5'), Decimal('264.5'), Decimal('274.5'), Decimal('299.5'), Decimal('324.5'), Decimal('349.5'), Decimal('374.5'), Decimal('399.5')] |
| 27056 | B.Young | fanduel | 256.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('174.5'), Decimal('199.5'), Decimal('224.5'), Decimal('249.5'), Decimal('252.5'), Decimal('274.5'), Decimal('299.5'), Decimal('324.5'), Decimal('349.5'), Decimal('374.5'), Decimal('399.5')] |
| 27058 | J.Jacobs | fanduel | 58.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('60.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5')] |
| 27059 | C.Hubbard | fanduel | 68.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('67.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 27074 | J.Metchie | fanduel | 15.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('16.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5')] |
| 27076 | C.Watson | fanduel | 71.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('70.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 27078 | T.Kraft | fanduel | 46.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('43.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5')] |
| 27082 | T.McMillan | fanduel | 74.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('70.5'), Decimal('72.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 27116 | J.Jacobs | fanduel | 58.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('60.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5')] |
| 27137 | B.Tremayne | fanduel | 6.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('2.5'), Decimal('4.5'), Decimal('5.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
| 27142 | T.McMillan | fanduel | 71.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('70.5'), Decimal('72.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 27192 | J.Smith | fanduel | 3.5 | player_market_not_observed_in_capture / completed_window | None |
| 27200 | B.Tremayne | fanduel | 5.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('2.5'), Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5')] |
| 27205 | T.McMillan | fanduel | 72.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('70.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 27206 | M.Evans | fanduel | 6.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('5.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5')] |
| 27211 | J.Smith | fanduel | 0.5 | player_market_not_observed_in_capture / completed_window | None |
