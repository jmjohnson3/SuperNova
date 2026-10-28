# NFL Close Capture Diagnostic

Date: 2026-10-26; completed-window valid closes: 50/58
Observed alternative lines do not establish availability or price CLV at the locked line. Unknown CLV stays null.

Future games are not counted as failed closes. Counts are locked forecast rows, not independent bets.

| Scope | Completed rows | Valid | Coverage | Meets 90% |
|---|---:|---:|---:|---|
| all_locked | 58 | 50 | 86.2% | False |
| current_fanduel | 31 | 28 | 90.3% | True |
| selected_real_tier | 0 | 0 | pending | None |
| fanduel | 58 | 50 | 86.2% | False |

| Prediction | Player | Book | Locked line | Status / Phase | Observed lines |
|---|---|---|---:|---|---|
| 26896 | D.Prescott | fanduel | 267.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('174.5'), Decimal('199.5'), Decimal('224.5'), Decimal('249.5'), Decimal('263.5'), Decimal('274.5'), Decimal('299.5'), Decimal('324.5'), Decimal('349.5'), Decimal('374.5'), Decimal('399.5')] |
| 26899 | S.Barkley | fanduel | 68.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('64.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 26900 | J.Williams | fanduel | 64.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('62.5'), Decimal('65.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 26901 | T.Goodson | fanduel | 10.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('12.5'), Decimal('13.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
| 26915 | C.Lamb | fanduel | 78.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5'), Decimal('174.5')] |
| 26946 | D.Prescott | fanduel | 265.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('174.5'), Decimal('199.5'), Decimal('224.5'), Decimal('249.5'), Decimal('263.5'), Decimal('274.5'), Decimal('299.5'), Decimal('324.5'), Decimal('349.5'), Decimal('374.5'), Decimal('399.5')] |
| 26965 | C.Lamb | fanduel | 78.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5'), Decimal('174.5')] |
| 26972 | B.Spann-Ford | fanduel | 12.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('13.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5')] |
