# NFL Close Capture Diagnostic

Date: 2026-10-12; completed-window valid closes: 114/121
Observed alternative lines do not establish availability or price CLV at the locked line. Unknown CLV stays null.

Future games are not counted as failed closes. Counts are locked forecast rows, not independent bets.

| Scope | Completed rows | Valid | Coverage | Meets 90% |
|---|---:|---:|---:|---|
| all_locked | 121 | 114 | 94.2% | True |
| current_fanduel | 45 | 44 | 97.8% | True |
| selected_real_tier | 0 | 0 | pending | None |
| fanduel | 121 | 114 | 94.2% | True |

| Prediction | Player | Book | Locked line | Status / Phase | Observed lines |
|---|---|---|---:|---|---|
| 18305 | K.Williams | fanduel | 68.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 18318 | T.Johnson | fanduel | 12.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5')] |
| 18322 | K.Shakir | fanduel | 45.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('46.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5')] |
| 18323 | K.Williams | fanduel | 21.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('22.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5')] |
| 18327 | B.Corum | fanduel | 3.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('2.5'), Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5')] |
| 18392 | Skyler Bell | fanduel | 6.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('14.5')] |
| 18424 | J.Allen | fanduel | 36.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('35.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5')] |
