# NFL Close Capture Diagnostic

Date: 2026-10-05; completed-window valid closes: 59/66
Observed alternative lines do not establish availability or price CLV at the locked line. Unknown CLV stays null.

Future games are not counted as failed closes. Counts are locked forecast rows, not independent bets.

| Scope | Completed rows | Valid | Coverage | Meets 90% |
|---|---:|---:|---:|---|
| all_locked | 66 | 59 | 89.4% | False |
| current_fanduel | 37 | 37 | 100.0% | True |
| selected_real_tier | 0 | 0 | pending | None |
| fanduel | 66 | 59 | 89.4% | False |

| Prediction | Player | Book | Locked line | Status / Phase | Observed lines |
|---|---|---|---:|---|---|
| 14658 | A.Kamara | fanduel | 37.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('35.5'), Decimal('36.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5')] |
| 14659 | Br.Robinson | fanduel | 29.5 | same_line_missing_side_price / completed_window | [Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('28.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5')] |
| 14661 | K.Miller | fanduel | 23.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('25.5'), Decimal('26.5'), Decimal('27.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5')] |
| 14670 | A.Hooper | fanduel | 10.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('6.5'), Decimal('7.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
| 14672 | O.Zaccheaus | fanduel | 7.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('8.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
| 14678 | D.London | fanduel | 79.5 | same_line_missing_side_price / completed_window | [Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('80.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5'), Decimal('174.5')] |
| 14681 | Bi.Robinson | fanduel | 37.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('35.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5')] |
