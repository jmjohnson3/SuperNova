# NFL Close Capture Diagnostic

Date: 2026-09-28; completed-window valid closes: 35/39
Observed alternative lines do not establish availability or price CLV at the locked line. Unknown CLV stays null.

Future games are not counted as failed closes. Counts are locked forecast rows, not independent bets.

| Scope | Completed rows | Valid | Coverage | Meets 90% |
|---|---:|---:|---:|---|
| all_locked | 39 | 35 | 89.7% | False |
| current_fanduel | 20 | 20 | 100.0% | True |
| selected_real_tier | 0 | 0 | pending | None |
| fanduel | 39 | 35 | 89.7% | False |

| Prediction | Player | Book | Locked line | Status / Phase | Observed lines |
|---|---|---|---:|---|---|
| 9621 | C.Kmet | fanduel | 7.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('8.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5')] |
| 9622 | D.Smith | fanduel | 71.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('70.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 9624 | R.Odunze | fanduel | 25.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('26.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5')] |
| 9628 | M.Lemon | fanduel | 27.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('21.5'), Decimal('24.5'), Decimal('26.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5')] |
