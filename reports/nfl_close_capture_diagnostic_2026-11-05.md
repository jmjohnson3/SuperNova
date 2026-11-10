# NFL Close Capture Diagnostic

Date: 2026-11-05; completed-window valid closes: 119/145
Observed alternative lines do not establish availability or price CLV at the locked line. Unknown CLV stays null.

Future games are not counted as failed closes. Counts are locked forecast rows, not independent bets.

| Scope | Completed rows | Valid | Coverage | Meets 90% |
|---|---:|---:|---:|---|
| all_locked | 145 | 119 | 82.1% | False |
| current_fanduel | 40 | 40 | 100.0% | True |
| selected_real_tier | 0 | 0 | pending | None |
| fanduel | 145 | 119 | 82.1% | False |

| Prediction | Player | Book | Locked line | Status / Phase | Observed lines |
|---|---|---|---:|---|---|
| 30123 | T.Lawrence | fanduel | 237.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('149.5'), Decimal('174.5'), Decimal('199.5'), Decimal('224.5'), Decimal('233.5'), Decimal('249.5'), Decimal('274.5'), Decimal('299.5'), Decimal('324.5'), Decimal('349.5'), Decimal('374.5')] |
| 30124 | J.Fagnano | fanduel | 160.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('99.5'), Decimal('124.5'), Decimal('149.5'), Decimal('162.5'), Decimal('163.5'), Decimal('165.5'), Decimal('174.5'), Decimal('199.5'), Decimal('224.5'), Decimal('249.5'), Decimal('274.5'), Decimal('299.5'), Decimal('324.5')] |
| 30140 | J.Meyers | fanduel | 40.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('38.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5')] |
| 30141 | J.Hill | fanduel | 13.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('15.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5')] |
| 30143 | P.Washington | fanduel | 63.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('60.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 30145 | Z.Flowers | fanduel | 47.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('48.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5')] |
| 30147 | Bri.Thomas | fanduel | 10.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('15.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5')] |
| 30150 | J.Lane | fanduel | 16.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('15.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5')] |
| 30152 | J.Cameron | fanduel | 10.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('8.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
| 30153 | N.Boerkircher | fanduel | 7.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('8.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
| 30188 | T.Lawrence | fanduel | 237.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('149.5'), Decimal('174.5'), Decimal('199.5'), Decimal('224.5'), Decimal('233.5'), Decimal('249.5'), Decimal('274.5'), Decimal('299.5'), Decimal('324.5'), Decimal('349.5'), Decimal('374.5')] |
| 30189 | J.Fagnano | fanduel | 160.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('99.5'), Decimal('124.5'), Decimal('149.5'), Decimal('162.5'), Decimal('163.5'), Decimal('165.5'), Decimal('174.5'), Decimal('199.5'), Decimal('224.5'), Decimal('249.5'), Decimal('274.5'), Decimal('299.5'), Decimal('324.5')] |
| 30205 | J.Meyers | fanduel | 40.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('38.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5')] |
| 30206 | J.Hill | fanduel | 13.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('15.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5')] |
| 30208 | P.Washington | fanduel | 63.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('60.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 30210 | Z.Flowers | fanduel | 47.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('48.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5')] |
| 30212 | Bri.Thomas | fanduel | 12.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('15.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5')] |
| 30215 | J.Lane | fanduel | 16.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('15.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5')] |
| 30216 | J.Cuevas | fanduel | 7.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('8.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
| 30217 | J.Cameron | fanduel | 10.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('8.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
| 30218 | N.Boerkircher | fanduel | 7.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('8.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
| 30256 | D.Henry | fanduel | 68.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('71.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5'), Decimal('174.5')] |
| 30273 | P.Washington | fanduel | 63.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('60.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 30277 | Bri.Thomas | fanduel | 12.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('15.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5')] |
| 30281 | J.Cuevas | fanduel | 7.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('8.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
| 30282 | J.Cameron | fanduel | 10.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('8.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
