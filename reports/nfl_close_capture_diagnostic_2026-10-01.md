# NFL Close Capture Diagnostic

Date: 2026-10-01; completed-window valid closes: 87/113
Observed alternative lines do not establish availability or price CLV at the locked line. Unknown CLV stays null.

Future games are not counted as failed closes. Counts are locked forecast rows, not independent bets.

| Scope | Completed rows | Valid | Coverage | Meets 90% |
|---|---:|---:|---:|---|
| all_locked | 113 | 87 | 77.0% | False |
| current_fanduel | 33 | 28 | 84.8% | False |
| selected_real_tier | 0 | 0 | pending | None |
| fanduel | 113 | 87 | 77.0% | False |

| Prediction | Player | Book | Locked line | Status / Phase | Observed lines |
|---|---|---|---:|---|---|
| 9735 | A.Rodgers | fanduel | 214.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('124.5'), Decimal('149.5'), Decimal('174.5'), Decimal('199.5'), Decimal('216.5'), Decimal('224.5'), Decimal('249.5'), Decimal('274.5'), Decimal('299.5'), Decimal('324.5'), Decimal('349.5')] |
| 9739 | D.Watson | fanduel | 29.5 | same_line_missing_side_price / completed_window | [Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('27.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5')] |
| 9744 | Q.Judkins | fanduel | 55.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('53.5'), Decimal('54.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5')] |
| 9759 | J.Jeudy | fanduel | 13.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5')] |
| 9760 | P.Freiermuth | fanduel | 28.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('27.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5')] |
| 9764 | R.Wilson | fanduel | 19.5 | same_line_missing_side_price / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('20.5'), Decimal('22.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5')] |
| 9765 | R.Sanders | fanduel | 11.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('12.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
| 9770 | G.Bernard | fanduel | 9.5 | same_line_missing_side_price / completed_window | [Decimal('4.5'), Decimal('8.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
| 9681 | A.Rodgers | fanduel | 212.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('124.5'), Decimal('149.5'), Decimal('174.5'), Decimal('199.5'), Decimal('216.5'), Decimal('224.5'), Decimal('249.5'), Decimal('274.5'), Decimal('299.5'), Decimal('324.5'), Decimal('349.5')] |
| 9682 | D.Watson | fanduel | 187.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('124.5'), Decimal('149.5'), Decimal('174.5'), Decimal('185.5'), Decimal('199.5'), Decimal('224.5'), Decimal('249.5'), Decimal('274.5'), Decimal('299.5')] |
| 9684 | D.Watson | fanduel | 29.5 | same_line_missing_side_price / completed_window | [Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('27.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5')] |
| 9686 | J.Warren | fanduel | 65.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('75.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 9689 | Q.Judkins | fanduel | 55.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('53.5'), Decimal('54.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5')] |
| 9701 | D.Metcalf | fanduel | 41.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('45.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5')] |
| 9704 | J.Jeudy | fanduel | 13.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5')] |
| 9706 | J.Warren | fanduel | 22.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('25.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5')] |
| 9713 | Q.Judkins | fanduel | 12.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('8.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5')] |
| 9714 | D.Boston | fanduel | 35.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('37.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5')] |
| 9831 | D.Washington | fanduel | 15.5 | missing_close / completed_window | None |
| 9837 | Q.Judkins | fanduel | 9.5 | missing_close / completed_window | None |
| 9840 | K.Concepcion | fanduel | 36.5 | missing_close / completed_window | None |
| 9884 | Q.Judkins | fanduel | 53.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('54.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5')] |
| 9900 | D.Washington | fanduel | 15.5 | missing_close / completed_window | None |
| 9902 | R.Wilson | fanduel | 22.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('20.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5')] |
| 9906 | Q.Judkins | fanduel | 9.5 | missing_close / completed_window | None |
| 9909 | K.Concepcion | fanduel | 36.5 | missing_close / completed_window | None |
