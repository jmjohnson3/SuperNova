# NFL Close Capture Diagnostic

Date: 2026-10-08; completed-window valid closes: 76/102
Observed alternative lines do not establish availability or price CLV at the locked line. Unknown CLV stays null.

Future games are not counted as failed closes. Counts are locked forecast rows, not independent bets.

| Scope | Completed rows | Valid | Coverage | Meets 90% |
|---|---:|---:|---:|---|
| all_locked | 102 | 76 | 74.5% | False |
| current_fanduel | 37 | 37 | 100.0% | True |
| selected_real_tier | 0 | 0 | pending | None |
| fanduel | 102 | 76 | 74.5% | False |

| Prediction | Player | Book | Locked line | Status / Phase | Observed lines |
|---|---|---|---:|---|---|
| 14770 | D.Prescott | fanduel | 263.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('174.5'), Decimal('199.5'), Decimal('224.5'), Decimal('249.5'), Decimal('268.5'), Decimal('274.5'), Decimal('299.5'), Decimal('324.5'), Decimal('349.5'), Decimal('374.5'), Decimal('399.5')] |
| 14774 | K.Gainwell | fanduel | 9.5 | same_line_missing_side_price / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('12.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5')] |
| 14775 | J.Williams | fanduel | 73.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('66.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 14776 | T.Goodson | fanduel | 18.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('13.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5')] |
| 14779 | B.Irving | fanduel | 56.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('52.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5')] |
| 14781 | D.Prescott | fanduel | 1.5 | same_line_missing_side_price / completed_window | [Decimal('1.5'), Decimal('2.5'), Decimal('3.5'), Decimal('4.5')] |
| 14789 | C.Godwin | fanduel | 27.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('31.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5')] |
| 14790 | C.Lamb | fanduel | 81.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('83.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5'), Decimal('174.5')] |
| 14791 | K.Gainwell | fanduel | 8.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('12.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5')] |
| 14793 | G.Pickens | fanduel | 61.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('63.5'), Decimal('65.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 14795 | C.Otton | fanduel | 27.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5')] |
| 14800 | E.Egbuka | fanduel | 30.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('35.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5')] |
| 14801 | T.Johnson | fanduel | 11.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('12.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5')] |
| 14802 | T.Hurst | fanduel | 22.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('16.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5')] |
| 14827 | D.Prescott | fanduel | 267.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('174.5'), Decimal('199.5'), Decimal('224.5'), Decimal('249.5'), Decimal('268.5'), Decimal('274.5'), Decimal('299.5'), Decimal('324.5'), Decimal('349.5'), Decimal('374.5'), Decimal('399.5')] |
| 14830 | J.Williams | fanduel | 70.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('66.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5')] |
| 14831 | T.Goodson | fanduel | 16.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('13.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5')] |
| 14834 | S.Tucker | fanduel | 8.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5')] |
| 14835 | B.Irving | fanduel | 51.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('52.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5')] |
| 14836 | J.Daniels | fanduel | 44.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('43.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5')] |
| 14846 | C.Godwin | fanduel | 28.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('31.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('89.5')] |
| 14847 | C.Lamb | fanduel | 81.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5'), Decimal('83.5'), Decimal('89.5'), Decimal('99.5'), Decimal('109.5'), Decimal('124.5'), Decimal('149.5'), Decimal('174.5')] |
| 14854 | B.Irving | fanduel | 12.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('13.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5')] |
| 14856 | B.Spann-Ford | fanduel | 5.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5')] |
| 14858 | T.Johnson | fanduel | 13.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('12.5'), Decimal('14.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5')] |
| 14859 | T.Hurst | fanduel | 20.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('4.5'), Decimal('9.5'), Decimal('14.5'), Decimal('16.5'), Decimal('19.5'), Decimal('24.5'), Decimal('29.5'), Decimal('39.5'), Decimal('49.5'), Decimal('59.5'), Decimal('69.5'), Decimal('79.5')] |
