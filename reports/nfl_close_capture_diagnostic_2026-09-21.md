# NFL Close Capture Diagnostic

Date: 2026-09-21; completed-window valid closes: 19/23
Observed alternative lines do not establish availability or price CLV at the locked line. Unknown CLV stays null.

Future games are not counted as failed closes. Counts are locked forecast rows, not independent bets.

| Scope | Completed rows | Valid | Coverage | Meets 90% |
|---|---:|---:|---:|---|
| all_locked | 23 | 19 | 82.6% | False |
| current_fanduel | 4 | 4 | 100.0% | True |
| selected_real_tier | 0 | 0 | pending | None |
| draftkings | 19 | 15 | 78.9% | False |
| fanduel | 4 | 4 | 100.0% | True |

| Prediction | Player | Book | Locked line | Status / Phase | Observed lines |
|---|---|---|---:|---|---|
| 6842 | M.Stafford | draftkings | 238.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('239.5'), Decimal('240.5')] |
| 6849 | C.Skattebo | draftkings | 51.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('52.5')] |
| 6859 | T.Higbee | draftkings | 13.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('10.5'), Decimal('11.5'), Decimal('12.5')] |
| 6865 | K.Williams | draftkings | 17.5 | exact_line_unavailable_in_captured_feed / completed_window | [Decimal('16.5')] |
