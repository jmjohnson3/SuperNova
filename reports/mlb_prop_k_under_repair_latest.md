# MLB Prop K-Under Repair Report

Generated UTC: 2026-07-30T14:38:59Z
Status: **ready**
Usage: K-under micro gate; paper/watch output remains visible.

- Rows: 2486
- Line/book groups allowed for micro: 1

| Level | Key | Rows | Dates | Win | Model P | Cal Err | Brier | CLV Rows | CLV Beat | Avg CLV | BF MAE | PC MAE | Micro Allowed | Blockers |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| book | *|draftkings | 1330 | 54 | 51.0% | 50.6% | -0.4% | 0.24733 | 1013 | 46.7% | 0.081 | 2.884 | 10.995 | False | clv_beat_below_micro_gate |
| book | *|fanduel | 1156 | 52 | 50.1% | 49.9% | -0.2% | 0.24665 | 986 | 42.0% | 0.157 | 2.906 | 11.135 | False | clv_beat_below_micro_gate |
| global | *|* | 2486 | 54 | 50.6% | 50.2% | -0.3% | 0.24701 | 1999 | 44.4% | 0.118 | 2.895 | 11.060 | False | clv_beat_below_micro_gate |
| line | K 4.5-6.0|* | 1605 | 54 | 51.4% | 50.9% | -0.5% | 0.24726 | 1293 | 44.5% | 0.072 | 2.856 | 10.928 | False | clv_beat_below_micro_gate |
| line | K 6.5-8.0|* | 370 | 51 | 50.5% | 51.0% | 0.5% | 0.24823 | 296 | 37.5% | -0.271 | 2.712 | 9.716 | False | clv_beat_below_micro_gate, avg_clv_below_micro_gate |
| line | K 8.5+|* | 19 | 8 | 57.9% | 51.1% | -6.8% | 0.24956 | 14 | 21.4% | -0.116 | 3.695 | 18.543 | False | rows<75, clv_rows<25, clv_beat_below_micro_gate, avg_clv_below_micro_gate |
| line | K <4.5|* | 492 | 52 | 47.6% | 47.3% | -0.2% | 0.24520 | 396 | 49.7% | 0.568 | 3.129 | 12.219 | False | clv_beat_below_micro_gate |
| line_book | K 4.5-6.0|draftkings | 858 | 54 | 51.7% | 51.4% | -0.3% | 0.24622 | 656 | 47.1% | 0.010 | 2.840 | 10.828 | False | clv_beat_below_micro_gate |
| line_book | K 4.5-6.0|fanduel | 747 | 52 | 51.0% | 50.4% | -0.6% | 0.24845 | 637 | 41.9% | 0.137 | 2.874 | 11.043 | False | clv_beat_below_micro_gate |
| line_book | K 6.5-8.0|draftkings | 203 | 51 | 51.2% | 51.4% | 0.2% | 0.24884 | 152 | 34.9% | -0.465 | 2.694 | 9.602 | False | clv_beat_below_micro_gate, avg_clv_below_micro_gate |
| line_book | K 6.5-8.0|fanduel | 167 | 51 | 49.7% | 50.5% | 0.8% | 0.24748 | 144 | 40.3% | -0.067 | 2.734 | 9.854 | False | clv_beat_below_micro_gate, avg_clv_below_micro_gate |
| line_book | K 8.5+|draftkings | 10 | 8 | 60.0% | 51.6% | -8.4% | 0.25429 | 7 | 14.3% | 0.019 | 3.680 | 18.112 | False | rows<75, clv_rows<25, clv_beat_below_micro_gate, calibration_error_above_gate |
| line_book | K 8.5+|fanduel | 9 | 7 | 55.6% | 50.5% | -5.0% | 0.24430 | 7 | 28.6% | -0.251 | 3.711 | 19.022 | False | rows<75, clv_rows<25, clv_beat_below_micro_gate, avg_clv_below_micro_gate |
| line_book | K <4.5|draftkings | 259 | 52 | 47.9% | 47.0% | -0.9% | 0.24955 | 198 | 55.6% | 0.737 | 3.152 | 12.377 | True | - |
| line_book | K <4.5|fanduel | 233 | 49 | 47.2% | 47.7% | 0.5% | 0.24036 | 198 | 43.9% | 0.399 | 3.103 | 12.044 | False | clv_beat_below_micro_gate |
