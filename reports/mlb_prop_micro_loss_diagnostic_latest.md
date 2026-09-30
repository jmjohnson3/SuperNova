# MLB Prop Micro Loss Diagnostic

Generated UTC: 2026-09-01T11:24:43+00:00
Window: 2026-07-19 through 2026-09-01
Locked micro rows: 35

## Loss / CLV Drivers

| Driver | Rows | Graded | Pending | Record | ROI | Cal Err | CLV Rows | CLV Beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| clv_model_weak | 34 | 34 | 0 | 17-17-0 | +7.5% | +7.1% | 29 | 48.3% | +2.334 |
| bucket_policy_no_bet | 29 | 29 | 0 | 16-13-0 | +18.7% | +3.5% | 24 | 45.8% | +0.979 |
| projection_overconfident | 18 | 18 | 0 | 0-18-0 | -100.0% | +55.4% | 16 | 56.2% | +4.238 |
| tb_hr_line_gate_failed | 17 | 17 | 0 | 9-8-0 | +21.8% | +3.0% | 14 | 50.0% | +1.179 |
| did_not_beat_close | 15 | 15 | 0 | 8-7-0 | +18.2% | +3.5% | 15 | 0.0% | -0.781 |
| opportunity_weak | 14 | 14 | 0 | 9-5-0 | +40.0% | -8.3% | 11 | 36.4% | +0.730 |
| bucket_roi_negative | 13 | 13 | 0 | 6-7-0 | +10.3% | -2.5% | 10 | 30.0% | +2.829 |
| bookability_weak | 12 | 12 | 0 | 6-6-0 | +18.4% | -2.4% | 9 | 11.1% | +0.000 |
| close_capture_weak | 12 | 12 | 0 | 6-6-0 | +18.4% | -2.4% | 9 | 11.1% | +0.000 |
| bucket_clv_weak | 7 | 7 | 0 | 3-4-0 | +2.4% | -0.2% | 5 | 20.0% | +0.000 |
| clv_unknown_or_invalid | 5 | 5 | 0 | 3-2-0 | +33.1% | -12.3% | 0 | - | - |
| selector_no_bet_locked | 2 | 2 | 0 | 1-1-0 | +0.5% | +23.7% | 2 | 0.0% | +0.000 |

## By Market / Side

| Market / Side | Rows | Graded | Record | ROI | Win Rate | Avg Model P | Cal Err | CLV Beat |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| batter_total_bases | over | 21 | 21 | 9-12-0 | -1.4% | 42.9% | 53.3% | +10.4% | 61.1% |
| pitcher_strikeouts | under | 14 | 14 | 8-6-0 | +13.2% | 57.1% | 61.8% | +4.7% | 33.3% |

## Recent Locked Micro Rows

| Date | Player | Market | Side | Line | Price | Result | Actual | CLV | Model P | Drivers |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|
| 2026-07-29 | Sal Stewart | batter_total_bases | over | 1.5 | +210 | L | 1.0 | +13.20 | 41.9% | bucket_roi_negative, clv_model_weak, projection_overconfident |
| 2026-07-29 | Alec Burleson | batter_total_bases | over | 1.5 | +255 | L | 1.0 | +12.65 | 39.0% | clv_model_weak, projection_overconfident |
| 2026-07-29 | Jordan Walker | batter_total_bases | over | 1.5 | +205 | L | 1.0 | +15.52 | 43.7% | bucket_roi_negative, projection_overconfident |
| 2026-07-29 | Matt Olson | batter_total_bases | over | 1.5 | +275 | L | 1.0 | +19.20 | 42.9% | clv_model_weak, projection_overconfident |
| 2026-07-29 | Zack Littell | pitcher_strikeouts | under | 2.5 | +114 | W | 2.0 | -0.43 | 57.4% | bucket_roi_negative, clv_model_weak, did_not_beat_close |
| 2026-07-29 | Randy Dobnak | pitcher_strikeouts | under | 2.5 | -117 | L | 4.0 | -0.43 | 57.7% | clv_model_weak, did_not_beat_close, projection_overconfident |
| 2026-07-28 | Michael Lorenzen | pitcher_strikeouts | under | 3.5 | -104 | W | 2.0 | +3.15 | 57.7% | bucket_policy_no_bet, clv_model_weak |
| 2026-07-27 | Andrew Alvarez | pitcher_strikeouts | under | 3.5 | -108 | W | 3.0 | - | 56.3% | bucket_policy_no_bet, clv_model_weak, clv_unknown_or_invalid |
| 2026-07-27 | Mitch Keller | pitcher_strikeouts | under | 3.5 | +136 | L | 4.0 | +4.14 | 57.1% | bucket_policy_no_bet, clv_model_weak, projection_overconfident |
| 2026-07-26 | Ernie Clement | batter_total_bases | over | 1.5 | +138 | W | 2.0 | -1.03 | 44.1% | bookability_weak, bucket_clv_weak, bucket_policy_no_bet, bucket_roi_negative, close_capture_weak, clv_model_weak, did_not_beat_close, tb_hr_line_gate_failed |
| 2026-07-26 | Erick Fedde | pitcher_strikeouts | under | 3.5 | -136 | W | 2.0 | -2.47 | 57.8% | bucket_policy_no_bet, clv_model_weak, did_not_beat_close, opportunity_weak |
| 2026-07-26 | Jeffrey Springs | pitcher_strikeouts | under | 3.5 | +121 | W | 0.0 | - | 55.1% | bucket_policy_no_bet, clv_model_weak, clv_unknown_or_invalid |
| 2026-07-26 | Andrew Abbott | pitcher_strikeouts | under | 3.5 | +114 | L | 4.0 | -3.99 | 57.4% | bucket_policy_no_bet, clv_model_weak, did_not_beat_close, projection_overconfident |
| 2026-07-25 | Ernie Clement | batter_total_bases | over | 1.5 | +151 | W | 4.0 | +0.00 | 43.5% | bookability_weak, bucket_policy_no_bet, bucket_roi_negative, close_capture_weak, clv_model_weak, did_not_beat_close, opportunity_weak, tb_hr_line_gate_failed |
| 2026-07-25 | Mitch Bratt | pitcher_strikeouts | under | 3.5 | +105 | W | 3.0 | +3.37 | 55.9% | bucket_policy_no_bet, clv_model_weak, opportunity_weak |
| 2026-07-24 | Masataka Yoshida | batter_total_bases | over | 1.5 | +150 | L | 0.0 | +0.00 | 40.6% | bookability_weak, bucket_policy_no_bet, bucket_roi_negative, close_capture_weak, clv_model_weak, did_not_beat_close, opportunity_weak, projection_overconfident, tb_hr_line_gate_failed |
| 2026-07-23 | Ernie Clement | batter_total_bases | over | 1.5 | +132 | L | 1.0 | +0.00 | 43.1% | bookability_weak, bucket_clv_weak, bucket_policy_no_bet, bucket_roi_negative, close_capture_weak, clv_model_weak, did_not_beat_close, projection_overconfident, tb_hr_line_gate_failed |
| 2026-07-22 | Vladimir Guerrero Jr. | batter_total_bases | over | 1.5 | +133 | W | 2.0 | +0.00 | 43.3% | bookability_weak, bucket_clv_weak, bucket_policy_no_bet, bucket_roi_negative, close_capture_weak, clv_model_weak, did_not_beat_close, opportunity_weak, tb_hr_line_gate_failed |
| 2026-07-22 | Jackson Merrill | batter_total_bases | over | 1.5 | +146 | W | 5.0 | +0.00 | 41.2% | bookability_weak, bucket_clv_weak, bucket_policy_no_bet, bucket_roi_negative, close_capture_weak, clv_model_weak, did_not_beat_close, tb_hr_line_gate_failed |
| 2026-07-22 | Travis Bazzana | batter_total_bases | over | 1.5 | +139 | L | 0.0 | - | 43.4% | bookability_weak, bucket_clv_weak, bucket_policy_no_bet, bucket_roi_negative, close_capture_weak, clv_model_weak, clv_unknown_or_invalid, opportunity_weak, projection_overconfident, tb_hr_line_gate_failed |
| 2026-07-22 | Carlos Cortes | batter_total_bases | over | 1.5 | +144 | L | 0.0 | +1.03 | 41.1% | bookability_weak, bucket_clv_weak, bucket_policy_no_bet, bucket_roi_negative, close_capture_weak, clv_model_weak, opportunity_weak, projection_overconfident, tb_hr_line_gate_failed |
| 2026-07-21 | Matt Vierling | batter_total_bases | over | 1.5 | +152 | W | 2.0 | - | 40.8% | bookability_weak, bucket_policy_no_bet, bucket_roi_negative, close_capture_weak, clv_model_weak, clv_unknown_or_invalid, opportunity_weak, tb_hr_line_gate_failed |
| 2026-07-21 | Troy Johnston | batter_total_bases | over | 1.5 | +138 | L | 1.0 | - | 42.7% | bookability_weak, bucket_clv_weak, bucket_policy_no_bet, bucket_roi_negative, close_capture_weak, clv_model_weak, clv_unknown_or_invalid, opportunity_weak, projection_overconfident, tb_hr_line_gate_failed |
| 2026-07-21 | Tyler Phillips | pitcher_strikeouts | under | 3.5 | -122 | W | 2.0 | +3.55 | 56.2% | bucket_policy_no_bet, clv_model_weak |
| 2026-07-19 | Jacob Lopez | pitcher_strikeouts | under | 3.5 | -103 | L | 6.0 | +0.00 | 74.0% | bucket_policy_no_bet, clv_model_weak, did_not_beat_close, opportunity_weak, projection_overconfident |
| 2026-07-19 | Logan Gilbert | pitcher_strikeouts | under | 6.5 | +105 | L | 10.0 | +0.00 | 69.6% | bookability_weak, bucket_policy_no_bet, close_capture_weak, clv_model_weak, did_not_beat_close, projection_overconfident, selector_no_bet_locked |
| 2026-07-19 | Grant Holmes | pitcher_strikeouts | under | 4.5 | +101 | W | 2.0 | +0.00 | 77.8% | bookability_weak, bucket_policy_no_bet, close_capture_weak, clv_model_weak, did_not_beat_close, opportunity_weak, selector_no_bet_locked |
| 2026-07-19 | Daylen Lile | batter_total_bases | over | 1.5 | +125 | L | 0.0 | +0.80 | 66.8% | bucket_policy_no_bet, clv_model_weak, projection_overconfident, tb_hr_line_gate_failed |
| 2026-07-19 | Shea Langeliers | batter_total_bases | over | 1.5 | -111 | W | 3.0 | +2.35 | 84.6% | bucket_policy_no_bet, clv_model_weak, opportunity_weak, tb_hr_line_gate_failed |
| 2026-07-19 | Joc Pederson | batter_total_bases | over | 1.5 | +117 | W | 3.0 | -2.60 | 69.3% | bucket_policy_no_bet, clv_model_weak, did_not_beat_close, opportunity_weak, tb_hr_line_gate_failed |
| 2026-07-19 | Ben Rice | batter_total_bases | over | 1.5 | +137 | W | 2.0 | +6.35 | 68.3% | bucket_policy_no_bet, clv_model_weak, opportunity_weak, tb_hr_line_gate_failed |
| 2026-07-19 | Eugenio Suárez | batter_total_bases | over | 1.5 | +106 | L | 1.0 | +1.21 | 78.9% | bucket_policy_no_bet, clv_model_weak, projection_overconfident, tb_hr_line_gate_failed |
| 2026-07-19 | Fernando Tatis Jr. | batter_total_bases | over | 1.5 | +106 | W | 5.0 | +3.15 | 78.2% | bucket_policy_no_bet, clv_model_weak, tb_hr_line_gate_failed |
| 2026-07-19 | James Wood | batter_total_bases | over | 1.5 | -106 | L | 0.0 | +5.25 | 81.2% | bucket_policy_no_bet, clv_model_weak, projection_overconfident, tb_hr_line_gate_failed |
| 2026-07-19 | Paul Skenes | pitcher_strikeouts | under | 6.5 | +126 | L | 8.0 | -0.77 | 75.8% | bucket_policy_no_bet, clv_model_weak, did_not_beat_close, projection_overconfident |
