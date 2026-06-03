# MLB Prop Failure Diagnostics

- Generated: 2026-06-03T05:56:46Z
- Lookback days: 365
- Graded rows: 612
- Min bucket rows: 10

## Worst Stat / Side

| stat | side | n | settled | priced | priced_rate | win_rate | units | roi | avg_prob | cal_error | avg_line_clv | avg_price_clv | clv_price_beat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batter_hits | over | 137 | 137 | 137 | 100.0% | 59.9% | -12.37 | -9.0% | 56.8% | +3.0% | +0.07 | -3.47 | 29.2% |
| batter_total_bases | over | 42 | 42 | 42 | 100.0% | 42.9% | -1.85 | -4.4% | 49.8% | -6.9% | +0.00 | -1.82 | 9.5% |
| batter_total_bases | under | 140 | 140 | 79 | 56.4% | 62.1% | +1.69 | +2.1% | 65.2% | -3.0% | +0.00 | +0.00 | 0.0% |
| batter_home_runs | over | 16 | 16 | 16 | 100.0% | 18.8% | +1.70 | +10.6% | 16.6% | +2.2% | +0.00 | +0.00 | 0.0% |
| batter_hits | under | 107 | 107 | 107 | 100.0% | 47.7% | +19.50 | +18.2% | 43.8% | +3.9% | +0.00 | +0.00 | 0.0% |
| pitcher_strikeouts | over | 23 | 23 | 23 | 100.0% | 73.9% | +10.42 | +45.3% | 55.8% | +18.1% | +0.00 | +1.76 | 65.2% |
| pitcher_strikeouts | under | 12 | 12 | 12 | 100.0% | 83.3% | +7.61 | +63.4% | 58.4% | +24.9% | +0.00 | +0.00 | 0.0% |
| batter_home_runs | under | 135 | 135 | 0 | 0.0% | 84.4% | +0.00 |  | 89.8% | -5.4% | +0.00 |  | 0.0% |


## Worst Line Buckets

| stat | side | line_bucket | n | settled | priced | priced_rate | win_rate | units | roi | avg_prob | cal_error | avg_line_clv | avg_price_clv | clv_price_beat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batter_hits | over | H 0.5 | 137 | 137 | 137 | 100.0% | 59.9% | -12.37 | -9.0% | 56.8% | +3.0% | +0.07 | -3.47 | 29.2% |
| batter_total_bases | over | TB 1.5 | 42 | 42 | 42 | 100.0% | 42.9% | -1.85 | -4.4% | 49.8% | -6.9% | +0.00 | -1.82 | 9.5% |
| batter_total_bases | under | TB 1.5 | 140 | 140 | 79 | 56.4% | 62.1% | +1.69 | +2.1% | 65.2% | -3.0% | +0.00 | +0.00 | 0.0% |
| batter_home_runs | over | HR 0.5 | 16 | 16 | 16 | 100.0% | 18.8% | +1.70 | +10.6% | 16.6% | +2.2% | +0.00 | +0.00 | 0.0% |
| batter_hits | under | H 0.5 | 107 | 107 | 107 | 100.0% | 47.7% | +19.50 | +18.2% | 43.8% | +3.9% | +0.00 | +0.00 | 0.0% |
| pitcher_strikeouts | over | K 4.5-6.0 | 12 | 12 | 12 | 100.0% | 100.0% | +12.16 | +101.3% | 58.1% | +41.9% | +0.00 | +2.94 | 66.7% |
| batter_home_runs | under | HR 0.5 | 135 | 135 | 0 | 0.0% | 84.4% | +0.00 |  | 89.8% | -5.4% | +0.00 |  | 0.0% |


## Worst Price Buckets

| stat | side | price_bucket | n | settled | priced | priced_rate | win_rate | units | roi | avg_prob | cal_error | avg_line_clv | avg_price_clv | clv_price_beat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batter_hits | over | lay_150_180 | 42 | 42 | 42 | 100.0% | 52.4% | -6.74 | -16.0% | 54.2% | -1.8% | +0.00 | -0.97 | 28.6% |
| batter_total_bases | under | lay_130_149 | 13 | 13 | 13 | 100.0% | 53.8% | -0.89 | -6.8% | 63.3% | -9.5% | +0.00 | +0.00 | 0.0% |
| batter_total_bases | over | plus_money | 34 | 34 | 34 | 100.0% | 41.2% | -1.28 | -3.8% | 49.0% | -7.8% | +0.00 | -1.22 | 11.8% |
| batter_hits | over | heavy_lay | 78 | 78 | 78 | 100.0% | 66.7% | -2.92 | -3.7% | 60.4% | +6.3% | +0.13 | -5.36 | 34.6% |
| batter_total_bases | under | lay_150_180 | 42 | 42 | 42 | 100.0% | 61.9% | -0.36 | -0.9% | 64.0% | -2.1% | +0.00 | +0.00 | 0.0% |
| batter_home_runs | over | plus_money | 16 | 16 | 16 | 100.0% | 18.8% | +1.70 | +10.6% | 16.6% | +2.2% | +0.00 | +0.00 | 0.0% |
| batter_total_bases | under | fair_lay | 10 | 10 | 10 | 100.0% | 60.0% | +1.17 | +11.7% | 60.9% | -0.9% | +0.00 | +0.00 | 0.0% |
| batter_total_bases | under | heavy_lay | 12 | 12 | 12 | 100.0% | 75.0% | +1.68 | +14.0% | 64.1% | +10.9% | +0.00 | +0.00 | 0.0% |


## Worst Model Families

| stat | side | model_family | n | settled | priced | priced_rate | win_rate | units | roi | avg_prob | cal_error | avg_line_clv | avg_price_clv | clv_price_beat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batter_total_bases | over | regression | 39 | 39 | 39 | 100.0% | 41.0% | -3.87 | -9.9% | 50.6% | -9.6% | +0.00 | -1.96 | 10.3% |
| batter_total_bases | under | regression | 73 | 73 | 12 | 16.4% | 61.6% | -1.13 | -9.4% | 66.4% | -4.7% | +0.00 | +0.00 | 0.0% |
| batter_hits | over | regression | 136 | 136 | 136 | 100.0% | 60.3% | -11.37 | -8.4% | 56.8% | +3.5% | +0.07 | -3.49 | 28.7% |
| batter_total_bases | under | alt_clf | 67 | 67 | 67 | 100.0% | 62.7% | +2.81 | +4.2% | 63.9% | -1.2% | +0.00 | +0.00 | 0.0% |
| batter_home_runs | over | clf | 16 | 16 | 16 | 100.0% | 18.8% | +1.70 | +10.6% | 16.6% | +2.2% | +0.00 | +0.00 | 0.0% |
| batter_hits | under | alt_clf | 103 | 103 | 103 | 100.0% | 47.6% | +19.55 | +19.0% | 43.2% | +4.4% | +0.00 | +0.00 | 0.0% |
| pitcher_strikeouts | over | alt_clf | 13 | 13 | 13 | 100.0% | 61.5% | +3.66 | +28.2% | 55.3% | +6.2% | +0.00 | +0.84 | 61.5% |
| batter_home_runs | under | clf | 135 | 135 | 0 | 0.0% | 84.4% | +0.00 |  | 89.8% | -5.4% | +0.00 |  | 0.0% |


## Worst Probability Buckets

| stat | side | prob_bucket | n | settled | priced | priced_rate | win_rate | units | roi | avg_prob | cal_error | avg_line_clv | avg_price_clv | clv_price_beat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batter_hits | over | 50-55 | 19 | 19 | 19 | 100.0% | 42.1% | -6.26 | -32.9% | 53.0% | -10.8% | +0.00 | -1.41 | 10.5% |
| batter_total_bases | under | 55-60 | 18 | 18 | 7 | 38.9% | 61.1% | -2.30 | -32.8% | 58.3% | +2.8% | +0.00 | +0.00 | 0.0% |
| batter_total_bases | over | 50-55 | 10 | 10 | 10 | 100.0% | 30.0% | -2.95 | -29.5% | 52.3% | -22.3% | +0.00 | -2.54 | 0.0% |
| batter_hits | over | 55-60 | 42 | 42 | 42 | 100.0% | 57.1% | -5.69 | -13.5% | 57.8% | -0.7% | +0.07 | -2.96 | 45.2% |
| batter_hits | over | 45-50 | 22 | 22 | 22 | 100.0% | 54.5% | -2.41 | -10.9% | 48.2% | +6.3% | +0.00 | -0.78 | 18.2% |
| batter_total_bases | under | 60-65 | 64 | 64 | 52 | 81.2% | 56.2% | -1.46 | -2.8% | 63.2% | -7.0% | +0.00 | +0.00 | 0.0% |
| batter_total_bases | over | 45-50 | 20 | 20 | 20 | 100.0% | 45.0% | -0.19 | -1.0% | 47.1% | -2.1% | +0.00 | -1.36 | 20.0% |
| batter_hits | over | 60-65 | 35 | 35 | 35 | 100.0% | 68.6% | -0.10 | -0.3% | 62.2% | +6.3% | +0.09 | -4.06 | 34.3% |


## Worst Teams

| stat | side | team_abbr | n | settled | priced | priced_rate | win_rate | units | roi | avg_prob | cal_error | avg_line_clv | avg_price_clv | clv_price_beat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batter_hits | over | COL | 10 | 10 | 10 | 100.0% | 60.0% | -0.23 | -2.3% | 54.1% | +5.9% | +0.00 | -1.09 | 10.0% |
| batter_total_bases | under | CWS | 11 | 11 | 6 | 54.5% | 45.5% | +0.16 | +2.6% | 66.0% | -20.5% | +0.00 | +0.00 | 0.0% |
| batter_total_bases | under | COL | 13 | 13 | 6 | 46.2% | 61.5% | +0.91 | +15.2% | 64.8% | -3.2% | +0.00 | +0.00 | 0.0% |


## Worst Players

_No buckets met the sample threshold._


## Worst Bankroll Reasons

| stat | side | bankroll_reasons | n | settled | priced | priced_rate | win_rate | units | roi | avg_prob | cal_error | avg_line_clv | avg_price_clv | clv_price_beat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batter_hits | under | threshold_disabled:threshold_clf | 19 | 19 | 19 | 100.0% | 36.8% | -2.31 | -12.2% | 42.1% | -5.2% | +0.00 | +0.00 | 0.0% |
| batter_total_bases | over | no_model_edge | 39 | 39 | 39 | 100.0% | 41.0% | -3.87 | -9.9% | 50.6% | -9.6% | +0.00 | -1.96 | 10.3% |
| batter_total_bases | under | no_model_edge | 73 | 73 | 12 | 16.4% | 61.6% | -1.13 | -9.4% | 66.4% | -4.7% | +0.00 | +0.00 | 0.0% |
| batter_hits | over | no_model_edge | 136 | 136 | 136 | 100.0% | 60.3% | -11.37 | -8.4% | 56.8% | +3.5% | +0.07 | -3.49 | 28.7% |
| batter_home_runs | over | hr_longshot_variance | 14 | 14 | 14 | 100.0% | 14.3% | -0.90 | -6.4% | 16.3% | -2.1% | +0.00 | +0.00 | 0.0% |
| batter_total_bases | under | daily_exposure_cap | 51 | 51 | 51 | 100.0% | 56.9% | -1.42 | -2.8% | 63.5% | -6.7% | +0.00 | +0.00 | 0.0% |
| batter_total_bases | under | threshold_disabled:threshold_clf | 14 | 14 | 14 | 100.0% | 78.6% | +3.21 | +22.9% | 64.5% | +14.1% | +0.00 | +0.00 | 0.0% |
| batter_hits | under | daily_exposure_cap | 83 | 83 | 83 | 100.0% | 50.6% | +22.84 | +27.5% | 43.1% | +7.5% | +0.00 | +0.00 | 0.0% |


## Bookability Gaps

| stat | side | price_bucket | n | settled | priced | priced_rate | win_rate | avg_prob |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batter_home_runs | under | missing_price | 135 | 135 | 0 | 0.0% | 84.4% | 89.8% |
| batter_total_bases | under | missing_price | 61 | 61 | 0 | 0.0% | 62.3% | 67.5% |
| batter_hits | under | plus_money | 101 | 101 | 101 | 100.0% | 46.5% | 42.8% |
| batter_hits | over | heavy_lay | 78 | 78 | 78 | 100.0% | 66.7% | 60.4% |
| batter_hits | over | lay_150_180 | 42 | 42 | 42 | 100.0% | 52.4% | 54.2% |
| batter_total_bases | under | lay_150_180 | 42 | 42 | 42 | 100.0% | 61.9% | 64.0% |
| batter_total_bases | over | plus_money | 34 | 34 | 34 | 100.0% | 41.2% | 49.0% |
| batter_home_runs | over | plus_money | 16 | 16 | 16 | 100.0% | 18.8% | 16.6% |

