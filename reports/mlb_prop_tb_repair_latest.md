# MLB TB Prop Repair Report

Generated: 2026-06-19T19:02:22.921818+00:00
Rows: 31530 | Dates: 2026-05-31 to 2026-06-18

## Top Repair Targets

| Bucket | Rows | ROI | CLV | Brier M/Mkt | TB Bias | PA MAE | Issues |
|---|---:|---:|---:|---:|---:|---:|---|
| over | common | TB 1.5 | plus_150_249 | draftkings | 120 | -0.305 | 0.952 | 0.208/0.207 | 0.145 | 0.768 | market_beats_model_brier |
| over | common | TB 2.5 | plus_500_plus | fanduel | 393 | -0.232 | 0.271 | 0.108/0.105 | 0.086 | 0.711 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_250_499 | fanduel | 3633 | -0.120 | 0.105 | 0.164/0.158 | -0.064 | 0.705 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_250_499 | fanduel | 76 | 0.070 | 0.501 | 0.207/0.204 | 0.091 | 0.653 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_150_249 | fanduel | 2059 | -0.184 | 0.249 | 0.208/0.199 | -0.039 | 0.755 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | fanduel | 2979 | -0.155 | 0.129 | 0.237/0.231 | -0.129 | 0.650 | market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 4.5+ | plus_500_plus | fanduel | 5837 | -0.268 | 0.063 | 0.066/0.063 | -0.117 | 0.695 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_500_plus | fanduel | 2467 | -0.181 | 0.057 | 0.100/0.098 | -0.012 | 0.724 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_150_249 | fanduel | 1935 | -0.143 | 0.123 | 0.202/0.191 | -0.252 | 0.668 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | heavy_lay | draftkings | 830 | -0.032 | -0.617 | 0.231/0.231 | -0.044 | 0.673 | market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | draftkings | 2655 | -0.109 | 0.383 | 0.241/0.239 | -0.153 | 0.637 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | fanduel | 186 | -0.012 | 0.504 | 0.243/0.223 | -0.771 | 0.699 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | draftkings | 73 | -0.116 | -0.053 | 0.247/0.254 | -0.391 | 0.837 | tb_projection_low, pa_projection_error, negative_or_flat_clv |
| over | common | TB 1.5 | fair_lay | draftkings | 467 | -0.098 | 0.043 | 0.251/0.249 | -0.333 | 0.710 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | fanduel | 192 | -0.197 | 0.172 | 0.252/0.240 | -0.282 | 0.735 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | draftkings | 27 | 0.001 | 0.768 | 0.268/0.249 | -0.484 | 0.507 | sample_small, tb_projection_low, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_250_499 | fanduel | 320 | -0.113 | 0.375 | 0.144/0.137 | -0.489 | 0.668 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_250_499 | fanduel | 3266 | -0.137 | 0.087 | 0.153/0.149 | -0.178 | 0.671 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_150_249 | fanduel | 404 | -0.023 | 0.413 | 0.212/0.205 | -0.508 | 0.698 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | lay_130_149 | draftkings | 652 | -0.046 | -0.285 | 0.250/0.246 | -0.348 | 0.666 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | fanduel | 746 | -0.112 | -0.077 | 0.252/0.246 | -0.380 | 0.691 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_100_149 | fanduel | 30 | -0.082 | 1.051 | 0.255/0.232 | -0.688 | 0.457 | sample_small, alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier |
| under | common | TB 1.5 | fair_lay | draftkings | 436 | -0.049 | -0.133 | 0.256/0.248 | -0.419 | 0.733 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | plus_100_149 | draftkings | 63 | 0.076 | -0.024 | 0.261/0.251 | -0.078 | 0.943 | pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | heavy_lay | fanduel | 30 | -0.109 | -0.075 | 0.226/0.204 | -0.243 | 0.603 | sample_small, tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 2.5 | lay_130_149 | draftkings | 24 | -0.073 | -0.610 | 0.263/0.249 | -0.322 | 0.492 | sample_small, tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |

## Exact Buckets

| Bucket | Rows | Dates | Win | ROI | CLV Beat | Avg CLV | Pred/Actual TB | PA MAE | Issues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| over | common | TB 1.5 | plus_150_249 | draftkings | 120 | 16 | 0.275 | -0.305 | 0.559 | 0.952 | 1.320/1.175 | 0.768 | market_beats_model_brier |
| over | common | TB 2.5 | plus_500_plus | fanduel | 393 | 15 | 0.120 | -0.232 | 0.375 | 0.271 | 1.066/0.980 | 0.711 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_250_499 | fanduel | 3633 | 16 | 0.207 | -0.120 | 0.402 | 0.105 | 1.326/1.390 | 0.705 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_250_499 | fanduel | 76 | 15 | 0.289 | 0.070 | 0.477 | 0.501 | 1.064/0.974 | 0.653 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_150_249 | fanduel | 2059 | 19 | 0.296 | -0.184 | 0.423 | 0.249 | 1.191/1.230 | 0.755 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_150_180 | fanduel | 101 | 14 | 0.545 | -0.118 | 0.444 | 0.088 | 1.891/2.396 | 0.690 | tb_projection_low, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | fanduel | 2979 | 18 | 0.384 | -0.155 | 0.388 | 0.129 | 1.468/1.598 | 0.650 | market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | lay_150_180 | draftkings | 1459 | 19 | 0.609 | -0.021 | 0.299 | -0.330 | 1.515/1.617 | 0.617 | negative_or_flat_clv, weak_clv_beat |
| over | common | TB 2.5 | fair_lay | draftkings | 10 | 4 | 0.400 | -0.227 | 0.700 | 0.174 | 2.319/2.400 | 0.440 | sample_small, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_500_plus | fanduel | 5837 | 16 | 0.071 | -0.268 | 0.332 | 0.063 | 1.387/1.504 | 0.695 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_500_plus | fanduel | 2467 | 16 | 0.113 | -0.181 | 0.308 | 0.057 | 1.221/1.233 | 0.724 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_150_249 | fanduel | 1935 | 16 | 0.282 | -0.143 | 0.398 | 0.123 | 1.592/1.844 | 0.668 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | heavy_lay | draftkings | 830 | 19 | 0.636 | -0.032 | 0.235 | -0.617 | 1.421/1.465 | 0.673 | market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | draftkings | 2655 | 19 | 0.398 | -0.109 | 0.466 | 0.383 | 1.512/1.665 | 0.637 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | fanduel | 186 | 16 | 0.430 | -0.012 | 0.461 | 0.504 | 1.891/2.661 | 0.699 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | draftkings | 73 | 17 | 0.507 | -0.116 | 0.548 | -0.053 | 1.965/2.356 | 0.837 | tb_projection_low, pa_projection_error, negative_or_flat_clv |
| over | common | TB 1.5 | fair_lay | draftkings | 467 | 17 | 0.475 | -0.098 | 0.407 | 0.043 | 1.750/2.084 | 0.710 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | fanduel | 192 | 16 | 0.464 | -0.197 | 0.461 | 0.172 | 1.833/2.115 | 0.735 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | draftkings | 27 | 7 | 0.481 | 0.001 | 0.750 | 0.768 | 2.183/2.667 | 0.507 | sample_small, tb_projection_low, market_beats_model_brier |
| under | common | TB 2.5 | plus_100_149 | draftkings | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.920 | 2.367/2.000 | 0.200 | sample_small, tb_projection_high, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_150_249 | fanduel | 11 | 3 | 0.000 | -1.000 | 0.727 | 1.083 | 2.399/2.091 | 0.291 | sample_small, alt_tail_requires_separate_proof, tb_projection_high, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_250_499 | fanduel | 320 | 16 | 0.181 | -0.113 | 0.474 | 0.375 | 1.855/2.344 | 0.668 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_250_499 | fanduel | 3266 | 16 | 0.187 | -0.137 | 0.416 | 0.087 | 1.505/1.682 | 0.671 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | lay_130_149 | draftkings | 1 | 1 | 0.000 | -1.000 | 0.000 | -1.920 | 2.367/2.000 | 0.200 | sample_small, tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_150_249 | fanduel | 404 | 16 | 0.317 | -0.023 | 0.427 | 0.413 | 1.787/2.295 | 0.698 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | lay_130_149 | fanduel | 4 | 2 | 0.000 | -1.000 | 0.000 | -0.910 | 2.503/1.750 | 0.250 | sample_small, tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | lay_130_149 | draftkings | 652 | 18 | 0.557 | -0.046 | 0.339 | -0.285 | 1.583/1.931 | 0.666 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | fanduel | 746 | 16 | 0.471 | -0.112 | 0.326 | -0.077 | 1.634/2.013 | 0.691 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_100_149 | fanduel | 30 | 5 | 0.400 | -0.082 | 0.533 | 1.051 | 2.112/2.800 | 0.457 | sample_small, alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier |
| under | common | TB 1.5 | fair_lay | draftkings | 436 | 18 | 0.514 | -0.049 | 0.378 | -0.133 | 1.785/2.204 | 0.733 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |

## Diagnostic Groups

| Group | Rows | Win | ROI | TB Bias | PA MAE | Model/Mkt Brier | Issues |
|---|---:|---:|---:|---:|---:|---:|---|
| over | 0 TB | 10536 | 0.000 | -1.000 | 1.390 | 0.733 | 0.077/0.073 | tb_projection_high, market_beats_model_brier, weak_clv_beat |
| over | 0-2 PA | 1692 | 0.033 | -0.893 | 0.977 | 1.884 | 0.077/0.078 | tb_projection_high, pa_projection_error, negative_or_flat_clv, weak_clv_beat, low_valid_clv_coverage |
| over | 1 TB | 6997 | 0.000 | -1.000 | 0.423 | 0.642 | 0.082/0.080 | tb_projection_high, market_beats_model_brier, weak_clv_beat |
| over | 2-3 TB | 5739 | 0.415 | 0.059 | -0.866 | 0.664 | 0.188/0.177 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | 3 PA | 3420 | 0.089 | -0.649 | 0.567 | 0.670 | 0.096/0.092 | tb_projection_high, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | 4 PA | 13981 | 0.220 | -0.191 | -0.054 | 0.404 | 0.160/0.155 | market_beats_model_brier, weak_clv_beat |
| over | 4+ TB non-HR/unknown | 880 | 0.850 | 2.480 | -2.934 | 0.790 | 0.423/0.413 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | 5+ PA | 8153 | 0.364 | 0.230 | -0.823 | 0.936 | 0.215/0.208 | tb_projection_low, pa_projection_error, market_beats_model_brier, weak_clv_beat |
| over | HR driven | 3900 | 0.890 | 2.651 | -3.591 | 0.669 | 0.450/0.446 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | PA unknown | 806 | 0.246 | -0.182 | -0.098 | n/a | 0.167/0.162 | market_beats_model_brier |
| over | cross_book | 5059 | 0.403 | 0.190 | -0.645 | 0.660 | 0.249/0.243 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | same_book | 3365 | 0.409 | -0.112 | -0.176 | 0.656 | 0.241/0.240 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | synthetic | 19628 | 0.162 | -0.268 | -0.005 | 0.701 | 0.128/0.123 | market_beats_model_brier, weak_clv_beat |
| under | 0 TB | 1157 | 1.000 | 0.635 | 1.527 | 0.682 | 0.158/0.183 | tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | 0-2 PA | 121 | 0.917 | 0.503 | 1.199 | 2.340 | 0.172/0.193 | tb_projection_high, pa_projection_error, weak_clv_beat, low_valid_clv_coverage |
| under | 1 TB | 893 | 1.000 | 0.643 | 0.538 | 0.639 | 0.163/0.185 | tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | 2-3 TB | 760 | 0.011 | -0.981 | -0.738 | 0.643 | 0.361/0.324 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | 3 PA | 211 | 0.810 | 0.305 | 0.827 | 0.902 | 0.191/0.204 | tb_projection_high, pa_projection_error, negative_or_flat_clv, weak_clv_beat |
| under | 4 PA | 1706 | 0.638 | 0.041 | 0.044 | 0.350 | 0.232/0.235 | negative_or_flat_clv, weak_clv_beat |
| under | 4+ TB non-HR/unknown | 120 | 0.000 | -1.000 | -2.848 | 0.810 | 0.356/0.317 | tb_projection_low, pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | 5+ PA | 1350 | 0.468 | -0.222 | -0.745 | 0.860 | 0.267/0.256 | tb_projection_low, pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | HR driven | 548 | 0.000 | -1.000 | -3.574 | 0.635 | 0.354/0.316 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | PA unknown | 90 | 0.611 | -0.000 | -0.028 | n/a | 0.240/0.240 | negative_or_flat_clv, weak_clv_beat |
| under | same_book | 3478 | 0.592 | -0.030 | -0.176 | 0.659 | 0.241/0.240 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
