# MLB TB Prop Repair Report

Generated: 2026-06-19T04:55:04.154973+00:00
Rows: 30634 | Dates: 2026-05-31 to 2026-06-17

## Top Repair Targets

| Bucket | Rows | ROI | CLV | Brier M/Mkt | TB Bias | PA MAE | Issues |
|---|---:|---:|---:|---:|---:|---:|---|
| over | common | TB 1.5 | plus_150_249 | draftkings | 118 | -0.293 | 0.864 | 0.210/0.209 | 0.121 | 0.768 | market_beats_model_brier |
| over | common | TB 2.5 | plus_500_plus | fanduel | 390 | -0.226 | 0.283 | 0.108/0.105 | 0.077 | 0.711 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_250_499 | fanduel | 3525 | -0.129 | 0.087 | 0.163/0.157 | -0.063 | 0.705 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_150_249 | fanduel | 2008 | -0.187 | 0.226 | 0.207/0.199 | -0.038 | 0.755 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_250_499 | fanduel | 75 | 0.084 | 0.555 | 0.209/0.206 | 0.075 | 0.653 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | fanduel | 2884 | -0.156 | 0.106 | 0.237/0.231 | -0.134 | 0.650 | market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 4.5+ | plus_500_plus | fanduel | 5674 | -0.255 | 0.050 | 0.067/0.064 | -0.120 | 0.695 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_500_plus | fanduel | 2403 | -0.187 | 0.047 | 0.099/0.097 | -0.008 | 0.724 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_150_249 | fanduel | 1874 | -0.141 | 0.087 | 0.203/0.191 | -0.267 | 0.668 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | heavy_lay | draftkings | 804 | -0.031 | -0.602 | 0.231/0.231 | -0.053 | 0.673 | market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | fanduel | 179 | -0.053 | 0.461 | 0.240/0.219 | -0.673 | 0.699 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | draftkings | 2580 | -0.110 | 0.365 | 0.241/0.239 | -0.157 | 0.637 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | draftkings | 71 | -0.140 | -0.037 | 0.247/0.255 | -0.310 | 0.837 | tb_projection_low, pa_projection_error, negative_or_flat_clv |
| over | common | TB 1.5 | fair_lay | draftkings | 456 | -0.089 | 0.029 | 0.252/0.249 | -0.346 | 0.710 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | fanduel | 187 | -0.194 | 0.160 | 0.253/0.240 | -0.263 | 0.735 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | draftkings | 27 | 0.001 | 0.768 | 0.268/0.249 | -0.484 | 0.507 | sample_small, tb_projection_low, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_250_499 | fanduel | 304 | -0.136 | 0.338 | 0.143/0.134 | -0.463 | 0.668 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_250_499 | fanduel | 3172 | -0.138 | 0.065 | 0.153/0.148 | -0.184 | 0.671 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_150_249 | fanduel | 383 | -0.017 | 0.338 | 0.214/0.207 | -0.502 | 0.698 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | lay_130_149 | draftkings | 636 | -0.055 | -0.270 | 0.250/0.247 | -0.364 | 0.666 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | fanduel | 722 | -0.112 | -0.125 | 0.252/0.246 | -0.389 | 0.691 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_100_149 | fanduel | 30 | -0.082 | 1.051 | 0.255/0.232 | -0.688 | 0.457 | sample_small, alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier |
| under | common | TB 1.5 | fair_lay | draftkings | 427 | -0.059 | -0.123 | 0.256/0.248 | -0.431 | 0.733 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | plus_100_149 | draftkings | 61 | 0.112 | -0.051 | 0.263/0.252 | 0.026 | 0.943 | pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | heavy_lay | fanduel | 30 | -0.109 | -0.075 | 0.226/0.204 | -0.243 | 0.603 | sample_small, tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 2.5 | lay_130_149 | draftkings | 24 | -0.073 | -0.610 | 0.263/0.249 | -0.322 | 0.492 | sample_small, tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |

## Exact Buckets

| Bucket | Rows | Dates | Win | ROI | CLV Beat | Avg CLV | Pred/Actual TB | PA MAE | Issues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| over | common | TB 1.5 | plus_150_249 | draftkings | 118 | 15 | 0.280 | -0.293 | 0.552 | 0.864 | 1.316/1.195 | 0.768 | market_beats_model_brier |
| over | common | TB 2.5 | plus_500_plus | fanduel | 390 | 14 | 0.121 | -0.226 | 0.378 | 0.283 | 1.064/0.987 | 0.711 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_250_499 | fanduel | 3525 | 15 | 0.205 | -0.129 | 0.399 | 0.087 | 1.324/1.386 | 0.705 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_150_249 | fanduel | 2008 | 18 | 0.295 | -0.187 | 0.418 | 0.226 | 1.188/1.226 | 0.755 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_250_499 | fanduel | 75 | 14 | 0.293 | 0.084 | 0.484 | 0.555 | 1.062/0.987 | 0.653 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_150_180 | fanduel | 98 | 13 | 0.531 | -0.141 | 0.448 | 0.075 | 1.896/2.327 | 0.690 | tb_projection_low, weak_clv_beat |
| under | common | TB 1.5 | lay_150_180 | draftkings | 1422 | 18 | 0.611 | -0.018 | 0.303 | -0.308 | 1.514/1.613 | 0.617 | negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | fanduel | 2884 | 17 | 0.384 | -0.156 | 0.383 | 0.106 | 1.468/1.601 | 0.650 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | fair_lay | draftkings | 10 | 4 | 0.400 | -0.227 | 0.700 | 0.174 | 2.319/2.400 | 0.440 | sample_small, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_500_plus | fanduel | 5674 | 15 | 0.072 | -0.255 | 0.327 | 0.050 | 1.385/1.505 | 0.695 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_500_plus | fanduel | 2403 | 15 | 0.113 | -0.187 | 0.305 | 0.047 | 1.219/1.227 | 0.724 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_150_249 | fanduel | 1874 | 15 | 0.283 | -0.141 | 0.392 | 0.087 | 1.594/1.861 | 0.668 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | heavy_lay | draftkings | 804 | 18 | 0.637 | -0.031 | 0.238 | -0.602 | 1.422/1.475 | 0.673 | market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | fanduel | 179 | 15 | 0.413 | -0.053 | 0.462 | 0.461 | 1.897/2.570 | 0.699 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | draftkings | 2580 | 18 | 0.398 | -0.110 | 0.461 | 0.365 | 1.513/1.669 | 0.637 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | draftkings | 71 | 16 | 0.493 | -0.140 | 0.567 | -0.037 | 1.971/2.282 | 0.837 | tb_projection_low, pa_projection_error, negative_or_flat_clv |
| over | common | TB 1.5 | fair_lay | draftkings | 456 | 16 | 0.480 | -0.089 | 0.404 | 0.029 | 1.755/2.101 | 0.710 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | fanduel | 187 | 15 | 0.465 | -0.194 | 0.469 | 0.160 | 1.838/2.102 | 0.735 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | draftkings | 27 | 7 | 0.481 | 0.001 | 0.750 | 0.768 | 2.183/2.667 | 0.507 | sample_small, tb_projection_low, market_beats_model_brier |
| under | common | TB 2.5 | plus_100_149 | draftkings | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.920 | 2.367/2.000 | 0.200 | sample_small, tb_projection_high, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_150_249 | fanduel | 11 | 3 | 0.000 | -1.000 | 0.727 | 1.083 | 2.399/2.091 | 0.291 | sample_small, alt_tail_requires_separate_proof, tb_projection_high, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_250_499 | fanduel | 304 | 15 | 0.178 | -0.136 | 0.469 | 0.338 | 1.866/2.329 | 0.668 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_250_499 | fanduel | 3172 | 15 | 0.187 | -0.138 | 0.411 | 0.065 | 1.504/1.689 | 0.671 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | lay_130_149 | draftkings | 1 | 1 | 0.000 | -1.000 | 0.000 | -1.920 | 2.367/2.000 | 0.200 | sample_small, tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_150_249 | fanduel | 383 | 15 | 0.319 | -0.017 | 0.416 | 0.338 | 1.795/2.298 | 0.698 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | lay_130_149 | fanduel | 4 | 2 | 0.000 | -1.000 | 0.000 | -0.910 | 2.503/1.750 | 0.250 | sample_small, tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | lay_130_149 | draftkings | 636 | 17 | 0.552 | -0.055 | 0.343 | -0.270 | 1.584/1.948 | 0.666 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | fanduel | 722 | 15 | 0.471 | -0.112 | 0.311 | -0.125 | 1.636/2.025 | 0.691 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_100_149 | fanduel | 30 | 5 | 0.400 | -0.082 | 0.533 | 1.051 | 2.112/2.800 | 0.457 | sample_small, alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier |
| under | common | TB 1.5 | fair_lay | draftkings | 427 | 17 | 0.508 | -0.059 | 0.383 | -0.123 | 1.789/2.220 | 0.733 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |

## Diagnostic Groups

| Group | Rows | Win | ROI | TB Bias | PA MAE | Model/Mkt Brier | Issues |
|---|---:|---:|---:|---:|---:|---:|---|
| over | 0 TB | 10198 | 0.000 | -1.000 | 1.388 | 0.733 | 0.076/0.072 | tb_projection_high, market_beats_model_brier, weak_clv_beat |
| over | 0-2 PA | 1692 | 0.033 | -0.893 | 0.977 | 1.884 | 0.077/0.078 | tb_projection_high, pa_projection_error, negative_or_flat_clv, weak_clv_beat, low_valid_clv_coverage |
| over | 1 TB | 6848 | 0.000 | -1.000 | 0.422 | 0.642 | 0.082/0.080 | tb_projection_high, market_beats_model_brier, weak_clv_beat |
| over | 2-3 TB | 5567 | 0.415 | 0.056 | -0.863 | 0.664 | 0.187/0.177 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | 3 PA | 3420 | 0.089 | -0.649 | 0.567 | 0.670 | 0.096/0.092 | tb_projection_high, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | 4 PA | 13981 | 0.220 | -0.191 | -0.054 | 0.404 | 0.160/0.155 | market_beats_model_brier, weak_clv_beat |
| over | 4+ TB non-HR/unknown | 733 | 0.853 | 2.546 | -2.880 | 0.790 | 0.431/0.415 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | 5+ PA | 8153 | 0.364 | 0.230 | -0.823 | 0.936 | 0.215/0.208 | tb_projection_low, pa_projection_error, market_beats_model_brier, weak_clv_beat |
| over | HR driven | 3900 | 0.890 | 2.651 | -3.591 | 0.669 | 0.450/0.446 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | cross_book | 4883 | 0.402 | 0.189 | -0.641 | 0.660 | 0.249/0.242 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | same_book | 3275 | 0.409 | -0.112 | -0.180 | 0.656 | 0.241/0.240 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | synthetic | 19088 | 0.162 | -0.267 | -0.008 | 0.701 | 0.128/0.123 | market_beats_model_brier, weak_clv_beat |
| under | 0 TB | 1115 | 1.000 | 0.635 | 1.530 | 0.682 | 0.157/0.183 | tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | 0-2 PA | 121 | 0.917 | 0.503 | 1.199 | 2.340 | 0.172/0.193 | tb_projection_high, pa_projection_error, weak_clv_beat, low_valid_clv_coverage |
| under | 1 TB | 880 | 1.000 | 0.642 | 0.538 | 0.639 | 0.162/0.185 | tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | 2-3 TB | 740 | 0.011 | -0.981 | -0.733 | 0.643 | 0.362/0.323 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | 3 PA | 211 | 0.810 | 0.305 | 0.827 | 0.902 | 0.191/0.204 | tb_projection_high, pa_projection_error, negative_or_flat_clv, weak_clv_beat |
| under | 4 PA | 1706 | 0.638 | 0.041 | 0.044 | 0.350 | 0.232/0.235 | negative_or_flat_clv, weak_clv_beat |
| under | 4+ TB non-HR/unknown | 105 | 0.000 | -1.000 | -2.752 | 0.810 | 0.362/0.318 | tb_projection_low, pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | 5+ PA | 1350 | 0.468 | -0.222 | -0.745 | 0.860 | 0.267/0.256 | tb_projection_low, pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | HR driven | 548 | 0.000 | -1.000 | -3.574 | 0.635 | 0.354/0.316 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | same_book | 3388 | 0.591 | -0.031 | -0.180 | 0.659 | 0.241/0.240 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
