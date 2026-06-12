# MLB TB Prop Repair Report

Generated: 2026-06-12T15:52:10.587501+00:00
Rows: 15270 | Dates: 2026-05-31 to 2026-06-11

## Top Repair Targets

| Bucket | Rows | ROI | CLV | Brier M/Mkt | TB Bias | PA MAE | Issues |
|---|---:|---:|---:|---:|---:|---:|---|
| over | common | TB 1.5 | plus_150_249 | draftkings | 57 | -0.423 | 0.612 | 0.189/0.196 | 0.310 | 0.744 | tb_projection_high |
| over | common | TB 2.5 | plus_500_plus | fanduel | 198 | -0.169 | 0.233 | 0.112/0.109 | -0.057 | 0.789 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_250_499 | fanduel | 1800 | -0.107 | 0.037 | 0.165/0.161 | -0.128 | 0.726 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_150_249 | fanduel | 870 | -0.222 | 0.100 | 0.194/0.181 | -0.130 | 0.682 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_150_249 | fanduel | 1036 | -0.140 | 0.186 | 0.212/0.206 | -0.117 | 0.780 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | fanduel | 1437 | -0.151 | 0.060 | 0.239/0.232 | -0.127 | 0.665 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | draftkings | 1285 | -0.096 | 0.296 | 0.242/0.240 | -0.133 | 0.655 | market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 4.5+ | plus_500_plus | fanduel | 2827 | -0.247 | 0.034 | 0.067/0.064 | -0.125 | 0.720 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_500_plus | fanduel | 1279 | -0.128 | 0.033 | 0.105/0.104 | -0.073 | 0.743 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | fanduel | 87 | 0.070 | 0.404 | 0.256/0.233 | -0.630 | 0.768 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | fair_lay | draftkings | 233 | -0.067 | 0.079 | 0.262/0.248 | -0.391 | 0.774 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 4.5+ | plus_250_499 | fanduel | 134 | -0.108 | 0.406 | 0.149/0.139 | -0.424 | 0.696 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_250_499 | fanduel | 1501 | -0.144 | 0.050 | 0.151/0.147 | -0.163 | 0.696 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_150_249 | fanduel | 171 | 0.070 | 0.420 | 0.227/0.219 | -0.353 | 0.752 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | heavy_lay | draftkings | 418 | -0.068 | -0.492 | 0.237/0.236 | -0.198 | 0.679 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | plus_250_499 | fanduel | 40 | 0.372 | 0.334 | 0.243/0.248 | -0.364 | 0.818 | sample_small, tb_projection_low, pa_projection_error, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | fanduel | 95 | -0.159 | 0.377 | 0.250/0.242 | -0.217 | 0.820 | tb_projection_low, pa_projection_error, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | fanduel | 309 | -0.136 | -0.152 | 0.253/0.243 | -0.211 | 0.726 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | draftkings | 232 | -0.095 | -0.230 | 0.254/0.250 | -0.212 | 0.746 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | lay_130_149 | draftkings | 305 | -0.103 | -0.098 | 0.256/0.250 | -0.279 | 0.719 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | plus_100_149 | draftkings | 38 | -0.016 | 0.260 | 0.266/0.248 | 0.263 | 0.942 | sample_small, tb_projection_high, pa_projection_error, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | draftkings | 44 | 0.033 | -0.360 | 0.262/0.250 | -0.336 | 0.873 | sample_small, tb_projection_low, pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |

## Exact Buckets

| Bucket | Rows | Dates | Win | ROI | CLV Beat | Avg CLV | Pred/Actual TB | PA MAE | Issues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| over | common | TB 1.5 | plus_150_249 | draftkings | 57 | 9 | 0.228 | -0.423 | 0.562 | 0.612 | 1.328/1.018 | 0.744 | tb_projection_high |
| over | common | TB 2.5 | plus_500_plus | fanduel | 198 | 8 | 0.126 | -0.169 | 0.342 | 0.233 | 0.989/1.045 | 0.789 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_250_499 | fanduel | 1800 | 9 | 0.211 | -0.107 | 0.389 | 0.037 | 1.300/1.428 | 0.726 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_150_249 | fanduel | 870 | 9 | 0.256 | -0.222 | 0.379 | 0.100 | 1.627/1.756 | 0.682 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_150_249 | fanduel | 1036 | 12 | 0.312 | -0.140 | 0.411 | 0.186 | 1.150/1.267 | 0.780 | market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | lay_150_180 | draftkings | 746 | 12 | 0.630 | 0.012 | 0.306 | -0.265 | 1.525/1.520 | 0.627 | negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | fanduel | 1437 | 11 | 0.386 | -0.151 | 0.373 | 0.060 | 1.469/1.596 | 0.665 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | draftkings | 1285 | 12 | 0.402 | -0.096 | 0.427 | 0.296 | 1.523/1.656 | 0.655 | market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 4.5+ | plus_500_plus | fanduel | 2827 | 9 | 0.072 | -0.247 | 0.317 | 0.034 | 1.372/1.497 | 0.720 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_500_plus | fanduel | 1279 | 9 | 0.121 | -0.128 | 0.295 | 0.033 | 1.192/1.264 | 0.743 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | draftkings | 16 | 4 | 0.500 | 0.034 | 0.692 | 0.242 | 2.343/2.500 | 0.550 | sample_small, tb_projection_low, market_beats_model_brier |
| over | common | TB 2.5 | plus_100_149 | fanduel | 87 | 9 | 0.471 | 0.070 | 0.464 | 0.404 | 2.014/2.644 | 0.768 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | fair_lay | draftkings | 233 | 11 | 0.502 | -0.067 | 0.398 | 0.079 | 1.823/2.215 | 0.774 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 4.5+ | plus_150_249 | fanduel | 2 | 1 | 0.000 | -1.000 | 1.000 | 1.930 | 2.837/1.000 | 0.400 | sample_small, alt_tail_requires_separate_proof, tb_projection_high, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_250_499 | fanduel | 134 | 9 | 0.187 | -0.108 | 0.508 | 0.406 | 1.957/2.381 | 0.696 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_250_499 | fanduel | 1501 | 9 | 0.185 | -0.144 | 0.413 | 0.050 | 1.515/1.678 | 0.696 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_150_249 | fanduel | 171 | 9 | 0.351 | 0.070 | 0.422 | 0.420 | 1.881/2.234 | 0.752 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | heavy_lay | draftkings | 418 | 12 | 0.612 | -0.068 | 0.244 | -0.492 | 1.422/1.620 | 0.679 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | lay_150_180 | fanduel | 46 | 7 | 0.630 | 0.021 | 0.304 | -0.562 | 2.006/2.326 | 0.746 | sample_small, tb_projection_low, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | plus_250_499 | fanduel | 40 | 8 | 0.375 | 0.372 | 0.379 | 0.334 | 0.911/1.275 | 0.818 | sample_small, tb_projection_low, pa_projection_error, weak_clv_beat |
| over | common | TB 1.5 | heavy_lay | fanduel | 15 | 4 | 0.600 | -0.093 | 0.467 | 0.252 | 2.322/2.667 | 0.580 | sample_small, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | fair_lay | fanduel | 7 | 3 | 0.571 | 0.103 | 0.143 | -1.256 | 2.415/2.143 | 0.643 | sample_small, tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | fanduel | 95 | 9 | 0.484 | -0.159 | 0.494 | 0.377 | 1.920/2.137 | 0.820 | tb_projection_low, pa_projection_error, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | fanduel | 309 | 9 | 0.460 | -0.136 | 0.282 | -0.152 | 1.686/1.896 | 0.726 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | draftkings | 232 | 10 | 0.478 | -0.095 | 0.357 | -0.230 | 1.796/2.009 | 0.746 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | lay_130_149 | draftkings | 305 | 11 | 0.525 | -0.103 | 0.349 | -0.098 | 1.600/1.879 | 0.719 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 2.5 | lay_130_149 | draftkings | 14 | 4 | 0.500 | -0.146 | 0.308 | -0.332 | 2.331/2.357 | 0.543 | sample_small, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 2.5 | fair_lay | draftkings | 1 | 1 | 0.000 | -1.000 | 1.000 | 4.140 | 2.042/5.000 | 1.600 | sample_small, tb_projection_low, pa_projection_error, market_beats_model_brier |
| under | common | TB 2.5 | lay_150_180 | draftkings | 3 | 2 | 0.667 | 0.091 | 0.000 | -0.890 | 2.548/2.667 | 0.533 | sample_small, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat, low_valid_clv_coverage |
| under | common | TB 1.5 | plus_100_149 | draftkings | 38 | 9 | 0.474 | -0.016 | 0.467 | 0.260 | 2.079/1.816 | 0.942 | sample_small, tb_projection_high, pa_projection_error, market_beats_model_brier, weak_clv_beat |

## Diagnostic Groups

| Group | Rows | Win | ROI | TB Bias | PA MAE | Model/Mkt Brier | Issues |
|---|---:|---:|---:|---:|---:|---:|---|
| over | 0 TB | 5047 | 0.000 | -1.000 | 1.381 | 0.739 | 0.073/0.074 | tb_projection_high, weak_clv_beat |
| over | 0-2 PA | 905 | 0.038 | -0.860 | 0.922 | 1.880 | 0.078/0.084 | tb_projection_high, pa_projection_error, weak_clv_beat, low_valid_clv_coverage |
| over | 1 TB | 3327 | 0.000 | -1.000 | 0.404 | 0.689 | 0.076/0.078 | tb_projection_high, weak_clv_beat |
| over | 2-3 TB | 2821 | 0.409 | 0.039 | -0.833 | 0.696 | 0.186/0.174 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | 3 PA | 1545 | 0.089 | -0.607 | 0.510 | 0.673 | 0.097/0.095 | tb_projection_high, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | 4 PA | 6796 | 0.218 | -0.183 | -0.043 | 0.393 | 0.159/0.155 | market_beats_model_brier, weak_clv_beat |
| over | 4+ TB non-HR/unknown | 431 | 0.858 | 2.668 | -2.985 | 0.823 | 0.457/0.431 | tb_projection_low, pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | 5+ PA | 4266 | 0.362 | 0.211 | -0.750 | 0.991 | 0.214/0.204 | tb_projection_low, pa_projection_error, market_beats_model_brier, weak_clv_beat |
| over | HR driven | 1886 | 0.889 | 2.678 | -3.470 | 0.692 | 0.460/0.445 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | cross_book | 2359 | 0.408 | 0.205 | -0.594 | 0.685 | 0.252/0.243 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | same_book | 1645 | 0.416 | -0.100 | -0.138 | 0.678 | 0.243/0.240 | market_beats_model_brier, weak_clv_beat |
| over | synthetic | 9508 | 0.163 | -0.251 | -0.025 | 0.727 | 0.128/0.124 | market_beats_model_brier, weak_clv_beat |
| under | 0 TB | 592 | 1.000 | 0.641 | 1.549 | 0.676 | 0.138/0.185 | tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | 0-2 PA | 74 | 0.905 | 0.499 | 1.170 | 2.311 | 0.170/0.203 | tb_projection_high, pa_projection_error, low_valid_clv_coverage |
| under | 1 TB | 435 | 1.000 | 0.636 | 0.539 | 0.675 | 0.139/0.183 | tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | 2-3 TB | 390 | 0.008 | -0.987 | -0.694 | 0.666 | 0.394/0.322 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | 3 PA | 100 | 0.850 | 0.373 | 0.804 | 0.938 | 0.176/0.202 | tb_projection_high, pa_projection_error, negative_or_flat_clv, weak_clv_beat |
| under | 4 PA | 869 | 0.638 | 0.038 | 0.097 | 0.324 | 0.230/0.234 | negative_or_flat_clv, weak_clv_beat |
| under | 4+ TB non-HR/unknown | 59 | 0.000 | -1.000 | -2.882 | 0.873 | 0.398/0.319 | tb_projection_low, pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | 5+ PA | 715 | 0.453 | -0.248 | -0.699 | 0.911 | 0.277/0.257 | tb_projection_low, pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | HR driven | 282 | 0.000 | -1.000 | -3.403 | 0.682 | 0.385/0.314 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | same_book | 1758 | 0.586 | -0.040 | -0.142 | 0.681 | 0.243/0.240 | market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
