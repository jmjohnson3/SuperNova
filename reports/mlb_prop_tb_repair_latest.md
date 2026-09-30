# MLB TB Prop Repair Report

Generated: 2026-08-29T12:52:20.110140+00:00
Rows: 123390 | Dates: 2026-06-01 to 2026-07-30

## Top Repair Targets

| Bucket | Rows | ROI | CLV | Brier M/Mkt | TB Bias | 2B/PA | XBH/PA | PA MAE | Issues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| over | common | TB 1.5 | plus_150_249 | draftkings | 521 | -0.125 | 0.909 | 0.227/0.226 | 0.136 | 0.037 | 0.058 | 0.670 | market_beats_model_brier, double_xbh_structure_repair_needed |
| over | common | TB 2.5 | plus_500_plus | fanduel | 1690 | -0.187 | 0.155 | 0.111/0.093 | 0.119 | 0.035 | 0.057 | 0.793 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 2.5 | plus_250_499 | fanduel | 14246 | -0.199 | 0.088 | 0.155/0.125 | 0.113 | 0.040 | 0.071 | 0.700 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 2.5 | plus_150_249 | fanduel | 7728 | -0.171 | 0.001 | 0.204/0.160 | 0.082 | 0.046 | 0.093 | 0.632 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 1.5 | plus_150_249 | fanduel | 8183 | -0.210 | 0.168 | 0.208/0.177 | 0.085 | 0.037 | 0.066 | 0.763 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | draftkings | 10231 | -0.089 | 0.349 | 0.241/0.241 | 0.067 | 0.044 | 0.085 | 0.619 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | fanduel | 741 | -0.160 | 0.242 | 0.244/0.174 | -0.045 | 0.047 | 0.120 | 0.661 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | draftkings | 1689 | -0.118 | 0.244 | 0.249/0.248 | 0.076 | 0.043 | 0.098 | 0.648 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | alt_tail | TB 4.5+ | plus_500_plus | fanduel | 23156 | -0.327 | 0.047 | 0.062/0.055 | 0.103 | 0.042 | 0.077 | 0.686 | alt_tail_requires_separate_proof, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_500_plus | fanduel | 9788 | -0.245 | 0.066 | 0.092/0.075 | 0.093 | 0.040 | 0.066 | 0.734 | alt_tail_requires_separate_proof, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | alt_tail | TB 4.5+ | plus_250_499 | fanduel | 1221 | -0.277 | 0.276 | 0.126/0.102 | 0.062 | 0.043 | 0.109 | 0.639 | alt_tail_requires_separate_proof, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_250_499 | fanduel | 12851 | -0.211 | 0.072 | 0.145/0.113 | 0.108 | 0.043 | 0.084 | 0.652 | alt_tail_requires_separate_proof, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 1.5 | plus_250_499 | fanduel | 366 | -0.059 | 0.356 | 0.194/0.171 | 0.109 | 0.036 | 0.066 | 0.842 | pa_projection_error, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_150_249 | fanduel | 1724 | -0.166 | 0.237 | 0.201/0.152 | 0.071 | 0.040 | 0.106 | 0.634 | alt_tail_requires_separate_proof, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| under | common | TB 1.5 | heavy_lay | draftkings | 3216 | -0.037 | -0.634 | 0.234/0.232 | 0.105 | 0.041 | 0.072 | 0.637 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | fanduel | 11772 | -0.155 | -0.016 | 0.239/0.226 | 0.107 | 0.043 | 0.081 | 0.634 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | heavy_lay | fanduel | 54 | -0.142 | 0.076 | 0.242/0.222 | -0.319 | 0.071 | 0.161 | 0.676 | tb_projection_low, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| under | common | TB 1.5 | lay_150_180 | draftkings | 5409 | -0.059 | -0.286 | 0.243/0.243 | 0.046 | 0.046 | 0.088 | 0.601 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | lay_130_149 | draftkings | 2429 | -0.015 | -0.226 | 0.247/0.246 | 0.103 | 0.040 | 0.087 | 0.647 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | draftkings | 45 | -0.025 | 0.595 | 0.250/0.246 | -0.205 | 0.077 | 0.165 | 0.739 | sample_small, tb_projection_low, market_beats_model_brier, double_xbh_structure_repair_needed |
| under | common | TB 1.5 | fair_lay | draftkings | 1507 | -0.044 | -0.263 | 0.252/0.249 | 0.006 | 0.045 | 0.105 | 0.655 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | fanduel | 2960 | -0.168 | -0.189 | 0.254/0.243 | 0.118 | 0.045 | 0.093 | 0.643 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | fanduel | 739 | -0.137 | -0.066 | 0.258/0.242 | 0.016 | 0.049 | 0.109 | 0.679 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_100_149 | fanduel | 59 | -0.133 | 0.247 | 0.260/0.185 | -0.088 | 0.040 | 0.135 | 0.569 | alt_tail_requires_separate_proof, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 1.5 | lay_150_180 | fanduel | 369 | -0.199 | -0.045 | 0.271/0.239 | 0.128 | 0.046 | 0.113 | 0.693 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | plus_100_149 | draftkings | 251 | -0.013 | 0.057 | 0.260/0.249 | 0.186 | 0.063 | 0.123 | 0.827 | tb_projection_high, pa_projection_error, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | draftkings | 269 | -0.156 | -0.014 | 0.268/0.253 | 0.258 | 0.055 | 0.112 | 0.769 | tb_projection_high, market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | lay_150_180 | draftkings | 50 | 0.216 | -0.440 | 0.233/0.222 | -0.502 | 0.081 | 0.171 | 0.870 | tb_projection_low, pa_projection_error, market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 2.5 | lay_130_149 | draftkings | 31 | -0.170 | -0.447 | 0.266/0.255 | -0.635 | 0.062 | 0.166 | 0.569 | sample_small, tb_projection_low, market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 2.5 | fair_lay | fanduel | 29 | -0.017 | -0.394 | 0.282/0.238 | -0.298 | 0.053 | 0.158 | 0.499 | sample_small, tb_projection_low, market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |

## Exact Buckets

| Bucket | Rows | Dates | Win | ROI | CLV Beat | Avg CLV | Pred/Actual TB | 2B/PA | PA MAE | Issues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| over | common | TB 1.5 | plus_150_249 | draftkings | 521 | 54 | 0.345 | -0.125 | 0.592 | 0.909 | 1.420/1.284 | 0.037 | 0.670 | market_beats_model_brier, double_xbh_structure_repair_needed |
| over | common | TB 2.5 | plus_500_plus | fanduel | 1690 | 53 | 0.125 | -0.187 | 0.335 | 0.155 | 1.134/1.015 | 0.035 | 0.793 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 2.5 | plus_250_499 | fanduel | 14246 | 55 | 0.188 | -0.199 | 0.410 | 0.088 | 1.434/1.321 | 0.040 | 0.700 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | alt_tail | TB 4.5+ | plus_150_249 | fanduel | 13 | 4 | 0.154 | -0.477 | 0.615 | 0.665 | 2.444/2.692 | 0.077 | 0.402 | sample_small, alt_tail_requires_separate_proof, tb_projection_low |
| over | common | TB 2.5 | plus_150_249 | fanduel | 7728 | 55 | 0.272 | -0.171 | 0.376 | 0.001 | 1.865/1.782 | 0.046 | 0.632 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 1.5 | plus_150_249 | fanduel | 8183 | 57 | 0.285 | -0.210 | 0.412 | 0.168 | 1.251/1.166 | 0.037 | 0.763 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | draftkings | 10231 | 57 | 0.407 | -0.089 | 0.457 | 0.349 | 1.729/1.662 | 0.044 | 0.619 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | fanduel | 741 | 55 | 0.366 | -0.160 | 0.447 | 0.242 | 2.252/2.297 | 0.047 | 0.661 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | draftkings | 1689 | 56 | 0.464 | -0.118 | 0.454 | 0.244 | 2.067/1.991 | 0.043 | 0.648 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 2.5 | fair_lay | draftkings | 10 | 4 | 0.400 | -0.227 | 0.700 | 0.174 | 2.319/2.400 | 0.080 | 0.440 | sample_small, market_beats_model_brier, double_xbh_structure_repair_needed |
| over | alt_tail | TB 4.5+ | plus_500_plus | fanduel | 23156 | 55 | 0.065 | -0.327 | 0.342 | 0.047 | 1.541/1.439 | 0.042 | 0.686 | alt_tail_requires_separate_proof, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_500_plus | fanduel | 9788 | 55 | 0.101 | -0.245 | 0.319 | 0.066 | 1.306/1.212 | 0.040 | 0.734 | alt_tail_requires_separate_proof, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | alt_tail | TB 4.5+ | plus_250_499 | fanduel | 1221 | 55 | 0.145 | -0.277 | 0.474 | 0.276 | 2.193/2.131 | 0.043 | 0.639 | alt_tail_requires_separate_proof, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_250_499 | fanduel | 12851 | 55 | 0.172 | -0.211 | 0.420 | 0.072 | 1.704/1.596 | 0.043 | 0.652 | alt_tail_requires_separate_proof, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 1.5 | plus_250_499 | fanduel | 366 | 51 | 0.257 | -0.059 | 0.431 | 0.356 | 1.089/0.981 | 0.036 | 0.842 | pa_projection_error, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_150_249 | fanduel | 1724 | 55 | 0.272 | -0.166 | 0.434 | 0.237 | 2.124/2.053 | 0.040 | 0.634 | alt_tail_requires_separate_proof, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 2.5 | lay_130_149 | draftkings | 1 | 1 | 0.000 | -1.000 | 0.000 | -1.920 | 2.367/2.000 | 0.200 | 0.200 | sample_small, tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 2.5 | lay_130_149 | fanduel | 4 | 2 | 0.000 | -1.000 | 0.000 | -0.910 | 2.503/1.750 | 0.150 | 0.250 | sample_small, tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | heavy_lay | draftkings | 3216 | 57 | 0.633 | -0.037 | 0.259 | -0.634 | 1.548/1.444 | 0.041 | 0.637 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | fanduel | 11772 | 57 | 0.383 | -0.155 | 0.367 | -0.016 | 1.661/1.554 | 0.043 | 0.634 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | heavy_lay | fanduel | 54 | 14 | 0.574 | -0.142 | 0.491 | 0.076 | 2.385/2.704 | 0.071 | 0.676 | tb_projection_low, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| under | common | TB 1.5 | lay_150_180 | draftkings | 5409 | 57 | 0.586 | -0.059 | 0.334 | -0.286 | 1.745/1.699 | 0.046 | 0.601 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | lay_130_149 | draftkings | 2429 | 56 | 0.574 | -0.015 | 0.360 | -0.226 | 1.884/1.782 | 0.040 | 0.647 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | draftkings | 45 | 14 | 0.467 | -0.025 | 0.711 | 0.595 | 2.373/2.578 | 0.077 | 0.739 | sample_small, tb_projection_low, market_beats_model_brier, double_xbh_structure_repair_needed |
| under | common | TB 1.5 | fair_lay | draftkings | 1507 | 56 | 0.517 | -0.044 | 0.338 | -0.263 | 2.092/2.086 | 0.045 | 0.655 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | fanduel | 2960 | 55 | 0.439 | -0.168 | 0.313 | -0.189 | 1.949/1.831 | 0.045 | 0.643 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | fanduel | 739 | 54 | 0.498 | -0.137 | 0.383 | -0.066 | 2.143/2.127 | 0.049 | 0.679 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_100_149 | fanduel | 59 | 19 | 0.373 | -0.133 | 0.339 | 0.247 | 2.353/2.441 | 0.040 | 0.569 | alt_tail_requires_separate_proof, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | common | TB 1.5 | lay_150_180 | fanduel | 369 | 47 | 0.493 | -0.199 | 0.404 | -0.045 | 2.309/2.182 | 0.046 | 0.693 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 2.5 | plus_100_149 | draftkings | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.920 | 2.367/2.000 | 0.200 | 0.200 | sample_small, tb_projection_high, market_beats_model_brier, double_xbh_structure_repair_needed |

## Diagnostic Groups

| Group | Rows | Win | ROI | TB Bias | 2B/PA | PA MAE | Model/Mkt Brier | Issues |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| over | 0 TB | 42764 | 0.000 | -1.000 | 1.560 | 0.000 | 0.721 | 0.086/0.053 | tb_projection_high, market_beats_model_brier, weak_clv_beat |
| over | 0-2 PA | 7565 | 0.045 | -0.838 | 1.030 | 0.030 | 1.980 | 0.084/0.071 | tb_projection_high, pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat, low_valid_clv_coverage |
| over | 1 TB | 27204 | 0.000 | -1.000 | 0.580 | 0.000 | 0.622 | 0.089/0.062 | tb_projection_high, market_beats_model_brier, weak_clv_beat |
| over | 2-3 TB | 23418 | 0.412 | 0.051 | -0.682 | 0.132 | 0.644 | 0.176/0.174 | tb_projection_low, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | 3 PA | 13469 | 0.088 | -0.657 | 0.633 | 0.031 | 0.665 | 0.099/0.075 | tb_projection_high, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | 4 PA | 58358 | 0.207 | -0.261 | 0.184 | 0.040 | 0.332 | 0.156/0.135 | tb_projection_high, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | 4+ TB non-HR/unknown | 3191 | 0.848 | 2.461 | -2.683 | 0.232 | 0.888 | 0.412/0.398 | tb_projection_low, pa_projection_error, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | 5+ PA | 31128 | 0.359 | 0.226 | -0.528 | 0.049 | 1.014 | 0.210/0.194 | tb_projection_low, pa_projection_error, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | HR driven | 13943 | 0.892 | 2.693 | -3.396 | 0.031 | 0.662 | 0.435/0.439 | tb_projection_low, weak_clv_beat |
| over | cross_book | 20803 | 0.404 | 0.216 | -0.397 | 0.063 | 0.640 | 0.252/0.242 | tb_projection_low, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | one_sided | 76899 | 0.144 | -0.347 | 0.233 | 0.036 | 0.696 | 0.120/0.095 | tb_projection_high, market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| over | same_book | 12818 | 0.415 | -0.094 | 0.072 | 0.044 | 0.630 | 0.242/0.242 | market_beats_model_brier, double_xbh_structure_repair_needed, weak_clv_beat |
| under | 0 TB | 4376 | 1.000 | 0.636 | 1.766 | 0.000 | 0.632 | 0.181/0.183 | tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | 0-2 PA | 444 | 0.905 | 0.472 | 1.277 | 0.020 | 2.424 | 0.188/0.192 | tb_projection_high, pa_projection_error, negative_or_flat_clv, weak_clv_beat, low_valid_clv_coverage |
| under | 1 TB | 3145 | 1.000 | 0.637 | 0.757 | 0.000 | 0.593 | 0.182/0.183 | tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | 2-3 TB | 2976 | 0.003 | -0.995 | -0.543 | 0.125 | 0.619 | 0.332/0.328 | tb_projection_low, market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| under | 3 PA | 741 | 0.798 | 0.277 | 0.838 | 0.036 | 0.893 | 0.206/0.207 | tb_projection_high, pa_projection_error, negative_or_flat_clv, weak_clv_beat |
| under | 4 PA | 6900 | 0.635 | 0.037 | 0.303 | 0.041 | 0.270 | 0.236/0.235 | tb_projection_high, market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| under | 4+ TB non-HR/unknown | 452 | 0.000 | -1.000 | -2.573 | 0.223 | 0.879 | 0.330/0.324 | tb_projection_low, pa_projection_error, market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| under | 5+ PA | 4785 | 0.450 | -0.254 | -0.501 | 0.050 | 0.942 | 0.263/0.260 | tb_projection_low, pa_projection_error, market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
| under | HR driven | 1921 | 0.000 | -1.000 | -3.354 | 0.033 | 0.643 | 0.324/0.318 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | same_book | 12870 | 0.585 | -0.042 | 0.069 | 0.044 | 0.630 | 0.243/0.242 | market_beats_model_brier, double_xbh_structure_repair_needed, negative_or_flat_clv, weak_clv_beat |
