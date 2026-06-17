# MLB TB Prop Repair Report

Generated: 2026-06-17T17:55:33.927918+00:00
Rows: 25882 | Dates: 2026-05-31 to 2026-06-15

## Top Repair Targets

| Bucket | Rows | ROI | CLV | Brier M/Mkt | TB Bias | PA MAE | Issues |
|---|---:|---:|---:|---:|---:|---:|---|
| over | common | TB 2.5 | plus_500_plus | fanduel | 320 | -0.323 | 0.325 | 0.096/0.093 | 0.091 | 0.738 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_250_499 | fanduel | 2979 | -0.148 | 0.096 | 0.161/0.155 | -0.064 | 0.709 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_150_249 | fanduel | 1678 | -0.185 | 0.256 | 0.207/0.200 | -0.034 | 0.761 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_250_499 | fanduel | 63 | 0.163 | 0.668 | 0.219/0.218 | -0.071 | 0.687 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | fanduel | 2423 | -0.163 | 0.118 | 0.236/0.231 | -0.134 | 0.655 | market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 4.5+ | plus_500_plus | fanduel | 4755 | -0.273 | 0.051 | 0.066/0.064 | -0.125 | 0.700 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_500_plus | fanduel | 2042 | -0.197 | 0.052 | 0.098/0.097 | -0.009 | 0.730 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_150_249 | fanduel | 1555 | -0.152 | 0.106 | 0.201/0.190 | -0.279 | 0.668 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | heavy_lay | draftkings | 703 | -0.039 | -0.551 | 0.233/0.233 | -0.086 | 0.672 | market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | draftkings | 2202 | -0.111 | 0.365 | 0.240/0.239 | -0.156 | 0.645 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | fanduel | 157 | -0.025 | 0.450 | 0.245/0.223 | -0.774 | 0.716 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | draftkings | 62 | -0.100 | -0.083 | 0.247/0.254 | -0.405 | 0.887 | tb_projection_low, pa_projection_error, negative_or_flat_clv |
| over | common | TB 1.5 | fair_lay | draftkings | 399 | -0.058 | 0.059 | 0.253/0.250 | -0.404 | 0.701 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | fanduel | 166 | -0.175 | 0.176 | 0.254/0.240 | -0.286 | 0.739 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | draftkings | 27 | 0.001 | 0.768 | 0.268/0.249 | -0.484 | 0.507 | sample_small, tb_projection_low, market_beats_model_brier |
| over | alt_tail | TB 3.5 | plus_250_499 | fanduel | 2634 | -0.134 | 0.061 | 0.153/0.149 | -0.195 | 0.675 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 4.5+ | plus_250_499 | fanduel | 266 | -0.013 | 0.352 | 0.158/0.149 | -0.502 | 0.660 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_150_249 | fanduel | 325 | -0.047 | 0.327 | 0.210/0.202 | -0.522 | 0.694 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | lay_130_149 | draftkings | 539 | -0.060 | -0.280 | 0.251/0.247 | -0.380 | 0.666 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | fanduel | 598 | -0.098 | -0.054 | 0.252/0.247 | -0.441 | 0.685 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_100_149 | fanduel | 30 | -0.082 | 1.051 | 0.255/0.232 | -0.688 | 0.457 | sample_small, alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier |
| under | common | TB 1.5 | fair_lay | draftkings | 372 | -0.082 | -0.165 | 0.258/0.249 | -0.509 | 0.728 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | plus_100_149 | draftkings | 57 | 0.048 | -0.044 | 0.262/0.250 | -0.016 | 0.995 | pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | heavy_lay | fanduel | 30 | -0.109 | -0.075 | 0.226/0.204 | -0.243 | 0.603 | sample_small, tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 2.5 | lay_130_149 | draftkings | 24 | -0.073 | -0.610 | 0.263/0.249 | -0.322 | 0.492 | sample_small, tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |

## Exact Buckets

| Bucket | Rows | Dates | Win | ROI | CLV Beat | Avg CLV | Pred/Actual TB | PA MAE | Issues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| over | common | TB 1.5 | plus_150_249 | draftkings | 100 | 13 | 0.310 | -0.216 | 0.545 | 0.821 | 1.303/1.290 | 0.706 |  |
| over | common | TB 2.5 | plus_500_plus | fanduel | 320 | 12 | 0.103 | -0.323 | 0.394 | 0.325 | 1.038/0.947 | 0.738 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_250_499 | fanduel | 2979 | 13 | 0.201 | -0.148 | 0.403 | 0.096 | 1.317/1.381 | 0.709 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_150_249 | fanduel | 1678 | 16 | 0.296 | -0.185 | 0.421 | 0.256 | 1.176/1.210 | 0.761 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_250_499 | fanduel | 63 | 12 | 0.317 | 0.163 | 0.500 | 0.668 | 1.024/1.095 | 0.687 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_150_180 | fanduel | 89 | 11 | 0.528 | -0.146 | 0.437 | 0.031 | 1.910/2.326 | 0.717 | tb_projection_low, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | fanduel | 2423 | 15 | 0.381 | -0.163 | 0.386 | 0.118 | 1.462/1.596 | 0.655 | market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | lay_150_180 | draftkings | 1217 | 16 | 0.613 | -0.015 | 0.310 | -0.311 | 1.513/1.596 | 0.628 | negative_or_flat_clv, weak_clv_beat |
| over | common | TB 2.5 | fair_lay | draftkings | 10 | 4 | 0.400 | -0.227 | 0.700 | 0.174 | 2.319/2.400 | 0.440 | sample_small, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_500_plus | fanduel | 4755 | 13 | 0.071 | -0.273 | 0.329 | 0.051 | 1.377/1.502 | 0.700 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_500_plus | fanduel | 2042 | 13 | 0.113 | -0.197 | 0.305 | 0.052 | 1.210/1.219 | 0.730 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_150_249 | fanduel | 1555 | 13 | 0.279 | -0.152 | 0.400 | 0.106 | 1.589/1.868 | 0.668 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | heavy_lay | draftkings | 703 | 16 | 0.632 | -0.039 | 0.243 | -0.551 | 1.416/1.502 | 0.672 | market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | draftkings | 2202 | 16 | 0.396 | -0.111 | 0.459 | 0.365 | 1.507/1.663 | 0.645 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | fanduel | 157 | 13 | 0.427 | -0.025 | 0.470 | 0.450 | 1.914/2.688 | 0.716 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | draftkings | 62 | 14 | 0.516 | -0.100 | 0.569 | -0.083 | 1.966/2.371 | 0.887 | tb_projection_low, pa_projection_error, negative_or_flat_clv |
| over | common | TB 1.5 | fair_lay | draftkings | 399 | 14 | 0.496 | -0.058 | 0.417 | 0.059 | 1.756/2.160 | 0.701 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | fanduel | 166 | 13 | 0.476 | -0.175 | 0.481 | 0.176 | 1.841/2.127 | 0.739 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | draftkings | 27 | 7 | 0.481 | 0.001 | 0.750 | 0.768 | 2.183/2.667 | 0.507 | sample_small, tb_projection_low, market_beats_model_brier |
| under | common | TB 2.5 | plus_100_149 | draftkings | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.920 | 2.367/2.000 | 0.200 | sample_small, tb_projection_high, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_150_249 | fanduel | 11 | 3 | 0.000 | -1.000 | 0.727 | 1.083 | 2.399/2.091 | 0.291 | sample_small, alt_tail_requires_separate_proof, tb_projection_high, market_beats_model_brier |
| over | alt_tail | TB 3.5 | plus_250_499 | fanduel | 2634 | 13 | 0.189 | -0.134 | 0.413 | 0.061 | 1.501/1.697 | 0.675 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 4.5+ | plus_250_499 | fanduel | 266 | 13 | 0.203 | -0.013 | 0.465 | 0.352 | 1.877/2.380 | 0.660 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | lay_130_149 | draftkings | 1 | 1 | 0.000 | -1.000 | 0.000 | -1.920 | 2.367/2.000 | 0.200 | sample_small, tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_150_249 | fanduel | 325 | 13 | 0.311 | -0.047 | 0.426 | 0.327 | 1.795/2.317 | 0.694 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | lay_130_149 | fanduel | 4 | 2 | 0.000 | -1.000 | 0.000 | -0.910 | 2.503/1.750 | 0.250 | sample_small, tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | lay_130_149 | draftkings | 539 | 15 | 0.549 | -0.060 | 0.339 | -0.280 | 1.572/1.952 | 0.666 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | fanduel | 598 | 13 | 0.478 | -0.098 | 0.321 | -0.054 | 1.627/2.069 | 0.685 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_100_149 | fanduel | 30 | 5 | 0.400 | -0.082 | 0.533 | 1.051 | 2.112/2.800 | 0.457 | sample_small, alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier |
| under | common | TB 1.5 | fair_lay | draftkings | 372 | 15 | 0.495 | -0.082 | 0.375 | -0.165 | 1.784/2.293 | 0.728 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |

## Diagnostic Groups

| Group | Rows | Win | ROI | TB Bias | PA MAE | Model/Mkt Brier | Issues |
|---|---:|---:|---:|---:|---:|---:|---|
| over | 0 TB | 8590 | 0.000 | -1.000 | 1.382 | 0.740 | 0.075/0.073 | tb_projection_high, market_beats_model_brier, weak_clv_beat |
| over | 0-2 PA | 1459 | 0.037 | -0.880 | 0.944 | 1.863 | 0.077/0.079 | tb_projection_high, pa_projection_error, negative_or_flat_clv, weak_clv_beat, low_valid_clv_coverage |
| over | 1 TB | 5741 | 0.000 | -1.000 | 0.415 | 0.643 | 0.081/0.080 | tb_projection_high, market_beats_model_brier, weak_clv_beat |
| over | 2-3 TB | 4711 | 0.412 | 0.040 | -0.851 | 0.672 | 0.187/0.175 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | 3 PA | 2791 | 0.085 | -0.650 | 0.564 | 0.674 | 0.096/0.093 | tb_projection_high, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | 4 PA | 11694 | 0.223 | -0.186 | -0.074 | 0.403 | 0.160/0.156 | market_beats_model_brier, weak_clv_beat |
| over | 4+ TB non-HR/unknown | 646 | 0.856 | 2.471 | -2.891 | 0.799 | 0.433/0.416 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | 5+ PA | 7012 | 0.358 | 0.187 | -0.790 | 0.940 | 0.212/0.204 | tb_projection_low, pa_projection_error, market_beats_model_brier, weak_clv_beat |
| over | HR driven | 3268 | 0.891 | 2.627 | -3.622 | 0.664 | 0.450/0.444 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | cross_book | 4130 | 0.404 | 0.194 | -0.650 | 0.663 | 0.249/0.243 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | same_book | 2813 | 0.413 | -0.104 | -0.195 | 0.661 | 0.242/0.240 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | synthetic | 16013 | 0.161 | -0.279 | -0.014 | 0.706 | 0.127/0.122 | market_beats_model_brier, weak_clv_beat |
| under | 0 TB | 959 | 1.000 | 0.638 | 1.531 | 0.685 | 0.154/0.184 | tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | 0-2 PA | 104 | 0.904 | 0.485 | 1.204 | 2.325 | 0.172/0.196 | tb_projection_high, pa_projection_error, low_valid_clv_coverage |
| under | 1 TB | 753 | 1.000 | 0.638 | 0.535 | 0.637 | 0.157/0.184 | tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | 2-3 TB | 648 | 0.012 | -0.978 | -0.731 | 0.653 | 0.367/0.323 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | 3 PA | 174 | 0.856 | 0.377 | 0.847 | 0.917 | 0.182/0.198 | tb_projection_high, pa_projection_error, negative_or_flat_clv, weak_clv_beat |
| under | 4 PA | 1459 | 0.631 | 0.030 | 0.023 | 0.348 | 0.233/0.236 | negative_or_flat_clv, weak_clv_beat |
| under | 4+ TB non-HR/unknown | 94 | 0.000 | -1.000 | -2.808 | 0.826 | 0.368/0.318 | tb_projection_low, pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | 5+ PA | 1189 | 0.468 | -0.225 | -0.736 | 0.868 | 0.268/0.256 | tb_projection_low, pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | HR driven | 472 | 0.000 | -1.000 | -3.607 | 0.645 | 0.359/0.315 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | same_book | 2926 | 0.588 | -0.037 | -0.195 | 0.664 | 0.242/0.240 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
