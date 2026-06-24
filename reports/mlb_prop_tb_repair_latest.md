# MLB TB Prop Repair Report

Generated: 2026-06-24T15:11:13.994198+00:00
Rows: 43148 | Dates: 2026-05-31 to 2026-06-23

## Top Repair Targets

| Bucket | Rows | ROI | CLV | Brier M/Mkt | TB Bias | PA MAE | Issues |
|---|---:|---:|---:|---:|---:|---:|---|
| over | common | TB 2.5 | plus_500_plus | fanduel | 592 | -0.151 | 0.195 | 0.117/0.113 | 0.064 | 0.759 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_250_499 | fanduel | 4996 | -0.159 | 0.075 | 0.160/0.153 | 0.022 | 0.704 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_250_499 | fanduel | 114 | 0.009 | 0.422 | 0.205/0.199 | 0.136 | 0.733 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_150_249 | fanduel | 2919 | -0.203 | 0.141 | 0.206/0.197 | 0.058 | 0.751 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_150_249 | draftkings | 158 | -0.184 | 0.699 | 0.221/0.220 | 0.096 | 0.758 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | fanduel | 4020 | -0.145 | 0.060 | 0.237/0.231 | -0.089 | 0.649 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | draftkings | 3545 | -0.080 | 0.365 | 0.243/0.242 | -0.122 | 0.634 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | draftkings | 635 | -0.121 | 0.074 | 0.251/0.249 | -0.114 | 0.670 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | draftkings | 35 | -0.041 | 1.053 | 0.255/0.246 | -0.097 | 0.749 | sample_small, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_500_plus | fanduel | 8055 | -0.280 | 0.040 | 0.066/0.062 | -0.040 | 0.693 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_500_plus | fanduel | 3450 | -0.237 | 0.046 | 0.095/0.092 | 0.058 | 0.729 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_250_499 | fanduel | 4455 | -0.140 | 0.069 | 0.154/0.148 | -0.099 | 0.664 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_150_249 | fanduel | 2629 | -0.136 | 0.034 | 0.204/0.191 | -0.171 | 0.652 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | heavy_lay | draftkings | 1109 | -0.050 | -0.519 | 0.234/0.234 | -0.027 | 0.661 | market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | fanduel | 263 | -0.104 | 0.291 | 0.240/0.214 | -0.359 | 0.711 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | fanduel | 269 | -0.195 | -0.034 | 0.262/0.241 | -0.024 | 0.683 | market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | lay_150_180 | fanduel | 150 | -0.158 | 0.018 | 0.273/0.245 | -0.251 | 0.705 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | draftkings | 104 | -0.098 | -0.100 | 0.281/0.252 | -0.011 | 0.747 | market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | alt_tail | TB 4.5+ | plus_250_499 | fanduel | 436 | -0.266 | 0.237 | 0.129/0.119 | -0.190 | 0.673 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_150_249 | fanduel | 562 | -0.136 | 0.271 | 0.204/0.191 | -0.236 | 0.686 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | heavy_lay | fanduel | 31 | -0.138 | -0.133 | 0.227/0.206 | -0.147 | 0.597 | sample_small, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | lay_130_149 | draftkings | 863 | -0.052 | -0.279 | 0.250/0.247 | -0.191 | 0.676 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | fanduel | 1014 | -0.112 | -0.163 | 0.252/0.246 | -0.198 | 0.687 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | fair_lay | draftkings | 580 | -0.025 | -0.142 | 0.254/0.248 | -0.203 | 0.686 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | plus_100_149 | draftkings | 95 | 0.021 | 0.150 | 0.257/0.250 | 0.216 | 0.902 | tb_projection_high, pa_projection_error, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_100_149 | fanduel | 34 | 0.094 | 1.004 | 0.284/0.244 | -0.795 | 0.503 | sample_small, alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier |
| under | common | TB 2.5 | lay_130_149 | draftkings | 24 | -0.073 | -0.685 | 0.263/0.249 | -0.322 | 0.492 | sample_small, tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |

## Exact Buckets

| Bucket | Rows | Dates | Win | ROI | CLV Beat | Avg CLV | Pred/Actual TB | PA MAE | Issues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| over | common | TB 2.5 | plus_500_plus | fanduel | 592 | 20 | 0.133 | -0.151 | 0.349 | 0.195 | 1.124/1.061 | 0.759 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_250_499 | fanduel | 4996 | 21 | 0.198 | -0.159 | 0.394 | 0.075 | 1.386/1.363 | 0.704 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_250_499 | fanduel | 114 | 20 | 0.272 | 0.009 | 0.471 | 0.422 | 1.101/0.965 | 0.733 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_150_249 | fanduel | 2919 | 24 | 0.287 | -0.203 | 0.397 | 0.141 | 1.251/1.193 | 0.751 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_150_249 | draftkings | 158 | 21 | 0.323 | -0.184 | 0.495 | 0.699 | 1.361/1.266 | 0.758 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | fanduel | 4020 | 23 | 0.388 | -0.145 | 0.369 | 0.060 | 1.535/1.624 | 0.649 | market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | lay_150_180 | draftkings | 1926 | 24 | 0.592 | -0.049 | 0.308 | -0.321 | 1.584/1.701 | 0.612 | negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | plus_100_149 | draftkings | 3545 | 24 | 0.410 | -0.080 | 0.456 | 0.365 | 1.587/1.710 | 0.634 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | draftkings | 635 | 22 | 0.463 | -0.121 | 0.404 | 0.074 | 1.853/1.967 | 0.670 | market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | draftkings | 35 | 10 | 0.457 | -0.041 | 0.781 | 1.053 | 2.360/2.457 | 0.749 | sample_small, market_beats_model_brier |
| over | common | TB 2.5 | fair_lay | draftkings | 10 | 4 | 0.400 | -0.227 | 0.700 | 0.174 | 2.319/2.400 | 0.440 | sample_small, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_500_plus | fanduel | 8055 | 21 | 0.070 | -0.280 | 0.321 | 0.040 | 1.455/1.495 | 0.693 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_500_plus | fanduel | 3450 | 21 | 0.105 | -0.237 | 0.308 | 0.046 | 1.278/1.221 | 0.729 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_250_499 | fanduel | 4455 | 21 | 0.187 | -0.140 | 0.402 | 0.069 | 1.582/1.681 | 0.664 | alt_tail_requires_separate_proof, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | plus_150_249 | fanduel | 2629 | 21 | 0.283 | -0.136 | 0.373 | 0.034 | 1.683/1.854 | 0.652 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| under | common | TB 1.5 | heavy_lay | draftkings | 1109 | 24 | 0.624 | -0.050 | 0.255 | -0.519 | 1.486/1.513 | 0.661 | market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 2.5 | plus_100_149 | fanduel | 263 | 21 | 0.392 | -0.104 | 0.409 | 0.291 | 2.052/2.411 | 0.711 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | fanduel | 269 | 21 | 0.465 | -0.195 | 0.379 | -0.034 | 1.947/1.970 | 0.683 | market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | lay_150_180 | fanduel | 150 | 19 | 0.520 | -0.158 | 0.376 | 0.018 | 2.102/2.353 | 0.705 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 1.5 | lay_130_149 | draftkings | 104 | 22 | 0.519 | -0.098 | 0.485 | -0.100 | 2.220/2.231 | 0.747 | market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 2.5 | plus_100_149 | draftkings | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.920 | 2.367/2.000 | 0.200 | sample_small, tb_projection_high, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_150_249 | fanduel | 11 | 3 | 0.000 | -1.000 | 0.727 | 1.083 | 2.399/2.091 | 0.291 | sample_small, alt_tail_requires_separate_proof, tb_projection_high, market_beats_model_brier |
| over | alt_tail | TB 4.5+ | plus_250_499 | fanduel | 436 | 21 | 0.149 | -0.266 | 0.443 | 0.237 | 1.978/2.167 | 0.673 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | lay_130_149 | draftkings | 1 | 1 | 0.000 | -1.000 | 0.000 | -1.920 | 2.367/2.000 | 0.200 | sample_small, tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| over | alt_tail | TB 3.5 | plus_150_249 | fanduel | 562 | 21 | 0.281 | -0.136 | 0.400 | 0.271 | 1.906/2.142 | 0.686 | alt_tail_requires_separate_proof, tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | common | TB 2.5 | lay_130_149 | fanduel | 4 | 2 | 0.000 | -1.000 | 0.000 | -0.910 | 2.503/1.750 | 0.250 | sample_small, tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | heavy_lay | fanduel | 31 | 8 | 0.581 | -0.138 | 0.387 | -0.133 | 2.370/2.516 | 0.597 | sample_small, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | lay_130_149 | draftkings | 863 | 23 | 0.553 | -0.052 | 0.338 | -0.279 | 1.691/1.882 | 0.676 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | common | TB 1.5 | fair_lay | fanduel | 1014 | 21 | 0.470 | -0.112 | 0.315 | -0.163 | 1.742/1.940 | 0.687 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | common | TB 1.5 | fair_lay | draftkings | 580 | 23 | 0.526 | -0.025 | 0.363 | -0.142 | 1.868/2.071 | 0.686 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |

## Diagnostic Groups

| Group | Rows | Win | ROI | TB Bias | PA MAE | Model/Mkt Brier | Issues |
|---|---:|---:|---:|---:|---:|---:|---|
| over | 0 TB | 14432 | 0.000 | -1.000 | 1.458 | 0.744 | 0.079/0.072 | tb_projection_high, market_beats_model_brier, weak_clv_beat |
| over | 0-2 PA | 2480 | 0.033 | -0.893 | 1.042 | 1.931 | 0.079/0.078 | tb_projection_high, pa_projection_error, market_beats_model_brier, weak_clv_beat |
| over | 1 TB | 9714 | 0.000 | -1.000 | 0.494 | 0.639 | 0.085/0.078 | tb_projection_high, market_beats_model_brier, weak_clv_beat |
| over | 2-3 TB | 8042 | 0.414 | 0.063 | -0.794 | 0.647 | 0.186/0.177 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | 3 PA | 4705 | 0.091 | -0.644 | 0.600 | 0.659 | 0.098/0.092 | tb_projection_high, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | 4 PA | 19662 | 0.220 | -0.206 | 0.011 | 0.400 | 0.160/0.154 | market_beats_model_brier, weak_clv_beat |
| over | 4+ TB non-HR/unknown | 1213 | 0.857 | 2.594 | -2.914 | 0.799 | 0.429/0.422 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| over | 5+ PA | 10355 | 0.369 | 0.263 | -0.797 | 0.944 | 0.216/0.210 | tb_projection_low, pa_projection_error, market_beats_model_brier, weak_clv_beat |
| over | HR driven | 5128 | 0.890 | 2.645 | -3.522 | 0.653 | 0.442/0.445 | tb_projection_low, weak_clv_beat |
| over | PA unknown | 1327 | 0.203 | -0.262 | 0.376 | n/a | 0.151/0.143 | tb_projection_high, market_beats_model_brier, weak_clv_beat |
| over | cross_book | 6945 | 0.410 | 0.219 | -0.577 | 0.652 | 0.251/0.244 | tb_projection_low, market_beats_model_brier, weak_clv_beat |
| over | same_book | 4506 | 0.419 | -0.088 | -0.111 | 0.649 | 0.244/0.242 | market_beats_model_brier, weak_clv_beat |
| over | synthetic | 27078 | 0.155 | -0.297 | 0.088 | 0.701 | 0.126/0.119 | market_beats_model_brier, weak_clv_beat |
| under | 0 TB | 1493 | 1.000 | 0.639 | 1.603 | 0.688 | 0.165/0.184 | tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | 0-2 PA | 165 | 0.927 | 0.532 | 1.270 | 2.427 | 0.179/0.194 | tb_projection_high, pa_projection_error, weak_clv_beat |
| under | 1 TB | 1186 | 1.000 | 0.643 | 0.621 | 0.630 | 0.168/0.185 | tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | 2-3 TB | 1051 | 0.008 | -0.987 | -0.647 | 0.621 | 0.351/0.324 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | 3 PA | 284 | 0.796 | 0.279 | 0.809 | 0.902 | 0.198/0.208 | tb_projection_high, pa_projection_error, negative_or_flat_clv, weak_clv_beat |
| under | 4 PA | 2359 | 0.621 | 0.015 | 0.099 | 0.347 | 0.236/0.237 | negative_or_flat_clv, weak_clv_beat |
| under | 4+ TB non-HR/unknown | 169 | 0.000 | -1.000 | -2.825 | 0.793 | 0.351/0.323 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | 5+ PA | 1672 | 0.458 | -0.242 | -0.740 | 0.862 | 0.267/0.258 | tb_projection_low, pa_projection_error, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | HR driven | 720 | 0.000 | -1.000 | -3.465 | 0.625 | 0.346/0.318 | tb_projection_low, market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
| under | PA unknown | 139 | 0.561 | -0.062 | 0.296 | n/a | 0.254/0.254 | tb_projection_high, negative_or_flat_clv, weak_clv_beat |
| under | same_book | 4619 | 0.582 | -0.045 | -0.113 | 0.651 | 0.243/0.242 | market_beats_model_brier, negative_or_flat_clv, weak_clv_beat |
