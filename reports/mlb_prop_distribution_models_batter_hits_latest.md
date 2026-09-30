# MLB Prop Distribution Models

Generated UTC: 2026-09-01T10:34:29Z
Rows: 10224
Raw rows before locked-offer dedupe: 67088
Collapsed duplicate locked-offer rows: 40802
Date range: 2026-07-02 to 2026-07-12
Status: ready

## Expanding Walk-Forward OOF

| Variant | Rows | Brier | Log Loss | Cal Err | Selected | ROI | CLV Beat |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 1965 | 0.206 | 0.590 | -4.6% | 9 | 43.9% | 72.7% |
| market_no_vig | 1965 | 0.188 | 0.543 | -0.2% | 169 | 75.4% | 40.3% |
| distribution | 1965 | 0.198 | 0.572 | -1.6% | 210 | 15.1% | 46.7% |
| distribution_calibrated | 1965 | 0.198 | 0.572 | -1.6% | 206 | 15.3% | 45.5% |
| distribution_empirical_blend | 1965 | 0.199 | 0.575 | -2.0% | 163 | -2.1% | 47.5% |
| event_side_line | 1965 | 0.180 | 0.524 | -3.1% | 811 | 3.6% | 40.8% |
| k_v3 | 0 | - | - | - | 0 | - | - |

## Pitcher K v3

- Status: missing
- Enabled: False
- Player-games: 0
- Holdout offers: 0
- K v3 Brier: -
- Poisson Brier: -
- Gain vs Poisson: -
- BF bias / sigma: - / -
- Beta-binomial concentration: -

## Market Holdout

| Market | Rows | Model Brier | Distribution Brier | Cal Dist Brier | Blend Brier | Side-Line Brier | Model ROI | Blend ROI | Side-Line ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_hits | 1965 | 0.206 | 0.198 | 0.198 | 0.199 | 0.180 | 43.9% | -2.1% | 3.6% |

## TB/HR True-Pair Production Gates

These gates use holdout rows with true, non-synthetic paired prices only.

| Market / Side / Line | Rows | Brier Gain | Cal Err | Selected | CLV Beat | Avg CLV | Pass | Reasons |
|---|---:|---:|---:|---:|---:|---:|---|---|

## Hitter Outcome Shrinkage

| Group | Rows | PA | Hit Mult | TB Mult | HR Mult | XBH Mult | Actual H/PA | Pred H/PA | Actual TB/PA | Pred TB/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|global | 1688 | 6750.0 | 1.578 | 1.165 | 0.807 | 0.801 | 0.223 | 0.137 | 0.379 | 0.322 | 0.036 | 0.046 |
| batter_hits|power=power_high | 1591 | 6380.0 | 1.616 | 1.163 | 0.818 | 0.770 | 0.222 | 0.132 | 0.380 | 0.324 | 0.037 | 0.047 |
| batter_hits|power=power_mid | 83 | 317.0 | 1.051 | 1.088 | 0.945 | 1.117 | 0.237 | 0.205 | 0.360 | 0.288 | 0.022 | 0.038 |
| batter_hits|slot=slot_bottom | 540 | 1929.0 | 1.617 | 1.216 | 0.882 | 0.833 | 0.227 | 0.123 | 0.362 | 0.284 | 0.029 | 0.036 |
| batter_hits|slot=slot_middle | 367 | 1433.0 | 1.410 | 1.105 | 0.867 | 0.841 | 0.214 | 0.133 | 0.357 | 0.313 | 0.033 | 0.044 |
| batter_hits|slot=slot_top | 781 | 3388.0 | 1.458 | 1.129 | 0.826 | 0.849 | 0.226 | 0.146 | 0.397 | 0.347 | 0.041 | 0.053 |
| batter_hits|slot_power=slot_bottom|power_high | 487 | 1743.0 | 1.658 | 1.196 | 0.890 | 0.787 | 0.222 | 0.114 | 0.356 | 0.285 | 0.029 | 0.036 |
| batter_hits|slot_power=slot_bottom|power_mid | 46 | 163.0 | 1.065 | 1.093 | 0.993 | 1.044 | 0.270 | 0.194 | 0.423 | 0.283 | 0.031 | 0.036 |
| batter_hits|slot_power=slot_middle|power_high | 343 | 1342.0 | 1.471 | 1.130 | 0.897 | 0.832 | 0.218 | 0.127 | 0.369 | 0.314 | 0.036 | 0.045 |
| batter_hits|slot_power=slot_top|power_high | 761 | 3295.0 | 1.466 | 1.124 | 0.829 | 0.834 | 0.225 | 0.144 | 0.397 | 0.348 | 0.042 | 0.053 |

## Direct Hitter Event Model

- Status: loaded
- Method: hierarchical_conditional_lgbm
- Trained UTC: 2026-07-05T07:32:10.149811+00:00
- Classes: out, walk, single, double, triple, hr
- Production gate: False
- Production eligible artifact: False
- Leakage-safe player priors: 783 players
- PA uncertainty groups: 6
- Direct event TB MAE gain vs independent rates: -0.005
- Explicit TB-state rows: 7395
- Explicit TB-state Brier: 0.698
- Explicit TB-state log loss: 1.344
- Direct-state selected candidate: convolution
- Direct-state blend alpha: 0.000
- HR-driven 4+ tail Brier gain: 0.000081

## True-Pair Hitter Line Calibration

- Status: insufficient_true_pair_rows
- Evidence: market_split_temporal_train_true_pair_non_synthetic_only
- Calibrated line/side groups: 0
- Enabled line/side groups: 0
- Synthetic and one-sided FanDuel prices are display-only and cannot train these calibrators.

## Event-Curve Side/Line Models

| Target | Status | Train | Holdout | Model Brier | Baseline Brier | Model Avg | Baseline Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| win_probability | trained | 6071 | 1521 | 0.222 | 0.242 | 50.7% | 48.2% |
| clv_beat_probability | trained | 5663 | 1413 | 0.245 | 0.264 | 46.0% | 46.8% |

## TB Component Structure

| Group | Rows | PA | 1B Mult | 2B Mult | 3B Mult | HR Mult | TB Mult | Actual 0 TB | Pred 0 TB | Actual 2B/PA | Pred 2B/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|global | 2295 | 9175.0 | 2.772 | 0.796 | 0.876 | 0.915 | 1.105 | 38.8% | 56.4% | 0.039 | 0.051 | 0.037 | 0.041 |
| batter_hits|pa=pa_high | 532 | 2344.0 | 1.942 | 0.898 | 0.926 | 0.835 | 0.979 | 34.8% | 47.6% | 0.044 | 0.053 | 0.037 | 0.050 |
| batter_hits|pa=pa_low | 309 | 1046.0 | 1.868 | 0.894 | 1.071 | 0.943 | 1.186 | 43.0% | 65.9% | 0.036 | 0.049 | 0.023 | 0.028 |
| batter_hits|pa=pa_mid | 1454 | 5785.0 | 2.637 | 0.787 | 0.832 | 1.000 | 1.133 | 39.3% | 57.7% | 0.037 | 0.050 | 0.039 | 0.039 |
| batter_hits|pa_power=pa_high|power_high | 483 | 2128.0 | 2.267 | 0.869 | 0.925 | 0.831 | 0.974 | 33.7% | 48.7% | 0.045 | 0.056 | 0.039 | 0.054 |
| batter_hits|pa_power=pa_low|power_high | 254 | 866.0 | 1.454 | 0.872 | 1.033 | 0.922 | 1.117 | 45.3% | 68.0% | 0.037 | 0.055 | 0.022 | 0.031 |
| batter_hits|pa_power=pa_mid|power_high | 1279 | 5092.0 | 2.455 | 0.759 | 0.858 | 0.944 | 1.092 | 39.1% | 59.1% | 0.038 | 0.054 | 0.039 | 0.042 |
| batter_hits|pa_power=pa_mid|power_low | 88 | 353.0 | 1.038 | 1.036 | 0.991 | 1.051 | 1.224 | 38.6% | 49.2% | 0.028 | 0.018 | 0.042 | 0.004 |
| batter_hits|pa_power=pa_mid|power_mid | 87 | 340.0 | 1.058 | 1.021 | 0.998 | 1.017 | 1.103 | 43.7% | 45.6% | 0.038 | 0.031 | 0.035 | 0.029 |
| batter_hits|power=power_high | 2016 | 8086.0 | 2.649 | 0.759 | 0.852 | 0.867 | 1.068 | 38.6% | 57.7% | 0.040 | 0.055 | 0.037 | 0.044 |
| batter_hits|power=power_low | 152 | 599.0 | 0.990 | 1.131 | 0.989 | 1.110 | 1.292 | 38.8% | 48.4% | 0.032 | 0.014 | 0.037 | 0.005 |
| batter_hits|power=power_mid | 127 | 490.0 | 1.075 | 1.039 | 1.020 | 0.997 | 1.130 | 41.7% | 45.7% | 0.039 | 0.031 | 0.029 | 0.029 |
| batter_hits|slot=slot_bottom | 715 | 2554.0 | 2.274 | 0.774 | 1.043 | 0.971 | 1.165 | 42.0% | 63.3% | 0.035 | 0.052 | 0.030 | 0.032 |
| batter_hits|slot=slot_middle | 501 | 1956.0 | 2.223 | 0.822 | 0.949 | 0.921 | 1.053 | 40.7% | 58.0% | 0.036 | 0.051 | 0.032 | 0.038 |
| batter_hits|slot=slot_top | 1079 | 4665.0 | 2.382 | 0.886 | 0.859 | 0.925 | 1.075 | 35.8% | 51.2% | 0.042 | 0.050 | 0.042 | 0.047 |
| batter_hits|slot_power=slot_bottom|power_high | 612 | 2193.0 | 1.896 | 0.727 | 1.019 | 0.914 | 1.095 | 43.1% | 65.2% | 0.034 | 0.057 | 0.028 | 0.034 |
| batter_hits|slot_power=slot_bottom|power_mid | 62 | 216.0 | 1.065 | 1.026 | 1.008 | 1.008 | 1.120 | 37.1% | 50.6% | 0.042 | 0.025 | 0.037 | 0.031 |
| batter_hits|slot_power=slot_middle|power_high | 439 | 1718.0 | 1.919 | 0.815 | 0.936 | 0.922 | 1.046 | 39.6% | 59.8% | 0.038 | 0.055 | 0.034 | 0.041 |
| batter_hits|slot_power=slot_top|power_high | 965 | 4175.0 | 2.504 | 0.862 | 0.875 | 0.885 | 1.047 | 35.2% | 52.0% | 0.043 | 0.053 | 0.043 | 0.051 |
| batter_hits|slot_power=slot_top|power_low | 76 | 321.0 | 0.981 | 1.042 | 0.998 | 1.057 | 1.166 | 40.8% | 45.3% | 0.028 | 0.015 | 0.037 | 0.005 |

## Hitter Outcome Policy

| Market | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_hits | 1521 | 0.251 | 0.239 | 0.012 | use_learned_outcome |
| batter_total_bases | 0 | - | - | - | use_baseline_curve |
| batter_home_runs | 0 | - | - | - | use_baseline_curve |

## TB Event Model Bucket Policy

| Bucket | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|

## Line-Bucket Probability Calibration

| Group | Rows | Columns | Method | Internal Gain | Holdout Gain | Enabled |
|---|---:|---|---|---:|---:|---|
| batter_hits / market / side / line_bucket / batter_hits / over / H 0.5 | 3167 | market, side, line_bucket | isotonic | 0.000 | -0.003 | False |
| batter_hits / market / side / line_surface / line_bucket / batter_hits / over / common / H 0.5 | 3167 | market, side, line_surface, line_bucket | isotonic | 0.000 | -0.003 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / heavy_lay | 1859 | market, side, line_surface, line_bucket, price_bucket | raw | -0.001 | - | False |
| batter_hits / market / side / line_bucket / batter_hits / under / H 0.5 | 1529 | market, side, line_bucket | raw | -0.000 | - | False |
| batter_hits / market / side / line_surface / line_bucket / batter_hits / under / common / H 0.5 | 1529 | market, side, line_surface, line_bucket | raw | -0.000 | - | False |
| batter_hits / market / side / line_bucket / batter_hits / over / H 1.5 | 1071 | market, side, line_bucket | beta | 0.033 | -0.038 | False |
| batter_hits / market / side / line_surface / line_bucket / batter_hits / over / common / H 1.5 | 1071 | market, side, line_surface, line_bucket | beta | 0.033 | -0.038 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / heavy_lay / fanduel | 1014 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | -0.001 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / heavy_lay / draftkings | 845 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | 0.000 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / lay_150_180 | 789 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.001 | -0.008 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / under / common / H 0.5 / plus_100_149 | 707 | market, side, line_surface, line_bucket, price_bucket | raw | -0.000 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / under / common / H 0.5 / plus_100_149 / draftkings | 707 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | -0.000 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / under / common / H 0.5 / plus_150_249 | 643 | market, side, line_surface, line_bucket, price_bucket | beta | 0.001 | -0.004 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / under / common / H 0.5 / plus_150_249 / draftkings | 643 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.001 | -0.004 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 1.5 / plus_150_249 | 610 | market, side, line_surface, line_bucket, price_bucket | beta | 0.008 | -0.038 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / lay_150_180 / fanduel | 433 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.003 | -0.019 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 1.5 / plus_150_249 / fanduel | 390 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.010 | -0.039 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 1.5 / plus_250_499 | 374 | market, side, line_surface, line_bucket, price_bucket | beta | 0.050 | -0.034 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 1.5 / plus_250_499 / fanduel | 374 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.050 | -0.034 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / lay_150_180 / draftkings | 356 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | 0.000 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / lay_130_149 | 302 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.001 | -0.011 | False |
| batter_hits / market / side / line_bucket / batter_hits / under / H 1.5 | 250 | market, side, line_bucket | raw | -0.001 | - | False |
| batter_hits / market / side / line_surface / line_bucket / batter_hits / under / common / H 1.5 | 250 | market, side, line_surface, line_bucket | raw | -0.001 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / under / common / H 1.5 / heavy_lay | 235 | market, side, line_surface, line_bucket, price_bucket | raw | -0.001 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / under / common / H 1.5 / heavy_lay / draftkings | 235 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | -0.001 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 1.5 / plus_150_249 / draftkings | 220 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | -0.002 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / fair_lay | 188 | market, side, line_surface, line_bucket, price_bucket | beta | 0.007 | 0.001 | True |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / lay_130_149 / draftkings | 178 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | -0.005 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / under / common / H 0.5 / fair_lay | 157 | market, side, line_surface, line_bucket, price_bucket | raw | -0.003 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / under / common / H 0.5 / fair_lay / draftkings | 157 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | -0.003 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / fair_lay / draftkings | 129 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.009 | -0.005 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / lay_130_149 / fanduel | 124 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | -0.008 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / fair_lay / fanduel | 59 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| batter_hits / market / side / line_bucket / batter_hits / over / H 2.5+ | 54 | market, side, line_bucket | raw | - | - | False |
| batter_hits / market / side / line_surface / line_bucket / batter_hits / over / alt_tail / H 2.5+ | 54 | market, side, line_surface, line_bucket | raw | - | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / over / alt_tail / H 2.5+ / plus_500_plus | 54 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / alt_tail / H 2.5+ / plus_500_plus / fanduel | 54 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 1.5 / plus_100_149 | 53 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 1.5 / plus_500_plus | 31 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 1.5 / plus_500_plus / fanduel | 31 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |

## Exact Bucket Model Selection

| Bucket | Rows | Decision | Best | Best Brier | ROI | Model | Market | Distribution | Cal Dist | Blend | Side-Line | K v3 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 261 | use_event_curve_side_line | event_side_line | 0.174 | 30.9% | 0.253 | 0.215 | 0.241 | 0.241 | 0.242 | 0.174 | - |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 212 | use_distribution | distribution | 0.243 | 14.6% | 0.248 | 0.248 | 0.243 | 0.243 | 0.243 | 0.256 | - |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 194 | use_distribution | distribution | 0.254 | 14.4% | 0.257 | 0.254 | 0.254 | 0.254 | 0.255 | 0.257 | - |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 150 | use_distribution | distribution | 0.242 | 40.3% | 0.253 | 0.253 | 0.242 | 0.242 | 0.244 | 0.250 | - |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 120 | use_event_curve_side_line | event_side_line | 0.213 | 55.5% | 0.243 | 0.234 | 0.262 | 0.262 | 0.262 | 0.213 | - |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 116 | use_event_curve_side_line | event_side_line | 0.193 | 36.7% | 0.271 | 0.224 | 0.259 | 0.259 | 0.259 | 0.193 | - |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 98 | use_distribution_market_blend | distribution_blend | 0.257 | 64.9% | 0.259 | 0.259 | 0.258 | 0.258 | 0.257 | 0.268 | - |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 84 | use_market_only | market_only | 0.184 | 155.1% | 0.202 | 0.184 | 0.204 | 0.204 | 0.200 | 0.192 | - |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 49 | use_event_curve_side_line | event_side_line | 0.133 | - | 0.162 | 0.162 | 0.144 | 0.144 | 0.149 | 0.133 | - |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 49 | use_distribution | distribution | 0.144 | 19.5% | 0.162 | 0.162 | 0.144 | 0.144 | 0.150 | 0.145 | - |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 42 | use_distribution | distribution | 0.259 | - | 0.266 | 0.266 | 0.259 | 0.259 | 0.267 | 0.305 | - |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 41 | use_event_curve_side_line | event_side_line | 0.237 | 36.1% | 0.273 | 0.250 | 0.252 | 0.252 | 0.253 | 0.237 | - |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 34 | use_event_curve_side_line | event_side_line | 0.213 | 31.5% | 0.258 | 0.232 | 0.245 | 0.245 | 0.247 | 0.213 | - |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 34 | no_bet_no_edge | distribution_calibrated | 0.250 | - | 0.253 | 0.253 | 0.251 | 0.250 | 0.251 | 0.253 | - |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 20 | no_bet_sample | event_side_line | 0.190 | 376.5% | 0.259 | 0.249 | 0.274 | 0.274 | 0.278 | 0.190 | - |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 8 | no_bet_sample | event_side_line | 0.175 | -8.3% | 0.277 | 0.196 | 0.248 | 0.249 | 0.240 | 0.175 | - |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 2 | no_bet_sample | model_only | 0.178 | - | 0.178 | 0.178 | 0.219 | 0.219 | 0.201 | 0.200 | - |
| batter_hits|over|common|H 0.5|plus_100_149|fanduel | 2 | no_bet_sample | market_only | 0.120 | - | 0.227 | 0.120 | 0.219 | 0.219 | 0.198 | 0.150 | - |
| batter_hits|under|common|H 0.5|lay_130_149|draftkings | 2 | no_bet_sample | event_side_line | 0.231 | - | 0.252 | 0.241 | 0.276 | 0.276 | 0.274 | 0.231 | - |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 1 | no_bet_sample | event_side_line | 0.283 | 145.0% | 0.350 | 0.354 | 0.560 | 0.560 | 0.522 | 0.283 | - |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 1 | no_bet_sample | distribution_blend | 0.023 | - | 0.028 | 0.049 | 0.028 | 0.028 | 0.023 | 0.132 | - |
| batter_hits|under|common|H 0.5|heavy_lay|draftkings | 1 | no_bet_sample | event_side_line | 0.058 | 51.8% | 0.258 | 0.147 | 0.143 | 0.143 | 0.200 | 0.058 | - |
