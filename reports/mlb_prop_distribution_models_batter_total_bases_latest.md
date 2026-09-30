# MLB Prop Distribution Models

Generated UTC: 2026-09-01T10:35:01Z
Rows: 10756
Raw rows before locked-offer dedupe: 67088
Collapsed duplicate locked-offer rows: 40802
Date range: 2026-07-02 to 2026-07-12
Status: ready

## Expanding Walk-Forward OOF

| Variant | Rows | Brier | Log Loss | Cal Err | Selected | ROI | CLV Beat |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 2144 | 0.151 | 0.468 | -6.0% | 612 | -27.1% | 35.7% |
| market_no_vig | 2144 | 0.128 | 0.395 | 1.1% | 186 | 216.8% | 37.2% |
| distribution | 2144 | 0.151 | 0.469 | -5.5% | 991 | -25.3% | 39.9% |
| distribution_calibrated | 2144 | 0.150 | 0.466 | -6.9% | 964 | -18.7% | 40.4% |
| distribution_empirical_blend | 2144 | 0.148 | 0.459 | -5.5% | 811 | -19.4% | 41.3% |
| event_side_line | 2144 | 0.124 | 0.399 | 1.2% | 431 | 56.5% | 36.9% |
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
| batter_total_bases | 2144 | 0.151 | 0.151 | 0.150 | 0.148 | 0.124 | -27.1% | -19.4% | 56.5% |

## TB/HR True-Pair Production Gates

These gates use holdout rows with true, non-synthetic paired prices only.

| Market / Side / Line | Rows | Brier Gain | Cal Err | Selected | CLV Beat | Avg CLV | Pass | Reasons |
|---|---:|---:|---:|---:|---:|---:|---|---|
| batter_total_bases / over / TB 1.5 | 492 | 0.004 | -2.4% | 47 | 36.5% | 17.9% | False | clv_beat_rate<0.55 |
| batter_total_bases / over / TB 2.5+ | 102 | 0.012 | -7.6% | 100 | 39.7% | 8.9% | False | abs_calibration_error>0.05, clv_beat_rate<0.55 |
| batter_total_bases / under / TB 1.5 | 214 | 0.005 | 0.8% | 66 | 42.9% | 0.3% | False | clv_beat_rate<0.55 |

## Hitter Outcome Shrinkage

| Group | Rows | PA | Hit Mult | TB Mult | HR Mult | XBH Mult | Actual H/PA | Pred H/PA | Actual TB/PA | Pred TB/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|global | 1688 | 6750.0 | 1.578 | 1.165 | 0.807 | 0.801 | 0.223 | 0.137 | 0.379 | 0.322 | 0.036 | 0.046 |
| batter_total_bases|power=power_high | 1591 | 6380.0 | 1.616 | 1.163 | 0.818 | 0.770 | 0.222 | 0.132 | 0.380 | 0.324 | 0.037 | 0.047 |
| batter_total_bases|power=power_mid | 83 | 317.0 | 1.051 | 1.088 | 0.945 | 1.117 | 0.237 | 0.205 | 0.360 | 0.288 | 0.022 | 0.038 |
| batter_total_bases|slot=slot_bottom | 540 | 1929.0 | 1.617 | 1.216 | 0.882 | 0.833 | 0.227 | 0.123 | 0.362 | 0.284 | 0.029 | 0.036 |
| batter_total_bases|slot=slot_middle | 367 | 1433.0 | 1.410 | 1.105 | 0.867 | 0.841 | 0.214 | 0.133 | 0.357 | 0.313 | 0.033 | 0.044 |
| batter_total_bases|slot=slot_top | 781 | 3388.0 | 1.458 | 1.129 | 0.826 | 0.849 | 0.226 | 0.146 | 0.397 | 0.347 | 0.041 | 0.053 |
| batter_total_bases|slot_power=slot_bottom|power_high | 487 | 1743.0 | 1.658 | 1.196 | 0.890 | 0.787 | 0.222 | 0.114 | 0.356 | 0.285 | 0.029 | 0.036 |
| batter_total_bases|slot_power=slot_bottom|power_mid | 46 | 163.0 | 1.065 | 1.093 | 0.993 | 1.044 | 0.270 | 0.194 | 0.423 | 0.283 | 0.031 | 0.036 |
| batter_total_bases|slot_power=slot_middle|power_high | 343 | 1342.0 | 1.471 | 1.130 | 0.897 | 0.832 | 0.218 | 0.127 | 0.369 | 0.314 | 0.036 | 0.045 |
| batter_total_bases|slot_power=slot_top|power_high | 761 | 3295.0 | 1.466 | 1.124 | 0.829 | 0.834 | 0.225 | 0.144 | 0.397 | 0.348 | 0.042 | 0.053 |

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

- Status: trained
- Evidence: market_split_temporal_train_true_pair_non_synthetic_only
- Calibrated line/side groups: 7
- Enabled line/side groups: 3
- Synthetic and one-sided FanDuel prices are display-only and cannot train these calibrators.
- `batter_total_bases|batter_total_bases|over|TB 1.5`: 2069 rows, method=beta, internal_gain=0.003, holdout_gain=0.000, cal_before=-2.0%, cal_after=-2.4%, enabled=True
- `batter_total_bases|batter_total_bases|over|TB 2.5`: 199 rows, method=beta, internal_gain=0.018, holdout_gain=0.014, cal_before=10.8%, cal_after=-2.1%, enabled=True
- `batter_total_bases|batter_total_bases|over|TB 3.5`: 59 rows, method=raw, internal_gain=-, holdout_gain=-, cal_before=-, cal_after=-, enabled=False
- `batter_total_bases|batter_total_bases|over|TB 3.5+`: 188 rows, method=isotonic, internal_gain=0.056, holdout_gain=0.010, cal_before=16.0%, cal_after=-14.4%, enabled=True
- `batter_total_bases|batter_total_bases|over|TB 4.5`: 129 rows, method=beta, internal_gain=0.039, holdout_gain=-, cal_before=-, cal_after=-, enabled=False
- `batter_total_bases|batter_total_bases|under|TB 1.5`: 927 rows, method=beta, internal_gain=0.001, holdout_gain=-0.003, cal_before=0.8%, cal_after=3.3%, enabled=False
- `batter_total_bases|batter_total_bases|under|TB 2.5`: 3 rows, method=raw, internal_gain=-, holdout_gain=-, cal_before=-, cal_after=-, enabled=False

## Event-Curve Side/Line Models

| Target | Status | Train | Holdout | Model Brier | Baseline Brier | Model Avg | Baseline Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| win_probability | trained | 3386 | 808 | 0.229 | 0.234 | 45.1% | 45.0% |
| clv_beat_probability | trained | 3159 | 757 | 0.245 | 0.259 | 42.5% | 43.2% |

## TB Component Structure

| Group | Rows | PA | 1B Mult | 2B Mult | 3B Mult | HR Mult | TB Mult | Actual 0 TB | Pred 0 TB | Actual 2B/PA | Pred 2B/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|global | 1730 | 6918.0 | 2.703 | 0.877 | 0.910 | 0.824 | 1.158 | 38.3% | 56.4% | 0.040 | 0.047 | 0.036 | 0.046 |
| batter_total_bases|pa=pa_high | 403 | 1784.0 | 1.929 | 0.952 | 0.959 | 0.802 | 1.025 | 34.0% | 47.8% | 0.044 | 0.048 | 0.035 | 0.055 |
| batter_total_bases|pa=pa_low | 230 | 781.0 | 1.630 | 0.919 | 1.027 | 0.921 | 1.158 | 43.5% | 66.1% | 0.037 | 0.049 | 0.023 | 0.034 |
| batter_total_bases|pa=pa_mid | 1097 | 4353.0 | 2.549 | 0.882 | 0.902 | 0.908 | 1.188 | 38.7% | 57.6% | 0.039 | 0.046 | 0.039 | 0.044 |
| batter_total_bases|pa_power=pa_high|power_high | 387 | 1709.0 | 2.056 | 0.950 | 0.962 | 0.815 | 1.036 | 33.3% | 48.7% | 0.045 | 0.050 | 0.037 | 0.056 |
| batter_total_bases|pa_power=pa_low|power_high | 209 | 713.0 | 1.409 | 0.913 | 1.019 | 0.923 | 1.137 | 44.5% | 67.8% | 0.038 | 0.052 | 0.022 | 0.034 |
| batter_total_bases|pa_power=pa_mid|power_high | 1026 | 4081.0 | 2.435 | 0.842 | 0.900 | 0.920 | 1.186 | 38.3% | 58.8% | 0.038 | 0.048 | 0.040 | 0.045 |
| batter_total_bases|pa_power=pa_mid|power_mid | 59 | 223.0 | 1.011 | 1.068 | 1.002 | 0.976 | 1.054 | 44.1% | 41.1% | 0.054 | 0.014 | 0.022 | 0.040 |
| batter_total_bases|power=power_high | 1622 | 6503.0 | 2.625 | 0.845 | 0.900 | 0.835 | 1.158 | 37.9% | 57.5% | 0.040 | 0.049 | 0.037 | 0.046 |
| batter_total_bases|power=power_mid | 89 | 344.0 | 1.016 | 1.089 | 1.005 | 0.953 | 1.059 | 42.7% | 41.0% | 0.044 | 0.017 | 0.020 | 0.038 |
| batter_total_bases|slot=slot_bottom | 551 | 1970.0 | 2.148 | 0.854 | 1.025 | 0.900 | 1.190 | 42.8% | 63.1% | 0.038 | 0.050 | 0.029 | 0.036 |
| batter_total_bases|slot=slot_middle | 382 | 1492.0 | 2.073 | 0.918 | 0.960 | 0.899 | 1.104 | 38.7% | 57.7% | 0.039 | 0.046 | 0.034 | 0.044 |
| batter_total_bases|slot=slot_top | 797 | 3456.0 | 2.418 | 0.943 | 0.918 | 0.845 | 1.118 | 34.9% | 51.2% | 0.042 | 0.046 | 0.041 | 0.052 |
| batter_total_bases|slot_power=slot_bottom|power_high | 497 | 1780.0 | 1.885 | 0.818 | 1.007 | 0.908 | 1.169 | 43.3% | 65.0% | 0.037 | 0.053 | 0.029 | 0.036 |
| batter_total_bases|slot_power=slot_middle|power_high | 355 | 1389.0 | 1.904 | 0.902 | 0.963 | 0.927 | 1.126 | 37.2% | 59.3% | 0.040 | 0.049 | 0.037 | 0.044 |
| batter_total_bases|slot_power=slot_top|power_high | 770 | 3334.0 | 2.463 | 0.927 | 0.926 | 0.847 | 1.116 | 34.8% | 51.9% | 0.042 | 0.047 | 0.041 | 0.053 |

## Hitter Outcome Policy

| Market | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_hits | 0 | - | - | - | use_baseline_curve |
| batter_total_bases | 808 | 0.238 | 0.243 | -0.004 | use_baseline_curve |
| batter_home_runs | 0 | - | - | - | use_baseline_curve |

## TB Event Model Bucket Policy

| Bucket | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_total_bases / under / common / TB 1.5 / lay_130_149 / draftkings | 175 | 0.263 | 0.257 | 0.006 | use_direct_event_curve |
| batter_total_bases / under / common / TB 1.5 / heavy_lay / draftkings | 248 | 0.232 | 0.227 | 0.005 | use_direct_event_curve |
| batter_total_bases / over / common / TB 1.5 / plus_100_149 / draftkings | 744 | 0.249 | 0.247 | 0.002 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / plus_150_249 / fanduel | 176 | 0.230 | 0.229 | 0.002 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / plus_100_149 / fanduel | 675 | 0.240 | 0.238 | 0.002 | use_baseline_curve |
| batter_total_bases / over / alt_tail / TB 2.5+ / plus_500_plus / fanduel | 138 | 0.370 | 0.369 | 0.002 | use_baseline_curve |
| batter_total_bases / under / common / TB 1.5 / lay_150_180 / draftkings | 382 | 0.249 | 0.248 | 0.002 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / fair_lay / fanduel | 203 | 0.256 | 0.261 | -0.004 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / fair_lay / draftkings | 115 | 0.250 | 0.254 | -0.004 | use_baseline_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_250_499 / fanduel | 90 | 0.274 | 0.280 | -0.007 | use_baseline_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_150_249 / fanduel | 81 | 0.262 | 0.272 | -0.010 | use_baseline_curve |
| batter_total_bases / under / common / TB 1.5 / fair_lay / draftkings | 103 | 0.244 | 0.255 | -0.010 | use_baseline_curve |

## Line-Bucket Probability Calibration

| Group | Rows | Columns | Method | Internal Gain | Holdout Gain | Enabled |
|---|---:|---|---|---:|---:|---|
| batter_total_bases / market / side / line_bucket / batter_total_bases / over / TB 1.5 | 2069 | market, side, line_bucket | beta | 0.003 | -0.002 | False |
| batter_total_bases / market / side / line_surface / line_bucket / batter_total_bases / over / common / TB 1.5 | 2069 | market, side, line_surface, line_bucket | beta | 0.003 | -0.002 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 1.5 / plus_100_149 | 1419 | market, side, line_surface, line_bucket, price_bucket | beta | 0.002 | 0.001 | True |
| batter_total_bases / market / side / line_bucket / batter_total_bases / under / TB 1.5 | 927 | market, side, line_bucket | beta | 0.001 | -0.003 | False |
| batter_total_bases / market / side / line_surface / line_bucket / batter_total_bases / under / common / TB 1.5 | 927 | market, side, line_surface, line_bucket | beta | 0.001 | -0.003 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / plus_100_149 / draftkings | 744 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.003 | 0.005 | True |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / plus_100_149 / fanduel | 675 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.002 | -0.002 | False |
| batter_total_bases / market / side / line_bucket / batter_total_bases / over / TB 2.5+ | 387 | market, side, line_bucket | isotonic | 0.033 | -0.082 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / under / common / TB 1.5 / lay_150_180 | 382 | market, side, line_surface, line_bucket, price_bucket | beta | 0.002 | -0.012 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / under / common / TB 1.5 / lay_150_180 / draftkings | 382 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.002 | -0.012 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 1.5 / fair_lay | 318 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.002 | -0.038 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / under / common / TB 1.5 / heavy_lay | 248 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.002 | -0.003 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / under / common / TB 1.5 / heavy_lay / draftkings | 248 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.002 | -0.003 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 1.5 / plus_150_249 | 220 | market, side, line_surface, line_bucket, price_bucket | beta | 0.002 | -0.005 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / fair_lay / fanduel | 203 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.005 | -0.040 | False |
| batter_total_bases / market / side / line_surface / line_bucket / batter_total_bases / over / common / TB 2.5+ | 199 | market, side, line_surface, line_bucket | beta | 0.018 | -0.042 | False |
| batter_total_bases / market / side / line_surface / line_bucket / batter_total_bases / over / alt_tail / TB 2.5+ | 188 | market, side, line_surface, line_bucket | isotonic | 0.056 | -0.129 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / plus_150_249 / fanduel | 176 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.001 | -0.006 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / under / common / TB 1.5 / lay_130_149 | 175 | market, side, line_surface, line_bucket, price_bucket | beta | 0.014 | -0.003 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / under / common / TB 1.5 / lay_130_149 / draftkings | 175 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.014 | -0.003 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / alt_tail / TB 2.5+ / plus_500_plus | 138 | market, side, line_surface, line_bucket, price_bucket | beta | 0.079 | -0.154 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / alt_tail / TB 2.5+ / plus_500_plus / fanduel | 138 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.079 | -0.154 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / fair_lay / draftkings | 115 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / under / common / TB 1.5 / fair_lay | 103 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / under / common / TB 1.5 / fair_lay / draftkings | 103 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 2.5+ / plus_250_499 | 90 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 2.5+ / plus_250_499 / fanduel | 90 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 2.5+ / plus_150_249 | 81 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 2.5+ / plus_150_249 / fanduel | 81 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 1.5 / lay_130_149 | 72 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / lay_130_149 / fanduel | 51 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / plus_150_249 / draftkings | 44 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / alt_tail / TB 2.5+ / plus_250_499 | 43 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / alt_tail / TB 2.5+ / plus_250_499 / fanduel | 43 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 1.5 / lay_150_180 | 29 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / lay_150_180 / fanduel | 26 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / lay_130_149 / draftkings | 21 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / under / common / TB 1.5 / plus_100_149 | 19 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / under / common / TB 1.5 / plus_100_149 / draftkings | 19 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 2.5+ / plus_100_149 | 16 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |

## Exact Bucket Model Selection

| Bucket | Rows | Decision | Best | Best Brier | ROI | Model | Market | Distribution | Cal Dist | Blend | Side-Line | K v3 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 180 | use_event_curve_side_line | event_side_line | 0.235 | 45.4% | 0.248 | 0.240 | 0.241 | 0.240 | 0.241 | 0.235 | - |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 97 | use_distribution | distribution | 0.226 | 9.6% | 0.238 | 0.238 | 0.226 | 0.226 | 0.229 | 0.240 | - |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 64 | use_market_only | market_only | 0.238 | - | 0.238 | 0.238 | 0.245 | 0.245 | 0.244 | 0.244 | - |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 44 | use_event_curve_side_line | event_side_line | 0.174 | 21.2% | 0.212 | 0.197 | 0.208 | 0.212 | 0.212 | 0.174 | - |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 43 | use_event_curve_side_line | event_side_line | 0.225 | 16.3% | 0.263 | 0.236 | 0.253 | 0.238 | 0.239 | 0.225 | - |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 34 | use_event_curve_side_line | event_side_line | 0.205 | 222.2% | 0.252 | 0.241 | 0.251 | 0.233 | 0.211 | 0.205 | - |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 187 | no_bet_negative_roi | distribution_blend | 0.237 | -13.8% | 0.238 | 0.239 | 0.243 | 0.238 | 0.237 | 0.241 | - |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 37 | no_bet_negative_roi | distribution_blend | 0.230 | -3.1% | 0.235 | 0.235 | 0.230 | 0.230 | 0.230 | 0.238 | - |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 27 | no_bet_sample | event_side_line | 0.224 | 61.0% | 0.255 | 0.247 | 0.262 | 0.246 | 0.249 | 0.224 | - |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 27 | no_bet_sample | event_side_line | 0.208 | 52.9% | 0.239 | 0.229 | 0.247 | 0.229 | 0.228 | 0.208 | - |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 19 | no_bet_sample | event_side_line | 0.238 | 36.7% | 0.245 | 0.245 | 0.244 | 0.250 | 0.250 | 0.238 | - |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 16 | no_bet_sample | event_side_line | 0.226 | - | 0.246 | 0.246 | 0.232 | 0.232 | 0.238 | 0.226 | - |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 10 | no_bet_sample | distribution | 0.218 | 136.2% | 0.263 | 0.261 | 0.218 | 0.243 | 0.236 | 0.250 | - |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 8 | no_bet_sample | model_only | 0.229 | 1.5% | 0.229 | 0.230 | 0.238 | 0.251 | 0.257 | 0.241 | - |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 7 | no_bet_sample | event_side_line | 0.204 | - | 0.212 | 0.211 | 0.221 | 0.220 | 0.210 | 0.204 | - |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | 3 | no_bet_sample | event_side_line | 0.153 | - | 0.222 | 0.196 | 0.244 | 0.233 | 0.227 | 0.153 | - |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 2 | no_bet_sample | distribution_calibrated | 0.246 | - | 0.257 | 0.250 | 0.248 | 0.246 | 0.248 | 0.258 | - |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 1 | no_bet_sample | distribution | 0.165 | 76.9% | 0.223 | 0.223 | 0.165 | 0.260 | 0.277 | 0.184 | - |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 1 | no_bet_sample | market_only | 0.068 | - | 0.082 | 0.068 | 0.085 | 0.137 | 0.109 | 0.080 | - |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 1 | no_bet_sample | distribution | 0.008 | - | 0.016 | 0.029 | 0.008 | 0.120 | 0.083 | 0.162 | - |
