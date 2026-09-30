# MLB Prop Distribution Models

Generated UTC: 2026-07-17T10:43:07Z
Rows: 26286
Raw rows before locked-offer dedupe: 67088
Collapsed duplicate locked-offer rows: 40802
Date range: 2026-07-02 to 2026-07-12
Status: ready

## Expanding Walk-Forward OOF

| Variant | Rows | Brier | Log Loss | Cal Err | Selected | ROI | CLV Beat |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 5179 | 0.160 | 0.478 | -4.6% | 631 | -25.8% | 37.1% |
| market_no_vig | 5179 | 0.143 | 0.425 | 0.8% | 400 | 186.1% | 39.4% |
| distribution | 5179 | 0.160 | 0.480 | -4.2% | 2072 | -22.0% | 38.6% |
| distribution_calibrated | 5179 | 0.158 | 0.475 | -4.7% | 2023 | -20.2% | 38.7% |
| distribution_empirical_blend | 5179 | 0.156 | 0.471 | -4.0% | 1838 | -25.2% | 39.4% |
| event_side_line | 4321 | 0.156 | 0.471 | -1.4% | 1454 | 17.2% | 38.8% |
| k_v3 | 212 | 0.310 | 0.862 | 0.0% | 94 | -8.4% | 36.4% |

## Pitcher K v3

- Status: trained
- Enabled: False
- Player-games: 225
- Holdout offers: 212
- K v3 Brier: 0.310
- Poisson Brier: 0.291
- Gain vs Poisson: -0.020
- BF bias / sigma: -0.943 / 3.878
- Beta-binomial concentration: 20.0

## Market Holdout

| Market | Rows | Model Brier | Distribution Brier | Cal Dist Brier | Blend Brier | Side-Line Brier | Model ROI | Blend ROI | Side-Line ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases | 2144 | 0.151 | 0.151 | 0.151 | 0.148 | 0.124 | -27.1% | -19.8% | 40.9% |
| batter_hits | 1965 | 0.206 | 0.199 | 0.198 | 0.199 | 0.179 | 43.9% | -1.4% | 3.3% |
| batter_home_runs | 858 | 0.052 | 0.060 | 0.060 | 0.056 | - | - | -39.9% | - |
| pitcher_strikeouts | 212 | 0.259 | 0.291 | 0.250 | 0.251 | 0.260 | -61.0% | 3.8% | -6.5% |

## TB/HR True-Pair Production Gates

These gates use holdout rows with true, non-synthetic paired prices only.

| Market / Side / Line | Rows | Brier Gain | Cal Err | Selected | CLV Beat | Avg CLV | Pass | Reasons |
|---|---:|---:|---:|---:|---:|---:|---|---|
| batter_total_bases / over / TB 1.5 | 492 | 0.004 | -2.2% | 40 | 33.8% | -21.8% | False | clv_beat_rate<0.55, avg_clv_price<=0 |
| batter_total_bases / over / TB 2.5+ | 102 | 0.010 | -7.8% | 100 | 39.7% | 8.9% | False | abs_calibration_error>0.05, clv_beat_rate<0.55 |
| batter_total_bases / under / TB 1.5 | 214 | 0.005 | 1.0% | 59 | 40.4% | -6.6% | False | clv_beat_rate<0.55, avg_clv_price<=0 |

## Hitter Outcome Shrinkage

| Group | Rows | PA | Hit Mult | TB Mult | HR Mult | XBH Mult | Actual H/PA | Pred H/PA | Actual TB/PA | Pred TB/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|global | 1688 | 6750.0 | 1.567 | 1.158 | 0.802 | 0.796 | 0.223 | 0.138 | 0.379 | 0.324 | 0.036 | 0.047 |
| batter_hits|power=power_high | 1591 | 6380.0 | 1.605 | 1.155 | 0.813 | 0.766 | 0.222 | 0.133 | 0.380 | 0.326 | 0.037 | 0.047 |
| batter_hits|power=power_mid | 83 | 317.0 | 1.048 | 1.085 | 0.944 | 1.117 | 0.237 | 0.207 | 0.360 | 0.291 | 0.022 | 0.038 |
| batter_hits|slot=slot_bottom | 540 | 1929.0 | 1.607 | 1.208 | 0.878 | 0.828 | 0.227 | 0.124 | 0.362 | 0.287 | 0.029 | 0.037 |
| batter_hits|slot=slot_middle | 367 | 1433.0 | 1.402 | 1.099 | 0.864 | 0.837 | 0.214 | 0.134 | 0.357 | 0.315 | 0.033 | 0.045 |
| batter_hits|slot=slot_top | 781 | 3388.0 | 1.449 | 1.122 | 0.822 | 0.846 | 0.226 | 0.147 | 0.397 | 0.349 | 0.041 | 0.053 |
| batter_hits|slot_power=slot_bottom|power_high | 487 | 1743.0 | 1.647 | 1.187 | 0.885 | 0.783 | 0.222 | 0.115 | 0.356 | 0.287 | 0.029 | 0.037 |
| batter_hits|slot_power=slot_bottom|power_mid | 46 | 163.0 | 1.064 | 1.092 | 0.993 | 1.044 | 0.270 | 0.195 | 0.423 | 0.284 | 0.031 | 0.036 |
| batter_hits|slot_power=slot_middle|power_high | 343 | 1342.0 | 1.464 | 1.124 | 0.894 | 0.829 | 0.218 | 0.127 | 0.369 | 0.316 | 0.036 | 0.045 |
| batter_hits|slot_power=slot_top|power_high | 761 | 3295.0 | 1.458 | 1.117 | 0.824 | 0.830 | 0.225 | 0.145 | 0.397 | 0.351 | 0.042 | 0.054 |
| batter_home_runs|global | 1688 | 6750.0 | 1.567 | 1.158 | 0.802 | 0.796 | 0.223 | 0.138 | 0.379 | 0.324 | 0.036 | 0.047 |
| batter_home_runs|power=power_high | 1591 | 6380.0 | 1.605 | 1.155 | 0.813 | 0.766 | 0.222 | 0.133 | 0.380 | 0.326 | 0.037 | 0.047 |
| batter_home_runs|power=power_mid | 83 | 317.0 | 1.048 | 1.085 | 0.944 | 1.117 | 0.237 | 0.207 | 0.360 | 0.291 | 0.022 | 0.038 |
| batter_home_runs|slot=slot_bottom | 540 | 1929.0 | 1.607 | 1.208 | 0.878 | 0.828 | 0.227 | 0.124 | 0.362 | 0.287 | 0.029 | 0.037 |
| batter_home_runs|slot=slot_middle | 367 | 1433.0 | 1.402 | 1.099 | 0.864 | 0.837 | 0.214 | 0.134 | 0.357 | 0.315 | 0.033 | 0.045 |
| batter_home_runs|slot=slot_top | 781 | 3388.0 | 1.449 | 1.122 | 0.822 | 0.846 | 0.226 | 0.147 | 0.397 | 0.349 | 0.041 | 0.053 |
| batter_home_runs|slot_power=slot_bottom|power_high | 487 | 1743.0 | 1.647 | 1.187 | 0.885 | 0.783 | 0.222 | 0.115 | 0.356 | 0.287 | 0.029 | 0.037 |
| batter_home_runs|slot_power=slot_bottom|power_mid | 46 | 163.0 | 1.064 | 1.092 | 0.993 | 1.044 | 0.270 | 0.195 | 0.423 | 0.284 | 0.031 | 0.036 |
| batter_home_runs|slot_power=slot_middle|power_high | 343 | 1342.0 | 1.464 | 1.124 | 0.894 | 0.829 | 0.218 | 0.127 | 0.369 | 0.316 | 0.036 | 0.045 |
| batter_home_runs|slot_power=slot_top|power_high | 761 | 3295.0 | 1.458 | 1.117 | 0.824 | 0.830 | 0.225 | 0.145 | 0.397 | 0.351 | 0.042 | 0.054 |
| batter_total_bases|global | 1688 | 6750.0 | 1.567 | 1.158 | 0.802 | 0.796 | 0.223 | 0.138 | 0.379 | 0.324 | 0.036 | 0.047 |
| batter_total_bases|power=power_high | 1591 | 6380.0 | 1.605 | 1.155 | 0.813 | 0.766 | 0.222 | 0.133 | 0.380 | 0.326 | 0.037 | 0.047 |
| batter_total_bases|power=power_mid | 83 | 317.0 | 1.048 | 1.085 | 0.944 | 1.117 | 0.237 | 0.207 | 0.360 | 0.291 | 0.022 | 0.038 |
| batter_total_bases|slot=slot_bottom | 540 | 1929.0 | 1.607 | 1.208 | 0.878 | 0.828 | 0.227 | 0.124 | 0.362 | 0.287 | 0.029 | 0.037 |
| batter_total_bases|slot=slot_middle | 367 | 1433.0 | 1.402 | 1.099 | 0.864 | 0.837 | 0.214 | 0.134 | 0.357 | 0.315 | 0.033 | 0.045 |
| batter_total_bases|slot=slot_top | 781 | 3388.0 | 1.449 | 1.122 | 0.822 | 0.846 | 0.226 | 0.147 | 0.397 | 0.349 | 0.041 | 0.053 |
| batter_total_bases|slot_power=slot_bottom|power_high | 487 | 1743.0 | 1.647 | 1.187 | 0.885 | 0.783 | 0.222 | 0.115 | 0.356 | 0.287 | 0.029 | 0.037 |
| batter_total_bases|slot_power=slot_bottom|power_mid | 46 | 163.0 | 1.064 | 1.092 | 0.993 | 1.044 | 0.270 | 0.195 | 0.423 | 0.284 | 0.031 | 0.036 |
| batter_total_bases|slot_power=slot_middle|power_high | 343 | 1342.0 | 1.464 | 1.124 | 0.894 | 0.829 | 0.218 | 0.127 | 0.369 | 0.316 | 0.036 | 0.045 |
| batter_total_bases|slot_power=slot_top|power_high | 761 | 3295.0 | 1.458 | 1.117 | 0.824 | 0.830 | 0.225 | 0.145 | 0.397 | 0.351 | 0.042 | 0.054 |

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
- Enabled line/side groups: 2
- Synthetic and one-sided FanDuel prices are display-only and cannot train these calibrators.
- `batter_total_bases|batter_total_bases|over|TB 1.5`: 2069 rows, method=beta, internal_gain=0.003, holdout_gain=-0.000, cal_before=-2.2%, cal_after=-2.5%, enabled=False
- `batter_total_bases|batter_total_bases|over|TB 2.5`: 199 rows, method=beta, internal_gain=0.019, holdout_gain=0.014, cal_before=10.6%, cal_after=-2.2%, enabled=True
- `batter_total_bases|batter_total_bases|over|TB 3.5`: 59 rows, method=raw, internal_gain=-, holdout_gain=-, cal_before=-, cal_after=-, enabled=False
- `batter_total_bases|batter_total_bases|over|TB 3.5+`: 188 rows, method=isotonic, internal_gain=0.056, holdout_gain=0.004, cal_before=15.8%, cal_after=-14.8%, enabled=True
- `batter_total_bases|batter_total_bases|over|TB 4.5`: 129 rows, method=beta, internal_gain=0.038, holdout_gain=-, cal_before=-, cal_after=-, enabled=False
- `batter_total_bases|batter_total_bases|under|TB 1.5`: 927 rows, method=beta, internal_gain=0.001, holdout_gain=-0.002, cal_before=1.0%, cal_after=3.5%, enabled=False
- `batter_total_bases|batter_total_bases|under|TB 2.5`: 3 rows, method=raw, internal_gain=-, holdout_gain=-, cal_before=-, cal_after=-, enabled=False

## Event-Curve Side/Line Models

| Target | Status | Train | Holdout | Model Brier | Baseline Brier | Model Avg | Baseline Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| win_probability | trained | 6071 | 1521 | 0.221 | 0.242 | 50.7% | 48.3% |
| clv_beat_probability | trained | 5663 | 1413 | 0.247 | 0.264 | 46.1% | 46.8% |

## TB Component Structure

| Group | Rows | PA | 1B Mult | 2B Mult | 3B Mult | HR Mult | TB Mult | Actual 0 TB | Pred 0 TB | Actual 2B/PA | Pred 2B/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|global | 2295 | 9175.0 | 2.774 | 0.791 | 0.873 | 0.909 | 1.098 | 38.8% | 56.2% | 0.039 | 0.051 | 0.037 | 0.041 |
| batter_hits|pa=pa_high | 552 | 2451.0 | 1.931 | 0.946 | 0.920 | 0.841 | 1.005 | 32.6% | 47.2% | 0.047 | 0.051 | 0.037 | 0.050 |
| batter_hits|pa=pa_low | 271 | 905.0 | 1.785 | 0.929 | 1.052 | 0.945 | 1.168 | 43.2% | 66.1% | 0.041 | 0.051 | 0.022 | 0.028 |
| batter_hits|pa=pa_mid | 1472 | 5819.0 | 2.635 | 0.744 | 0.855 | 0.988 | 1.113 | 40.3% | 57.7% | 0.035 | 0.051 | 0.039 | 0.039 |
| batter_hits|pa_power=pa_high|power_high | 495 | 2198.0 | 2.297 | 0.911 | 0.921 | 0.834 | 0.998 | 31.5% | 48.3% | 0.047 | 0.055 | 0.039 | 0.054 |
| batter_hits|pa_power=pa_low|power_high | 224 | 751.0 | 1.378 | 0.901 | 1.034 | 0.931 | 1.114 | 45.5% | 68.3% | 0.041 | 0.058 | 0.021 | 0.031 |
| batter_hits|pa_power=pa_mid|power_high | 1297 | 5137.0 | 2.452 | 0.721 | 0.853 | 0.933 | 1.070 | 40.1% | 59.1% | 0.036 | 0.054 | 0.039 | 0.042 |
| batter_hits|pa_power=pa_mid|power_low | 87 | 342.0 | 1.029 | 1.015 | 0.991 | 1.056 | 1.202 | 40.2% | 48.9% | 0.023 | 0.019 | 0.044 | 0.004 |
| batter_hits|pa_power=pa_mid|power_mid | 88 | 340.0 | 1.071 | 1.020 | 1.008 | 1.016 | 1.124 | 43.2% | 45.7% | 0.038 | 0.032 | 0.035 | 0.029 |
| batter_hits|power=power_high | 2016 | 8086.0 | 2.651 | 0.754 | 0.849 | 0.862 | 1.061 | 38.6% | 57.5% | 0.040 | 0.055 | 0.037 | 0.044 |
| batter_hits|power=power_low | 152 | 599.0 | 0.985 | 1.129 | 0.988 | 1.111 | 1.284 | 38.8% | 47.9% | 0.032 | 0.015 | 0.037 | 0.005 |
| batter_hits|power=power_mid | 127 | 490.0 | 1.072 | 1.038 | 1.019 | 0.996 | 1.126 | 41.7% | 45.4% | 0.039 | 0.031 | 0.029 | 0.029 |
| batter_hits|slot=slot_bottom | 715 | 2554.0 | 2.279 | 0.770 | 1.041 | 0.965 | 1.157 | 42.0% | 63.1% | 0.035 | 0.052 | 0.030 | 0.032 |
| batter_hits|slot=slot_middle | 501 | 1956.0 | 2.228 | 0.819 | 0.948 | 0.918 | 1.047 | 40.7% | 57.7% | 0.036 | 0.052 | 0.032 | 0.038 |
| batter_hits|slot=slot_top | 1079 | 4665.0 | 2.366 | 0.881 | 0.857 | 0.920 | 1.068 | 35.8% | 50.9% | 0.042 | 0.050 | 0.042 | 0.047 |
| batter_hits|slot_power=slot_bottom|power_high | 612 | 2193.0 | 1.901 | 0.723 | 1.018 | 0.909 | 1.088 | 43.1% | 65.0% | 0.034 | 0.058 | 0.028 | 0.034 |
| batter_hits|slot_power=slot_bottom|power_mid | 62 | 216.0 | 1.064 | 1.026 | 1.008 | 1.008 | 1.118 | 37.1% | 50.3% | 0.042 | 0.026 | 0.037 | 0.031 |
| batter_hits|slot_power=slot_middle|power_high | 439 | 1718.0 | 1.923 | 0.812 | 0.936 | 0.919 | 1.040 | 39.6% | 59.6% | 0.038 | 0.056 | 0.034 | 0.041 |
| batter_hits|slot_power=slot_top|power_high | 965 | 4175.0 | 2.506 | 0.858 | 0.873 | 0.880 | 1.042 | 35.2% | 51.8% | 0.043 | 0.053 | 0.043 | 0.051 |
| batter_hits|slot_power=slot_top|power_low | 76 | 321.0 | 0.977 | 1.041 | 0.998 | 1.058 | 1.162 | 40.8% | 44.8% | 0.028 | 0.015 | 0.037 | 0.005 |
| batter_home_runs|global | 1690 | 6757.0 | 2.689 | 0.882 | 0.918 | 0.815 | 1.153 | 38.3% | 56.4% | 0.041 | 0.047 | 0.036 | 0.046 |
| batter_home_runs|pa=pa_high | 408 | 1821.0 | 1.937 | 1.000 | 0.957 | 0.813 | 1.054 | 31.6% | 47.5% | 0.047 | 0.047 | 0.037 | 0.055 |
| batter_home_runs|pa=pa_low | 196 | 656.0 | 1.564 | 0.943 | 1.029 | 0.919 | 1.128 | 44.4% | 66.1% | 0.040 | 0.050 | 0.021 | 0.034 |
| batter_home_runs|pa=pa_mid | 1086 | 4280.0 | 2.519 | 0.854 | 0.906 | 0.892 | 1.172 | 39.7% | 58.0% | 0.038 | 0.047 | 0.038 | 0.044 |
| batter_home_runs|pa_power=pa_high|power_high | 395 | 1760.0 | 2.031 | 0.994 | 0.960 | 0.824 | 1.062 | 31.1% | 48.2% | 0.048 | 0.048 | 0.038 | 0.056 |
| batter_home_runs|pa_power=pa_low|power_high | 178 | 598.0 | 1.352 | 0.934 | 1.020 | 0.920 | 1.104 | 46.1% | 67.9% | 0.040 | 0.053 | 0.020 | 0.034 |
| batter_home_runs|pa_power=pa_mid|power_high | 1020 | 4029.0 | 2.408 | 0.820 | 0.901 | 0.905 | 1.168 | 39.2% | 59.1% | 0.037 | 0.049 | 0.039 | 0.045 |
| batter_home_runs|pa_power=pa_mid|power_mid | 57 | 215.0 | 1.025 | 1.056 | 1.001 | 0.980 | 1.056 | 45.6% | 42.2% | 0.051 | 0.017 | 0.023 | 0.039 |
| batter_home_runs|power=power_high | 1593 | 6387.0 | 2.619 | 0.852 | 0.904 | 0.826 | 1.150 | 38.0% | 57.4% | 0.041 | 0.049 | 0.037 | 0.047 |
| batter_home_runs|power=power_mid | 83 | 317.0 | 1.024 | 1.081 | 1.005 | 0.962 | 1.065 | 43.4% | 42.0% | 0.044 | 0.017 | 0.022 | 0.038 |
| batter_home_runs|slot=slot_bottom | 540 | 1929.0 | 2.133 | 0.861 | 1.028 | 0.895 | 1.190 | 42.4% | 63.0% | 0.038 | 0.049 | 0.029 | 0.036 |
| batter_home_runs|slot=slot_middle | 368 | 1436.0 | 2.059 | 0.909 | 0.963 | 0.888 | 1.088 | 39.7% | 57.8% | 0.038 | 0.047 | 0.033 | 0.044 |
| batter_home_runs|slot=slot_top | 782 | 3392.0 | 2.452 | 0.954 | 0.922 | 0.842 | 1.116 | 34.8% | 51.2% | 0.043 | 0.046 | 0.041 | 0.053 |
| batter_home_runs|slot_power=slot_bottom|power_high | 487 | 1743.0 | 1.866 | 0.829 | 1.009 | 0.902 | 1.170 | 42.7% | 64.9% | 0.037 | 0.053 | 0.029 | 0.037 |
| batter_home_runs|slot_power=slot_middle|power_high | 344 | 1345.0 | 1.879 | 0.899 | 0.965 | 0.914 | 1.109 | 38.1% | 59.2% | 0.039 | 0.050 | 0.036 | 0.044 |
| batter_home_runs|slot_power=slot_top|power_high | 762 | 3299.0 | 2.459 | 0.934 | 0.927 | 0.845 | 1.111 | 34.9% | 51.7% | 0.043 | 0.047 | 0.042 | 0.053 |
| batter_total_bases|global | 1730 | 6918.0 | 2.704 | 0.872 | 0.907 | 0.819 | 1.150 | 38.3% | 56.2% | 0.040 | 0.047 | 0.036 | 0.046 |
| batter_total_bases|pa=pa_high | 414 | 1849.0 | 1.924 | 0.991 | 0.955 | 0.808 | 1.050 | 31.9% | 47.4% | 0.047 | 0.047 | 0.036 | 0.055 |
| batter_total_bases|pa=pa_low | 200 | 671.0 | 1.568 | 0.933 | 1.028 | 0.927 | 1.138 | 44.0% | 66.1% | 0.039 | 0.050 | 0.022 | 0.034 |
| batter_total_bases|pa=pa_mid | 1116 | 4398.0 | 2.547 | 0.848 | 0.897 | 0.896 | 1.168 | 39.6% | 57.7% | 0.038 | 0.047 | 0.038 | 0.044 |
| batter_total_bases|pa_power=pa_high|power_high | 397 | 1769.0 | 2.049 | 0.989 | 0.959 | 0.823 | 1.062 | 31.2% | 48.2% | 0.047 | 0.048 | 0.038 | 0.056 |
| batter_total_bases|pa_power=pa_low|power_high | 182 | 613.0 | 1.356 | 0.924 | 1.020 | 0.928 | 1.114 | 45.6% | 67.8% | 0.039 | 0.054 | 0.021 | 0.034 |
| batter_total_bases|pa_power=pa_mid|power_high | 1043 | 4121.0 | 2.428 | 0.809 | 0.896 | 0.907 | 1.165 | 39.1% | 58.9% | 0.037 | 0.049 | 0.039 | 0.045 |
| batter_total_bases|pa_power=pa_mid|power_mid | 59 | 223.0 | 1.022 | 1.065 | 1.001 | 0.977 | 1.059 | 44.1% | 41.3% | 0.054 | 0.016 | 0.022 | 0.039 |
| batter_total_bases|power=power_high | 1622 | 6503.0 | 2.628 | 0.840 | 0.897 | 0.830 | 1.150 | 37.9% | 57.3% | 0.040 | 0.049 | 0.037 | 0.047 |
| batter_total_bases|power=power_mid | 89 | 344.0 | 1.014 | 1.089 | 1.005 | 0.953 | 1.057 | 42.7% | 40.8% | 0.044 | 0.017 | 0.020 | 0.038 |
| batter_total_bases|slot=slot_bottom | 551 | 1970.0 | 2.152 | 0.850 | 1.024 | 0.896 | 1.182 | 42.8% | 62.8% | 0.038 | 0.050 | 0.029 | 0.036 |
| batter_total_bases|slot=slot_middle | 382 | 1492.0 | 2.063 | 0.915 | 0.960 | 0.896 | 1.099 | 38.7% | 57.4% | 0.039 | 0.047 | 0.034 | 0.044 |
| batter_total_bases|slot=slot_top | 797 | 3456.0 | 2.406 | 0.939 | 0.917 | 0.841 | 1.112 | 34.9% | 51.0% | 0.042 | 0.046 | 0.041 | 0.052 |
| batter_total_bases|slot_power=slot_bottom|power_high | 497 | 1780.0 | 1.889 | 0.814 | 1.006 | 0.904 | 1.162 | 43.3% | 64.7% | 0.037 | 0.053 | 0.029 | 0.036 |

## Hitter Outcome Policy

| Market | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_hits | 0 | - | - | - | use_baseline_curve |
| batter_total_bases | 0 | - | - | - | use_baseline_curve |
| batter_home_runs | 0 | - | - | - | use_baseline_curve |

## TB Event Model Bucket Policy

| Bucket | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_total_bases / under / common / TB 1.5 / lay_130_149 / draftkings | 175 | 0.262 | 0.257 | 0.006 | use_direct_event_curve |
| batter_total_bases / under / common / TB 1.5 / heavy_lay / draftkings | 248 | 0.231 | 0.227 | 0.005 | use_direct_event_curve |
| batter_total_bases / over / common / TB 1.5 / plus_150_249 / fanduel | 176 | 0.230 | 0.228 | 0.002 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / plus_100_149 / fanduel | 675 | 0.239 | 0.238 | 0.002 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / plus_100_149 / draftkings | 744 | 0.248 | 0.246 | 0.001 | use_baseline_curve |
| batter_total_bases / over / alt_tail / TB 2.5+ / plus_500_plus / fanduel | 138 | 0.369 | 0.368 | 0.001 | use_baseline_curve |
| batter_total_bases / under / common / TB 1.5 / lay_150_180 / draftkings | 382 | 0.248 | 0.247 | 0.001 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / fair_lay / draftkings | 115 | 0.249 | 0.254 | -0.005 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / fair_lay / fanduel | 203 | 0.255 | 0.260 | -0.005 | use_baseline_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_250_499 / fanduel | 90 | 0.272 | 0.280 | -0.008 | use_baseline_curve |
| batter_total_bases / under / common / TB 1.5 / fair_lay / draftkings | 103 | 0.244 | 0.255 | -0.011 | use_baseline_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_150_249 / fanduel | 81 | 0.260 | 0.272 | -0.011 | use_baseline_curve |

## Line-Bucket Probability Calibration

| Group | Rows | Columns | Method | Internal Gain | Holdout Gain | Enabled |
|---|---:|---|---|---:|---:|---|
| batter_hits / market / side / line_bucket / batter_hits / over / H 0.5 | 3167 | market, side, line_bucket | isotonic | 0.000 | -0.002 | False |
| batter_hits / market / side / line_surface / line_bucket / batter_hits / over / common / H 0.5 | 3167 | market, side, line_surface, line_bucket | isotonic | 0.000 | -0.002 | False |
| batter_total_bases / market / side / line_bucket / batter_total_bases / over / TB 1.5 | 2069 | market, side, line_bucket | beta | 0.003 | -0.002 | False |
| batter_total_bases / market / side / line_surface / line_bucket / batter_total_bases / over / common / TB 1.5 | 2069 | market, side, line_surface, line_bucket | beta | 0.003 | -0.002 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / heavy_lay | 1859 | market, side, line_surface, line_bucket, price_bucket | raw | -0.001 | - | False |
| batter_hits / market / side / line_bucket / batter_hits / under / H 0.5 | 1529 | market, side, line_bucket | raw | -0.000 | - | False |
| batter_hits / market / side / line_surface / line_bucket / batter_hits / under / common / H 0.5 | 1529 | market, side, line_surface, line_bucket | raw | -0.000 | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 1.5 / plus_100_149 | 1419 | market, side, line_surface, line_bucket, price_bucket | beta | 0.003 | 0.001 | True |
| batter_hits / market / side / line_bucket / batter_hits / over / H 1.5 | 1071 | market, side, line_bucket | beta | 0.033 | -0.037 | False |
| batter_hits / market / side / line_surface / line_bucket / batter_hits / over / common / H 1.5 | 1071 | market, side, line_surface, line_bucket | beta | 0.033 | -0.037 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / heavy_lay / fanduel | 1014 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | -0.000 | - | False |
| batter_total_bases / market / side / line_bucket / batter_total_bases / under / TB 1.5 | 927 | market, side, line_bucket | beta | 0.001 | -0.002 | False |
| batter_total_bases / market / side / line_surface / line_bucket / batter_total_bases / under / common / TB 1.5 | 927 | market, side, line_surface, line_bucket | beta | 0.001 | -0.002 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / heavy_lay / draftkings | 845 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | 0.000 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / lay_150_180 | 789 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.003 | 0.001 | True |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / plus_100_149 / draftkings | 744 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.003 | 0.005 | True |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / under / common / H 0.5 / plus_100_149 | 707 | market, side, line_surface, line_bucket, price_bucket | raw | -0.002 | - | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / under / common / H 0.5 / plus_100_149 / draftkings | 707 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | -0.002 | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / plus_100_149 / fanduel | 675 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.002 | -0.002 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / under / common / H 0.5 / plus_150_249 | 643 | market, side, line_surface, line_bucket, price_bucket | beta | 0.001 | -0.004 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / under / common / H 0.5 / plus_150_249 / draftkings | 643 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.001 | -0.004 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 1.5 / plus_150_249 | 610 | market, side, line_surface, line_bucket, price_bucket | beta | 0.008 | -0.037 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / lay_150_180 / fanduel | 433 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.001 | -0.003 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 1.5 / plus_150_249 / fanduel | 390 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.010 | -0.039 | False |
| batter_total_bases / market / side / line_bucket / batter_total_bases / over / TB 2.5+ | 387 | market, side, line_bucket | beta | 0.031 | -0.073 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / under / common / TB 1.5 / lay_150_180 | 382 | market, side, line_surface, line_bucket, price_bucket | beta | 0.002 | -0.010 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / under / common / TB 1.5 / lay_150_180 / draftkings | 382 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.002 | -0.010 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 1.5 / plus_250_499 | 374 | market, side, line_surface, line_bucket, price_bucket | beta | 0.049 | -0.034 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 1.5 / plus_250_499 / fanduel | 374 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.049 | -0.034 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / lay_150_180 / draftkings | 356 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.003 | -0.002 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 1.5 / fair_lay | 318 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.000 | -0.028 | False |
| batter_hits / market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / lay_130_149 | 302 | market, side, line_surface, line_bucket, price_bucket | raw | -0.001 | - | False |
| pitcher_strikeouts / market / side / line_bucket / pitcher_strikeouts / over / K 4.5-6.0 | 277 | market, side, line_bucket | beta | 0.016 | 0.016 | True |
| pitcher_strikeouts / market / side / line_bucket / pitcher_strikeouts / under / K 4.5-6.0 | 277 | market, side, line_bucket | beta | 0.016 | 0.016 | True |
| pitcher_strikeouts / market / side / line_surface / line_bucket / pitcher_strikeouts / over / common / K 4.5-6.0 | 277 | market, side, line_surface, line_bucket | beta | 0.016 | 0.016 | True |
| pitcher_strikeouts / market / side / line_surface / line_bucket / pitcher_strikeouts / under / common / K 4.5-6.0 | 277 | market, side, line_surface, line_bucket | beta | 0.016 | 0.016 | True |
| batter_hits / market / side / line_bucket / batter_hits / under / H 1.5 | 250 | market, side, line_bucket | raw | -0.001 | - | False |
| batter_hits / market / side / line_surface / line_bucket / batter_hits / under / common / H 1.5 | 250 | market, side, line_surface, line_bucket | raw | -0.001 | - | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / batter_total_bases / under / common / TB 1.5 / heavy_lay | 248 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.001 | -0.005 | False |
| batter_total_bases / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / under / common / TB 1.5 / heavy_lay / draftkings | 248 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.001 | -0.005 | False |

## Exact Bucket Model Selection

| Bucket | Rows | Decision | Best | Best Brier | ROI | Model | Market | Distribution | Cal Dist | Blend | Side-Line | K v3 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 261 | use_event_curve_side_line | event_side_line | 0.174 | 31.2% | 0.253 | 0.215 | 0.242 | 0.242 | 0.242 | 0.174 | - |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 212 | use_distribution | distribution | 0.244 | 14.6% | 0.248 | 0.248 | 0.244 | 0.244 | 0.244 | 0.256 | - |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 194 | use_market_only | market_only | 0.254 | - | 0.257 | 0.254 | 0.254 | 0.254 | 0.255 | 0.257 | - |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 180 | use_event_curve_side_line | event_side_line | 0.234 | 46.2% | 0.248 | 0.240 | 0.240 | 0.241 | 0.242 | 0.234 | - |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 150 | use_distribution | distribution | 0.243 | 40.3% | 0.253 | 0.253 | 0.243 | 0.243 | 0.244 | 0.250 | - |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 120 | use_event_curve_side_line | event_side_line | 0.212 | 54.6% | 0.243 | 0.234 | 0.261 | 0.261 | 0.262 | 0.212 | - |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 116 | use_event_curve_side_line | event_side_line | 0.189 | 37.1% | 0.271 | 0.224 | 0.258 | 0.259 | 0.257 | 0.189 | - |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 97 | use_distribution | distribution | 0.228 | 0.9% | 0.238 | 0.238 | 0.228 | 0.228 | 0.230 | 0.243 | - |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 84 | use_market_only | market_only | 0.184 | 155.1% | 0.202 | 0.184 | 0.203 | 0.203 | 0.200 | 0.194 | - |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 64 | use_market_only | market_only | 0.238 | - | 0.238 | 0.238 | 0.246 | 0.246 | 0.245 | 0.246 | - |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 49 | use_event_curve_side_line | event_side_line | 0.133 | - | 0.162 | 0.162 | 0.144 | 0.144 | 0.149 | 0.133 | - |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 49 | use_distribution | distribution | 0.144 | 19.5% | 0.162 | 0.162 | 0.144 | 0.144 | 0.150 | 0.145 | - |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 44 | use_event_curve_side_line | event_side_line | 0.172 | 9.0% | 0.212 | 0.197 | 0.207 | 0.207 | 0.209 | 0.172 | - |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 41 | use_event_curve_side_line | event_side_line | 0.238 | 53.4% | 0.273 | 0.250 | 0.252 | 0.252 | 0.252 | 0.238 | - |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 37 | use_distribution_market_blend | distribution_blend | 0.230 | 19.2% | 0.235 | 0.235 | 0.230 | 0.230 | 0.230 | 0.237 | - |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 34 | use_distribution | distribution | 0.250 | 2.3% | 0.253 | 0.253 | 0.250 | 0.250 | 0.251 | 0.251 | - |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 34 | use_event_curve_side_line | event_side_line | 0.211 | 31.5% | 0.258 | 0.232 | 0.246 | 0.246 | 0.248 | 0.211 | - |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 34 | use_event_curve_side_line | event_side_line | 0.208 | 222.2% | 0.252 | 0.241 | 0.250 | 0.239 | 0.214 | 0.208 | - |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 187 | no_bet_negative_roi | distribution_blend | 0.238 | -39.2% | 0.238 | 0.239 | 0.243 | 0.238 | 0.238 | 0.239 | - |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 98 | no_bet_negative_roi | distribution_blend | 0.254 | -17.5% | 0.259 | 0.259 | 0.258 | 0.255 | 0.254 | 0.258 | - |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 43 | no_bet_negative_roi | event_side_line | 0.234 | -31.7% | 0.263 | 0.236 | 0.253 | 0.253 | 0.249 | 0.234 | - |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 42 | no_bet_negative_roi | distribution | 0.259 | -100.0% | 0.266 | 0.266 | 0.259 | 0.259 | 0.267 | 0.305 | - |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 27 | no_bet_sample | event_side_line | 0.222 | 61.0% | 0.255 | 0.247 | 0.263 | 0.246 | 0.249 | 0.222 | - |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 27 | no_bet_sample | event_side_line | 0.208 | 52.9% | 0.239 | 0.229 | 0.245 | 0.229 | 0.227 | 0.208 | - |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 20 | no_bet_sample | event_side_line | 0.190 | 376.5% | 0.259 | 0.249 | 0.274 | 0.274 | 0.278 | 0.190 | - |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 20 | no_bet_sample | distribution | 0.259 | 79.5% | 0.264 | 0.264 | 0.259 | 0.269 | 0.274 | 0.304 | 0.292 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 20 | no_bet_sample | market_only | 0.260 | - | 0.260 | 0.260 | 0.265 | 0.262 | 0.267 | 0.293 | 0.269 |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 19 | no_bet_sample | event_side_line | 0.237 | 7.9% | 0.245 | 0.245 | 0.241 | 0.241 | 0.242 | 0.237 | - |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 16 | no_bet_sample | event_side_line | 0.227 | - | 0.246 | 0.246 | 0.228 | 0.228 | 0.235 | 0.227 | - |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 13 | no_bet_sample | distribution | 0.260 | -3.3% | 0.271 | 0.271 | 0.260 | 0.281 | 0.284 | 0.312 | 0.314 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 12 | no_bet_sample | distribution_calibrated | 0.253 | 18.5% | 0.259 | 0.259 | 0.338 | 0.253 | 0.254 | 0.280 | 0.299 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 11 | no_bet_sample | market_only | 0.250 | - | 0.269 | 0.250 | 0.266 | 0.255 | 0.258 | 0.279 | 0.264 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 10 | no_bet_sample | distribution | 0.216 | 136.2% | 0.263 | 0.261 | 0.216 | 0.244 | 0.236 | 0.244 | - |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 10 | no_bet_sample | market_only | 0.257 | - | 0.282 | 0.257 | 0.324 | 0.269 | 0.266 | 0.293 | 0.412 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 10 | no_bet_sample | market_only | 0.252 | - | 0.272 | 0.252 | 0.334 | 0.270 | 0.267 | 0.286 | 0.422 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 10 | no_bet_sample | k_v3 | 0.188 | 45.2% | 0.265 | 0.265 | 0.271 | 0.239 | 0.244 | 0.267 | 0.188 |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 8 | no_bet_sample | event_side_line | 0.170 | -8.3% | 0.277 | 0.196 | 0.244 | 0.244 | 0.236 | 0.170 | - |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 8 | no_bet_sample | event_side_line | 0.219 | - | 0.229 | 0.230 | 0.230 | 0.230 | 0.237 | 0.219 | - |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 8 | no_bet_sample | distribution | 0.256 | -45.3% | 0.272 | 0.278 | 0.256 | 0.282 | 0.301 | 0.358 | 0.295 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 7 | no_bet_sample | event_side_line | 0.205 | - | 0.212 | 0.211 | 0.220 | 0.220 | 0.213 | 0.205 | - |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 7 | no_bet_sample | distribution | 0.239 | 28.7% | 0.275 | 0.262 | 0.239 | 0.267 | 0.269 | 0.290 | 0.429 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 6 | no_bet_sample | k_v3 | 0.167 | - | 0.283 | 0.283 | 0.307 | 0.235 | 0.252 | 0.318 | 0.167 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|fanduel | 6 | no_bet_sample | k_v3 | 0.200 | - | 0.253 | 0.253 | 0.290 | 0.265 | 0.255 | 0.267 | 0.200 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 6 | no_bet_sample | distribution | 0.230 | 94.3% | 0.271 | 0.249 | 0.230 | 0.281 | 0.287 | 0.310 | 0.412 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 6 | no_bet_sample | model_only | 0.256 | - | 0.256 | 0.256 | 0.263 | 0.258 | 0.261 | 0.282 | 0.269 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 5 | no_bet_sample | distribution | 0.172 | - | 0.215 | 0.215 | 0.172 | 0.238 | 0.238 | 0.301 | 0.186 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 5 | no_bet_sample | event_side_line | 0.165 | 115.4% | 0.287 | 0.287 | 0.346 | 0.197 | 0.196 | 0.165 | 0.355 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|fanduel | 4 | no_bet_sample | market_only | 0.255 | - | 0.255 | 0.255 | 0.410 | 0.275 | 0.266 | 0.295 | 0.403 |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_150_180|draftkings | 4 | no_bet_sample | k_v3 | 0.101 | 63.7% | 0.178 | 0.178 | 0.115 | 0.205 | 0.206 | 0.276 | 0.101 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 4 | no_bet_sample | event_side_line | 0.136 | 118.5% | 0.322 | 0.322 | 0.327 | 0.164 | 0.140 | 0.136 | 0.240 |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | 3 | no_bet_sample | event_side_line | 0.151 | - | 0.222 | 0.196 | 0.247 | 0.235 | 0.228 | 0.151 | - |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 3 | no_bet_sample | model_only | 0.230 | - | 0.230 | 0.230 | 0.271 | 0.276 | 0.263 | 0.238 | 0.277 |
| pitcher_strikeouts|over|common|K <4.5|fair_lay|draftkings | 3 | no_bet_sample | event_side_line | 0.147 | - | 0.236 | 0.236 | 0.350 | 0.210 | 0.214 | 0.147 | 0.350 |
| pitcher_strikeouts|over|common|K <4.5|lay_130_149|fanduel | 3 | no_bet_sample | event_side_line | 0.179 | - | 0.262 | 0.262 | 0.307 | 0.216 | 0.218 | 0.179 | 0.365 |
| pitcher_strikeouts|over|common|K <4.5|plus_100_149|fanduel | 3 | no_bet_sample | event_side_line | 0.081 | - | 0.187 | 0.187 | 0.362 | 0.159 | 0.151 | 0.081 | 0.425 |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|draftkings | 3 | no_bet_sample | event_side_line | 0.144 | 85.7% | 0.236 | 0.236 | 0.350 | 0.210 | 0.214 | 0.144 | 0.350 |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 2 | no_bet_sample | model_only | 0.178 | - | 0.178 | 0.178 | 0.219 | 0.219 | 0.201 | 0.196 | - |
| batter_hits|over|common|H 0.5|plus_100_149|fanduel | 2 | no_bet_sample | market_only | 0.120 | - | 0.227 | 0.120 | 0.219 | 0.219 | 0.198 | 0.145 | - |
| batter_hits|under|common|H 0.5|lay_130_149|draftkings | 2 | no_bet_sample | event_side_line | 0.231 | 72.5% | 0.252 | 0.241 | 0.278 | 0.278 | 0.275 | 0.231 | - |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 2 | no_bet_sample | distribution | 0.243 | - | 0.257 | 0.250 | 0.243 | 0.243 | 0.249 | 0.248 | - |
