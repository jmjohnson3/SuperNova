# MLB Prop Distribution Models

Generated UTC: 2026-09-01T10:35:34Z
Rows: 4234
Raw rows before locked-offer dedupe: 67088
Collapsed duplicate locked-offer rows: 40802
Date range: 2026-07-02 to 2026-07-12
Status: ready

## Expanding Walk-Forward OOF

| Variant | Rows | Brier | Log Loss | Cal Err | Selected | ROI | CLV Beat |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 858 | 0.052 | 0.190 | -2.4% | 0 | - | - |
| market_no_vig | 858 | 0.047 | 0.162 | 2.7% | 45 | 443.8% | 46.2% |
| distribution | 858 | 0.060 | 0.220 | -7.2% | 752 | -30.9% | 35.0% |
| distribution_calibrated | 858 | 0.060 | 0.220 | -7.2% | 752 | -30.9% | 35.0% |
| distribution_empirical_blend | 858 | 0.056 | 0.207 | -5.4% | 738 | -38.9% | 34.8% |
| event_side_line | 0 | - | - | - | 0 | - | - |
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
| batter_home_runs | 858 | 0.052 | 0.060 | 0.060 | 0.056 | - | - | -38.9% | - |

## TB/HR True-Pair Production Gates

These gates use holdout rows with true, non-synthetic paired prices only.

| Market / Side / Line | Rows | Brier Gain | Cal Err | Selected | CLV Beat | Avg CLV | Pass | Reasons |
|---|---:|---:|---:|---:|---:|---:|---|---|

## Hitter Outcome Shrinkage

| Group | Rows | PA | Hit Mult | TB Mult | HR Mult | XBH Mult | Actual H/PA | Pred H/PA | Actual TB/PA | Pred TB/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_home_runs|global | 1688 | 6750.0 | 1.578 | 1.165 | 0.807 | 0.801 | 0.223 | 0.137 | 0.379 | 0.322 | 0.036 | 0.046 |
| batter_home_runs|power=power_high | 1591 | 6380.0 | 1.616 | 1.163 | 0.818 | 0.770 | 0.222 | 0.132 | 0.380 | 0.324 | 0.037 | 0.047 |
| batter_home_runs|power=power_mid | 83 | 317.0 | 1.051 | 1.088 | 0.945 | 1.117 | 0.237 | 0.205 | 0.360 | 0.288 | 0.022 | 0.038 |
| batter_home_runs|slot=slot_bottom | 540 | 1929.0 | 1.617 | 1.216 | 0.882 | 0.833 | 0.227 | 0.123 | 0.362 | 0.284 | 0.029 | 0.036 |
| batter_home_runs|slot=slot_middle | 367 | 1433.0 | 1.410 | 1.105 | 0.867 | 0.841 | 0.214 | 0.133 | 0.357 | 0.313 | 0.033 | 0.044 |
| batter_home_runs|slot=slot_top | 781 | 3388.0 | 1.458 | 1.129 | 0.826 | 0.849 | 0.226 | 0.146 | 0.397 | 0.347 | 0.041 | 0.053 |
| batter_home_runs|slot_power=slot_bottom|power_high | 487 | 1743.0 | 1.658 | 1.196 | 0.890 | 0.787 | 0.222 | 0.114 | 0.356 | 0.285 | 0.029 | 0.036 |
| batter_home_runs|slot_power=slot_bottom|power_mid | 46 | 163.0 | 1.065 | 1.093 | 0.993 | 1.044 | 0.270 | 0.194 | 0.423 | 0.283 | 0.031 | 0.036 |
| batter_home_runs|slot_power=slot_middle|power_high | 343 | 1342.0 | 1.471 | 1.130 | 0.897 | 0.832 | 0.218 | 0.127 | 0.369 | 0.314 | 0.036 | 0.045 |
| batter_home_runs|slot_power=slot_top|power_high | 761 | 3295.0 | 1.466 | 1.124 | 0.829 | 0.834 | 0.225 | 0.144 | 0.397 | 0.348 | 0.042 | 0.053 |

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
| win_probability | insufficient_rows | 0 | 0 | - | - | - | - |
| clv_beat_probability | insufficient_rows | 0 | 0 | - | - | - | - |

## TB Component Structure

| Group | Rows | PA | 1B Mult | 2B Mult | 3B Mult | HR Mult | TB Mult | Actual 0 TB | Pred 0 TB | Actual 2B/PA | Pred 2B/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_home_runs|global | 1690 | 6757.0 | 2.687 | 0.887 | 0.920 | 0.820 | 1.160 | 38.3% | 56.7% | 0.041 | 0.047 | 0.036 | 0.046 |
| batter_home_runs|pa=pa_high | 397 | 1756.0 | 1.942 | 0.961 | 0.960 | 0.806 | 1.029 | 33.8% | 48.0% | 0.045 | 0.048 | 0.036 | 0.055 |
| batter_home_runs|pa=pa_low | 226 | 766.0 | 1.627 | 0.929 | 1.028 | 0.913 | 1.149 | 43.8% | 66.1% | 0.038 | 0.049 | 0.022 | 0.034 |
| batter_home_runs|pa=pa_mid | 1067 | 4235.0 | 2.522 | 0.889 | 0.912 | 0.904 | 1.192 | 38.8% | 57.9% | 0.039 | 0.046 | 0.038 | 0.044 |
| batter_home_runs|pa_power=pa_high|power_high | 385 | 1700.0 | 2.038 | 0.955 | 0.963 | 0.816 | 1.036 | 33.2% | 48.7% | 0.045 | 0.049 | 0.037 | 0.056 |
| batter_home_runs|pa_power=pa_low|power_high | 205 | 698.0 | 1.405 | 0.922 | 1.019 | 0.915 | 1.127 | 44.9% | 67.8% | 0.039 | 0.052 | 0.021 | 0.034 |
| batter_home_runs|pa_power=pa_mid|power_high | 1003 | 3989.0 | 2.415 | 0.853 | 0.905 | 0.918 | 1.189 | 38.4% | 58.9% | 0.039 | 0.048 | 0.040 | 0.045 |
| batter_home_runs|pa_power=pa_mid|power_mid | 57 | 215.0 | 1.013 | 1.059 | 1.002 | 0.978 | 1.051 | 45.6% | 41.9% | 0.051 | 0.015 | 0.023 | 0.040 |
| batter_home_runs|power=power_high | 1593 | 6387.0 | 2.617 | 0.857 | 0.906 | 0.831 | 1.158 | 38.0% | 57.6% | 0.041 | 0.049 | 0.037 | 0.047 |
| batter_home_runs|power=power_mid | 83 | 317.0 | 1.026 | 1.081 | 1.005 | 0.963 | 1.067 | 43.4% | 42.3% | 0.044 | 0.017 | 0.022 | 0.038 |
| batter_home_runs|slot=slot_bottom | 540 | 1929.0 | 2.129 | 0.865 | 1.028 | 0.899 | 1.197 | 42.4% | 63.2% | 0.038 | 0.049 | 0.029 | 0.036 |
| batter_home_runs|slot=slot_middle | 368 | 1436.0 | 2.054 | 0.912 | 0.964 | 0.891 | 1.093 | 39.7% | 58.0% | 0.038 | 0.046 | 0.033 | 0.044 |
| batter_home_runs|slot=slot_top | 782 | 3392.0 | 2.464 | 0.958 | 0.923 | 0.846 | 1.122 | 34.8% | 51.5% | 0.043 | 0.046 | 0.041 | 0.052 |
| batter_home_runs|slot_power=slot_bottom|power_high | 487 | 1743.0 | 1.861 | 0.833 | 1.009 | 0.907 | 1.177 | 42.7% | 65.1% | 0.037 | 0.052 | 0.029 | 0.036 |
| batter_home_runs|slot_power=slot_middle|power_high | 344 | 1345.0 | 1.876 | 0.902 | 0.966 | 0.917 | 1.114 | 38.1% | 59.5% | 0.039 | 0.049 | 0.036 | 0.044 |
| batter_home_runs|slot_power=slot_top|power_high | 762 | 3299.0 | 2.457 | 0.938 | 0.928 | 0.849 | 1.117 | 34.9% | 52.0% | 0.043 | 0.047 | 0.042 | 0.053 |

## Hitter Outcome Policy

| Market | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_hits | 0 | - | - | - | use_baseline_curve |
| batter_total_bases | 0 | - | - | - | use_baseline_curve |
| batter_home_runs | 0 | - | - | - | use_baseline_curve |

## TB Event Model Bucket Policy

| Bucket | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|

## Line-Bucket Probability Calibration

| Group | Rows | Columns | Method | Internal Gain | Holdout Gain | Enabled |
|---|---:|---|---|---:|---:|---|

## Exact Bucket Model Selection

| Bucket | Rows | Decision | Best | Best Brier | ROI | Model | Market | Distribution | Cal Dist | Blend | Side-Line | K v3 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
