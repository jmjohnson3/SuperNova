# MLB Pitcher K-Per-BF Challenger

Generated UTC: 2026-09-01T11:08:19Z
Status: **ready**
Usage: challenger diagnostic only; production artifacts are unchanged.

- Rows: 1110
- OOF evaluation rows: 1017
- Offer rows for line-level Brier: 11809
- Dates: 55
- Baseline K MAE, all loaded rows: 1.800
- Baseline K MAE: 1.814
- Raw boosted K MAE: 2.057
- Residual v2 K MAE: 1.850
- Residual v3 K MAE: 1.822
- Challenger K MAE: 1.814
- Selected challenger variant: baseline
- MAE gain: 0.000
- Line-level Brier gain: 0.00000
- Selected alpha: 0.000
- Accepted: False
- Gate reason: no_challenger_variant_improved

## Conservative Blend Gate

| Alpha | Rows | MAE | MAE Gain | Bias | Line Brier Rows | Line Brier | Brier Gain | Mean Shift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 1017 | 1.814 | 0.000 | -0.023 | 11370 | 0.25228 | 0.00000 | 0.000 |
| 0.050 | 1017 | 1.813 | 0.000 | -0.023 | 11370 | 0.25207 | 0.00021 | 0.000 |
| 0.100 | 1017 | 1.814 | -0.001 | -0.023 | 11370 | 0.25216 | 0.00012 | 0.000 |
| 0.200 | 1017 | 1.821 | -0.007 | -0.023 | 11370 | 0.25321 | -0.00093 | 0.000 |
| 0.350 | 1017 | 1.841 | -0.028 | -0.022 | 11370 | 0.25684 | -0.00455 | 0.001 |
| 0.500 | 1017 | 1.873 | -0.059 | -0.022 | 11370 | 0.26259 | -0.01031 | 0.001 |

## Residual v2 Gate

| Alpha | Rows | MAE | MAE Gain | Bias | Line Brier Rows | Line Brier | Brier Gain | Mean Shift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 1017 | 1.814 | 0.000 | -0.023 | 11370 | 0.25228 | 0.00000 | 0.000 |
| 0.020 | 1017 | 1.813 | 0.000 | -0.024 | 11370 | 0.25226 | 0.00003 | -0.001 |
| 0.050 | 1017 | 1.813 | 0.000 | -0.025 | 11370 | 0.25222 | 0.00006 | -0.002 |
| 0.100 | 1017 | 1.813 | 0.000 | -0.026 | 11370 | 0.25220 | 0.00008 | -0.003 |
| 0.150 | 1017 | 1.814 | 0.000 | -0.028 | 11370 | 0.25222 | 0.00006 | -0.005 |
| 0.200 | 1017 | 1.814 | -0.000 | -0.030 | 11370 | 0.25227 | 0.00001 | -0.007 |

- Residual v2 accepted: False
- Residual v2 reason: mae_not_improved,line_brier_not_improved

## Empirical-Bayes Residual v3 Gate

| Alpha | Rows | MAE | MAE Gain | Bias | Line Brier Rows | Line Brier | Brier Gain | Mean Shift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 1017 | 1.814 | 0.000 | -0.023 | 11370 | 0.25228 | 0.00000 | 0.000 |
| 0.050 | 1017 | 1.814 | -0.000 | -0.023 | 11370 | 0.25234 | -0.00006 | -0.000 |
| 0.100 | 1017 | 1.814 | -0.000 | -0.024 | 11370 | 0.25241 | -0.00013 | -0.001 |
| 0.150 | 1017 | 1.814 | -0.001 | -0.024 | 11370 | 0.25248 | -0.00019 | -0.001 |
| 0.200 | 1017 | 1.814 | -0.001 | -0.024 | 11370 | 0.25255 | -0.00027 | -0.001 |
| 0.300 | 1017 | 1.815 | -0.002 | -0.025 | 11370 | 0.25271 | -0.00043 | -0.002 |

- Residual v3 accepted: False
- Residual v3 reason: mae_not_improved,line_brier_not_improved

## Slices

## BF Error

| Bucket | Rows | Dates | Base MAE | Challenger MAE | Gain | Base Bias | Challenger Bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| near_bf | 614 | 49 | 1.750 | 1.750 | 0.000 | -0.190 | -0.190 |
| over_bf_by_3_plus | 216 | 49 | 2.020 | 2.020 | 0.000 | 0.808 | 0.808 |
| under_bf_by_3_plus | 187 | 50 | 1.785 | 1.785 | 0.000 | -0.436 | -0.436 |

## Pitch Count Error

| Bucket | Rows | Dates | Base MAE | Challenger MAE | Gain | Base Bias | Challenger Bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| near_pitches | 651 | 50 | 1.733 | 1.733 | 0.000 | -0.229 | -0.229 |
| over_pitches_by_12_plus | 200 | 47 | 2.069 | 2.069 | 0.000 | 1.311 | 1.311 |
| under_pitches_by_12_plus | 166 | 48 | 1.822 | 1.822 | 0.000 | -0.822 | -0.822 |

## Opponent K Profile

| Bucket | Rows | Dates | Base MAE | Challenger MAE | Gain | Base Bias | Challenger Bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| high_opp_k | 491 | 48 | 1.808 | 1.808 | 0.000 | -0.085 | -0.085 |
| low_opp_k | 60 | 33 | 1.885 | 1.885 | 0.000 | 0.633 | 0.633 |
| mid_opp_k | 453 | 49 | 1.817 | 1.817 | 0.000 | -0.035 | -0.035 |
