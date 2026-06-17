# MLB Prop Opportunity Feature Report

- Generated UTC: 2026-06-17T14:44:45Z
- Source: features.mlb_prop_market_training_examples
- Date range: 2026-05-31 to 2026-06-15
- Rows: 65072
- Unique dates: 16
- Holdout days: 28
- Minimum Brier gain: 0.001

This is a holdout diagnostic. It does not reopen bankroll buckets by itself.

## Projection Accuracy

| Metric | Rows | Coverage | MAE | RMSE | Bias | Pred Avg | Actual Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hitters PA | 62807 | 100.0% | 0.696 | 0.908 | -0.145 | 3.912 | 4.057 |
| Pitch Count Proxy | 2265 | 100.0% | 11.327 | 15.052 | +1.173 | 89.277 | 88.105 |
| Pitcher BF | 2265 | 100.0% | 2.942 | 3.910 | +0.305 | 23.189 | 22.884 |
| batter_hits PA | 26819 | 100.0% | 0.701 | 0.915 | -0.149 | 3.892 | 4.041 |
| batter_home_runs PA | 10106 | 100.0% | 0.698 | 0.908 | -0.139 | 3.887 | 4.027 |
| batter_total_bases PA | 25882 | 100.0% | 0.690 | 0.901 | -0.144 | 3.942 | 4.086 |

## Hitter Opportunity Breakdowns

These rows isolate confirmed lineup, batting-order, projected-PA, and pinch-hit risk effects. `Low-PA Miss` means the model expected at least 3.8 PA and the hitter finished with 2 or fewer.

| Level | Bucket | Rows | PA Cov | PA MAE | PA Bias | Avg PA | Actual PA | Low-PA | Low-PA Miss | ROI | CLV Beat |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batting_order | order_3_5 | 23003 | 100.0% | 0.630 | -0.189 | 4.00 | 4.19 | 3.0% | 1.7% | -13.3% | 38.8% |
| batting_order | order_6_9 | 22873 | 100.0% | 0.781 | -0.075 | 3.54 | 3.61 | 12.2% | 2.6% | -21.1% | 34.2% |
| batting_order | order_1_2 | 16931 | 100.0% | 0.670 | -0.181 | 4.29 | 4.47 | 3.0% | 1.5% | -7.7% | 39.7% |
| confirmed_lineup | confirmed_lineup | 62807 | 100.0% | 0.696 | -0.145 | 3.91 | 4.06 | 6.4% | 2.0% | -14.6% | 37.4% |
| market_confirmed_lineup | batter_hits|confirmed_lineup | 26819 | 100.0% | 0.701 | -0.149 | 3.89 | 4.04 | 6.6% | 2.0% | -10.2% | 38.3% |
| market_confirmed_lineup | batter_total_bases|confirmed_lineup | 25882 | 100.0% | 0.690 | -0.144 | 3.94 | 4.09 | 6.0% | 2.0% | -15.7% | 37.8% |
| market_confirmed_lineup | batter_home_runs|confirmed_lineup | 10106 | 100.0% | 0.698 | -0.139 | 3.89 | 4.03 | 6.7% | 2.0% | -23.6% | 34.0% |
| pinch_hit_risk | pinch_low | 39934 | 100.0% | 0.647 | -0.186 | 4.13 | 4.31 | 3.0% | 1.6% | -10.9% | 39.2% |
| pinch_hit_risk | pinch_medium | 22873 | 100.0% | 0.781 | -0.075 | 3.54 | 3.61 | 12.2% | 2.6% | -21.1% | 34.2% |
| projected_pa | projected_pa_3_8_to_4_3 | 27770 | 100.0% | 0.608 | -0.115 | 4.07 | 4.19 | 4.1% | 4.1% | -9.3% | 37.5% |
| projected_pa | projected_pa_3_2_to_3_7 | 15923 | 100.0% | 0.809 | -0.210 | 3.49 | 3.70 | 11.3% | 0.0% | -21.6% | 35.3% |
| projected_pa | projected_pa_4_4_plus | 13157 | 100.0% | 0.608 | +0.091 | 4.55 | 4.46 | 0.9% | 0.9% | -21.5% | 40.0% |
| projected_pa | projected_pa_under_3_2 | 5957 | 100.0% | 0.998 | -0.634 | 2.90 | 3.53 | 16.2% | 0.0% | -5.6% | 36.6% |

## Overall

| Bucket | Rows | Train | Holdout | Decision | Brier Gain | Base Brier | Opp Brier | Base ROI | Opp ROI | Base CLV | Opp CLV | PA Cov | BF Cov | Reasons |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| overall | 65072 | 59581 | 5491 | keep_baseline | -0.000 | 0.154 | 0.155 | +37.5% | +32.9% | +0.17 | +0.21 | 96.5% | 3.5% | brier_not_improved; roi_worse |

## Market

| Bucket | Rows | Train | Holdout | Decision | Brier Gain | Base Brier | Opp Brier | Base ROI | Opp ROI | Base CLV | Opp CLV | PA Cov | BF Cov | Reasons |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| batter_total_bases | 25882 | 23540 | 2342 | keep_baseline | -0.001 | 0.153 | 0.154 | +55.4% | +51.5% | +0.31 | +0.30 | 100.0% | 0.0% | brier_not_improved; roi_worse; avg_clv_worse; clv_beat_worse |
| batter_hits | 26819 | 24788 | 2031 | keep_baseline | -0.000 | 0.189 | 0.189 | +12.9% | +12.6% | +0.06 | +0.09 | 100.0% | 0.0% | brier_not_improved; roi_worse |
| batter_home_runs | 10106 | 9186 | 920 | keep_baseline | -0.000 | 0.058 | 0.058 | +57.6% | +36.3% | +0.50 | +0.39 | 100.0% | 0.0% | brier_not_improved; roi_worse; avg_clv_worse; clv_beat_worse |
| pitcher_strikeouts | 2265 | 2067 | 198 | keep_baseline | -0.000 | 0.273 | 0.273 | -57.5% | -50.2% | +0.28 | +0.27 | 0.0% | 100.0% | brier_not_improved; avg_clv_worse; clv_beat_worse |

## Market And Side

| Bucket | Rows | Train | Holdout | Decision | Brier Gain | Base Brier | Opp Brier | Base ROI | Opp ROI | Base CLV | Opp CLV | PA Cov | BF Cov | Reasons |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| batter_total_bases|over | 22956 | 20865 | 2091 | keep_baseline | -0.004 | 0.144 | 0.148 | +61.7% | +43.0% | +0.37 | +0.28 | 100.0% | 0.0% | brier_not_improved; roi_worse; avg_clv_worse; clv_beat_worse |
| batter_hits|over | 21394 | 19845 | 1549 | keep_baseline | +0.001 | 0.174 | 0.173 | +18.7% | +13.7% | +0.12 | +0.19 | 100.0% | 0.0% | brier_not_improved; roi_worse |
| batter_home_runs|over | 10106 | 9186 | 920 | keep_baseline | -0.000 | 0.058 | 0.058 | +57.6% | +36.3% | +0.50 | +0.39 | 100.0% | 0.0% | brier_not_improved; roi_worse; avg_clv_worse; clv_beat_worse |
| batter_hits|under | 5425 | 4943 | 482 | opportunity_helped | +0.003 | 0.236 | 0.232 | -3.7% | +29.6% | +0.08 | +0.43 | 100.0% | 0.0% |  |
| batter_total_bases|under | 2926 | 2675 | 251 | keep_baseline | -0.008 | 0.230 | 0.239 | -15.1% | -0.2% | -0.20 | +0.10 | 100.0% | 0.0% | brier_not_improved; clv_beat_worse |
| pitcher_strikeouts|over | 1139 | 1040 | 99 | keep_baseline | +0.022 | 0.290 | 0.268 | -18.8% | -4.5% | -0.06 | -0.39 | 0.0% | 100.0% | avg_clv_worse; clv_beat_worse |
| pitcher_strikeouts|under | 1126 | 1027 | 99 | keep_baseline | +0.030 | 0.285 | 0.255 | -49.7% | -37.9% | +0.39 | +0.13 | 0.0% | 100.0% | avg_clv_worse |

## Rule

Opportunity features are favored only when they improve holdout Brier and do not make ROI/CLV worse on enough selected rows.
