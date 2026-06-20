# MLB Prop Opportunity Feature Report

- Generated UTC: 2026-06-19T23:40:33Z
- Source: features.mlb_prop_market_training_examples
- Date range: 2026-05-31 to 2026-06-18
- Rows: 78285
- Unique dates: 19
- Holdout days: 28
- Minimum Brier gain: 0.001

This is a holdout diagnostic. It does not reopen bankroll buckets by itself.

## Projection Accuracy

| Metric | Rows | Coverage | MAE | RMSE | Bias | Pred Avg | Actual Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hitters PA | 75592 | 100.0% | 0.693 | 0.909 | -0.132 | 3.915 | 4.047 |
| Pitch Count Proxy | 2693 | 100.0% | 11.490 | 15.273 | +1.062 | 89.046 | 87.984 |
| Pitcher BF | 2693 | 100.0% | 2.984 | 3.967 | +0.276 | 23.129 | 22.853 |
| batter_hits PA | 31687 | 100.0% | 0.699 | 0.916 | -0.138 | 3.895 | 4.033 |
| batter_home_runs PA | 12375 | 100.0% | 0.695 | 0.910 | -0.126 | 3.891 | 4.017 |
| batter_total_bases PA | 31530 | 100.0% | 0.687 | 0.903 | -0.129 | 3.944 | 4.073 |

## Hitter Opportunity Breakdowns

These rows isolate confirmed lineup, batting-order, projected-PA, and pinch-hit risk effects. `Low-PA Miss` means the model expected at least 3.8 PA and the hitter finished with 2 or fewer.

| Level | Bucket | Rows | PA Cov | PA MAE | PA Bias | Avg PA | Actual PA | Low-PA | Low-PA Miss | ROI | CLV Beat |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batting_order | order_3_5 | 27594 | 100.0% | 0.627 | -0.177 | 4.00 | 4.18 | 3.0% | 1.6% | -13.3% | 38.9% |
| batting_order | order_6_9 | 27557 | 100.0% | 0.767 | -0.066 | 3.54 | 3.61 | 11.8% | 2.8% | -20.4% | 34.1% |
| batting_order | order_1_2 | 20441 | 100.0% | 0.683 | -0.160 | 4.30 | 4.46 | 3.3% | 2.0% | -9.0% | 40.0% |
| confirmed_lineup | confirmed_lineup | 75592 | 100.0% | 0.693 | -0.132 | 3.91 | 4.05 | 6.3% | 2.1% | -14.7% | 37.5% |
| market_confirmed_lineup | batter_hits|confirmed_lineup | 31687 | 100.0% | 0.699 | -0.138 | 3.89 | 4.03 | 6.5% | 2.1% | -10.6% | 38.2% |
| market_confirmed_lineup | batter_total_bases|confirmed_lineup | 31530 | 100.0% | 0.687 | -0.129 | 3.94 | 4.07 | 6.0% | 2.1% | -15.2% | 37.9% |
| market_confirmed_lineup | batter_home_runs|confirmed_lineup | 12375 | 100.0% | 0.695 | -0.126 | 3.89 | 4.02 | 6.6% | 2.1% | -24.2% | 34.5% |
| pinch_hit_risk | pinch_low | 48035 | 100.0% | 0.651 | -0.170 | 4.13 | 4.30 | 3.1% | 1.7% | -11.5% | 39.3% |
| pinch_hit_risk | pinch_medium | 27557 | 100.0% | 0.767 | -0.066 | 3.54 | 3.61 | 11.8% | 2.8% | -20.4% | 34.1% |
| projected_pa | projected_pa_3_8_to_4_3 | 33343 | 100.0% | 0.600 | -0.104 | 4.07 | 4.18 | 4.0% | 4.0% | -10.3% | 37.7% |
| projected_pa | projected_pa_3_2_to_3_7 | 19011 | 100.0% | 0.790 | -0.199 | 3.49 | 3.69 | 10.5% | 0.0% | -22.2% | 35.3% |
| projected_pa | projected_pa_4_4_plus | 15989 | 100.0% | 0.628 | +0.126 | 4.55 | 4.43 | 1.6% | 1.6% | -18.9% | 40.0% |
| projected_pa | projected_pa_under_3_2 | 7249 | 100.0% | 1.013 | -0.653 | 2.89 | 3.55 | 15.9% | 0.0% | -6.3% | 36.2% |

## Overall

| Bucket | Rows | Train | Holdout | Decision | Brier Gain | Base Brier | Opp Brier | Base ROI | Opp ROI | Base CLV | Opp CLV | PA Cov | BF Cov | Reasons |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| overall | 78285 | 76157 | 2128 | keep_baseline | +0.001 | 0.163 | 0.162 | +24.8% | +26.8% | +0.76 | +0.78 | 96.6% | 3.4% | brier_not_improved; clv_beat_worse |

## Market

| Bucket | Rows | Train | Holdout | Decision | Brier Gain | Base Brier | Opp Brier | Base ROI | Opp ROI | Base CLV | Opp CLV | PA Cov | BF Cov | Reasons |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| batter_total_bases | 31530 | 30634 | 896 | keep_baseline | +0.003 | 0.168 | 0.165 | +33.7% | +50.6% | +1.48 | +1.34 | 100.0% | 0.0% | avg_clv_worse; clv_beat_worse |
| batter_hits | 31687 | 30907 | 780 | keep_baseline | +0.000 | 0.193 | 0.193 | +10.1% | +11.6% | +0.29 | +0.31 | 100.0% | 0.0% | brier_not_improved |
| batter_home_runs | 12375 | 12017 | 358 | keep_baseline | +0.002 | 0.055 | 0.053 | +114.7% | +70.4% | +0.69 | +0.43 | 100.0% | 0.0% | roi_worse; avg_clv_worse; clv_beat_worse |
| pitcher_strikeouts | 2693 | 2599 | 94 | keep_baseline | -0.000 | 0.266 | 0.267 | -100.0% | -78.8% | +4.46 | +4.72 | 0.0% | 100.0% | brier_not_improved; selected_rows_too_small |

## Market And Side

| Bucket | Rows | Train | Holdout | Decision | Brier Gain | Base Brier | Opp Brier | Base ROI | Opp ROI | Base CLV | Opp CLV | PA Cov | BF Cov | Reasons |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| batter_total_bases|over | 28052 | 27246 | 806 | keep_baseline | +0.006 | 0.160 | 0.154 | +43.2% | +63.9% | +1.76 | +1.44 | 100.0% | 0.0% | avg_clv_worse; clv_beat_worse |
| batter_hits|over | 25128 | 24527 | 601 | keep_baseline | +0.001 | 0.179 | 0.178 | +10.2% | +17.9% | +0.63 | +0.66 | 100.0% | 0.0% | brier_not_improved; clv_beat_worse |
| batter_home_runs|over | 12375 | 12017 | 358 | keep_baseline | +0.002 | 0.055 | 0.053 | +114.7% | +70.4% | +0.69 | +0.43 | 100.0% | 0.0% | roi_worse; avg_clv_worse; clv_beat_worse |
| batter_hits|under | 6559 | 6380 | 179 | keep_baseline | +0.001 | 0.238 | 0.238 | +23.3% | +24.2% | -1.03 | -0.11 | 100.0% | 0.0% | brier_not_improved |
| batter_total_bases|under | 3478 | 3388 | 90 | opportunity_helped | +0.008 | 0.240 | 0.233 | +13.3% | +13.8% | -0.89 | -0.80 | 100.0% | 0.0% |  |
| pitcher_strikeouts|over | 1353 | 1306 | 47 | keep_baseline | -0.005 | 0.295 | 0.300 | - | -100.0% | - | -1.68 | 0.0% | 100.0% | brier_not_improved; selected_rows_too_small |
| pitcher_strikeouts|under | 1340 | 1293 | 47 | keep_baseline | -0.001 | 0.304 | 0.305 | -51.0% | -49.1% | +1.37 | +0.97 | 0.0% | 100.0% | brier_not_improved; avg_clv_worse; clv_beat_worse |

## Rule

Opportunity features are favored only when they improve holdout Brier and do not make ROI/CLV worse on enough selected rows.
