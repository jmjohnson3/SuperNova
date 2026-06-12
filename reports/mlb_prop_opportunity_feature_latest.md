# MLB Prop Opportunity Feature Report

- Generated UTC: 2026-06-12T14:50:41Z
- Source: features.mlb_prop_market_training_examples
- Date range: 2026-05-31 to 2026-06-11
- Rows: 40380
- Unique dates: 12
- Holdout days: 28
- Minimum Brier gain: 0.001

This is a holdout diagnostic. It does not reopen bankroll buckets by itself.

## Projection Accuracy

| Metric | Rows | Coverage | MAE | RMSE | Bias | Pred Avg | Actual Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hitters PA | 39121 | 100.0% | 0.717 | 0.941 | -0.166 | 3.907 | 4.073 |
| Pitch Count Proxy | 1259 | 100.0% | 10.242 | 13.486 | +0.878 | 89.935 | 89.058 |
| Pitcher BF | 1259 | 100.0% | 2.660 | 3.503 | +0.228 | 23.360 | 23.132 |
| batter_hits PA | 17881 | 100.0% | 0.722 | 0.946 | -0.167 | 3.889 | 4.056 |
| batter_home_runs PA | 5970 | 100.0% | 0.719 | 0.940 | -0.165 | 3.883 | 4.048 |
| batter_total_bases PA | 15270 | 100.0% | 0.710 | 0.934 | -0.165 | 3.938 | 4.103 |

## Hitter Opportunity Breakdowns

These rows isolate confirmed lineup, batting-order, projected-PA, and pinch-hit risk effects. `Low-PA Miss` means the model expected at least 3.8 PA and the hitter finished with 2 or fewer.

| Level | Bucket | Rows | PA Cov | PA MAE | PA Bias | Avg PA | Actual PA | Low-PA | Low-PA Miss | ROI | CLV Beat |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batting_order | order_3_5 | 14349 | 100.0% | 0.636 | -0.220 | 4.01 | 4.23 | 3.2% | 1.5% | -10.1% | 38.3% |
| batting_order | order_6_9 | 14325 | 100.0% | 0.820 | -0.088 | 3.54 | 3.63 | 12.7% | 2.9% | -21.0% | 33.1% |
| batting_order | order_1_2 | 10447 | 100.0% | 0.685 | -0.199 | 4.28 | 4.47 | 3.6% | 1.6% | -10.6% | 37.4% |
| confirmed_lineup | confirmed_lineup | 39121 | 100.0% | 0.717 | -0.166 | 3.91 | 4.07 | 6.8% | 2.1% | -14.2% | 36.2% |
| market_confirmed_lineup | batter_hits|confirmed_lineup | 17881 | 100.0% | 0.722 | -0.167 | 3.89 | 4.06 | 7.0% | 2.0% | -9.0% | 37.2% |
| market_confirmed_lineup | batter_total_bases|confirmed_lineup | 15270 | 100.0% | 0.710 | -0.165 | 3.94 | 4.10 | 6.4% | 2.1% | -14.0% | 36.5% |
| market_confirmed_lineup | batter_home_runs|confirmed_lineup | 5970 | 100.0% | 0.719 | -0.165 | 3.88 | 4.05 | 7.0% | 2.0% | -30.6% | 32.6% |
| pinch_hit_risk | pinch_low | 24796 | 100.0% | 0.657 | -0.211 | 4.12 | 4.33 | 3.3% | 1.6% | -10.3% | 37.9% |
| pinch_hit_risk | pinch_medium | 14325 | 100.0% | 0.820 | -0.088 | 3.54 | 3.63 | 12.7% | 2.9% | -21.0% | 33.1% |
| projected_pa | projected_pa_3_8_to_4_3 | 17832 | 100.0% | 0.618 | -0.128 | 4.07 | 4.20 | 4.0% | 4.0% | -9.6% | 36.9% |
| projected_pa | projected_pa_3_2_to_3_7 | 9959 | 100.0% | 0.856 | -0.216 | 3.48 | 3.70 | 12.8% | 0.0% | -19.5% | 34.2% |
| projected_pa | projected_pa_4_4_plus | 7733 | 100.0% | 0.624 | +0.056 | 4.54 | 4.49 | 1.2% | 1.2% | -19.3% | 38.7% |
| projected_pa | projected_pa_under_3_2 | 3597 | 100.0% | 1.022 | -0.693 | 2.90 | 3.59 | 15.7% | 0.0% | -11.8% | 33.1% |

## Overall

| Bucket | Rows | Train | Holdout | Decision | Brier Gain | Base Brier | Opp Brier | Base ROI | Opp ROI | Base CLV | Opp CLV | PA Cov | BF Cov | Reasons |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| overall | 40380 | 38710 | 1670 | keep_baseline | -0.000 | 0.166 | 0.166 | +8.3% | +17.1% | -0.25 | -0.13 | 96.9% | 3.1% | brier_not_improved |

## Market

| Bucket | Rows | Train | Holdout | Decision | Brier Gain | Base Brier | Opp Brier | Base ROI | Opp ROI | Base CLV | Opp CLV | PA Cov | BF Cov | Reasons |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| batter_total_bases | 15270 | 14570 | 700 | keep_baseline | -0.001 | 0.174 | 0.174 | +29.4% | +30.3% | -0.15 | -0.09 | 100.0% | 0.0% | brier_not_improved |
| batter_hits | 17881 | 17259 | 622 | keep_baseline | -0.001 | 0.181 | 0.182 | +1.5% | +2.3% | -0.24 | -0.29 | 100.0% | 0.0% | brier_not_improved; avg_clv_worse; clv_beat_worse |
| batter_home_runs | 5970 | 5706 | 264 | keep_baseline | -0.001 | 0.076 | 0.077 | +3.9% | -3.7% | -0.31 | -0.25 | 100.0% | 0.0% | brier_not_improved; roi_worse; clv_beat_worse |
| pitcher_strikeouts | 1259 | 1175 | 84 | keep_baseline | -0.000 | 0.227 | 0.227 | +35.3% | +37.8% | +2.07 | +2.02 | 0.0% | 100.0% | brier_not_improved; avg_clv_worse; clv_beat_worse |

## Market And Side

| Bucket | Rows | Train | Holdout | Decision | Brier Gain | Base Brier | Opp Brier | Base ROI | Opp ROI | Base CLV | Opp CLV | PA Cov | BF Cov | Reasons |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| batter_total_bases|over | 13512 | 12898 | 614 | keep_baseline | -0.003 | 0.164 | 0.166 | +35.4% | +28.8% | -0.20 | +0.03 | 100.0% | 0.0% | brier_not_improved; roi_worse |
| batter_hits|over | 14608 | 14135 | 473 | keep_baseline | -0.003 | 0.160 | 0.163 | +6.3% | +10.3% | -0.31 | -0.15 | 100.0% | 0.0% | brier_not_improved |
| batter_home_runs|over | 5970 | 5706 | 264 | keep_baseline | -0.001 | 0.076 | 0.077 | +3.9% | -3.7% | -0.31 | -0.25 | 100.0% | 0.0% | brier_not_improved; roi_worse; clv_beat_worse |
| batter_hits|under | 3273 | 3124 | 149 | keep_baseline | -0.004 | 0.246 | 0.250 | -13.6% | -2.9% | +0.49 | +1.03 | 100.0% | 0.0% | brier_not_improved |
| batter_total_bases|under | 1758 | 1672 | 86 | keep_baseline | -0.013 | 0.246 | 0.259 | +13.6% | -14.4% | +0.39 | +0.20 | 100.0% | 0.0% | brier_not_improved; roi_worse; avg_clv_worse; clv_beat_worse |
| pitcher_strikeouts|over | 636 | 594 | 42 | keep_baseline | -0.020 | 0.221 | 0.241 | +20.2% | -49.7% | +2.23 | +1.48 | 0.0% | 100.0% | brier_not_improved; roi_worse; avg_clv_worse; clv_beat_worse |
| pitcher_strikeouts|under | 623 | 581 | 42 | keep_baseline | -0.031 | 0.211 | 0.242 | +49.3% | +30.2% | +1.30 | +2.51 | 0.0% | 100.0% | brier_not_improved; roi_worse |

## Rule

Opportunity features are favored only when they improve holdout Brier and do not make ROI/CLV worse on enough selected rows.
