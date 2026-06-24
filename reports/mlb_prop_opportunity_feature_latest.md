# MLB Prop Opportunity Feature Report

- Generated UTC: 2026-06-24T14:27:13Z
- Source: features.mlb_prop_market_training_examples
- Date range: 2026-05-31 to 2026-06-22
- Rows: 102734
- Unique dates: 23
- Holdout days: 28
- Minimum Brier gain: 0.001

This is a holdout diagnostic. It does not reopen bankroll buckets by itself.

## Projection Accuracy

| Metric | Rows | Coverage | MAE | RMSE | Bias | Pred Avg | Actual Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hitters PA | 98965 | 100.0% | 0.688 | 0.910 | -0.103 | 3.913 | 4.015 |
| Pitch Count Proxy | 3769 | 100.0% | 11.404 | 15.398 | +0.592 | 88.948 | 88.356 |
| Pitcher BF | 3769 | 100.0% | 2.962 | 3.999 | +0.154 | 23.103 | 22.950 |
| batter_hits PA | 40838 | 100.0% | 0.691 | 0.914 | -0.110 | 3.893 | 4.003 |
| batter_home_runs PA | 16445 | 100.0% | 0.691 | 0.914 | -0.096 | 3.888 | 3.984 |
| batter_total_bases PA | 41682 | 100.0% | 0.682 | 0.905 | -0.098 | 3.941 | 4.040 |

## Hitter Opportunity Breakdowns

These rows isolate confirmed lineup, batting-order, projected-PA, and pinch-hit risk effects. `Low-PA Miss` means the model expected at least 3.8 PA and the hitter finished with 2 or fewer.

| Level | Bucket | Rows | PA Cov | PA MAE | PA Bias | Avg PA | Actual PA | Low-PA | Low-PA Miss | ROI | CLV Beat |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batting_order | order_6_9 | 36275 | 100.0% | 0.783 | -0.031 | 3.55 | 3.58 | 12.7% | 3.3% | -21.9% | 33.4% |
| batting_order | order_3_5 | 36039 | 100.0% | 0.602 | -0.163 | 4.00 | 4.17 | 3.0% | 1.5% | -15.5% | 37.9% |
| batting_order | order_1_2 | 26651 | 100.0% | 0.673 | -0.118 | 4.29 | 4.41 | 3.3% | 2.0% | -6.3% | 39.0% |
| confirmed_lineup | confirmed_lineup | 98965 | 100.0% | 0.688 | -0.103 | 3.91 | 4.02 | 6.6% | 2.3% | -15.4% | 36.6% |
| market_confirmed_lineup | batter_total_bases|confirmed_lineup | 41682 | 100.0% | 0.682 | -0.098 | 3.94 | 4.04 | 6.3% | 2.3% | -16.3% | 36.9% |
| market_confirmed_lineup | batter_hits|confirmed_lineup | 40838 | 100.0% | 0.691 | -0.110 | 3.89 | 4.00 | 6.8% | 2.3% | -10.5% | 37.3% |
| market_confirmed_lineup | batter_home_runs|confirmed_lineup | 16445 | 100.0% | 0.691 | -0.096 | 3.89 | 3.98 | 7.0% | 2.3% | -25.2% | 34.1% |
| pinch_hit_risk | pinch_low | 62690 | 100.0% | 0.632 | -0.144 | 4.13 | 4.27 | 3.1% | 1.7% | -11.6% | 38.4% |
| pinch_hit_risk | pinch_medium | 36275 | 100.0% | 0.783 | -0.031 | 3.55 | 3.58 | 12.7% | 3.3% | -21.9% | 33.4% |
| projected_pa | projected_pa_3_8_to_4_3 | 43787 | 100.0% | 0.588 | -0.063 | 4.08 | 4.14 | 4.3% | 4.3% | -12.3% | 36.6% |
| projected_pa | projected_pa_3_2_to_3_7 | 24796 | 100.0% | 0.782 | -0.203 | 3.49 | 3.69 | 10.5% | 0.0% | -19.8% | 34.7% |
| projected_pa | projected_pa_4_4_plus | 20704 | 100.0% | 0.636 | +0.160 | 4.55 | 4.39 | 2.0% | 2.0% | -15.2% | 39.3% |
| projected_pa | projected_pa_under_3_2 | 9678 | 100.0% | 1.007 | -0.588 | 2.89 | 3.48 | 17.4% | 0.0% | -18.1% | 35.1% |

## Overall

| Bucket | Rows | Train | Holdout | Decision | Brier Gain | Base Brier | Opp Brier | Base ROI | Opp ROI | Base CLV | Opp CLV | PA Cov | BF Cov | Reasons |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| overall | 102734 | 96107 | 6627 | keep_baseline | -0.000 | 0.141 | 0.141 | +0.5% | -0.2% | -0.09 | -0.09 | 96.3% | 3.7% | brier_not_improved; roi_worse; clv_beat_worse |

## Market

| Bucket | Rows | Train | Holdout | Decision | Brier Gain | Base Brier | Opp Brier | Base ROI | Opp ROI | Base CLV | Opp CLV | PA Cov | BF Cov | Reasons |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| batter_total_bases | 41682 | 38944 | 2738 | keep_baseline | -0.001 | 0.140 | 0.141 | -1.2% | +2.8% | +0.08 | -0.03 | 100.0% | 0.0% | brier_not_improved; avg_clv_worse; clv_beat_worse |
| batter_hits | 40838 | 38312 | 2526 | keep_baseline | -0.000 | 0.173 | 0.173 | +6.2% | +4.5% | -0.24 | -0.29 | 100.0% | 0.0% | brier_not_improved; roi_worse; avg_clv_worse; clv_beat_worse |
| batter_home_runs | 16445 | 15356 | 1089 | keep_baseline | -0.000 | 0.037 | 0.037 | -34.7% | -47.7% | -0.10 | -0.11 | 100.0% | 0.0% | brier_not_improved; roi_worse; avg_clv_worse; clv_beat_worse |
| pitcher_strikeouts | 3769 | 3495 | 274 | keep_baseline | -0.000 | 0.260 | 0.260 | -38.2% | -20.0% | -0.33 | -0.28 | 0.0% | 100.0% | brier_not_improved |

## Market And Side

| Bucket | Rows | Train | Holdout | Decision | Brier Gain | Base Brier | Opp Brier | Base ROI | Opp ROI | Base CLV | Opp CLV | PA Cov | BF Cov | Reasons |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| batter_total_bases|over | 37202 | 34741 | 2461 | keep_baseline | -0.002 | 0.126 | 0.128 | -1.2% | -2.0% | +0.11 | -0.08 | 100.0% | 0.0% | brier_not_improved; roi_worse; avg_clv_worse; clv_beat_worse |
| batter_hits|over | 32138 | 30173 | 1965 | keep_baseline | +0.000 | 0.153 | 0.152 | +11.2% | +15.0% | -0.24 | -0.15 | 100.0% | 0.0% | brier_not_improved |
| batter_home_runs|over | 16445 | 15356 | 1089 | keep_baseline | -0.000 | 0.037 | 0.037 | -34.7% | -47.7% | -0.10 | -0.11 | 100.0% | 0.0% | brier_not_improved; roi_worse; avg_clv_worse; clv_beat_worse |
| batter_hits|under | 8700 | 8139 | 561 | keep_baseline | +0.002 | 0.234 | 0.233 | +34.0% | +1.6% | -1.27 | +0.61 | 100.0% | 0.0% | roi_worse |
| batter_total_bases|under | 4480 | 4203 | 277 | keep_baseline | -0.020 | 0.255 | 0.274 | +86.9% | -29.7% | +0.00 | +0.03 | 100.0% | 0.0% | brier_not_improved |
| pitcher_strikeouts|over | 1891 | 1754 | 137 | keep_baseline | -0.019 | 0.277 | 0.296 | -27.7% | +1.9% | -0.00 | -0.26 | 0.0% | 100.0% | brier_not_improved; avg_clv_worse; clv_beat_worse |
| pitcher_strikeouts|under | 1878 | 1741 | 137 | keep_baseline | -0.023 | 0.278 | 0.301 | -46.1% | -42.9% | -0.03 | -0.69 | 0.0% | 100.0% | brier_not_improved; avg_clv_worse; clv_beat_worse |

## Rule

Opportunity features are favored only when they improve holdout Brier and do not make ROI/CLV worse on enough selected rows.
