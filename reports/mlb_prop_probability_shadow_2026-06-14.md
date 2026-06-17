# MLB Prop Probability Shadow - 2026-06-14

## Scope

- Locked side rows: 48398
- Date range: 2026-05-31 to 2026-06-12
- Unique graded dates: 13
- Minimum EV for simulated picks: 0.020
- Shadow winner: model_only

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 48398 | 0.155 | -1.1% | 8983 | -10.7% | 41.5% | +0.17 |
| model_plus_prior | 48398 | 0.154 | -0.7% | 6799 | -4.5% | 41.2% | +0.18 |
| market_no_vig | 48398 | 0.151 | -0.6% | 2991 | 214.6% | 40.5% | +0.24 |
| direct_side_model | 48398 | 0.140 | -0.2% | 10891 | 62.6% | 40.3% | +0.16 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 16853 | keep_model_only_clv | 0.150 | 0.148 | 2.9% | 13.4% | 44.2% |
| batter_hits|under | 3966 | keep_model_only | 0.234 | 0.233 | -4.8% | -3.1% | 41.6% |
| batter_home_runs|over | 7298 | keep_model_only | 0.062 | 0.062 | -32.3% | -12.4% | 42.0% |
| batter_total_bases|over | 16542 | keep_model_only | 0.162 | 0.162 | -17.2% | -12.1% | 40.5% |
| batter_total_bases|under | 2132 | keep_model_only | 0.244 | 0.244 | -3.7% | -4.0% | 38.9% |
| pitcher_strikeouts|over | 810 | keep_model_only_roi | 0.250 | 0.249 | -21.2% | -21.5% | 39.8% |
| pitcher_strikeouts|under | 797 | keep_model_only | 0.247 | 0.247 | 25.0% | 25.9% | 42.7% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 5702 | keep_model_only | 0.029 | 0.029 | 16.6% | 24.5% | 45.3% |
| batter_hits|over|common | 11151 | keep_model_only_clv | 0.212 | 0.208 | -25.2% | 4.9% | 43.4% |
| batter_hits|under|common | 3966 | keep_model_only | 0.234 | 0.233 | -4.8% | -3.1% | 41.6% |
| batter_home_runs|over|alt_tail | 3640 | keep_model_only | 0.008 | 0.008 | -38.4% | -10.2% | 40.9% |
| batter_home_runs|over|common | 3658 | keep_model_only | 0.115 | 0.115 | -19.9% | -16.1% | 43.9% |
| batter_total_bases|over|alt_tail | 7254 | keep_model_only | 0.101 | 0.100 | -20.0% | -9.6% | 38.8% |
| batter_total_bases|over|common | 9288 | keep_model_only | 0.210 | 0.210 | -8.4% | -17.8% | 44.3% |
| batter_total_bases|under|common | 2132 | keep_model_only | 0.244 | 0.244 | -3.7% | -4.0% | 38.9% |
| pitcher_strikeouts|over|common | 810 | keep_model_only_roi | 0.250 | 0.249 | -21.2% | -21.5% | 39.8% |
| pitcher_strikeouts|under|common | 797 | keep_model_only | 0.247 | 0.247 | 25.0% | 25.9% | 42.7% |

## Shadow Winner

No shadow variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | clv_beat_rate_not_improved |
| market_no_vig | false | clv_beat_rate_not_improved |
| direct_side_model | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
