# MLB Prop Probability Shadow - 2026-06-07

## Scope

- Locked side rows: 12085
- Date range: 2026-05-31 to 2026-06-05
- Unique graded dates: 6
- Minimum EV for simulated picks: 0.020
- Shadow winner: model_plus_prior

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 12085 | 0.158 | 0.3% | 2386 | -2.9% | 42.7% | +0.17 |
| model_plus_prior | 12085 | 0.157 | 0.5% | 1856 | 0.8% | 43.2% | +0.19 |
| market_no_vig | 12085 | 0.153 | -0.7% | 275 | 168.6% | 39.2% | +0.26 |
| direct_side_model | 12085 | 0.150 | 0.4% | 2464 | 62.7% | 38.4% | +0.16 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 4344 | keep_model_only_roi | 0.148 | 0.146 | 53.0% | 46.4% | 44.9% |
| batter_hits|under | 1072 | keep_model_only | 0.240 | 0.240 | -8.9% | -8.2% | 41.8% |
| batter_home_runs|over | 1738 | keep_model_only | 0.061 | 0.061 | -63.1% | -51.6% | 53.3% |
| batter_total_bases|over | 3891 | keep_model_only | 0.165 | 0.164 | -3.6% | 13.5% | 44.2% |
| batter_total_bases|under | 601 | keep_model_only | 0.244 | 0.244 | -7.4% | -9.2% | 36.7% |
| pitcher_strikeouts|over | 226 | use_model_plus_prior | 0.248 | 0.245 | -36.2% | -35.0% | 49.3% |
| pitcher_strikeouts|under | 213 | use_model_plus_prior | 0.249 | 0.247 | 1.0% | 4.6% | 43.6% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 1647 | keep_model_only | 0.031 | 0.031 | 69.7% | 51.2% | 45.5% |
| batter_hits|over|common | 2697 | keep_model_only_calibration | 0.220 | 0.216 | 3.7% | 38.9% | 44.1% |
| batter_hits|under|common | 1072 | keep_model_only | 0.240 | 0.240 | -8.9% | -8.2% | 41.8% |
| batter_home_runs|over|alt_tail | 860 | keep_model_only | 0.013 | 0.013 | -100.0% | -100.0% | 64.3% |
| batter_home_runs|over|common | 878 | keep_model_only | 0.109 | 0.109 | -23.6% | -4.4% | 46.0% |
| batter_total_bases|over|alt_tail | 1694 | keep_model_only | 0.103 | 0.103 | -0.8% | 13.3% | 43.2% |
| batter_total_bases|over|common | 2197 | keep_model_only | 0.212 | 0.212 | -12.6% | 14.1% | 47.7% |
| batter_total_bases|under|common | 601 | keep_model_only | 0.244 | 0.244 | -7.4% | -9.2% | 36.7% |
| pitcher_strikeouts|over|common | 226 | use_model_plus_prior | 0.248 | 0.245 | -36.2% | -35.0% | 49.3% |
| pitcher_strikeouts|under|common | 213 | use_model_plus_prior | 0.249 | 0.247 | 1.0% | 4.6% | 43.6% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | true | passes |
| market_no_vig | false | clv_beat_rate_not_improved |
| direct_side_model | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
