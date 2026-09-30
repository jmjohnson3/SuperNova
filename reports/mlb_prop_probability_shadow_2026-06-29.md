# MLB Prop Probability Shadow - 2026-06-29

## Scope

- Locked side rows: 126502
- Date range: 2026-05-31 to 2026-06-27
- Unique graded dates: 28
- Minimum EV for simulated picks: 0.020
- Shadow winner: direct_side_model

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 126502 | 0.162 | -1.9% | 29996 | -16.4% | 38.5% | +0.13 |
| model_plus_prior | 126502 | 0.161 | -1.8% | 25705 | -13.3% | 38.3% | +0.13 |
| market_no_vig | 126502 | 0.150 | 1.3% | 13335 | 130.5% | 48.6% | +0.28 |
| direct_side_model | 126502 | 0.143 | -0.1% | 26903 | 79.9% | 40.6% | +0.20 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 39164 | keep_model_only | 0.176 | 0.175 | -11.7% | -3.1% | 38.2% |
| batter_hits|under | 10718 | keep_model_only | 0.233 | 0.233 | -5.0% | -5.8% | 40.3% |
| batter_home_runs|over | 20405 | keep_model_only | 0.060 | 0.060 | -28.3% | -23.5% | 37.2% |
| batter_total_bases|over | 46040 | keep_model_only | 0.161 | 0.160 | -16.9% | -14.9% | 38.2% |
| batter_total_bases|under | 5418 | keep_model_only | 0.243 | 0.243 | -3.7% | -3.8% | 39.1% |
| pitcher_strikeouts|over | 2385 | keep_model_only_roi | 0.245 | 0.244 | -16.8% | -17.9% | 35.9% |
| pitcher_strikeouts|under | 2372 | keep_model_only_roi | 0.245 | 0.244 | 5.1% | 4.9% | 50.8% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 8143 | keep_model_only | 0.029 | 0.029 | -9.1% | -7.0% | 33.6% |
| batter_hits|over|common | 31021 | keep_model_only_clv | 0.214 | 0.213 | -13.1% | -1.9% | 39.7% |
| batter_hits|under|common | 10718 | keep_model_only | 0.233 | 0.233 | -5.0% | -5.8% | 40.3% |
| batter_home_runs|over|alt_tail | 10187 | keep_model_only | 0.009 | 0.009 | -31.8% | -26.3% | 37.7% |
| batter_home_runs|over|common | 10218 | keep_model_only | 0.111 | 0.110 | -17.9% | -16.6% | 35.9% |
| batter_total_bases|over|alt_tail | 20346 | keep_model_only | 0.099 | 0.098 | -20.2% | -17.0% | 37.6% |
| batter_total_bases|over|common | 25694 | keep_model_only | 0.210 | 0.210 | -11.2% | -11.3% | 39.2% |
| batter_total_bases|under|common | 5418 | keep_model_only | 0.243 | 0.243 | -3.7% | -3.8% | 39.1% |
| pitcher_strikeouts|over|common | 2385 | keep_model_only_roi | 0.245 | 0.244 | -16.8% | -17.9% | 35.9% |
| pitcher_strikeouts|under|common | 2372 | keep_model_only_roi | 0.245 | 0.244 | 5.1% | 4.9% | 50.8% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved, clv_beat_rate_not_improved, avg_clv_price_not_improved |
| market_no_vig | true | passes |
| direct_side_model | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
