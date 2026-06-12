# MLB Prop Probability Shadow - 2026-06-06

## Scope

- Locked side rows: 5918
- Date range: 2026-05-31 to 2026-06-04
- Unique graded dates: 5
- Minimum EV for simulated picks: 0.020
- Shadow winner: model_only

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 5918 | 0.162 | 1.5% | 1454 | 0.0% | 37.9% | +0.17 |
| model_plus_prior | 5918 | 0.161 | 1.7% | 1154 | 1.6% | 40.1% | +0.21 |
| market_no_vig | 5918 | 0.159 | 0.1% | 70 | 154.3% | 12.0% | -0.51 |
| direct_side_model | 5918 | 0.156 | 1.6% | 1272 | 40.4% | 37.8% | +0.16 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 2086 | keep_model_only | 0.150 | 0.149 | 79.3% | 40.2% | 23.6% |
| batter_hits|under | 601 | keep_model_only | 0.237 | 0.236 | -11.1% | -9.7% | 42.3% |
| batter_home_runs|over | 822 | keep_model_only | 0.061 | 0.061 | -66.2% | -58.2% | 40.4% |
| batter_total_bases|over | 1843 | keep_model_only | 0.170 | 0.170 | -0.5% | 29.2% | 33.3% |
| batter_total_bases|under | 357 | keep_model_only_roi | 0.245 | 0.244 | -7.3% | -8.0% | 45.3% |
| pitcher_strikeouts|over | 111 | use_model_plus_prior | 0.243 | 0.240 | -2.3% | -0.8% | 50.0% |
| pitcher_strikeouts|under | 98 | use_model_plus_prior | 0.246 | 0.244 | 10.0% | 17.5% | 40.0% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 762 | keep_model_only | 0.035 | 0.035 | 117.4% | 45.0% | 17.1% |
| batter_hits|over|common | 1324 | keep_model_only | 0.216 | 0.215 | 11.8% | 30.9% | 35.0% |
| batter_hits|under|common | 601 | keep_model_only | 0.237 | 0.236 | -11.1% | -9.7% | 42.3% |
| batter_home_runs|over|alt_tail | 395 | keep_model_only | 0.010 | 0.010 | -100.0% | -100.0% | 52.9% |
| batter_home_runs|over|common | 427 | keep_model_only | 0.109 | 0.109 | -34.5% | -19.7% | 35.0% |
| batter_total_bases|over|alt_tail | 792 | keep_model_only | 0.105 | 0.105 | -1.5% | 31.9% | 21.4% |
| batter_total_bases|over|common | 1051 | keep_model_only | 0.219 | 0.219 | 1.8% | 22.0% | 48.5% |
| batter_total_bases|under|common | 357 | keep_model_only_roi | 0.245 | 0.244 | -7.3% | -8.0% | 45.3% |
| pitcher_strikeouts|over|common | 111 | use_model_plus_prior | 0.243 | 0.240 | -2.3% | -0.8% | 50.0% |
| pitcher_strikeouts|under|common | 98 | use_model_plus_prior | 0.246 | 0.244 | 10.0% | 17.5% | 40.0% |

## Shadow Winner

No shadow variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |
| direct_side_model | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
