# MLB Prop Probability Shadow - 2026-07-10

## Scope

- Locked side rows: 197255
- Date range: 2026-05-31 to 2026-07-09
- Unique graded dates: 40
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 197255 | 0.165 | -2.4% | 42715 | -17.4% | 39.8% | +0.15 |
| model_plus_prior | 197255 | 0.164 | -2.3% | 36665 | -13.0% | 39.9% | +0.15 |
| market_no_vig | 197255 | 0.152 | 1.4% | 20214 | 136.5% | 50.5% | +0.34 |
| direct_side_model | 197255 | 0.207 | -2.6% | 92708 | -20.4% | 36.8% | +0.06 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 58924 | keep_model_only | 0.184 | 0.183 | -12.7% | -4.0% | 40.8% |
| batter_hits|under | 16882 | keep_model_only | 0.234 | 0.233 | -3.7% | 3.6% | 43.1% |
| batter_home_runs|over | 32194 | keep_model_only | 0.061 | 0.061 | -29.1% | -24.9% | 38.3% |
| batter_total_bases|over | 72846 | keep_model_only | 0.162 | 0.161 | -18.0% | -13.9% | 39.9% |
| batter_total_bases|under | 8630 | keep_model_only | 0.243 | 0.242 | -3.4% | -3.3% | 39.3% |
| pitcher_strikeouts|over | 3896 | keep_model_only | 0.245 | 0.244 | -16.6% | -17.0% | 36.5% |
| pitcher_strikeouts|under | 3883 | keep_model_only | 0.245 | 0.244 | 6.0% | 5.3% | 48.6% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 9941 | keep_model_only | 0.027 | 0.027 | -9.8% | -7.6% | 33.6% |
| batter_hits|over|common | 48983 | keep_model_only | 0.216 | 0.215 | -13.8% | -3.3% | 42.2% |
| batter_hits|under|common | 16882 | keep_model_only | 0.234 | 0.233 | -3.7% | 3.6% | 43.1% |
| batter_home_runs|over|alt_tail | 16077 | keep_model_only | 0.010 | 0.010 | -31.8% | -26.2% | 37.5% |
| batter_home_runs|over|common | 16117 | keep_model_only | 0.112 | 0.112 | -23.4% | -22.6% | 39.7% |
| batter_total_bases|over|alt_tail | 32142 | keep_model_only | 0.099 | 0.099 | -20.2% | -12.3% | 39.1% |
| batter_total_bases|over|common | 40704 | keep_model_only | 0.211 | 0.210 | -16.1% | -15.4% | 40.7% |
| batter_total_bases|under|common | 8630 | keep_model_only | 0.243 | 0.242 | -3.4% | -3.3% | 39.3% |
| pitcher_strikeouts|over|common | 3896 | keep_model_only | 0.245 | 0.244 | -16.6% | -17.0% | 36.5% |
| pitcher_strikeouts|under|common | 3883 | keep_model_only | 0.245 | 0.244 | 6.0% | 5.3% | 48.6% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | false | brier_not_improved, roi_not_improved, clv_beat_rate_not_improved, avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
