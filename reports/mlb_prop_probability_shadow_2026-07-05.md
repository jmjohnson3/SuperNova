# MLB Prop Probability Shadow - 2026-07-05

## Scope

- Locked side rows: 165867
- Date range: 2026-05-31 to 2026-07-04
- Unique graded dates: 35
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 165867 | 0.165 | -2.3% | 37596 | -16.9% | 39.3% | +0.13 |
| model_plus_prior | 165867 | 0.164 | -2.1% | 32229 | -13.7% | 39.2% | +0.13 |
| market_no_vig | 165867 | 0.152 | 1.3% | 17309 | 133.0% | 49.9% | +0.32 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 50154 | keep_model_only | 0.182 | 0.181 | -12.8% | -7.5% | 39.9% |
| batter_hits|under | 14204 | keep_model_only | 0.234 | 0.233 | -4.7% | 2.3% | 40.6% |
| batter_home_runs|over | 26908 | keep_model_only | 0.062 | 0.062 | -29.1% | -25.2% | 38.2% |
| batter_total_bases|over | 60875 | keep_model_only | 0.162 | 0.162 | -17.1% | -14.0% | 39.1% |
| batter_total_bases|under | 7243 | keep_model_only | 0.244 | 0.243 | -3.4% | -3.1% | 39.1% |
| pitcher_strikeouts|over | 3248 | keep_model_only | 0.245 | 0.245 | -15.2% | -16.0% | 34.2% |
| pitcher_strikeouts|under | 3235 | keep_model_only | 0.246 | 0.245 | 3.0% | 3.6% | 49.5% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 9141 | keep_model_only | 0.028 | 0.028 | -9.8% | -8.0% | 33.2% |
| batter_hits|over|common | 41013 | keep_model_only | 0.216 | 0.215 | -13.9% | -7.4% | 41.4% |
| batter_hits|under|common | 14204 | keep_model_only | 0.234 | 0.233 | -4.7% | 2.3% | 40.6% |
| batter_home_runs|over|alt_tail | 13437 | keep_model_only | 0.010 | 0.010 | -31.8% | -26.7% | 37.4% |
| batter_home_runs|over|common | 13471 | keep_model_only | 0.114 | 0.113 | -23.4% | -22.6% | 39.8% |
| batter_total_bases|over|alt_tail | 26850 | keep_model_only | 0.100 | 0.100 | -19.5% | -13.4% | 38.2% |
| batter_total_bases|over|common | 34025 | keep_model_only | 0.211 | 0.211 | -14.5% | -14.6% | 40.2% |
| batter_total_bases|under|common | 7243 | keep_model_only | 0.244 | 0.243 | -3.4% | -3.1% | 39.1% |
| pitcher_strikeouts|over|common | 3248 | keep_model_only | 0.245 | 0.245 | -15.2% | -16.0% | 34.2% |
| pitcher_strikeouts|under|common | 3235 | keep_model_only | 0.246 | 0.245 | 3.0% | 3.6% | 49.5% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved, clv_beat_rate_not_improved |
| market_no_vig | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
