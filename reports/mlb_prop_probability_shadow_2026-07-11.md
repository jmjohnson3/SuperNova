# MLB Prop Probability Shadow - 2026-07-11

## Scope

- Locked side rows: 202627
- Date range: 2026-05-31 to 2026-07-10
- Unique graded dates: 41
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 202627 | 0.165 | -2.4% | 43372 | -17.2% | 39.8% | +0.15 |
| model_plus_prior | 202627 | 0.165 | -2.3% | 37938 | -12.6% | 40.0% | +0.15 |
| market_no_vig | 202627 | 0.152 | 1.5% | 20630 | 136.7% | 50.5% | +0.34 |
| direct_side_model | 202627 | 0.245 | -10.3% | 73143 | 3.6% | 37.4% | +0.12 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 60435 | keep_model_only | 0.184 | 0.184 | -12.7% | -3.7% | 41.5% |
| batter_hits|under | 17351 | keep_model_only | 0.234 | 0.233 | -3.2% | 4.8% | 43.2% |
| batter_home_runs|over | 33094 | keep_model_only | 0.061 | 0.061 | -29.1% | -24.7% | 38.3% |
| batter_total_bases|over | 74884 | keep_model_only | 0.162 | 0.162 | -17.7% | -13.6% | 39.9% |
| batter_total_bases|under | 8868 | keep_model_only | 0.243 | 0.242 | -3.4% | -3.4% | 39.3% |
| pitcher_strikeouts|over | 4004 | keep_model_only | 0.245 | 0.244 | -15.1% | -16.1% | 36.1% |
| pitcher_strikeouts|under | 3991 | keep_model_only | 0.245 | 0.244 | 7.5% | 7.2% | 47.4% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 10083 | keep_model_only | 0.027 | 0.027 | -9.8% | -7.3% | 33.6% |
| batter_hits|over|common | 50352 | keep_model_only | 0.216 | 0.215 | -13.8% | -3.1% | 42.9% |
| batter_hits|under|common | 17351 | keep_model_only | 0.234 | 0.233 | -3.2% | 4.8% | 43.2% |
| batter_home_runs|over|alt_tail | 16527 | keep_model_only | 0.010 | 0.009 | -31.8% | -26.1% | 37.5% |
| batter_home_runs|over|common | 16567 | keep_model_only | 0.113 | 0.113 | -23.4% | -22.3% | 39.7% |
| batter_total_bases|over|alt_tail | 33042 | keep_model_only | 0.100 | 0.100 | -20.1% | -12.1% | 39.0% |
| batter_total_bases|over|common | 41842 | keep_model_only | 0.211 | 0.211 | -15.7% | -15.1% | 40.7% |
| batter_total_bases|under|common | 8868 | keep_model_only | 0.243 | 0.242 | -3.4% | -3.4% | 39.3% |
| pitcher_strikeouts|over|common | 4004 | keep_model_only | 0.245 | 0.244 | -15.1% | -16.1% | 36.1% |
| pitcher_strikeouts|under|common | 3991 | keep_model_only | 0.245 | 0.244 | 7.5% | 7.2% | 47.4% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | false | brier_not_improved, clv_beat_rate_not_improved, avg_clv_price_not_improved, calibration_worse |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
