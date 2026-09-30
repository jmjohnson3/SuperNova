# MLB Prop Probability Shadow - 2026-07-16

## Scope

- Locked side rows: 218392
- Date range: 2026-06-01 to 2026-07-12
- Unique graded dates: 42
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 218392 | 0.165 | -2.5% | 45236 | -17.6% | 39.6% | +0.14 |
| model_plus_prior | 218392 | 0.165 | -2.4% | 39166 | -12.6% | 39.8% | +0.15 |
| market_no_vig | 218392 | 0.152 | 1.4% | 21822 | 138.8% | 50.1% | +0.34 |
| direct_side_model | 218392 | 0.244 | -11.0% | 90529 | 2.7% | 37.9% | +0.11 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 64880 | keep_model_only | 0.186 | 0.185 | -12.7% | -0.8% | 40.8% |
| batter_hits|under | 18619 | keep_model_only | 0.235 | 0.234 | -4.1% | 4.8% | 44.3% |
| batter_home_runs|over | 35779 | keep_model_only | 0.061 | 0.060 | -29.2% | -25.3% | 38.3% |
| batter_total_bases|over | 80974 | keep_model_only | 0.161 | 0.161 | -18.2% | -14.0% | 39.6% |
| batter_total_bases|under | 9495 | keep_model_only | 0.242 | 0.242 | -3.7% | -3.4% | 39.2% |
| pitcher_strikeouts|over | 4325 | keep_model_only | 0.246 | 0.246 | -17.6% | -17.8% | 36.5% |
| pitcher_strikeouts|under | 4320 | keep_model_only | 0.247 | 0.246 | 4.6% | 4.7% | 49.2% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 10463 | keep_model_only | 0.027 | 0.027 | -9.8% | -4.3% | 33.4% |
| batter_hits|over|common | 54417 | keep_model_only_clv | 0.216 | 0.215 | -13.8% | -0.1% | 42.2% |
| batter_hits|under|common | 18619 | keep_model_only | 0.235 | 0.234 | -4.1% | 4.8% | 44.3% |
| batter_home_runs|over|alt_tail | 17876 | keep_model_only | 0.009 | 0.009 | -31.8% | -26.8% | 37.6% |
| batter_home_runs|over|common | 17903 | keep_model_only | 0.112 | 0.112 | -23.5% | -22.7% | 39.4% |
| batter_total_bases|over|alt_tail | 35744 | keep_model_only | 0.099 | 0.099 | -20.3% | -12.2% | 38.7% |
| batter_total_bases|over|common | 45230 | keep_model_only | 0.211 | 0.210 | -16.5% | -15.5% | 40.4% |
| batter_total_bases|under|common | 9495 | keep_model_only | 0.242 | 0.242 | -3.7% | -3.4% | 39.2% |
| pitcher_strikeouts|over|common | 4325 | keep_model_only | 0.246 | 0.246 | -17.6% | -17.8% | 36.5% |
| pitcher_strikeouts|under|common | 4320 | keep_model_only | 0.247 | 0.246 | 4.6% | 4.7% | 49.2% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | false | brier_not_improved, clv_beat_rate_not_improved, avg_clv_price_not_improved, calibration_worse |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
