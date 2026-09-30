# MLB Prop Probability Shadow - 2026-07-19

## Scope

- Locked side rows: 229961
- Date range: 2026-06-01 to 2026-07-18
- Unique graded dates: 45
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 229961 | 0.165 | -2.6% | 45625 | -17.7% | 39.6% | +0.14 |
| model_plus_prior | 229961 | 0.165 | -2.4% | 40031 | -12.5% | 39.9% | +0.15 |
| market_no_vig | 229961 | 0.152 | 1.5% | 22605 | 140.9% | 50.0% | +0.34 |
| direct_side_model | 229961 | 0.148 | -0.4% | 89320 | 7.1% | 37.4% | +0.08 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 67994 | keep_model_only | 0.186 | 0.186 | -12.7% | 1.0% | 41.1% |
| batter_hits|under | 19627 | keep_model_only | 0.235 | 0.234 | -4.2% | 2.7% | 44.8% |
| batter_home_runs|over | 37748 | keep_model_only | 0.061 | 0.060 | -29.5% | -26.2% | 38.6% |
| batter_total_bases|over | 85456 | keep_model_only | 0.161 | 0.161 | -18.2% | -13.8% | 39.7% |
| batter_total_bases|under | 10037 | keep_model_only | 0.242 | 0.242 | -3.7% | -4.0% | 39.0% |
| pitcher_strikeouts|over | 4552 | keep_model_only | 0.246 | 0.246 | -17.6% | -17.8% | 36.9% |
| pitcher_strikeouts|under | 4547 | keep_model_only | 0.246 | 0.246 | 4.7% | 4.8% | 48.8% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 10600 | keep_model_only | 0.027 | 0.027 | -9.8% | -4.0% | 33.4% |
| batter_hits|over|common | 57394 | use_model_plus_prior | 0.216 | 0.215 | -13.8% | 1.9% | 42.4% |
| batter_hits|under|common | 19627 | keep_model_only | 0.235 | 0.234 | -4.2% | 2.7% | 44.8% |
| batter_home_runs|over|alt_tail | 18860 | keep_model_only | 0.009 | 0.009 | -31.8% | -27.1% | 37.5% |
| batter_home_runs|over|common | 18888 | keep_model_only | 0.112 | 0.112 | -25.4% | -24.9% | 40.1% |
| batter_total_bases|over|alt_tail | 37716 | keep_model_only | 0.099 | 0.099 | -20.3% | -12.1% | 38.7% |
| batter_total_bases|over|common | 47740 | keep_model_only | 0.210 | 0.210 | -16.5% | -15.4% | 40.5% |
| batter_total_bases|under|common | 10037 | keep_model_only | 0.242 | 0.242 | -3.7% | -4.0% | 39.0% |
| pitcher_strikeouts|over|common | 4552 | keep_model_only | 0.246 | 0.246 | -17.6% | -17.8% | 36.9% |
| pitcher_strikeouts|under|common | 4547 | keep_model_only | 0.246 | 0.246 | 4.7% | 4.8% | 48.8% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
