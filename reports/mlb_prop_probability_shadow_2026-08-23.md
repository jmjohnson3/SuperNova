# MLB Prop Probability Shadow - 2026-08-23

## Scope

- Locked side rows: 295467
- Date range: 2026-06-01 to 2026-07-30
- Unique graded dates: 57
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 295467 | 0.165 | -2.8% | 50188 | -19.3% | 39.3% | +0.13 |
| model_plus_prior | 295467 | 0.165 | -2.7% | 46880 | -11.0% | 39.6% | +0.14 |
| market_no_vig | 295467 | 0.152 | 1.4% | 28626 | 141.2% | 49.6% | +0.31 |
| direct_side_model | 295467 | 0.148 | -0.8% | 117030 | 5.9% | 37.4% | +0.08 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 86201 | use_model_plus_prior | 0.189 | 0.188 | -12.7% | 12.7% | 41.2% |
| batter_hits|under | 25189 | keep_model_only | 0.235 | 0.235 | -4.2% | 5.2% | 45.3% |
| batter_home_runs|over | 48878 | keep_model_only | 0.059 | 0.059 | -33.3% | -31.5% | 37.3% |
| batter_total_bases|over | 110520 | keep_model_only | 0.159 | 0.159 | -18.2% | -11.8% | 39.6% |
| batter_total_bases|under | 12870 | keep_model_only | 0.243 | 0.243 | -2.9% | -3.4% | 40.1% |
| pitcher_strikeouts|over | 5907 | keep_model_only | 0.247 | 0.247 | -10.8% | -11.3% | 36.8% |
| pitcher_strikeouts|under | 5902 | keep_model_only | 0.247 | 0.247 | -2.6% | -2.5% | 49.6% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 12113 | keep_model_only | 0.027 | 0.027 | -9.8% | -4.0% | 33.1% |
| batter_hits|over|common | 74088 | keep_model_only_clv | 0.216 | 0.214 | -13.8% | 14.8% | 42.2% |
| batter_hits|under|common | 25189 | keep_model_only | 0.235 | 0.235 | -4.2% | 5.2% | 45.3% |
| batter_home_runs|over|alt_tail | 24424 | keep_model_only | 0.009 | 0.009 | -31.8% | -27.2% | 37.4% |
| batter_home_runs|over|common | 24454 | keep_model_only | 0.109 | 0.108 | -34.3% | -33.8% | 37.2% |
| batter_total_bases|over|alt_tail | 48815 | keep_model_only | 0.096 | 0.096 | -20.3% | -8.3% | 38.8% |
| batter_total_bases|over|common | 61705 | keep_model_only | 0.209 | 0.209 | -16.5% | -15.0% | 40.3% |
| batter_total_bases|under|common | 12870 | keep_model_only | 0.243 | 0.243 | -2.9% | -3.4% | 40.1% |
| pitcher_strikeouts|over|common | 5907 | keep_model_only | 0.247 | 0.247 | -10.8% | -11.3% | 36.8% |
| pitcher_strikeouts|under|common | 5902 | keep_model_only | 0.247 | 0.247 | -2.6% | -2.5% | 49.6% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
