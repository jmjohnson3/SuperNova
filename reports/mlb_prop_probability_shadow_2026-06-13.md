# MLB Prop Probability Shadow - 2026-06-13

## Scope

- Locked side rows: 40380
- Date range: 2026-05-31 to 2026-06-11
- Unique graded dates: 12
- Minimum EV for simulated picks: 0.020
- Shadow winner: model_only

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 40380 | 0.153 | -0.9% | 7334 | -9.2% | 42.8% | +0.19 |
| model_plus_prior | 40380 | 0.152 | -0.5% | 5323 | -4.0% | 42.7% | +0.20 |
| market_no_vig | 40380 | 0.149 | -0.5% | 2528 | 230.5% | 41.4% | +0.23 |
| direct_side_model | 40380 | 0.145 | -0.5% | 7474 | 48.8% | 42.4% | +0.20 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 14608 | keep_model_only_clv | 0.144 | 0.142 | 7.2% | 19.5% | 44.9% |
| batter_hits|under | 3273 | keep_model_only | 0.235 | 0.235 | -2.7% | -1.9% | 42.0% |
| batter_home_runs|over | 5970 | keep_model_only | 0.062 | 0.061 | -49.8% | -43.1% | 44.6% |
| batter_total_bases|over | 13512 | keep_model_only | 0.164 | 0.163 | -14.0% | -8.3% | 43.2% |
| batter_total_bases|under | 1758 | keep_model_only | 0.243 | 0.243 | -3.6% | -4.1% | 39.1% |
| pitcher_strikeouts|over | 636 | use_model_plus_prior | 0.251 | 0.249 | -32.0% | -29.7% | 44.5% |
| pitcher_strikeouts|under | 623 | keep_model_only | 0.247 | 0.246 | 27.2% | 28.2% | 40.7% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 5478 | keep_model_only | 0.029 | 0.029 | 19.1% | 23.6% | 46.8% |
| batter_hits|over|common | 9130 | keep_model_only_clv | 0.213 | 0.210 | -24.3% | 14.6% | 42.9% |
| batter_hits|under|common | 3273 | keep_model_only | 0.235 | 0.235 | -2.7% | -1.9% | 42.0% |
| batter_home_runs|over|alt_tail | 2976 | keep_model_only | 0.008 | 0.008 | -71.1% | -70.7% | 45.7% |
| batter_home_runs|over|common | 2994 | keep_model_only | 0.115 | 0.114 | -12.9% | -8.9% | 43.4% |
| batter_total_bases|over|alt_tail | 5926 | keep_model_only | 0.103 | 0.103 | -15.4% | -0.9% | 42.4% |
| batter_total_bases|over|common | 7586 | keep_model_only | 0.211 | 0.210 | -8.8% | -24.5% | 44.9% |
| batter_total_bases|under|common | 1758 | keep_model_only | 0.243 | 0.243 | -3.6% | -4.1% | 39.1% |
| pitcher_strikeouts|over|common | 636 | use_model_plus_prior | 0.251 | 0.249 | -32.0% | -29.7% | 44.5% |
| pitcher_strikeouts|under|common | 623 | keep_model_only | 0.247 | 0.246 | 27.2% | 28.2% | 40.7% |

## Shadow Winner

No shadow variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | clv_beat_rate_not_improved |
| market_no_vig | false | clv_beat_rate_not_improved |
| direct_side_model | false | clv_beat_rate_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
