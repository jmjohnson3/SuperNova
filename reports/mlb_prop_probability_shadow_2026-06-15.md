# MLB Prop Probability Shadow - 2026-06-15

## Scope

- Locked side rows: 54405
- Date range: 2026-05-31 to 2026-06-13
- Unique graded dates: 14
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 54405 | 0.157 | -1.1% | 10466 | -13.1% | 41.7% | +0.20 |
| model_plus_prior | 54405 | 0.156 | -0.8% | 7986 | -6.9% | 41.6% | +0.21 |
| market_no_vig | 54405 | 0.153 | -0.6% | 3326 | 215.0% | 42.3% | +0.29 |
| direct_side_model | 54405 | 0.143 | -0.2% | 11259 | 54.5% | 41.1% | +0.20 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 18461 | keep_model_only_clv | 0.155 | 0.153 | 3.7% | 21.7% | 43.6% |
| batter_hits|under | 4491 | keep_model_only | 0.234 | 0.233 | -7.2% | -5.5% | 41.5% |
| batter_home_runs|over | 8316 | keep_model_only | 0.063 | 0.063 | -40.5% | -27.2% | 41.5% |
| batter_total_bases|over | 18868 | keep_model_only | 0.163 | 0.163 | -19.2% | -14.6% | 41.6% |
| batter_total_bases|under | 2418 | keep_model_only | 0.243 | 0.242 | -3.7% | -4.1% | 39.0% |
| pitcher_strikeouts|over | 932 | use_model_plus_prior | 0.251 | 0.250 | -14.8% | -13.7% | 38.9% |
| pitcher_strikeouts|under | 919 | keep_model_only | 0.248 | 0.247 | 21.6% | 21.8% | 45.9% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 5765 | keep_model_only | 0.030 | 0.030 | 19.3% | 31.6% | 45.8% |
| batter_hits|over|common | 12696 | keep_model_only_clv | 0.212 | 0.208 | -22.3% | 14.1% | 42.0% |
| batter_hits|under|common | 4491 | keep_model_only | 0.234 | 0.233 | -7.2% | -5.5% | 41.5% |
| batter_home_runs|over|alt_tail | 4148 | keep_model_only | 0.009 | 0.009 | -49.6% | -32.8% | 40.6% |
| batter_home_runs|over|common | 4168 | keep_model_only | 0.116 | 0.116 | -20.6% | -16.5% | 43.4% |
| batter_total_bases|over|alt_tail | 8274 | keep_model_only | 0.103 | 0.102 | -21.5% | -13.3% | 39.4% |
| batter_total_bases|over|common | 10594 | keep_model_only | 0.210 | 0.210 | -12.2% | -17.7% | 46.9% |
| batter_total_bases|under|common | 2418 | keep_model_only | 0.243 | 0.242 | -3.7% | -4.1% | 39.0% |
| pitcher_strikeouts|over|common | 932 | use_model_plus_prior | 0.251 | 0.250 | -14.8% | -13.7% | 38.9% |
| pitcher_strikeouts|under|common | 919 | keep_model_only | 0.248 | 0.247 | 21.6% | 21.8% | 45.9% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | clv_beat_rate_not_improved |
| market_no_vig | true | passes |
| direct_side_model | false | clv_beat_rate_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
