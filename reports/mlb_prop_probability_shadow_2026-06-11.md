# MLB Prop Probability Shadow - 2026-06-11

## Scope

- Locked side rows: 38710
- Date range: 2026-05-31 to 2026-06-10
- Unique graded dates: 11
- Minimum EV for simulated picks: 0.020
- Shadow winner: model_plus_prior

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 38710 | 0.153 | -0.9% | 7012 | -7.8% | 42.9% | +0.20 |
| model_plus_prior | 38710 | 0.152 | -0.5% | 5174 | -4.5% | 43.4% | +0.21 |
| market_no_vig | 38710 | 0.148 | -0.5% | 2433 | 239.6% | 41.5% | +0.24 |
| direct_side_model | 38710 | 0.141 | 0.3% | 7405 | 60.9% | 38.6% | +0.13 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 14135 | keep_model_only_clv | 0.143 | 0.141 | 9.2% | 11.3% | 47.7% |
| batter_hits|under | 3124 | keep_model_only | 0.235 | 0.235 | -2.9% | -2.0% | 42.0% |
| batter_home_runs|over | 5706 | keep_model_only | 0.061 | 0.061 | -46.1% | -36.6% | 47.0% |
| batter_total_bases|over | 12898 | keep_model_only | 0.163 | 0.163 | -13.2% | -9.2% | 43.2% |
| batter_total_bases|under | 1672 | keep_model_only | 0.243 | 0.242 | -3.0% | -3.7% | 39.3% |
| pitcher_strikeouts|over | 594 | use_model_plus_prior | 0.251 | 0.250 | -31.6% | -29.8% | 45.5% |
| pitcher_strikeouts|under | 581 | keep_model_only | 0.247 | 0.246 | 27.0% | 26.0% | 39.8% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 5418 | keep_model_only | 0.029 | 0.029 | 20.6% | 24.2% | 49.1% |
| batter_hits|over|common | 8717 | keep_model_only_clv | 0.213 | 0.210 | -23.0% | -2.5% | 46.3% |
| batter_hits|under|common | 3124 | keep_model_only | 0.235 | 0.235 | -2.9% | -2.0% | 42.0% |
| batter_home_runs|over|alt_tail | 2844 | keep_model_only | 0.009 | 0.009 | -67.3% | -63.9% | 50.4% |
| batter_home_runs|over|common | 2862 | keep_model_only | 0.113 | 0.113 | -12.0% | -6.4% | 43.8% |
| batter_total_bases|over|alt_tail | 5662 | keep_model_only | 0.103 | 0.102 | -14.6% | -2.0% | 42.4% |
| batter_total_bases|over|common | 7236 | keep_model_only | 0.211 | 0.210 | -8.2% | -24.1% | 44.9% |
| batter_total_bases|under|common | 1672 | keep_model_only | 0.243 | 0.242 | -3.0% | -3.7% | 39.3% |
| pitcher_strikeouts|over|common | 594 | use_model_plus_prior | 0.251 | 0.250 | -31.6% | -29.8% | 45.5% |
| pitcher_strikeouts|under|common | 581 | keep_model_only | 0.247 | 0.246 | 27.0% | 26.0% | 39.8% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | true | passes |
| market_no_vig | false | clv_beat_rate_not_improved |
| direct_side_model | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
