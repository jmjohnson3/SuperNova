# MLB Prop Probability Shadow - 2026-06-16

## Scope

- Locked side rows: 59581
- Date range: 2026-05-31 to 2026-06-14
- Unique graded dates: 15
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 59581 | 0.159 | -1.1% | 11566 | -12.2% | 42.1% | +0.21 |
| model_plus_prior | 59581 | 0.158 | -0.8% | 8739 | -6.8% | 42.2% | +0.22 |
| market_no_vig | 59581 | 0.154 | -0.5% | 3708 | 206.0% | 42.1% | +0.28 |
| direct_side_model | 59581 | 0.146 | -0.2% | 11400 | 57.3% | 41.3% | +0.20 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 19845 | keep_model_only_clv | 0.158 | 0.157 | 3.3% | 26.0% | 45.1% |
| batter_hits|under | 4943 | keep_model_only | 0.233 | 0.232 | -8.0% | -6.4% | 41.2% |
| batter_home_runs|over | 9186 | keep_model_only | 0.064 | 0.064 | -38.6% | -29.2% | 43.0% |
| batter_total_bases|over | 20865 | keep_model_only | 0.164 | 0.163 | -16.1% | -12.0% | 42.0% |
| batter_total_bases|under | 2675 | keep_model_only | 0.243 | 0.243 | -3.7% | -4.3% | 38.9% |
| pitcher_strikeouts|over | 1040 | keep_model_only_roi | 0.252 | 0.250 | -16.1% | -16.5% | 37.9% |
| pitcher_strikeouts|under | 1027 | keep_model_only_roi | 0.250 | 0.248 | 14.3% | 13.3% | 50.5% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 5827 | keep_model_only | 0.030 | 0.030 | 18.9% | 26.9% | 47.5% |
| batter_hits|over|common | 14018 | keep_model_only_clv | 0.211 | 0.209 | -19.7% | 25.3% | 43.0% |
| batter_hits|under|common | 4943 | keep_model_only | 0.233 | 0.232 | -8.0% | -6.4% | 41.2% |
| batter_home_runs|over|alt_tail | 4583 | keep_model_only | 0.010 | 0.010 | -44.1% | -31.6% | 42.7% |
| batter_home_runs|over|common | 4603 | keep_model_only | 0.118 | 0.117 | -26.4% | -24.4% | 43.6% |
| batter_total_bases|over|alt_tail | 9144 | keep_model_only | 0.104 | 0.103 | -18.9% | -11.4% | 39.9% |
| batter_total_bases|over|common | 11721 | keep_model_only | 0.210 | 0.210 | -7.5% | -13.8% | 47.2% |
| batter_total_bases|under|common | 2675 | keep_model_only | 0.243 | 0.243 | -3.7% | -4.3% | 38.9% |
| pitcher_strikeouts|over|common | 1040 | keep_model_only_roi | 0.252 | 0.250 | -16.1% | -16.5% | 37.9% |
| pitcher_strikeouts|under|common | 1027 | keep_model_only_roi | 0.250 | 0.248 | 14.3% | 13.3% | 50.5% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
