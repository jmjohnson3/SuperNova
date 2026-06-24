# MLB Prop Probability Shadow - 2026-06-24

## Scope

- Locked side rows: 102734
- Date range: 2026-05-31 to 2026-06-22
- Unique graded dates: 23
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 102734 | 0.162 | -1.5% | 23863 | -13.9% | 39.4% | +0.15 |
| model_plus_prior | 102734 | 0.161 | -1.3% | 20172 | -9.1% | 39.7% | +0.16 |
| market_no_vig | 102734 | 0.156 | -0.6% | 6335 | 208.0% | 41.6% | +0.25 |
| direct_side_model | 102734 | 0.147 | -0.3% | 21645 | 54.7% | 40.6% | +0.19 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 32138 | keep_model_only_clv | 0.172 | 0.170 | -10.8% | 3.6% | 41.5% |
| batter_hits|under | 8700 | keep_model_only | 0.232 | 0.231 | -4.7% | -6.5% | 40.8% |
| batter_home_runs|over | 16445 | keep_model_only | 0.061 | 0.061 | -13.9% | -5.0% | 40.2% |
| batter_total_bases|over | 37202 | keep_model_only | 0.163 | 0.162 | -17.2% | -13.9% | 39.0% |
| batter_total_bases|under | 4480 | keep_model_only | 0.243 | 0.243 | -3.7% | -3.9% | 38.6% |
| pitcher_strikeouts|over | 1891 | keep_model_only_roi | 0.247 | 0.246 | -16.2% | -17.9% | 36.3% |
| pitcher_strikeouts|under | 1878 | keep_model_only_roi | 0.247 | 0.246 | 4.4% | 3.6% | 47.1% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 7097 | keep_model_only | 0.030 | 0.030 | -1.0% | 6.7% | 37.1% |
| batter_hits|over|common | 25041 | keep_model_only_clv | 0.212 | 0.210 | -17.4% | 2.5% | 43.0% |
| batter_hits|under|common | 8700 | keep_model_only | 0.232 | 0.231 | -4.7% | -6.5% | 40.8% |
| batter_home_runs|over|alt_tail | 8209 | keep_model_only | 0.010 | 0.010 | -16.0% | -7.0% | 40.3% |
| batter_home_runs|over|common | 8236 | keep_model_only | 0.113 | 0.113 | -5.2% | 3.0% | 39.9% |
| batter_total_bases|over|alt_tail | 16410 | keep_model_only | 0.102 | 0.101 | -20.2% | -15.9% | 37.6% |
| batter_total_bases|over|common | 20792 | keep_model_only | 0.211 | 0.210 | -9.1% | -8.9% | 42.5% |
| batter_total_bases|under|common | 4480 | keep_model_only | 0.243 | 0.243 | -3.7% | -3.9% | 38.6% |
| pitcher_strikeouts|over|common | 1891 | keep_model_only_roi | 0.247 | 0.246 | -16.2% | -17.9% | 36.3% |
| pitcher_strikeouts|under|common | 1878 | keep_model_only_roi | 0.247 | 0.246 | 4.4% | 3.6% | 47.1% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
