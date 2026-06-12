# MLB Prop Probability Shadow - 2026-06-05

## Scope

- Locked side rows: 3686
- Date range: 2026-05-31 to 2026-06-03
- Unique graded dates: 4
- Minimum EV for simulated picks: 0.020
- Shadow winner: direct_side_model

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 3686 | 0.163 | 1.1% | 955 | 11.3% | 47.2% | +0.33 |
| model_plus_prior | 3686 | 0.162 | 0.8% | 947 | -3.8% | 48.0% | +0.38 |
| market_no_vig | 3686 | 0.162 | -0.1% | 2 | 112.0% | N/A | N/A |
| direct_side_model | 3686 | 0.160 | 1.0% | 873 | 29.2% | 49.5% | +0.47 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 1263 | keep_model_only | 0.153 | 0.153 | 154.2% | 25.1% | 0.0% |
| batter_hits|under | 430 | keep_model_only | 0.233 | 0.233 | -6.4% | -4.6% | 47.4% |
| batter_home_runs|over | 494 | keep_model_only | 0.052 | 0.052 | -57.6% | -50.8% | 43.3% |
| batter_total_bases|over | 1084 | keep_model_only | 0.167 | 0.166 | 4.2% | -4.0% | 52.2% |
| batter_total_bases|under | 254 | use_model_plus_prior | 0.243 | 0.242 | -4.3% | -3.3% | 48.6% |
| pitcher_strikeouts|over | 87 | keep_model_only_calibration | 0.238 | 0.236 | 12.0% | 14.4% | 61.9% |
| pitcher_strikeouts|under | 74 | keep_model_only | 0.241 | 0.240 | 10.0% | 13.6% | 40.0% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 438 | keep_model_only | 0.044 | 0.044 | 249.3% | 38.8% | N/A |
| batter_hits|over|common | 825 | keep_model_only_clv | 0.211 | 0.210 | -14.5% | -14.3% | 0.0% |
| batter_hits|under|common | 430 | keep_model_only | 0.233 | 0.233 | -6.4% | -4.6% | 47.4% |
| batter_home_runs|over|alt_tail | 231 | keep_model_only | 0.000 | 0.000 | -100.0% | -100.0% | N/A |
| batter_home_runs|over|common | 263 | keep_model_only | 0.097 | 0.097 | -14.4% | -8.6% | 43.3% |
| batter_total_bases|over|alt_tail | 464 | keep_model_only | 0.098 | 0.098 | 10.8% | -7.1% | N/A |
| batter_total_bases|over|common | 620 | keep_model_only | 0.218 | 0.217 | -7.6% | 2.7% | 52.2% |
| batter_total_bases|under|common | 254 | use_model_plus_prior | 0.243 | 0.242 | -4.3% | -3.3% | 48.6% |
| pitcher_strikeouts|over|common | 87 | keep_model_only_calibration | 0.238 | 0.236 | 12.0% | 14.4% | 61.9% |
| pitcher_strikeouts|under|common | 74 | keep_model_only | 0.241 | 0.240 | 10.0% | 13.6% | 40.0% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved, roi_not_improved |
| market_no_vig | false | brier_not_improved, candidate_selected_rows_too_small |
| direct_side_model | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
