# MLB Prop Probability Shadow - 2026-06-20

## Scope

- Locked side rows: 78285
- Date range: 2026-05-31 to 2026-06-18
- Unique graded dates: 19
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 78285 | 0.161 | -1.3% | 16173 | -12.1% | 41.7% | +0.20 |
| model_plus_prior | 78285 | 0.160 | -1.0% | 12721 | -5.2% | 41.9% | +0.22 |
| market_no_vig | 78285 | 0.156 | -0.6% | 4868 | 205.5% | 42.8% | +0.32 |
| direct_side_model | 78285 | 0.150 | -0.5% | 15860 | 56.8% | 42.5% | +0.26 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 25128 | keep_model_only_clv | 0.166 | 0.165 | -3.8% | 18.1% | 44.8% |
| batter_hits|under | 6559 | keep_model_only | 0.233 | 0.233 | -6.2% | -5.3% | 41.0% |
| batter_home_runs|over | 12375 | keep_model_only | 0.064 | 0.064 | -19.6% | -3.8% | 43.4% |
| batter_total_bases|over | 28052 | keep_model_only | 0.163 | 0.163 | -15.3% | -10.3% | 41.4% |
| batter_total_bases|under | 3478 | keep_model_only | 0.241 | 0.241 | -3.7% | -4.1% | 38.7% |
| pitcher_strikeouts|over | 1353 | keep_model_only_roi | 0.250 | 0.249 | -16.2% | -17.0% | 37.6% |
| pitcher_strikeouts|under | 1340 | keep_model_only_clv | 0.250 | 0.248 | 4.4% | 4.9% | 48.9% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 6302 | keep_model_only | 0.030 | 0.030 | 11.1% | 20.2% | 42.5% |
| batter_hits|over|common | 18826 | keep_model_only_clv | 0.212 | 0.210 | -19.2% | 16.8% | 46.0% |
| batter_hits|under|common | 6559 | keep_model_only | 0.233 | 0.233 | -6.2% | -5.3% | 41.0% |
| batter_home_runs|over|alt_tail | 6176 | keep_model_only | 0.010 | 0.010 | -21.2% | -1.4% | 44.0% |
| batter_home_runs|over|common | 6199 | keep_model_only | 0.117 | 0.117 | -15.0% | -10.4% | 41.5% |
| batter_total_bases|over|alt_tail | 12336 | keep_model_only | 0.103 | 0.103 | -17.2% | -11.4% | 39.9% |
| batter_total_bases|over|common | 15716 | keep_model_only | 0.211 | 0.210 | -10.0% | -7.2% | 45.8% |
| batter_total_bases|under|common | 3478 | keep_model_only | 0.241 | 0.241 | -3.7% | -4.1% | 38.7% |
| pitcher_strikeouts|over|common | 1353 | keep_model_only_roi | 0.250 | 0.249 | -16.2% | -17.0% | 37.6% |
| pitcher_strikeouts|under|common | 1340 | keep_model_only_clv | 0.250 | 0.248 | 4.4% | 4.9% | 48.9% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
