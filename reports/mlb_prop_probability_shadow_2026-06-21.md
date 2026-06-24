# MLB Prop Probability Shadow - 2026-06-21

## Scope

- Locked side rows: 85628
- Date range: 2026-05-31 to 2026-06-19
- Unique graded dates: 20
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 85628 | 0.161 | -1.2% | 18383 | -9.4% | 40.9% | +0.19 |
| model_plus_prior | 85628 | 0.160 | -1.0% | 14656 | -2.1% | 40.9% | +0.19 |
| market_no_vig | 85628 | 0.156 | -0.5% | 5350 | 210.6% | 42.0% | +0.29 |
| direct_side_model | 85628 | 0.146 | -0.2% | 18869 | 61.0% | 41.8% | +0.24 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 27334 | keep_model_only_clv | 0.167 | 0.166 | -6.3% | 14.0% | 42.1% |
| batter_hits|under | 7222 | keep_model_only | 0.232 | 0.231 | -6.2% | -6.5% | 41.1% |
| batter_home_runs|over | 13573 | keep_model_only | 0.064 | 0.064 | -2.1% | 19.7% | 42.3% |
| batter_total_bases|over | 30739 | keep_model_only | 0.164 | 0.163 | -13.5% | -9.2% | 40.5% |
| batter_total_bases|under | 3769 | keep_model_only | 0.242 | 0.242 | -3.7% | -3.9% | 38.8% |
| pitcher_strikeouts|over | 1502 | keep_model_only_roi | 0.249 | 0.247 | -16.2% | -17.2% | 36.9% |
| pitcher_strikeouts|under | 1489 | keep_model_only_clv | 0.248 | 0.247 | 4.4% | 4.9% | 48.9% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 6647 | keep_model_only | 0.030 | 0.030 | 6.0% | 15.5% | 39.4% |
| batter_hits|over|common | 20687 | keep_model_only_clv | 0.212 | 0.210 | -17.6% | 13.1% | 43.5% |
| batter_hits|under|common | 7222 | keep_model_only | 0.232 | 0.231 | -6.2% | -6.5% | 41.1% |
| batter_home_runs|over|alt_tail | 6775 | keep_model_only | 0.010 | 0.010 | -1.0% | 25.5% | 43.0% |
| batter_home_runs|over|common | 6798 | keep_model_only | 0.117 | 0.117 | -5.5% | 2.3% | 40.1% |
| batter_total_bases|over|alt_tail | 13534 | keep_model_only | 0.103 | 0.103 | -15.5% | -10.1% | 39.4% |
| batter_total_bases|over|common | 17205 | keep_model_only | 0.211 | 0.211 | -7.9% | -6.4% | 43.8% |
| batter_total_bases|under|common | 3769 | keep_model_only | 0.242 | 0.242 | -3.7% | -3.9% | 38.8% |
| pitcher_strikeouts|over|common | 1502 | keep_model_only_roi | 0.249 | 0.247 | -16.2% | -17.2% | 36.9% |
| pitcher_strikeouts|under|common | 1489 | keep_model_only_clv | 0.248 | 0.247 | 4.4% | 4.9% | 48.9% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
