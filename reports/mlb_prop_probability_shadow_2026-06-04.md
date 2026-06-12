# MLB Prop Probability Shadow - 2026-06-04

## Scope

- Locked side rows: 826
- Date range: 2026-05-31 to 2026-06-02
- Unique graded dates: 3
- Minimum EV for simulated picks: 0.020
- Shadow winner: model_only

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 826 | 0.216 | -1.3% | 379 | 4.7% | 4.7% | +0.02 |
| model_plus_prior | 826 | 0.215 | -0.9% | 337 | 6.6% | 5.0% | +0.02 |
| market_no_vig | 630 | 0.235 | 4.4% | 2 | 112.0% | 100.0% | +13.97 |
| direct_side_model | 826 | 0.213 | -1.2% | 385 | 11.9% | 4.7% | +0.01 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 139 | keep_model_only | 0.237 | 0.236 | -17.9% | -39.8% | 33.3% |
| batter_hits|under | 206 | keep_model_only | 0.241 | 0.240 | 3.1% | 3.5% | 0.0% |
| batter_home_runs|over | 32 | insufficient_rows | 0.104 | 0.105 | -25.9% | -23.1% | 0.0% |
| batter_home_runs|under | 135 | keep_model_only | 0.134 | 0.134 | N/A | N/A | N/A |
| batter_total_bases|over | 48 | keep_model_only | 0.248 | 0.249 | 4.6% | 16.2% | 4.2% |
| batter_total_bases|under | 212 | use_model_plus_prior | 0.237 | 0.236 | 1.1% | 3.4% | 0.0% |
| pitcher_strikeouts|over | 33 | insufficient_rows | 0.223 | 0.223 | 43.9% | 43.9% | 57.7% |
| pitcher_strikeouts|under | 21 | insufficient_rows | 0.225 | 0.225 | 34.7% | 42.2% | 0.0% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|common | 139 | keep_model_only | 0.237 | 0.236 | -17.9% | -39.8% | 33.3% |
| batter_hits|under|common | 206 | keep_model_only | 0.241 | 0.240 | 3.1% | 3.5% | 0.0% |
| batter_home_runs|over|common | 32 | insufficient_rows | 0.104 | 0.105 | -25.9% | -23.1% | 0.0% |
| batter_home_runs|under|common | 135 | keep_model_only | 0.134 | 0.134 | N/A | N/A | N/A |
| batter_total_bases|over|common | 48 | keep_model_only | 0.248 | 0.249 | 4.6% | 16.2% | 4.2% |
| batter_total_bases|under|common | 212 | use_model_plus_prior | 0.237 | 0.236 | 1.1% | 3.4% | 0.0% |
| pitcher_strikeouts|over|common | 33 | insufficient_rows | 0.223 | 0.223 | 43.9% | 43.9% | 57.7% |
| pitcher_strikeouts|under|common | 21 | insufficient_rows | 0.225 | 0.225 | 34.7% | 42.2% | 0.0% |

## Shadow Winner

No shadow variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | false | brier_not_improved, candidate_selected_rows_too_small, calibration_worse |
| direct_side_model | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
