# MLB Prop Probability Shadow - 2026-06-03

## Scope

- Locked side rows: 2234
- Date range: 2026-05-31 to 2026-06-01
- Unique graded dates: 2
- Minimum EV for simulated picks: 0.020
- Shadow winner: model_only

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 2234 | 0.193 | 0.0% | 313 | 9.1% | 5.1% | -2.12 |
| model_plus_prior | 2234 | 0.193 | 0.0% | 256 | 16.9% | 5.5% | -1.92 |
| market_no_vig | 1683 | 0.211 | -0.4% | 2 | 4.0% | 50.0% | +6.58 |
| direct_side_model | 2234 | 0.186 | 0.0% | 328 | 36.0% | 5.8% | -2.31 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 363 | keep_model_only | 0.232 | 0.232 | -26.2% | -100.0% | 100.0% |
| batter_hits|under | 363 | keep_model_only | 0.232 | 0.232 | 12.8% | 13.6% | 0.0% |
| batter_home_runs|over | 355 | keep_model_only | 0.112 | 0.112 | -1.4% | 31.4% | 0.0% |
| batter_home_runs|under | 355 | keep_model_only | 0.112 | 0.112 | N/A | N/A | N/A |
| batter_total_bases|over | 356 | keep_model_only | 0.232 | 0.233 | -7.7% | 8.6% | 3.0% |
| batter_total_bases|under | 356 | keep_model_only | 0.232 | 0.231 | -1.6% | 6.1% | 0.0% |
| pitcher_strikeouts|over | 43 | keep_model_only | 0.205 | 0.206 | 48.9% | 48.9% | 66.7% |
| pitcher_strikeouts|under | 43 | keep_model_only | 0.205 | 0.206 | 36.7% | 41.6% | 0.0% |

## Shadow Winner

No shadow variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | false | brier_not_improved, candidate_selected_rows_too_small |
| direct_side_model | false | avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
