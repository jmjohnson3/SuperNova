# MLB Prop Probability Shadow - 2026-06-10

## Scope

- Locked side rows: 31086
- Date range: 2026-05-31 to 2026-06-09
- Unique graded dates: 10
- Minimum EV for simulated picks: 0.020
- Shadow winner: model_only

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 31086 | 0.152 | -0.7% | 5480 | -14.2% | 42.1% | +0.17 |
| model_plus_prior | 31086 | 0.152 | -0.5% | 3925 | -3.0% | 42.7% | +0.18 |
| market_no_vig | 31086 | 0.149 | -1.0% | 928 | 176.8% | 39.0% | +0.14 |
| direct_side_model | 31086 | 0.141 | -0.2% | 6259 | 54.6% | 37.1% | +0.06 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 11412 | keep_model_only_clv | 0.144 | 0.142 | -9.4% | 25.7% | 47.6% |
| batter_hits|under | 2527 | keep_model_only | 0.238 | 0.237 | -2.9% | -2.6% | 40.3% |
| batter_home_runs|over | 4582 | keep_model_only | 0.059 | 0.059 | -46.1% | -37.6% | 47.3% |
| batter_total_bases|over | 10312 | keep_model_only | 0.161 | 0.161 | -17.8% | -8.1% | 43.0% |
| batter_total_bases|under | 1334 | keep_model_only | 0.244 | 0.244 | -4.7% | -6.0% | 37.8% |
| pitcher_strikeouts|over | 466 | keep_model_only_clv | 0.253 | 0.251 | -41.5% | -40.9% | 48.6% |
| pitcher_strikeouts|under | 453 | keep_model_only_roi | 0.248 | 0.246 | 32.2% | 32.0% | 39.8% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 4416 | keep_model_only | 0.030 | 0.030 | -11.4% | 31.6% | 48.1% |
| batter_hits|over|common | 6996 | keep_model_only_clv | 0.216 | 0.213 | -3.0% | 18.2% | 47.1% |
| batter_hits|under|common | 2527 | keep_model_only | 0.238 | 0.237 | -2.9% | -2.6% | 40.3% |
| batter_home_runs|over|alt_tail | 2282 | keep_model_only | 0.010 | 0.009 | -67.3% | -63.9% | 50.4% |
| batter_home_runs|over|common | 2300 | keep_model_only | 0.109 | 0.109 | -12.0% | -9.4% | 44.4% |
| batter_total_bases|over|alt_tail | 4538 | keep_model_only | 0.100 | 0.100 | -18.9% | -7.4% | 40.5% |
| batter_total_bases|over|common | 5774 | keep_model_only | 0.209 | 0.209 | -14.0% | -10.3% | 51.5% |
| batter_total_bases|under|common | 1334 | keep_model_only | 0.244 | 0.244 | -4.7% | -6.0% | 37.8% |
| pitcher_strikeouts|over|common | 466 | keep_model_only_clv | 0.253 | 0.251 | -41.5% | -40.9% | 48.6% |
| pitcher_strikeouts|under|common | 453 | keep_model_only_roi | 0.248 | 0.246 | 32.2% | 32.0% | 39.8% |

## Shadow Winner

No shadow variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |
| direct_side_model | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
