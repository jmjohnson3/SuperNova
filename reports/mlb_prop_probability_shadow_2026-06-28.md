# MLB Prop Probability Shadow - 2026-06-28

## Scope

- Locked side rows: 119950
- Date range: 2026-05-31 to 2026-06-26
- Unique graded dates: 27
- Minimum EV for simulated picks: 0.020
- Shadow winner: direct_side_model

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 119950 | 0.161 | -1.7% | 27642 | -16.6% | 38.7% | +0.14 |
| model_plus_prior | 119950 | 0.161 | -1.6% | 23890 | -13.3% | 38.7% | +0.14 |
| market_no_vig | 119950 | 0.150 | 1.3% | 12725 | 131.3% | 48.1% | +0.28 |
| direct_side_model | 119950 | 0.143 | -0.0% | 24793 | 81.9% | 40.4% | +0.20 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 37282 | keep_model_only | 0.173 | 0.173 | -12.5% | -3.7% | 39.1% |
| batter_hits|under | 10161 | keep_model_only | 0.233 | 0.233 | -4.9% | -6.1% | 40.7% |
| batter_home_runs|over | 19286 | keep_model_only | 0.059 | 0.059 | -27.3% | -21.5% | 38.1% |
| batter_total_bases|over | 43568 | keep_model_only | 0.161 | 0.161 | -17.4% | -15.5% | 38.4% |
| batter_total_bases|under | 5162 | keep_model_only | 0.243 | 0.243 | -3.7% | -3.6% | 39.0% |
| pitcher_strikeouts|over | 2252 | keep_model_only_roi | 0.246 | 0.245 | -16.2% | -17.3% | 36.6% |
| pitcher_strikeouts|under | 2239 | keep_model_only_roi | 0.246 | 0.245 | 3.2% | 2.6% | 48.9% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 7940 | keep_model_only | 0.029 | 0.029 | -8.1% | -7.2% | 33.7% |
| batter_hits|over|common | 29342 | keep_model_only_clv | 0.213 | 0.211 | -15.3% | -2.4% | 41.0% |
| batter_hits|under|common | 10161 | keep_model_only | 0.233 | 0.233 | -4.9% | -6.1% | 40.7% |
| batter_home_runs|over|alt_tail | 9629 | keep_model_only | 0.009 | 0.009 | -31.8% | -26.0% | 37.7% |
| batter_home_runs|over|common | 9657 | keep_model_only | 0.110 | 0.110 | -5.0% | -2.0% | 39.6% |
| batter_total_bases|over|alt_tail | 19252 | keep_model_only | 0.099 | 0.099 | -20.2% | -17.3% | 37.7% |
| batter_total_bases|over|common | 24316 | keep_model_only | 0.210 | 0.209 | -11.7% | -11.9% | 39.9% |
| batter_total_bases|under|common | 5162 | keep_model_only | 0.243 | 0.243 | -3.7% | -3.6% | 39.0% |
| pitcher_strikeouts|over|common | 2252 | keep_model_only_roi | 0.246 | 0.245 | -16.2% | -17.3% | 36.6% |
| pitcher_strikeouts|under|common | 2239 | keep_model_only_roi | 0.246 | 0.245 | 3.2% | 2.6% | 48.9% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved, clv_beat_rate_not_improved |
| market_no_vig | true | passes |
| direct_side_model | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
