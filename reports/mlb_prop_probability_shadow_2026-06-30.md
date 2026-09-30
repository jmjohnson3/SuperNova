# MLB Prop Probability Shadow - 2026-06-30

## Scope

- Locked side rows: 131965
- Date range: 2026-05-31 to 2026-06-28
- Unique graded dates: 29
- Minimum EV for simulated picks: 0.020
- Shadow winner: direct_side_model

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 131965 | 0.162 | -2.2% | 31795 | -17.4% | 38.8% | +0.13 |
| model_plus_prior | 131965 | 0.162 | -2.0% | 27343 | -14.2% | 38.6% | +0.13 |
| market_no_vig | 131965 | 0.150 | 1.2% | 13829 | 128.3% | 49.0% | +0.29 |
| direct_side_model | 131965 | 0.143 | -0.2% | 27707 | 76.1% | 41.0% | +0.20 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 40797 | keep_model_only_clv | 0.178 | 0.177 | -13.2% | -4.9% | 38.5% |
| batter_hits|under | 11182 | keep_model_only | 0.234 | 0.234 | -4.7% | -5.6% | 40.4% |
| batter_home_runs|over | 21317 | keep_model_only | 0.060 | 0.060 | -29.2% | -24.8% | 38.3% |
| batter_total_bases|over | 48081 | keep_model_only | 0.160 | 0.160 | -17.6% | -15.6% | 38.5% |
| batter_total_bases|under | 5631 | keep_model_only | 0.243 | 0.242 | -3.7% | -3.9% | 38.9% |
| pitcher_strikeouts|over | 2485 | keep_model_only | 0.245 | 0.244 | -16.0% | -17.1% | 35.9% |
| pitcher_strikeouts|under | 2472 | keep_model_only_roi | 0.245 | 0.244 | 5.1% | 4.9% | 50.8% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 8398 | keep_model_only | 0.029 | 0.029 | -9.8% | -7.2% | 33.4% |
| batter_hits|over|common | 32399 | keep_model_only_clv | 0.216 | 0.215 | -14.8% | -4.2% | 39.8% |
| batter_hits|under|common | 11182 | keep_model_only | 0.234 | 0.234 | -4.7% | -5.6% | 40.4% |
| batter_home_runs|over|alt_tail | 10642 | keep_model_only | 0.009 | 0.009 | -31.8% | -26.2% | 37.8% |
| batter_home_runs|over|common | 10675 | keep_model_only | 0.111 | 0.110 | -23.5% | -22.2% | 39.4% |
| batter_total_bases|over|alt_tail | 21260 | keep_model_only | 0.098 | 0.097 | -20.2% | -16.7% | 37.7% |
| batter_total_bases|over|common | 26821 | keep_model_only | 0.210 | 0.209 | -13.8% | -13.9% | 39.7% |
| batter_total_bases|under|common | 5631 | keep_model_only | 0.243 | 0.242 | -3.7% | -3.9% | 38.9% |
| pitcher_strikeouts|over|common | 2485 | keep_model_only | 0.245 | 0.244 | -16.0% | -17.1% | 35.9% |
| pitcher_strikeouts|under|common | 2472 | keep_model_only_roi | 0.245 | 0.244 | 5.1% | 4.9% | 50.8% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved, clv_beat_rate_not_improved |
| market_no_vig | true | passes |
| direct_side_model | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
