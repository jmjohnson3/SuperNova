# MLB Prop Probability Shadow - 2026-07-01

## Scope

- Locked side rows: 138373
- Date range: 2026-05-31 to 2026-06-29
- Unique graded dates: 30
- Minimum EV for simulated picks: 0.020
- Shadow winner: direct_side_model

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 138373 | 0.163 | -2.3% | 33170 | -17.0% | 39.2% | +0.14 |
| model_plus_prior | 138373 | 0.162 | -2.1% | 28748 | -13.9% | 39.2% | +0.14 |
| market_no_vig | 138373 | 0.151 | 1.2% | 14467 | 132.5% | 49.4% | +0.30 |
| direct_side_model | 138373 | 0.143 | -0.2% | 28833 | 76.2% | 41.5% | +0.21 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 42527 | keep_model_only_clv | 0.179 | 0.178 | -12.5% | -4.2% | 39.4% |
| batter_hits|under | 11762 | keep_model_only | 0.234 | 0.234 | -4.7% | -6.7% | 40.9% |
| batter_home_runs|over | 22375 | keep_model_only | 0.060 | 0.060 | -29.1% | -25.1% | 38.4% |
| batter_total_bases|over | 50508 | keep_model_only | 0.160 | 0.160 | -17.4% | -15.3% | 39.1% |
| batter_total_bases|under | 5942 | keep_model_only | 0.243 | 0.243 | -3.7% | -3.9% | 39.2% |
| pitcher_strikeouts|over | 2636 | keep_model_only | 0.245 | 0.244 | -16.0% | -17.4% | 35.7% |
| pitcher_strikeouts|under | 2623 | keep_model_only | 0.245 | 0.244 | 5.6% | 5.4% | 51.7% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 8490 | keep_model_only | 0.029 | 0.029 | -9.8% | -6.8% | 33.3% |
| batter_hits|over|common | 34037 | keep_model_only_clv | 0.217 | 0.215 | -13.6% | -3.6% | 40.8% |
| batter_hits|under|common | 11762 | keep_model_only | 0.234 | 0.234 | -4.7% | -6.7% | 40.9% |
| batter_home_runs|over|alt_tail | 11171 | keep_model_only | 0.009 | 0.009 | -31.8% | -26.8% | 37.6% |
| batter_home_runs|over|common | 11204 | keep_model_only | 0.111 | 0.111 | -23.4% | -22.2% | 39.9% |
| batter_total_bases|over|alt_tail | 22318 | keep_model_only | 0.098 | 0.097 | -20.2% | -16.4% | 37.7% |
| batter_total_bases|over|common | 28190 | keep_model_only | 0.210 | 0.210 | -13.6% | -13.8% | 40.9% |
| batter_total_bases|under|common | 5942 | keep_model_only | 0.243 | 0.243 | -3.7% | -3.9% | 39.2% |
| pitcher_strikeouts|over|common | 2636 | keep_model_only | 0.245 | 0.244 | -16.0% | -17.4% | 35.7% |
| pitcher_strikeouts|under|common | 2623 | keep_model_only | 0.245 | 0.244 | 5.6% | 5.4% | 51.7% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved, clv_beat_rate_not_improved |
| market_no_vig | true | passes |
| direct_side_model | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
