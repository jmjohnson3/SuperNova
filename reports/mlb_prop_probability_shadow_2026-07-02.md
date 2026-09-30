# MLB Prop Probability Shadow - 2026-07-02

## Scope

- Locked side rows: 145601
- Date range: 2026-05-31 to 2026-06-30
- Unique graded dates: 31
- Minimum EV for simulated picks: 0.020
- Shadow winner: direct_side_model

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 145601 | 0.164 | -2.3% | 34177 | -16.9% | 39.3% | +0.14 |
| model_plus_prior | 145601 | 0.163 | -2.1% | 29133 | -14.5% | 39.2% | +0.14 |
| market_no_vig | 145601 | 0.151 | 1.3% | 15147 | 133.1% | 49.5% | +0.31 |
| direct_side_model | 145601 | 0.143 | -0.2% | 31361 | 75.6% | 41.3% | +0.21 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 44580 | keep_model_only | 0.180 | 0.179 | -12.4% | -9.5% | 39.4% |
| batter_hits|under | 12391 | keep_model_only | 0.234 | 0.234 | -4.7% | -0.9% | 42.3% |
| batter_home_runs|over | 23557 | keep_model_only | 0.061 | 0.060 | -29.1% | -24.9% | 38.4% |
| batter_total_bases|over | 53222 | keep_model_only | 0.161 | 0.160 | -17.3% | -14.7% | 39.1% |
| batter_total_bases|under | 6292 | keep_model_only | 0.243 | 0.243 | -3.7% | -3.7% | 39.2% |
| pitcher_strikeouts|over | 2786 | keep_model_only | 0.245 | 0.244 | -15.7% | -17.4% | 35.6% |
| pitcher_strikeouts|under | 2773 | keep_model_only_calibration | 0.245 | 0.244 | 3.2% | 3.2% | 51.4% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 8732 | keep_model_only | 0.029 | 0.029 | -9.8% | -6.7% | 33.2% |
| batter_hits|over|common | 35848 | keep_model_only | 0.217 | 0.216 | -13.5% | -10.2% | 40.8% |
| batter_hits|under|common | 12391 | keep_model_only | 0.234 | 0.234 | -4.7% | -0.9% | 42.3% |
| batter_home_runs|over|alt_tail | 11762 | keep_model_only | 0.010 | 0.010 | -31.8% | -26.4% | 37.6% |
| batter_home_runs|over|common | 11795 | keep_model_only | 0.112 | 0.111 | -23.4% | -22.2% | 39.9% |
| batter_total_bases|over|alt_tail | 23500 | keep_model_only | 0.098 | 0.098 | -20.2% | -15.7% | 37.7% |
| batter_total_bases|over|common | 29722 | keep_model_only | 0.210 | 0.210 | -13.7% | -13.6% | 40.8% |
| batter_total_bases|under|common | 6292 | keep_model_only | 0.243 | 0.243 | -3.7% | -3.7% | 39.2% |
| pitcher_strikeouts|over|common | 2786 | keep_model_only | 0.245 | 0.244 | -15.7% | -17.4% | 35.6% |
| pitcher_strikeouts|under|common | 2773 | keep_model_only_calibration | 0.245 | 0.244 | 3.2% | 3.2% | 51.4% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved, clv_beat_rate_not_improved |
| market_no_vig | true | passes |
| direct_side_model | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
