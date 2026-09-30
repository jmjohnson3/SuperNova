# MLB Prop Probability Shadow - 2026-07-04

## Scope

- Locked side rows: 155762
- Date range: 2026-05-31 to 2026-07-02
- Unique graded dates: 33
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 155762 | 0.164 | -2.3% | 35835 | -17.1% | 39.4% | +0.14 |
| model_plus_prior | 155762 | 0.164 | -2.1% | 30658 | -14.4% | 39.4% | +0.14 |
| market_no_vig | 155762 | 0.152 | 1.3% | 16381 | 128.8% | 50.4% | +0.33 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 47324 | keep_model_only | 0.181 | 0.180 | -12.8% | -9.9% | 39.7% |
| batter_hits|under | 13291 | keep_model_only | 0.234 | 0.233 | -4.7% | -0.2% | 42.1% |
| batter_home_runs|over | 25248 | keep_model_only | 0.061 | 0.061 | -29.1% | -24.7% | 38.4% |
| batter_total_bases|over | 57083 | keep_model_only | 0.162 | 0.162 | -17.4% | -14.6% | 39.3% |
| batter_total_bases|under | 6771 | keep_model_only | 0.244 | 0.244 | -3.4% | -3.5% | 39.3% |
| pitcher_strikeouts|over | 3029 | keep_model_only | 0.245 | 0.244 | -16.0% | -16.9% | 34.6% |
| pitcher_strikeouts|under | 3016 | keep_model_only_roi | 0.246 | 0.245 | 1.2% | 0.7% | 51.2% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 8884 | keep_model_only | 0.029 | 0.029 | -9.8% | -8.7% | 33.2% |
| batter_hits|over|common | 38440 | keep_model_only | 0.216 | 0.216 | -14.0% | -10.1% | 41.2% |
| batter_hits|under|common | 13291 | keep_model_only | 0.234 | 0.233 | -4.7% | -0.2% | 42.1% |
| batter_home_runs|over|alt_tail | 12607 | keep_model_only | 0.009 | 0.009 | -31.8% | -26.2% | 37.6% |
| batter_home_runs|over|common | 12641 | keep_model_only | 0.113 | 0.112 | -23.4% | -22.2% | 39.9% |
| batter_total_bases|over|alt_tail | 25190 | keep_model_only | 0.099 | 0.099 | -20.6% | -15.4% | 38.0% |
| batter_total_bases|over|common | 31893 | keep_model_only | 0.212 | 0.211 | -13.7% | -13.5% | 40.9% |
| batter_total_bases|under|common | 6771 | keep_model_only | 0.244 | 0.244 | -3.4% | -3.5% | 39.3% |
| pitcher_strikeouts|over|common | 3029 | keep_model_only | 0.245 | 0.244 | -16.0% | -16.9% | 34.6% |
| pitcher_strikeouts|under|common | 3016 | keep_model_only_roi | 0.246 | 0.245 | 1.2% | 0.7% | 51.2% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved, clv_beat_rate_not_improved |
| market_no_vig | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
