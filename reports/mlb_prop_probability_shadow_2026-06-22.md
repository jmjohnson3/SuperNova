# MLB Prop Probability Shadow - 2026-06-22

## Scope

- Locked side rows: 91317
- Date range: 2026-05-31 to 2026-06-20
- Unique graded dates: 21
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 91317 | 0.162 | -1.2% | 20210 | -9.4% | 40.5% | +0.18 |
| model_plus_prior | 91317 | 0.161 | -1.0% | 16499 | -3.9% | 40.6% | +0.19 |
| market_no_vig | 91317 | 0.157 | -0.5% | 5698 | 220.5% | 41.9% | +0.28 |
| direct_side_model | 91317 | 0.148 | -0.3% | 19410 | 61.1% | 41.8% | +0.23 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 28878 | keep_model_only_clv | 0.170 | 0.168 | -6.1% | 14.1% | 42.2% |
| batter_hits|under | 7723 | keep_model_only | 0.232 | 0.231 | -5.8% | -6.5% | 41.1% |
| batter_home_runs|over | 14535 | keep_model_only | 0.063 | 0.063 | 0.4% | 11.6% | 40.7% |
| batter_total_bases|over | 32899 | keep_model_only | 0.164 | 0.163 | -14.0% | -10.6% | 40.2% |
| batter_total_bases|under | 4005 | keep_model_only | 0.242 | 0.242 | -3.7% | -3.8% | 38.9% |
| pitcher_strikeouts|over | 1645 | keep_model_only_roi | 0.248 | 0.246 | -16.2% | -17.4% | 36.9% |
| pitcher_strikeouts|under | 1632 | use_model_plus_prior | 0.247 | 0.246 | 4.4% | 4.6% | 49.1% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 6728 | keep_model_only | 0.030 | 0.030 | 7.6% | 21.6% | 39.4% |
| batter_hits|over|common | 22150 | keep_model_only_clv | 0.212 | 0.210 | -17.3% | 10.4% | 43.5% |
| batter_hits|under|common | 7723 | keep_model_only | 0.232 | 0.231 | -5.8% | -6.5% | 41.1% |
| batter_home_runs|over|alt_tail | 7256 | keep_model_only | 0.011 | 0.010 | 1.6% | 14.1% | 41.0% |
| batter_home_runs|over|common | 7279 | keep_model_only | 0.116 | 0.116 | -4.1% | 3.1% | 39.6% |
| batter_total_bases|over|alt_tail | 14496 | keep_model_only | 0.103 | 0.103 | -16.1% | -11.3% | 38.8% |
| batter_total_bases|over|common | 18403 | keep_model_only | 0.211 | 0.211 | -8.5% | -8.8% | 43.8% |
| batter_total_bases|under|common | 4005 | keep_model_only | 0.242 | 0.242 | -3.7% | -3.8% | 38.9% |
| pitcher_strikeouts|over|common | 1645 | keep_model_only_roi | 0.248 | 0.246 | -16.2% | -17.4% | 36.9% |
| pitcher_strikeouts|under|common | 1632 | use_model_plus_prior | 0.247 | 0.246 | 4.4% | 4.6% | 49.1% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
