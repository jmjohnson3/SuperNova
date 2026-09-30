# MLB Prop Probability Shadow - 2026-06-26

## Scope

- Locked side rows: 112385
- Date range: 2026-05-31 to 2026-06-25
- Unique graded dates: 26
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 112385 | 0.161 | -1.6% | 25976 | -15.1% | 39.1% | +0.14 |
| model_plus_prior | 112385 | 0.161 | -1.4% | 22071 | -10.4% | 39.1% | +0.15 |
| market_no_vig | 112385 | 0.156 | -0.8% | 6846 | 203.2% | 41.1% | +0.26 |
| direct_side_model | 112385 | 0.152 | -0.7% | 21611 | 47.0% | 40.7% | +0.19 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 35114 | keep_model_only_clv | 0.172 | 0.171 | -11.9% | 1.1% | 40.1% |
| batter_hits|under | 9508 | keep_model_only | 0.233 | 0.233 | -5.2% | -6.3% | 40.8% |
| batter_home_runs|over | 18018 | keep_model_only | 0.060 | 0.060 | -18.8% | -10.8% | 38.9% |
| batter_total_bases|over | 40722 | keep_model_only | 0.162 | 0.161 | -17.5% | -14.5% | 38.7% |
| batter_total_bases|under | 4852 | keep_model_only | 0.243 | 0.243 | -3.7% | -3.6% | 39.0% |
| pitcher_strikeouts|over | 2092 | keep_model_only_roi | 0.246 | 0.245 | -16.2% | -17.6% | 36.8% |
| pitcher_strikeouts|under | 2079 | keep_model_only_roi | 0.247 | 0.245 | 4.4% | 3.6% | 48.9% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 7693 | keep_model_only | 0.029 | 0.028 | -7.1% | -2.6% | 34.5% |
| batter_hits|over|common | 27421 | keep_model_only_clv | 0.213 | 0.211 | -15.1% | 2.6% | 42.2% |
| batter_hits|under|common | 9508 | keep_model_only | 0.233 | 0.233 | -5.2% | -6.3% | 40.8% |
| batter_home_runs|over|alt_tail | 8995 | keep_model_only | 0.009 | 0.009 | -21.5% | -13.1% | 38.8% |
| batter_home_runs|over|common | 9023 | keep_model_only | 0.111 | 0.111 | -6.7% | -1.1% | 39.4% |
| batter_total_bases|over|alt_tail | 17984 | keep_model_only | 0.101 | 0.100 | -20.2% | -16.1% | 37.6% |
| batter_total_bases|over|common | 22738 | keep_model_only | 0.210 | 0.209 | -11.3% | -10.9% | 41.0% |
| batter_total_bases|under|common | 4852 | keep_model_only | 0.243 | 0.243 | -3.7% | -3.6% | 39.0% |
| pitcher_strikeouts|over|common | 2092 | keep_model_only_roi | 0.246 | 0.245 | -16.2% | -17.6% | 36.8% |
| pitcher_strikeouts|under|common | 2079 | keep_model_only_roi | 0.247 | 0.245 | 4.4% | 3.6% | 48.9% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
