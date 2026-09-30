# MLB Prop Probability Shadow - 2026-07-06

## Scope

- Locked side rows: 173503
- Date range: 2026-05-31 to 2026-07-05
- Unique graded dates: 36
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 173503 | 0.165 | -2.4% | 38989 | -16.9% | 39.3% | +0.13 |
| model_plus_prior | 173503 | 0.165 | -2.2% | 33708 | -13.2% | 39.5% | +0.14 |
| market_no_vig | 173503 | 0.152 | 1.3% | 17924 | 132.2% | 50.2% | +0.32 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 52148 | keep_model_only | 0.183 | 0.182 | -12.8% | -6.7% | 41.0% |
| batter_hits|under | 14865 | keep_model_only | 0.234 | 0.233 | -4.5% | 2.6% | 41.0% |
| batter_home_runs|over | 28195 | keep_model_only | 0.062 | 0.062 | -29.1% | -24.9% | 38.3% |
| batter_total_bases|over | 63853 | keep_model_only | 0.162 | 0.162 | -17.2% | -13.6% | 39.2% |
| batter_total_bases|under | 7645 | keep_model_only | 0.244 | 0.243 | -3.4% | -3.1% | 39.2% |
| pitcher_strikeouts|over | 3405 | keep_model_only | 0.245 | 0.244 | -15.2% | -16.8% | 35.0% |
| pitcher_strikeouts|under | 3392 | keep_model_only | 0.246 | 0.245 | 2.1% | 1.5% | 49.7% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 9186 | keep_model_only | 0.028 | 0.028 | -9.8% | -6.2% | 33.6% |
| batter_hits|over|common | 42962 | keep_model_only | 0.216 | 0.215 | -13.9% | -6.8% | 42.4% |
| batter_hits|under|common | 14865 | keep_model_only | 0.234 | 0.233 | -4.5% | 2.6% | 41.0% |
| batter_home_runs|over|alt_tail | 14080 | keep_model_only | 0.010 | 0.010 | -31.8% | -26.3% | 37.5% |
| batter_home_runs|over|common | 14115 | keep_model_only | 0.113 | 0.113 | -23.4% | -22.4% | 39.7% |
| batter_total_bases|over|alt_tail | 28138 | keep_model_only | 0.100 | 0.100 | -19.0% | -12.5% | 38.3% |
| batter_total_bases|over|common | 35715 | keep_model_only | 0.211 | 0.211 | -15.3% | -14.9% | 40.2% |
| batter_total_bases|under|common | 7645 | keep_model_only | 0.244 | 0.243 | -3.4% | -3.1% | 39.2% |
| pitcher_strikeouts|over|common | 3405 | keep_model_only | 0.245 | 0.244 | -15.2% | -16.8% | 35.0% |
| pitcher_strikeouts|under|common | 3392 | keep_model_only | 0.246 | 0.245 | 2.1% | 1.5% | 49.7% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
