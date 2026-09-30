# MLB Prop Probability Shadow - 2026-07-09

## Scope

- Locked side rows: 191755
- Date range: 2026-05-31 to 2026-07-08
- Unique graded dates: 39
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 191755 | 0.165 | -2.4% | 41888 | -17.0% | 39.5% | +0.14 |
| model_plus_prior | 191755 | 0.164 | -2.2% | 35957 | -12.5% | 39.7% | +0.14 |
| market_no_vig | 191755 | 0.152 | 1.4% | 19704 | 137.7% | 50.3% | +0.34 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 57367 | keep_model_only | 0.184 | 0.183 | -12.7% | -3.5% | 40.7% |
| batter_hits|under | 16414 | keep_model_only | 0.234 | 0.233 | -3.9% | 3.2% | 42.0% |
| batter_home_runs|over | 31281 | keep_model_only | 0.061 | 0.061 | -29.1% | -24.3% | 38.4% |
| batter_total_bases|over | 70773 | keep_model_only | 0.162 | 0.161 | -17.4% | -13.4% | 39.6% |
| batter_total_bases|under | 8385 | keep_model_only | 0.243 | 0.243 | -3.4% | -3.0% | 39.2% |
| pitcher_strikeouts|over | 3774 | keep_model_only | 0.245 | 0.244 | -15.4% | -13.8% | 35.8% |
| pitcher_strikeouts|under | 3761 | keep_model_only | 0.246 | 0.245 | 3.4% | 3.0% | 48.6% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 9766 | keep_model_only | 0.028 | 0.028 | -9.8% | -6.9% | 33.5% |
| batter_hits|over|common | 47601 | keep_model_only | 0.216 | 0.215 | -13.8% | -2.8% | 42.1% |
| batter_hits|under|common | 16414 | keep_model_only | 0.234 | 0.233 | -3.9% | 3.2% | 42.0% |
| batter_home_runs|over|alt_tail | 15621 | keep_model_only | 0.010 | 0.010 | -31.8% | -25.5% | 37.5% |
| batter_home_runs|over|common | 15660 | keep_model_only | 0.112 | 0.111 | -23.4% | -22.2% | 39.8% |
| batter_total_bases|over|alt_tail | 31228 | keep_model_only | 0.100 | 0.099 | -19.5% | -12.0% | 38.8% |
| batter_total_bases|over|common | 39545 | keep_model_only | 0.211 | 0.211 | -15.5% | -14.8% | 40.4% |
| batter_total_bases|under|common | 8385 | keep_model_only | 0.243 | 0.243 | -3.4% | -3.0% | 39.2% |
| pitcher_strikeouts|over|common | 3774 | keep_model_only | 0.245 | 0.244 | -15.4% | -13.8% | 35.8% |
| pitcher_strikeouts|under|common | 3761 | keep_model_only | 0.246 | 0.245 | 3.4% | 3.0% | 48.6% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
