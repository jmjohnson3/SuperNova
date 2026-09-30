# MLB Prop Probability Shadow - 2026-07-08

## Scope

- Locked side rows: 184188
- Date range: 2026-05-31 to 2026-07-07
- Unique graded dates: 38
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 184188 | 0.165 | -2.3% | 40755 | -16.9% | 39.3% | +0.13 |
| model_plus_prior | 184188 | 0.165 | -2.2% | 35506 | -12.6% | 39.5% | +0.14 |
| market_no_vig | 184188 | 0.152 | 1.4% | 18989 | 136.7% | 50.2% | +0.32 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 55200 | keep_model_only | 0.184 | 0.183 | -12.7% | -5.0% | 41.1% |
| batter_hits|under | 15764 | keep_model_only | 0.234 | 0.233 | -4.1% | 3.5% | 41.6% |
| batter_home_runs|over | 30019 | keep_model_only | 0.061 | 0.061 | -29.1% | -24.7% | 38.3% |
| batter_total_bases|over | 67917 | keep_model_only | 0.162 | 0.162 | -17.2% | -13.3% | 39.2% |
| batter_total_bases|under | 8057 | keep_model_only | 0.243 | 0.243 | -3.4% | -3.0% | 39.2% |
| pitcher_strikeouts|over | 3622 | keep_model_only | 0.245 | 0.245 | -14.6% | -16.1% | 34.8% |
| pitcher_strikeouts|under | 3609 | keep_model_only | 0.246 | 0.246 | 2.6% | 2.0% | 49.7% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 9513 | keep_model_only | 0.028 | 0.028 | -9.8% | -5.5% | 33.7% |
| batter_hits|over|common | 45687 | keep_model_only | 0.216 | 0.215 | -13.8% | -4.9% | 42.5% |
| batter_hits|under|common | 15764 | keep_model_only | 0.234 | 0.233 | -4.1% | 3.5% | 41.6% |
| batter_home_runs|over|alt_tail | 14991 | keep_model_only | 0.010 | 0.010 | -31.8% | -26.2% | 37.5% |
| batter_home_runs|over|common | 15028 | keep_model_only | 0.113 | 0.112 | -23.4% | -22.1% | 39.8% |
| batter_total_bases|over|alt_tail | 29964 | keep_model_only | 0.100 | 0.100 | -19.2% | -12.1% | 38.5% |
| batter_total_bases|over|common | 37953 | keep_model_only | 0.211 | 0.211 | -15.2% | -14.5% | 40.0% |
| batter_total_bases|under|common | 8057 | keep_model_only | 0.243 | 0.243 | -3.4% | -3.0% | 39.2% |
| pitcher_strikeouts|over|common | 3622 | keep_model_only | 0.245 | 0.245 | -14.6% | -16.1% | 34.8% |
| pitcher_strikeouts|under|common | 3609 | keep_model_only | 0.246 | 0.246 | 2.6% | 2.0% | 49.7% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
