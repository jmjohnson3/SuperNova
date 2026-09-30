# MLB Prop Probability Shadow - 2026-07-07

## Scope

- Locked side rows: 176515
- Date range: 2026-05-31 to 2026-07-06
- Unique graded dates: 37
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 176515 | 0.165 | -2.3% | 39467 | -16.7% | 39.2% | +0.13 |
| model_plus_prior | 176515 | 0.165 | -2.2% | 34227 | -12.7% | 39.4% | +0.14 |
| market_no_vig | 176515 | 0.153 | 1.4% | 18246 | 137.0% | 50.0% | +0.32 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 52978 | keep_model_only | 0.183 | 0.182 | -12.7% | -5.7% | 41.0% |
| batter_hits|under | 15117 | keep_model_only | 0.234 | 0.234 | -4.3% | 2.8% | 40.6% |
| batter_home_runs|over | 28693 | keep_model_only | 0.062 | 0.062 | -29.1% | -24.9% | 38.3% |
| batter_total_bases|over | 64990 | keep_model_only | 0.162 | 0.162 | -16.8% | -13.1% | 39.1% |
| batter_total_bases|under | 7782 | keep_model_only | 0.244 | 0.243 | -3.4% | -3.0% | 39.3% |
| pitcher_strikeouts|over | 3484 | keep_model_only | 0.245 | 0.245 | -14.6% | -16.5% | 34.9% |
| pitcher_strikeouts|under | 3471 | keep_model_only | 0.246 | 0.246 | 2.1% | 1.5% | 49.7% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 9264 | keep_model_only | 0.028 | 0.028 | -9.8% | -6.0% | 33.6% |
| batter_hits|over|common | 43714 | keep_model_only | 0.216 | 0.215 | -13.8% | -5.7% | 42.4% |
| batter_hits|under|common | 15117 | keep_model_only | 0.234 | 0.234 | -4.3% | 2.8% | 40.6% |
| batter_home_runs|over|alt_tail | 14328 | keep_model_only | 0.010 | 0.010 | -31.8% | -26.2% | 37.5% |
| batter_home_runs|over|common | 14365 | keep_model_only | 0.113 | 0.113 | -23.4% | -22.5% | 39.8% |
| batter_total_bases|over|alt_tail | 28638 | keep_model_only | 0.100 | 0.100 | -18.8% | -11.9% | 38.2% |
| batter_total_bases|over|common | 36352 | keep_model_only | 0.211 | 0.211 | -14.9% | -14.4% | 40.0% |
| batter_total_bases|under|common | 7782 | keep_model_only | 0.244 | 0.243 | -3.4% | -3.0% | 39.3% |
| pitcher_strikeouts|over|common | 3484 | keep_model_only | 0.245 | 0.245 | -14.6% | -16.5% | 34.9% |
| pitcher_strikeouts|under|common | 3471 | keep_model_only | 0.246 | 0.246 | 2.1% | 1.5% | 49.7% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
