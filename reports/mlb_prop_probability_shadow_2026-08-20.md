# MLB Prop Probability Shadow - 2026-08-20

## Scope

- Locked side rows: 292465
- Date range: 2026-06-01 to 2026-07-29
- Unique graded dates: 56
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 292465 | 0.165 | -2.7% | 49933 | -19.2% | 39.3% | +0.13 |
| model_plus_prior | 292465 | 0.165 | -2.6% | 46603 | -11.1% | 39.6% | +0.14 |
| market_no_vig | 292465 | 0.152 | 1.4% | 28266 | 142.7% | 49.6% | +0.31 |
| direct_side_model | 292465 | 0.147 | -0.5% | 112627 | 7.9% | 37.5% | +0.08 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 85402 | keep_model_only_clv | 0.189 | 0.188 | -12.7% | 12.3% | 41.1% |
| batter_hits|under | 24931 | keep_model_only | 0.235 | 0.234 | -4.2% | 5.2% | 45.1% |
| batter_home_runs|over | 48365 | keep_model_only | 0.059 | 0.059 | -33.1% | -31.3% | 37.4% |
| batter_total_bases|over | 109350 | keep_model_only | 0.160 | 0.159 | -18.2% | -12.1% | 39.6% |
| batter_total_bases|under | 12728 | keep_model_only | 0.243 | 0.243 | -2.9% | -3.5% | 40.1% |
| pitcher_strikeouts|over | 5847 | keep_model_only | 0.247 | 0.247 | -10.8% | -10.9% | 36.8% |
| pitcher_strikeouts|under | 5842 | keep_model_only | 0.247 | 0.247 | -2.2% | -2.9% | 49.4% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 12086 | keep_model_only | 0.027 | 0.027 | -9.8% | -2.7% | 33.1% |
| batter_hits|over|common | 73316 | keep_model_only_clv | 0.216 | 0.214 | -13.8% | 14.2% | 42.1% |
| batter_hits|under|common | 24931 | keep_model_only | 0.235 | 0.234 | -4.2% | 5.2% | 45.1% |
| batter_home_runs|over|alt_tail | 24168 | keep_model_only | 0.009 | 0.009 | -31.8% | -27.2% | 37.4% |
| batter_home_runs|over|common | 24197 | keep_model_only | 0.109 | 0.109 | -33.9% | -33.4% | 37.4% |
| batter_total_bases|over|alt_tail | 48301 | keep_model_only | 0.097 | 0.096 | -20.3% | -8.7% | 38.8% |
| batter_total_bases|over|common | 61049 | keep_model_only | 0.209 | 0.209 | -16.5% | -15.1% | 40.3% |
| batter_total_bases|under|common | 12728 | keep_model_only | 0.243 | 0.243 | -2.9% | -3.5% | 40.1% |
| pitcher_strikeouts|over|common | 5847 | keep_model_only | 0.247 | 0.247 | -10.8% | -10.9% | 36.8% |
| pitcher_strikeouts|under|common | 5842 | keep_model_only | 0.247 | 0.247 | -2.2% | -2.9% | 49.4% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
