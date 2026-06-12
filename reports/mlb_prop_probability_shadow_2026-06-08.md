# MLB Prop Probability Shadow - 2026-06-08

## Scope

- Locked side rows: 15819
- Date range: 2026-05-31 to 2026-06-06
- Unique graded dates: 7
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 15819 | 0.156 | -0.3% | 2832 | -6.2% | 42.0% | +0.14 |
| model_plus_prior | 15819 | 0.154 | 0.0% | 2113 | -0.0% | 43.2% | +0.17 |
| market_no_vig | 15819 | 0.151 | -1.0% | 409 | 164.7% | 45.8% | +0.39 |
| direct_side_model | 15819 | 0.145 | 0.0% | 3246 | 69.9% | 38.9% | +0.14 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 5725 | use_model_plus_prior | 0.146 | 0.144 | 32.7% | 40.6% | 47.2% |
| batter_hits|under | 1351 | keep_model_only | 0.241 | 0.241 | -9.0% | -8.6% | 41.7% |
| batter_home_runs|over | 2294 | keep_model_only | 0.062 | 0.062 | -66.1% | -53.4% | 50.4% |
| batter_total_bases|over | 5141 | keep_model_only | 0.163 | 0.162 | -1.8% | 15.2% | 45.5% |
| batter_total_bases|under | 739 | keep_model_only | 0.242 | 0.242 | -6.2% | -8.4% | 35.9% |
| pitcher_strikeouts|over | 291 | use_model_plus_prior | 0.248 | 0.244 | -44.3% | -43.5% | 47.3% |
| pitcher_strikeouts|under | 278 | keep_model_only_roi | 0.249 | 0.247 | 16.0% | 14.0% | 42.9% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 2193 | keep_model_only | 0.029 | 0.029 | 44.6% | 46.3% | 47.6% |
| batter_hits|over|common | 3532 | use_model_plus_prior | 0.219 | 0.215 | -6.7% | 32.5% | 46.7% |
| batter_hits|under|common | 1351 | keep_model_only | 0.241 | 0.241 | -9.0% | -8.6% | 41.7% |
| batter_home_runs|over|alt_tail | 1138 | keep_model_only | 0.012 | 0.012 | -100.0% | -100.0% | 62.8% |
| batter_home_runs|over|common | 1156 | keep_model_only | 0.112 | 0.111 | -30.2% | -15.0% | 43.8% |
| batter_total_bases|over|alt_tail | 2250 | keep_model_only | 0.104 | 0.103 | 1.6% | 15.1% | 43.9% |
| batter_total_bases|over|common | 2891 | keep_model_only | 0.209 | 0.208 | -12.4% | 15.4% | 51.9% |
| batter_total_bases|under|common | 739 | keep_model_only | 0.242 | 0.242 | -6.2% | -8.4% | 35.9% |
| pitcher_strikeouts|over|common | 291 | use_model_plus_prior | 0.248 | 0.244 | -44.3% | -43.5% | 47.3% |
| pitcher_strikeouts|under|common | 278 | keep_model_only_roi | 0.249 | 0.247 | 16.0% | 14.0% | 42.9% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | true | passes |
| market_no_vig | true | passes |
| direct_side_model | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
