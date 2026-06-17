# MLB Prop Probability Shadow - 2026-06-17

## Scope

- Locked side rows: 65072
- Date range: 2026-05-31 to 2026-06-15
- Unique graded dates: 16
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 65072 | 0.159 | -1.2% | 12754 | -8.5% | 42.4% | +0.20 |
| model_plus_prior | 65072 | 0.158 | -0.9% | 9590 | -0.2% | 42.5% | +0.21 |
| market_no_vig | 65072 | 0.154 | -0.6% | 4022 | 218.8% | 42.9% | +0.30 |
| direct_side_model | 65072 | 0.146 | -0.3% | 13374 | 66.2% | 41.1% | +0.19 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 21394 | keep_model_only_clv | 0.161 | 0.160 | 1.4% | 26.8% | 45.2% |
| batter_hits|under | 5425 | keep_model_only | 0.233 | 0.233 | -6.6% | -5.9% | 41.1% |
| batter_home_runs|over | 10106 | keep_model_only | 0.064 | 0.064 | -5.8% | 17.2% | 43.7% |
| batter_total_bases|over | 22956 | keep_model_only | 0.163 | 0.162 | -14.5% | -7.6% | 42.6% |
| batter_total_bases|under | 2926 | keep_model_only | 0.242 | 0.242 | -3.7% | -4.2% | 38.7% |
| pitcher_strikeouts|over | 1139 | keep_model_only_roi | 0.251 | 0.250 | -16.2% | -17.0% | 37.6% |
| pitcher_strikeouts|under | 1126 | keep_model_only_clv | 0.251 | 0.250 | 6.0% | 6.9% | 47.4% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 5974 | keep_model_only | 0.030 | 0.029 | 16.4% | 31.6% | 45.3% |
| batter_hits|over|common | 15420 | keep_model_only_clv | 0.212 | 0.210 | -18.4% | 23.0% | 45.2% |
| batter_hits|under|common | 5425 | keep_model_only | 0.233 | 0.233 | -6.6% | -5.9% | 41.1% |
| batter_home_runs|over|alt_tail | 5043 | keep_model_only | 0.011 | 0.011 | 1.5% | 35.5% | 43.8% |
| batter_home_runs|over|common | 5063 | keep_model_only | 0.117 | 0.116 | -23.2% | -22.2% | 43.5% |
| batter_total_bases|over|alt_tail | 10064 | keep_model_only | 0.103 | 0.102 | -15.9% | -7.1% | 40.8% |
| batter_total_bases|over|common | 12892 | keep_model_only | 0.209 | 0.208 | -9.9% | -8.8% | 48.2% |
| batter_total_bases|under|common | 2926 | keep_model_only | 0.242 | 0.242 | -3.7% | -4.2% | 38.7% |
| pitcher_strikeouts|over|common | 1139 | keep_model_only_roi | 0.251 | 0.250 | -16.2% | -17.0% | 37.6% |
| pitcher_strikeouts|under|common | 1126 | keep_model_only_clv | 0.251 | 0.250 | 6.0% | 6.9% | 47.4% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
