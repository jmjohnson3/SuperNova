# MLB Prop Probability Shadow - 2026-06-09

## Scope

- Locked side rows: 24974
- Date range: 2026-05-31 to 2026-06-08
- Unique graded dates: 9
- Minimum EV for simulated picks: 0.020
- Shadow winner: model_only

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 24974 | 0.152 | -0.9% | 4262 | -13.8% | 41.8% | +0.15 |
| model_plus_prior | 24974 | 0.151 | -0.6% | 3166 | -3.8% | 43.2% | +0.19 |
| market_no_vig | 24974 | 0.149 | -1.2% | 698 | 165.9% | 40.7% | +0.21 |
| direct_side_model | 24974 | 0.146 | -0.5% | 4231 | 52.0% | 39.9% | +0.14 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 9168 | use_model_plus_prior | 0.145 | 0.143 | -1.9% | 20.3% | 48.3% |
| batter_hits|under | 2065 | keep_model_only | 0.241 | 0.241 | -3.3% | -2.1% | 41.7% |
| batter_home_runs|over | 3676 | keep_model_only | 0.059 | 0.059 | -68.9% | -54.6% | 45.5% |
| batter_total_bases|over | 8269 | keep_model_only | 0.160 | 0.159 | -12.9% | -3.3% | 44.3% |
| batter_total_bases|under | 1103 | keep_model_only | 0.240 | 0.239 | -3.0% | -4.5% | 38.3% |
| pitcher_strikeouts|over | 353 | use_model_plus_prior | 0.252 | 0.250 | -44.3% | -43.5% | 47.3% |
| pitcher_strikeouts|under | 340 | keep_model_only_roi | 0.248 | 0.246 | 28.8% | 27.9% | 40.6% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 3540 | keep_model_only | 0.028 | 0.028 | -4.7% | 26.8% | 48.5% |
| batter_hits|over|common | 5628 | keep_model_only_clv | 0.218 | 0.216 | 7.9% | 12.9% | 48.0% |
| batter_hits|under|common | 2065 | keep_model_only | 0.241 | 0.241 | -3.3% | -2.1% | 41.7% |
| batter_home_runs|over|alt_tail | 1829 | keep_model_only | 0.009 | 0.009 | -100.0% | -100.0% | 49.4% |
| batter_home_runs|over|common | 1847 | keep_model_only | 0.109 | 0.108 | -22.7% | -10.2% | 42.5% |
| batter_total_bases|over|alt_tail | 3632 | keep_model_only | 0.100 | 0.099 | -12.5% | 1.1% | 41.8% |
| batter_total_bases|over|common | 4637 | keep_model_only | 0.207 | 0.206 | -14.2% | -17.7% | 52.9% |
| batter_total_bases|under|common | 1103 | keep_model_only | 0.240 | 0.239 | -3.0% | -4.5% | 38.3% |
| pitcher_strikeouts|over|common | 353 | use_model_plus_prior | 0.252 | 0.250 | -44.3% | -43.5% | 47.3% |
| pitcher_strikeouts|under|common | 340 | keep_model_only_roi | 0.248 | 0.246 | 28.8% | 27.9% | 40.6% |

## Shadow Winner

No shadow variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | false | clv_beat_rate_not_improved |
| direct_side_model | false | clv_beat_rate_not_improved, avg_clv_price_not_improved |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
