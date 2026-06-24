# MLB Prop Probability Shadow - 2026-06-23

## Scope

- Locked side rows: 96107
- Date range: 2026-05-31 to 2026-06-21
- Unique graded dates: 22
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 96107 | 0.162 | -1.3% | 21715 | -10.0% | 40.3% | +0.17 |
| model_plus_prior | 96107 | 0.161 | -1.1% | 18274 | -4.8% | 40.7% | +0.18 |
| market_no_vig | 96107 | 0.157 | -0.4% | 6018 | 215.5% | 42.0% | +0.27 |
| direct_side_model | 96107 | 0.149 | -0.3% | 19817 | 59.7% | 41.5% | +0.21 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 30173 | keep_model_only_clv | 0.171 | 0.169 | -5.7% | 9.1% | 43.2% |
| batter_hits|under | 8139 | keep_model_only | 0.231 | 0.231 | -5.5% | -5.9% | 40.5% |
| batter_home_runs|over | 15356 | keep_model_only | 0.063 | 0.063 | -5.7% | 4.2% | 40.6% |
| batter_total_bases|over | 34741 | keep_model_only | 0.164 | 0.163 | -13.7% | -10.0% | 40.2% |
| batter_total_bases|under | 4203 | keep_model_only | 0.243 | 0.242 | -3.7% | -3.7% | 38.6% |
| pitcher_strikeouts|over | 1754 | keep_model_only_roi | 0.247 | 0.246 | -16.2% | -17.5% | 37.0% |
| pitcher_strikeouts|under | 1741 | keep_model_only_roi | 0.247 | 0.246 | 4.4% | 4.3% | 47.5% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 6785 | keep_model_only | 0.030 | 0.030 | 6.4% | 19.1% | 40.0% |
| batter_hits|over|common | 23388 | keep_model_only_clv | 0.212 | 0.210 | -14.6% | 5.5% | 44.3% |
| batter_hits|under|common | 8139 | keep_model_only | 0.231 | 0.231 | -5.5% | -5.9% | 40.5% |
| batter_home_runs|over|alt_tail | 7666 | keep_model_only | 0.010 | 0.010 | -5.3% | 5.4% | 40.7% |
| batter_home_runs|over|common | 7690 | keep_model_only | 0.116 | 0.116 | -7.3% | -0.2% | 40.2% |
| batter_total_bases|over|alt_tail | 15318 | keep_model_only | 0.104 | 0.103 | -16.2% | -11.0% | 38.7% |
| batter_total_bases|over|common | 19423 | keep_model_only | 0.211 | 0.210 | -7.0% | -7.4% | 44.1% |
| batter_total_bases|under|common | 4203 | keep_model_only | 0.243 | 0.242 | -3.7% | -3.7% | 38.6% |
| pitcher_strikeouts|over|common | 1754 | keep_model_only_roi | 0.247 | 0.246 | -16.2% | -17.5% | 37.0% |
| pitcher_strikeouts|under|common | 1741 | keep_model_only_roi | 0.247 | 0.246 | 4.4% | 4.3% | 47.5% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
