# MLB Prop Probability Shadow - 2026-07-15

## Scope

- Locked side rows: 67088
- Date range: 2026-07-02 to 2026-07-12
- Unique graded dates: 11
- Minimum EV for simulated picks: 0.020
- Shadow winner: market_no_vig

## Overall

| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 67088 | 0.168 | -3.0% | 10227 | -18.0% | 40.1% | +0.12 |
| model_plus_prior | 67088 | 0.167 | -2.8% | 9242 | -7.1% | 40.9% | +0.13 |
| market_no_vig | 67088 | 0.153 | 1.9% | 5911 | 166.1% | 48.7% | +0.33 |
| direct_side_model | 67088 | 0.157 | -0.6% | 22546 | 20.8% | 40.5% | +0.13 |

## Market/Side Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over | 18719 | keep_model_only_clv | 0.197 | 0.196 | -10.3% | 55.6% | 50.2% |
| batter_hits|under | 5804 | keep_model_only | 0.236 | 0.236 | 58.6% | 67.7% | 68.2% |
| batter_home_runs|over | 11263 | keep_model_only | 0.060 | 0.060 | N/A | -100.0% | 16.7% |
| batter_total_bases|over | 25516 | keep_model_only | 0.161 | 0.161 | -19.0% | -12.6% | 40.0% |
| batter_total_bases|under | 2972 | keep_model_only | 0.240 | 0.240 | N/A | N/A | N/A |
| pitcher_strikeouts|over | 1407 | keep_model_only | 0.249 | 0.249 | -17.7% | -10.8% | 44.8% |
| pitcher_strikeouts|under | 1407 | keep_model_only | 0.250 | 0.249 | 25.4% | 26.0% | 46.3% |

## Line Surface Recommendations

| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |
|---|---:|---|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail | 1643 | keep_model_only | 0.021 | 0.021 | N/A | N/A | N/A |
| batter_hits|over|common | 17076 | keep_model_only_clv | 0.214 | 0.213 | -10.3% | 55.6% | 50.2% |
| batter_hits|under|common | 5804 | keep_model_only | 0.236 | 0.236 | 58.6% | 67.7% | 68.2% |
| batter_home_runs|over|alt_tail | 5627 | keep_model_only | 0.008 | 0.008 | N/A | -100.0% | 33.3% |
| batter_home_runs|over|common | 5636 | keep_model_only | 0.111 | 0.111 | N/A | -100.0% | 0.0% |
| batter_total_bases|over|alt_tail | 11272 | keep_model_only | 0.099 | 0.099 | -19.0% | -2.3% | 41.3% |
| batter_total_bases|over|common | 14244 | keep_model_only | 0.210 | 0.209 | -19.0% | -17.1% | 39.4% |
| batter_total_bases|under|common | 2972 | keep_model_only | 0.240 | 0.240 | N/A | N/A | N/A |
| pitcher_strikeouts|over|common | 1407 | keep_model_only | 0.249 | 0.249 | -17.7% | -10.8% | 44.8% |
| pitcher_strikeouts|under|common | 1407 | keep_model_only | 0.250 | 0.249 | 25.4% | 26.0% | 46.3% |

## Shadow Winner

Selected variant improved Brier, ROI, and CLV versus model_only.

| Candidate | Eligible | Reasons |
|---|---:|---|
| model_plus_prior | false | brier_not_improved |
| market_no_vig | true | passes |
| direct_side_model | true | passes |

## Rule

A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.
