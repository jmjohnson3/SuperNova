# MLB Prop Distribution Models

Generated UTC: 2026-09-01T10:34:02Z
Rows: 1072
Raw rows before locked-offer dedupe: 67088
Collapsed duplicate locked-offer rows: 40802
Date range: 2026-07-02 to 2026-07-12
Status: ready

## Expanding Walk-Forward OOF

| Variant | Rows | Brier | Log Loss | Cal Err | Selected | ROI | CLV Beat |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 212 | 0.259 | 0.712 | 0.2% | 10 | -61.0% | 80.0% |
| market_no_vig | 212 | 0.255 | 0.704 | 0.0% | 0 | - | - |
| distribution | 212 | 0.282 | 0.765 | 0.0% | 75 | -6.4% | 49.0% |
| distribution_calibrated | 212 | 0.253 | 0.700 | 0.0% | 64 | 8.0% | 44.5% |
| distribution_empirical_blend | 212 | 0.253 | 0.699 | 0.0% | 77 | 1.9% | 39.9% |
| event_side_line | 212 | 0.266 | 0.727 | -0.1% | 84 | -13.4% | 38.6% |
| k_v3 | 212 | 0.303 | 0.841 | 0.0% | 95 | -8.8% | 35.2% |

## Pitcher K v3

- Status: trained
- Enabled: False
- Player-games: 225
- Holdout offers: 212
- K v3 Brier: 0.303
- Poisson Brier: 0.282
- Gain vs Poisson: -0.021
- BF bias / sigma: -0.943 / 3.878
- Beta-binomial concentration: 20.0

## Market Holdout

| Market | Rows | Model Brier | Distribution Brier | Cal Dist Brier | Blend Brier | Side-Line Brier | Model ROI | Blend ROI | Side-Line ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pitcher_strikeouts | 212 | 0.259 | 0.282 | 0.253 | 0.253 | 0.266 | -61.0% | 1.9% | -13.4% |

## TB/HR True-Pair Production Gates

These gates use holdout rows with true, non-synthetic paired prices only.

| Market / Side / Line | Rows | Brier Gain | Cal Err | Selected | CLV Beat | Avg CLV | Pass | Reasons |
|---|---:|---:|---:|---:|---:|---:|---|---|

## Hitter Outcome Shrinkage

| Group | Rows | PA | Hit Mult | TB Mult | HR Mult | XBH Mult | Actual H/PA | Pred H/PA | Actual TB/PA | Pred TB/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Direct Hitter Event Model

- Status: missing
- Method: -
- Trained UTC: -
- Classes: -
- Production gate: False
- Production eligible artifact: False
- Leakage-safe player priors: 0 players
- PA uncertainty groups: 0
- Direct event TB MAE gain vs independent rates: -
- Explicit TB-state rows: 0
- Explicit TB-state Brier: -
- Explicit TB-state log loss: -
- Direct-state selected candidate: convolution
- Direct-state blend alpha: -
- HR-driven 4+ tail Brier gain: -

## True-Pair Hitter Line Calibration

- Status: insufficient_true_pair_rows
- Evidence: market_split_temporal_train_true_pair_non_synthetic_only
- Calibrated line/side groups: 0
- Enabled line/side groups: 0
- Synthetic and one-sided FanDuel prices are display-only and cannot train these calibrators.

## Event-Curve Side/Line Models

| Target | Status | Train | Holdout | Model Brier | Baseline Brier | Model Avg | Baseline Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| win_probability | trained | 860 | 212 | 0.266 | 0.253 | 50.2% | 50.0% |
| clv_beat_probability | trained | 706 | 188 | 0.234 | 0.263 | 44.2% | 50.0% |

## TB Component Structure

| Group | Rows | PA | 1B Mult | 2B Mult | 3B Mult | HR Mult | TB Mult | Actual 0 TB | Pred 0 TB | Actual 2B/PA | Pred 2B/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Hitter Outcome Policy

| Market | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_hits | 0 | - | - | - | use_baseline_curve |
| batter_total_bases | 0 | - | - | - | use_baseline_curve |
| batter_home_runs | 0 | - | - | - | use_baseline_curve |

## TB Event Model Bucket Policy

| Bucket | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|

## Line-Bucket Probability Calibration

| Group | Rows | Columns | Method | Internal Gain | Holdout Gain | Enabled |
|---|---:|---|---|---:|---:|---|
| pitcher_strikeouts / market / side / line_bucket / pitcher_strikeouts / over / K 4.5-6.0 | 277 | market, side, line_bucket | beta | 0.011 | 0.010 | True |
| pitcher_strikeouts / market / side / line_bucket / pitcher_strikeouts / under / K 4.5-6.0 | 277 | market, side, line_bucket | beta | 0.011 | 0.010 | True |
| pitcher_strikeouts / market / side / line_surface / line_bucket / pitcher_strikeouts / over / common / K 4.5-6.0 | 277 | market, side, line_surface, line_bucket | beta | 0.011 | 0.010 | True |
| pitcher_strikeouts / market / side / line_surface / line_bucket / pitcher_strikeouts / under / common / K 4.5-6.0 | 277 | market, side, line_surface, line_bucket | beta | 0.011 | 0.010 | True |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / pitcher_strikeouts / over / common / K 4.5-6.0 / plus_100_149 | 125 | market, side, line_surface, line_bucket, price_bucket | beta | 0.025 | -0.023 | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / pitcher_strikeouts / under / common / K 4.5-6.0 / plus_100_149 | 90 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_bucket / pitcher_strikeouts / over / K <4.5 | 79 | market, side, line_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_bucket / pitcher_strikeouts / under / K <4.5 | 79 | market, side, line_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / pitcher_strikeouts / over / common / K <4.5 | 79 | market, side, line_surface, line_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / pitcher_strikeouts / under / common / K <4.5 | 79 | market, side, line_surface, line_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / pitcher_strikeouts / under / common / K 4.5-6.0 / lay_150_180 | 74 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / pitcher_strikeouts / under / common / K 4.5-6.0 / fair_lay | 72 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_bucket / pitcher_strikeouts / over / K 6.5-8.0 | 71 | market, side, line_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_bucket / pitcher_strikeouts / under / K 6.5-8.0 | 71 | market, side, line_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / pitcher_strikeouts / over / common / K 6.5-8.0 | 71 | market, side, line_surface, line_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / pitcher_strikeouts / under / common / K 6.5-8.0 | 71 | market, side, line_surface, line_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / over / common / K 4.5-6.0 / plus_100_149 / draftkings | 68 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / pitcher_strikeouts / over / common / K 4.5-6.0 / fair_lay | 67 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / pitcher_strikeouts / under / common / K <4.5 / plus_100_149 | 60 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / over / common / K 4.5-6.0 / plus_100_149 / fanduel | 57 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / pitcher_strikeouts / over / common / K 4.5-6.0 / lay_130_149 | 50 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / under / common / K 4.5-6.0 / plus_100_149 / fanduel | 46 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / under / common / K 4.5-6.0 / plus_100_149 / draftkings | 44 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / under / common / K 4.5-6.0 / lay_150_180 / draftkings | 43 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / pitcher_strikeouts / over / common / K <4.5 / lay_150_180 | 41 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / pitcher_strikeouts / under / common / K 4.5-6.0 / lay_130_149 | 41 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / pitcher_strikeouts / over / common / K 6.5-8.0 / plus_100_149 | 38 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / under / common / K 4.5-6.0 / fair_lay / draftkings | 37 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / under / common / K 4.5-6.0 / fair_lay / fanduel | 35 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / pitcher_strikeouts / over / common / K 4.5-6.0 / lay_150_180 | 35 | market, side, line_surface, line_bucket, price_bucket | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / over / common / K 4.5-6.0 / fair_lay / draftkings | 34 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / over / common / K 4.5-6.0 / fair_lay / fanduel | 33 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / under / common / K <4.5 / plus_100_149 / draftkings | 32 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / under / common / K 4.5-6.0 / lay_150_180 / fanduel | 31 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / under / common / K <4.5 / plus_100_149 / fanduel | 28 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / over / common / K 4.5-6.0 / lay_130_149 / draftkings | 27 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / over / common / K 4.5-6.0 / lay_130_149 / fanduel | 23 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / over / common / K <4.5 / lay_150_180 / draftkings | 23 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / under / common / K 4.5-6.0 / lay_130_149 / draftkings | 21 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |
| pitcher_strikeouts / market / side / line_surface / line_bucket / price_bucket / bookmaker_key / pitcher_strikeouts / over / common / K 6.5-8.0 / plus_100_149 / draftkings | 20 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | - | - | False |

## Exact Bucket Model Selection

| Bucket | Rows | Decision | Best | Best Brier | ROI | Model | Market | Distribution | Cal Dist | Blend | Side-Line | K v3 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 20 | no_bet_sample | distribution | 0.261 | 70.4% | 0.264 | 0.264 | 0.261 | 0.269 | 0.274 | 0.304 | 0.291 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 20 | no_bet_sample | market_only | 0.260 | - | 0.260 | 0.260 | 0.267 | 0.263 | 0.268 | 0.294 | 0.270 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 13 | no_bet_sample | distribution | 0.257 | -25.2% | 0.271 | 0.271 | 0.257 | 0.280 | 0.283 | 0.311 | 0.310 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 12 | no_bet_sample | distribution_calibrated | 0.253 | 18.5% | 0.259 | 0.259 | 0.305 | 0.253 | 0.254 | 0.290 | 0.274 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 11 | no_bet_sample | market_only | 0.250 | - | 0.269 | 0.250 | 0.271 | 0.256 | 0.259 | 0.280 | 0.266 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 10 | no_bet_sample | market_only | 0.257 | - | 0.282 | 0.257 | 0.318 | 0.274 | 0.269 | 0.305 | 0.403 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 10 | no_bet_sample | market_only | 0.252 | - | 0.272 | 0.252 | 0.331 | 0.276 | 0.271 | 0.298 | 0.413 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 10 | no_bet_sample | k_v3 | 0.180 | 45.2% | 0.265 | 0.265 | 0.253 | 0.237 | 0.242 | 0.267 | 0.180 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 8 | no_bet_sample | distribution | 0.246 | 64.1% | 0.272 | 0.278 | 0.246 | 0.280 | 0.299 | 0.360 | 0.292 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 7 | no_bet_sample | distribution | 0.233 | -100.0% | 0.275 | 0.262 | 0.233 | 0.266 | 0.268 | 0.290 | 0.424 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 6 | no_bet_sample | k_v3 | 0.159 | - | 0.283 | 0.283 | 0.294 | 0.237 | 0.254 | 0.326 | 0.159 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|fanduel | 6 | no_bet_sample | k_v3 | 0.193 | - | 0.253 | 0.253 | 0.279 | 0.266 | 0.255 | 0.271 | 0.193 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 6 | no_bet_sample | distribution | 0.248 | 94.3% | 0.271 | 0.249 | 0.248 | 0.284 | 0.289 | 0.314 | 0.424 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 6 | no_bet_sample | model_only | 0.256 | - | 0.256 | 0.256 | 0.274 | 0.260 | 0.263 | 0.284 | 0.274 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 5 | no_bet_sample | distribution | 0.174 | - | 0.215 | 0.215 | 0.174 | 0.231 | 0.232 | 0.291 | 0.184 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 5 | no_bet_sample | event_side_line | 0.173 | 65.7% | 0.287 | 0.287 | 0.326 | 0.208 | 0.204 | 0.173 | 0.339 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|fanduel | 4 | no_bet_sample | market_only | 0.255 | - | 0.255 | 0.255 | 0.361 | 0.277 | 0.268 | 0.316 | 0.366 |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_150_180|draftkings | 4 | no_bet_sample | k_v3 | 0.101 | 63.7% | 0.178 | 0.178 | 0.126 | 0.191 | 0.196 | 0.251 | 0.101 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 4 | no_bet_sample | event_side_line | 0.133 | 118.5% | 0.322 | 0.322 | 0.305 | 0.173 | 0.146 | 0.133 | 0.222 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 3 | no_bet_sample | model_only | 0.230 | - | 0.230 | 0.230 | 0.242 | 0.268 | 0.258 | 0.236 | 0.265 |
| pitcher_strikeouts|over|common|K <4.5|fair_lay|draftkings | 3 | no_bet_sample | event_side_line | 0.157 | - | 0.236 | 0.236 | 0.338 | 0.220 | 0.220 | 0.157 | 0.344 |
| pitcher_strikeouts|over|common|K <4.5|lay_130_149|fanduel | 3 | no_bet_sample | event_side_line | 0.192 | - | 0.262 | 0.262 | 0.290 | 0.223 | 0.223 | 0.192 | 0.355 |
| pitcher_strikeouts|over|common|K <4.5|plus_100_149|fanduel | 3 | no_bet_sample | event_side_line | 0.114 | -100.0% | 0.187 | 0.187 | 0.321 | 0.174 | 0.162 | 0.114 | 0.375 |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|draftkings | 3 | no_bet_sample | event_side_line | 0.155 | 23.8% | 0.236 | 0.236 | 0.338 | 0.220 | 0.220 | 0.155 | 0.344 |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_130_149|fanduel | 2 | no_bet_sample | event_side_line | 0.194 | - | 0.209 | 0.209 | 0.363 | 0.328 | 0.318 | 0.194 | 0.403 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|fanduel | 2 | no_bet_sample | market_only | 0.249 | - | 0.249 | 0.249 | 0.261 | 0.272 | 0.274 | 0.270 | 0.254 |
| pitcher_strikeouts|over|common|K <4.5|fair_lay|fanduel | 2 | no_bet_sample | event_side_line | 0.108 | - | 0.231 | 0.231 | 0.309 | 0.168 | 0.158 | 0.108 | 0.153 |
| pitcher_strikeouts|over|common|K <4.5|lay_130_149|draftkings | 2 | no_bet_sample | distribution_blend | 0.161 | - | 0.300 | 0.300 | 0.238 | 0.173 | 0.161 | 0.178 | 0.169 |
| pitcher_strikeouts|over|common|K <4.5|lay_150_180|draftkings | 2 | no_bet_sample | event_side_line | 0.112 | - | 0.355 | 0.355 | 0.406 | 0.173 | 0.130 | 0.112 | 0.301 |
| pitcher_strikeouts|over|common|K <4.5|lay_150_180|fanduel | 2 | no_bet_sample | event_side_line | 0.135 | - | 0.344 | 0.344 | 0.406 | 0.173 | 0.142 | 0.135 | 0.301 |
| pitcher_strikeouts|over|common|K <4.5|plus_100_149|draftkings | 2 | no_bet_sample | event_side_line | 0.139 | -100.0% | 0.202 | 0.202 | 0.297 | 0.177 | 0.164 | 0.139 | 0.410 |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|draftkings | 2 | no_bet_sample | market_only | 0.198 | - | 0.198 | 0.198 | 0.363 | 0.328 | 0.318 | 0.199 | 0.403 |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|fanduel | 2 | no_bet_sample | model_only | 0.209 | - | 0.209 | 0.209 | 0.363 | 0.328 | 0.318 | 0.213 | 0.403 |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|fanduel | 2 | no_bet_sample | event_side_line | 0.096 | 81.7% | 0.231 | 0.231 | 0.309 | 0.168 | 0.158 | 0.096 | 0.153 |
| pitcher_strikeouts|under|common|K <4.5|lay_130_149|draftkings | 2 | no_bet_sample | event_side_line | 0.130 | 75.2% | 0.202 | 0.202 | 0.297 | 0.177 | 0.164 | 0.130 | 0.410 |
| pitcher_strikeouts|under|common|K <4.5|lay_130_149|fanduel | 2 | no_bet_sample | event_side_line | 0.047 | 69.1% | 0.195 | 0.195 | 0.445 | 0.164 | 0.155 | 0.047 | 0.243 |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_130_149|draftkings | 1 | no_bet_sample | market_only | 0.216 | - | 0.216 | 0.216 | 0.341 | 0.331 | 0.320 | 0.218 | 0.397 |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|draftkings | 1 | no_bet_sample | event_side_line | 0.170 | - | 0.186 | 0.186 | 0.378 | 0.326 | 0.316 | 0.170 | 0.408 |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|draftkings | 1 | no_bet_sample | distribution | 0.302 | - | 0.315 | 0.315 | 0.302 | 0.336 | 0.324 | 0.345 | 0.404 |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|fanduel | 1 | no_bet_sample | distribution | 0.302 | - | 0.303 | 0.303 | 0.302 | 0.336 | 0.324 | 0.347 | 0.404 |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_150_180|fanduel | 1 | no_bet_sample | k_v3 | 0.027 | 60.2% | 0.169 | 0.169 | 0.200 | 0.177 | 0.186 | 0.153 | 0.027 |
| pitcher_strikeouts|under|common|K <4.5|lay_150_180|fanduel | 1 | no_bet_sample | distribution | 0.114 | 61.7% | 0.174 | 0.174 | 0.114 | 0.192 | 0.173 | 0.196 | 0.594 |
