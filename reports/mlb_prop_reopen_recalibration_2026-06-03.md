# MLB Prop Reopen/Recalibration - 2026-06-03

## Current Status

- Historical prop market table built: 419,500 side examples from odds plus outcomes.
- Reopen policy history window loaded: 415,934 examples inside the active 540-day window.
- Historical market-side prior artifact trained: 265,280 priced examples, 92 models.
- Active bankroll prop reopen status: 0 exact buckets reopened.

The reopen result is still correct for real money. The locked model-pick sample has only 2,234 side rows and does not yet prove any prop bucket by train/holdout ROI, EV, CLV, and calibration.

## What The History Says

Market-side summary from the reopen diagnostics:

| Market side | Rows | Priced rate | Win rate | Avg market prob | Calibration error | ROI |
|---|---:|---:|---:|---:|---:|---:|
| batter hits over | 84,781 | 100.0% | 36.5% | 38.8% | -2.4% | -14.3% |
| batter hits under | 84,781 | 34.4% | 63.5% | 42.8% | +20.7% | -5.0% |
| batter HR over | 38,887 | 100.0% | 8.0% | 9.9% | -2.0% | -28.9% |
| batter HR under | 38,887 | 0.7% | 92.0% | 90.1% | +1.9% | -0.6% |
| batter total bases over | 77,365 | 100.0% | 27.9% | 31.7% | -3.8% | -17.9% |
| batter total bases under | 77,365 | 27.1% | 72.1% | 51.9% | +20.2% | -5.1% |
| pitcher strikeouts over | 6,934 | 100.0% | 48.8% | 49.6% | -0.8% | -7.5% |
| pitcher strikeouts under | 6,934 | 100.0% | 51.2% | 50.4% | +0.8% | -4.4% |

## Important Takeaways

- Total bases overs are structurally hostile in the raw market history. The prior learned to pull FanDuel TB over probabilities down hard, which is exactly what we want before reopening that bucket.
- Hit unders and TB unders are not truly "bad prediction" buckets yet; they are mostly bookability/price-data buckets. FanDuel often has over-only alt markets, so under pricing coverage is low and market calibration looks distorted.
- K props are the closest to reopenable on broad history. They still need model-pick proof and CLV proof from locked picks before bankroll use.
- HR overs remain lottery-only/research. The raw market history is deeply negative ROI.

## New Safety/Model Plumbing

- `prop_bucket_reopen_policy.json` now includes historical context and `history_reasons` for every bucket.
- The history guard is present but disabled by default via `enforce_history_guard=false`; it explains weak buckets without silently blocking future model-proofed buckets.
- `prop_market_side_priors.json` can be used as an optional runtime calibration prior.
- Runtime prior application is off by default via `apply_market_side_priors=False`, so live Discord output is unchanged until shadow testing says it helps.
- Fun/research reopen artifact added: `prop_bucket_reopen_policy_fun.json`.
  It uses `force_reopen_all=true` and `research_only=true`, so every prop bucket can flow for visibility while bankroll labels stay paper-only with `research_reopen_only`.

Run fun reopen mode with:

```powershell
.\.venv\Scripts\python.exe -m src.mlb_pipeline.modeling.predict_player_props --fun-reopen-props
```

## Prior Smoke Examples

- FanDuel TB over 1.5 at -110: model 62.0% -> prior-adjusted 57.0%.
- FanDuel hits over 0.5 at -150: model 68.0% -> prior-adjusted 64.4%.
- DraftKings K under 5.5 at -110: model 57.0% -> prior-adjusted 56.8%.

This is the shape we want: historically inflated batter overs get trimmed, while reasonably calibrated K sides barely move.
