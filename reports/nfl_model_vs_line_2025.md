# NFL Model vs Line (2025, honest week-by-week replay)

Backtest sides matched to a replayed projection: 7050 of 7710.

## 1. When the model disagrees with FanDuel's line, does the result follow it?

| Stat | Player-games | Corr(model-line, actual-line) | Same for baseline | Model's side wins (any gap) | 5+ | 10+ | 15+ | Model MAE | Model minus line (avg) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| receiving_yards | 947 | -0.014 | -0.033 | 51.7% (947) | 50.3% (382) | 48.5% (130) | 37.8% (45) | 20.8 | -0.9 |
| rushing_yards | 226 | -0.057 | -0.048 | 47.8% (226) | 43.8% (105) | 50.0% (44) | 47.4% (19) | 21.7 | -1.6 |

## 2. Do sharp gaps the model agrees with hold up better?

All snapshots (T-90/45/15). EV at the sharp close is the most reliable column; win rate and ROI are noisy.

| Stat | Gap | Model | Sides | EV when seen | EV at sharp close | Win rate | ROI (settled) |
|---|---|---|---:|---:|---:|---:|---:|
| receiving_yards | all sides | model agrees | 2850 | -5.24% | -5.29% | 52.0% | -1.58% (2850) |
| receiving_yards | all sides | model disagrees | 2850 | -5.49% | -5.43% | 48.0% | -9.15% (2850) |
| receiving_yards | all sides | agrees by 5+ | 1154 | -5.09% | -5.13% | 50.4% | -4.55% (1154) |
| receiving_yards | >= +1% | model agrees | 124 | +2.51% | +1.26% | 55.6% | +5.22% (124) |
| receiving_yards | >= +1% | model disagrees | 115 | +2.72% | +2.08% | 52.2% | -1.01% (115) |
| receiving_yards | >= +1% | agrees by 5+ | 60 | +2.29% | +1.19% | 46.7% | -11.89% (60) |
| receiving_yards | >= +3% | model agrees | 45 | +3.97% | +2.61% | 53.3% | +0.95% (45) |
| receiving_yards | >= +3% | model disagrees | 45 | +4.24% | +3.57% | 48.9% | -7.46% (45) |
| receiving_yards | >= +3% | agrees by 5+ | 22 | +3.67% | +2.81% | 36.4% | -31.17% (22) |
| rushing_yards | all sides | model agrees | 675 | -5.53% | -5.42% | 48.0% | -9.09% (675) |
| rushing_yards | all sides | model disagrees | 675 | -5.20% | -5.32% | 52.0% | -1.70% (675) |
| rushing_yards | all sides | agrees by 5+ | 317 | -5.62% | -5.49% | 45.4% | -13.64% (317) |
| rushing_yards | >= +1% | model agrees | 40 | +2.38% | +1.65% | 90.0% | +72.96% (40) |
| rushing_yards | >= +1% | model disagrees | 49 | +3.13% | +2.10% | 69.4% | +31.25% (49) |
| rushing_yards | >= +1% | agrees by 5+ | 19 | +2.33% | +2.26% | 84.2% | +64.88% (19) |
| rushing_yards | >= +3% | model agrees | 8 | +3.92% | +3.59% | 100.0% | +89.29% (8) |
| rushing_yards | >= +3% | model disagrees | 22 | +4.77% | +4.11% | 77.3% | +46.27% (22) |
| rushing_yards | >= +3% | agrees by 5+ | 4 | +3.87% | +3.21% | 100.0% | +89.29% (4) |
