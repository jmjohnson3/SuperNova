# NFL Model vs Line (2025, honest week-by-week replay)

Backtest sides matched to a replayed projection: 7044 of 7710.

## 1. When the model disagrees with FanDuel's line, does the result follow it?

| Stat | Player-games | Corr(model-line, actual-line) | Same for baseline | Model's side wins (any gap) | 5+ | 10+ | 15+ | Model MAE | Model minus line (avg) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| receiving_yards | 946 | -0.018 | -0.033 | 51.4% (946) | 52.5% (398) | 49.0% (157) | 43.3% (60) | 21.0 | -1.3 |
| rushing_yards | 226 | -0.146 | -0.066 | 47.8% (226) | 48.0% (100) | 38.9% (54) | 33.3% (24) | 22.2 | -2.2 |

## 2. Do sharp gaps the model agrees with hold up better?

All snapshots (T-90/45/15). EV at the sharp close is the most reliable column; win rate and ROI are noisy.

| Stat | Gap | Model | Sides | EV when seen | EV at sharp close | Win rate | ROI (settled) |
|---|---|---|---:|---:|---:|---:|---:|
| receiving_yards | all sides | model agrees | 2847 | -5.13% | -5.18% | 51.1% | -3.34% (2847) |
| receiving_yards | all sides | model disagrees | 2847 | -5.59% | -5.54% | 48.9% | -7.39% (2847) |
| receiving_yards | all sides | agrees by 5+ | 1205 | -5.20% | -5.25% | 52.0% | -1.52% (1205) |
| receiving_yards | >= +1% | model agrees | 129 | +2.62% | +1.90% | 54.3% | +2.61% (129) |
| receiving_yards | >= +1% | model disagrees | 110 | +2.61% | +1.35% | 53.6% | +1.76% (110) |
| receiving_yards | >= +1% | agrees by 5+ | 54 | +2.38% | +1.52% | 40.7% | -23.14% (54) |
| receiving_yards | >= +3% | model agrees | 52 | +3.96% | +3.38% | 50.0% | -5.36% (52) |
| receiving_yards | >= +3% | model disagrees | 38 | +4.31% | +2.65% | 52.6% | -0.38% (38) |
| receiving_yards | >= +3% | agrees by 5+ | 21 | +3.81% | +2.94% | 47.6% | -9.86% (21) |
| rushing_yards | all sides | model agrees | 675 | -5.62% | -5.47% | 47.9% | -9.22% (675) |
| rushing_yards | all sides | model disagrees | 675 | -5.12% | -5.27% | 52.1% | -1.57% (675) |
| rushing_yards | all sides | agrees by 5+ | 297 | -6.17% | -5.96% | 47.5% | -10.14% (297) |
| rushing_yards | >= +1% | model agrees | 39 | +2.17% | +1.46% | 92.3% | +77.40% (39) |
| rushing_yards | >= +1% | model disagrees | 50 | +3.29% | +2.24% | 68.0% | +28.62% (50) |
| rushing_yards | >= +1% | agrees by 5+ | 17 | +1.69% | +0.55% | 82.4% | +55.88% (17) |
| rushing_yards | >= +3% | model agrees | 4 | +4.38% | +3.79% | 100.0% | +89.29% (4) |
| rushing_yards | >= +3% | model disagrees | 26 | +4.57% | +4.00% | 80.8% | +52.88% (26) |
| rushing_yards | >= +3% | agrees by 5+ | 0 | - | - | - | - (0) |
