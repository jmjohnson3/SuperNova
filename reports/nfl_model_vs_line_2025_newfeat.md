# NFL Model vs Line (2025, honest week-by-week replay)

Backtest sides matched to a replayed projection: 7050 of 7710.

## 1. When the model disagrees with FanDuel's line, does the result follow it?

| Stat | Player-games | Corr(model-line, actual-line) | Same for baseline | Model's side wins (any gap) | 5+ | 10+ | 15+ | Model MAE | Model minus line (avg) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| receiving_yards | 947 | -0.053 | -0.033 | 51.5% (947) | 47.7% (398) | 44.8% (145) | 38.0% (50) | 21.0 | -1.1 |
| rushing_yards | 226 | -0.084 | -0.066 | 45.6% (226) | 44.8% (105) | 40.8% (49) | 38.1% (21) | 22.0 | -1.5 |

## 2. Do sharp gaps the model agrees with hold up better?

All snapshots (T-90/45/15). EV at the sharp close is the most reliable column; win rate and ROI are noisy.

| Stat | Gap | Model | Sides | EV when seen | EV at sharp close | Win rate | ROI (settled) |
|---|---|---|---:|---:|---:|---:|---:|
| receiving_yards | all sides | model agrees | 2850 | -5.19% | -5.23% | 51.4% | -2.70% (2850) |
| receiving_yards | all sides | model disagrees | 2850 | -5.53% | -5.49% | 48.6% | -8.02% (2850) |
| receiving_yards | all sides | agrees by 5+ | 1207 | -5.23% | -5.27% | 48.0% | -9.21% (1207) |
| receiving_yards | >= +1% | model agrees | 158 | +2.76% | +1.77% | 57.0% | +7.73% (158) |
| receiving_yards | >= +1% | model disagrees | 141 | +2.85% | +1.84% | 49.6% | -5.84% (141) |
| receiving_yards | >= +1% | agrees by 5+ | 60 | +2.68% | +2.14% | 48.3% | -8.74% (60) |
| receiving_yards | >= +3% | model agrees | 62 | +4.09% | +2.73% | 58.1% | +9.91% (62) |
| receiving_yards | >= +3% | model disagrees | 48 | +4.83% | +4.13% | 47.9% | -9.30% (48) |
| receiving_yards | >= +3% | agrees by 5+ | 25 | +4.00% | +3.23% | 36.0% | -31.86% (25) |
| rushing_yards | all sides | model agrees | 675 | -5.79% | -5.60% | 46.1% | -12.61% (675) |
| rushing_yards | all sides | model disagrees | 675 | -4.95% | -5.15% | 53.9% | +1.82% (675) |
| rushing_yards | all sides | agrees by 5+ | 317 | -5.81% | -5.81% | 45.1% | -14.61% (317) |
| rushing_yards | >= +1% | model agrees | 31 | +2.54% | +1.57% | 87.1% | +68.07% (31) |
| rushing_yards | >= +1% | model disagrees | 69 | +3.20% | +1.91% | 65.2% | +23.45% (69) |
| rushing_yards | >= +1% | agrees by 5+ | 12 | +2.42% | +0.41% | 75.0% | +41.96% (12) |
| rushing_yards | >= +3% | model agrees | 8 | +4.37% | +1.28% | 100.0% | +89.29% (8) |
| rushing_yards | >= +3% | model disagrees | 29 | +5.21% | +4.26% | 69.0% | +30.54% (29) |
| rushing_yards | >= +3% | agrees by 5+ | 4 | +3.58% | -2.29% | 100.0% | +89.29% (4) |
