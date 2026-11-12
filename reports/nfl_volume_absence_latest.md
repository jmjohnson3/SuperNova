# NFL volume: teammates ruled out (challenger)

Built 2026-11-12 20:38 UTC. Walk-forward 4-week blocks over 2023, 2024, 2025, 2026, each trained on earlier games only, same parameters and parity-safe features as the volume model, with and without the absence inputs. Research only; nothing live changes.

Absence inputs come from earlier box scores and the week's injury report (Out, Doubtful or injured reserve): vacated and newly vacated targets/carries, volume returning, the player's own volume without those teammates, vacated targets at his position, quarterback out.

## targets

| Player-games | Rows | Volume model MAE | With absence inputs | Change | Weekly difference (95% interval) |
|---|---:|---:|---:|---:|---:|
| All | 17900 | 1.656 | 1.644 | -0.7% | -0.010 (-0.014 to -0.005, 75 weeks) |
| 8+ targets newly vacated | 583 | 1.816 | 1.803 | -0.7% | -0.002 (-0.066 to +0.062, 39 weeks) |
| 8+ targets returning | 190 | 1.922 | 1.847 | -3.9% | -0.086 (-0.132 to -0.040, 22 weeks) |
| Season 2023 | 4841 | 1.684 | 1.682 | -0.1% | +0.001 (-0.008 to +0.010, 22 weeks) |
| Season 2024 | 4781 | 1.698 | 1.686 | -0.7% | -0.009 (-0.016 to -0.003, 22 weeks) |
| Season 2025 | 5629 | 1.537 | 1.523 | -0.9% | -0.014 (-0.020 to -0.008, 22 weeks) |
| Season 2026 | 2649 | 1.779 | 1.753 | -1.5% | -0.025 (-0.041 to -0.010, 9 weeks) |
| RB | 5194 | 1.332 | 1.322 | -0.7% | -0.007 (-0.013 to -0.002, 75 weeks) |
| TE | 4303 | 1.521 | 1.510 | -0.7% | -0.008 (-0.014 to -0.001, 75 weeks) |
| WR | 8403 | 1.925 | 1.911 | -0.7% | -0.012 (-0.019 to -0.005, 75 weeks) |

## carries

| Player-games | Rows | Volume model MAE | With absence inputs | Change | Weekly difference (95% interval) |
|---|---:|---:|---:|---:|---:|
| All | 7588 | 3.072 | 3.029 | -1.4% | -0.051 (-0.065 to -0.037, 75 weeks) |
| 8+ carries newly vacated | 399 | 3.598 | 3.235 | -10.1% | -0.443 (-0.640 to -0.247, 45 weeks) |
| 8+ carries returning | 130 | 3.377 | 3.129 | -7.3% | -0.327 (-0.629 to -0.025, 29 weeks) |
| Season 2023 | 2093 | 3.100 | 3.050 | -1.6% | -0.054 (-0.076 to -0.032, 22 weeks) |
| Season 2024 | 2105 | 3.047 | 3.005 | -1.4% | -0.054 (-0.087 to -0.022, 22 weeks) |
| Season 2025 | 2342 | 2.869 | 2.831 | -1.3% | -0.046 (-0.073 to -0.019, 22 weeks) |
| Season 2026 | 1048 | 3.521 | 3.474 | -1.3% | -0.046 (-0.097 to +0.005, 9 weeks) |
| QB | 2394 | 1.812 | 1.808 | -0.2% | -0.009 (-0.020 to +0.002, 75 weeks) |
| RB | 5194 | 3.653 | 3.591 | -1.7% | -0.070 (-0.089 to -0.050, 75 weeks) |

## Does it help yards? (live 2026 settled forecasts, same rows)

Rescaled: the live projection moved from the 5-game-average volume to the model's out-of-sample volume.

| Stat | Rows | Production MAE | Rescaled, volume model | Rescaled, with absence inputs |
|---|---:|---:|---:|---:|
| receiving_yards | 1564 | 20.67 | 21.27 | 21.09 |
| receptions | 1136 | 1.54 | 1.63 | 1.62 |
| rushing_yards | 681 | 20.52 | 21.27 | 21.11 |

Coverage: 43% of test player-games have a recent teammate ruled out; 3% have 8+ targets newly vacated.

Negative is better. The weekly difference is the mean over NFL weeks of the MAE with the absence inputs minus without.
