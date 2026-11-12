# NFL efficiency model (challenger)

Built 2026-11-12 21:53 UTC. Walk-forward 4-week blocks over 2023, 2024, 2025, 2026, each trained on earlier games only. Each estimate is a per-opportunity rate, scored as rate x the targets or carries the player actually had, so only efficiency is judged. Research only; nothing live changes.

Baseline = the player's last-10 rate shrunk toward his position's rate. Weekly difference is against the baseline (negative is better).

## receiving_yards with the volume known

| Player-games | Rows | Baseline | Model | + opponent | + opponent and weather | Best model, weekly difference (95%) |
|---|---:|---:|---:|---:|---:|---:|
| All | 15501 | 13.852 | 13.767 | 13.747 | 13.746 | + opponent and weather: -0.123 (-0.233 to -0.013) |
| Season 2023 | 4515 | 14.323 | 14.234 | 14.195 | 14.134 | + opponent and weather: -0.273 (-0.520 to -0.025) |
| Season 2024 | 4375 | 13.752 | 13.813 | 13.747 | 13.819 | + opponent: +0.042 (-0.175 to +0.260) |
| Season 2025 | 4451 | 13.496 | 13.307 | 13.343 | 13.348 | model: -0.160 (-0.302 to -0.018) |
| Season 2026 | 2160 | 13.804 | 13.645 | 13.645 | 13.606 | + opponent and weather: -0.197 (-0.417 to +0.022) |
| RB | 3777 | 9.472 | 9.423 | 9.414 | 9.415 | + opponent: -0.175 (-0.379 to +0.030) |
| TE | 3998 | 11.248 | 11.115 | 11.087 | 11.133 | + opponent: -0.176 (-0.334 to -0.019) |
| WR | 7726 | 17.340 | 17.263 | 17.243 | 17.215 | + opponent and weather: -0.099 (-0.256 to +0.058) |

## receptions with the volume known

| Player-games | Rows | Baseline | Model | + opponent | + opponent and weather | Best model, weekly difference (95%) |
|---|---:|---:|---:|---:|---:|---:|
| All | 15501 | 0.733 | 0.729 | 0.728 | 0.727 | + opponent and weather: -0.005 (-0.010 to +0.000) |
| Season 2023 | 4515 | 0.742 | 0.737 | 0.737 | 0.735 | + opponent and weather: -0.003 (-0.013 to +0.006) |
| Season 2024 | 4375 | 0.737 | 0.736 | 0.733 | 0.732 | + opponent and weather: -0.003 (-0.012 to +0.006) |
| Season 2025 | 4451 | 0.718 | 0.712 | 0.712 | 0.710 | + opponent and weather: -0.010 (-0.023 to +0.002) |
| Season 2026 | 2160 | 0.736 | 0.736 | 0.736 | 0.734 | + opponent and weather: -0.003 (-0.016 to +0.011) |
| RB | 3777 | 0.566 | 0.567 | 0.567 | 0.565 | + opponent and weather: +0.000 (-0.006 to +0.007) |
| TE | 3998 | 0.672 | 0.666 | 0.666 | 0.665 | + opponent and weather: -0.003 (-0.010 to +0.005) |
| WR | 7726 | 0.846 | 0.842 | 0.839 | 0.838 | + opponent and weather: -0.009 (-0.019 to +0.001) |

## rushing_yards with the volume known

| Player-games | Rows | Baseline | Model | + opponent | + opponent and weather | Best model, weekly difference (95%) |
|---|---:|---:|---:|---:|---:|---:|
| All | 6768 | 11.969 | 11.750 | 11.730 | 11.753 | + opponent: -0.351 (-0.614 to -0.088) |
| Season 2023 | 1945 | 11.381 | 11.202 | 11.213 | 11.163 | + opponent and weather: -0.448 (-0.955 to +0.058) |
| Season 2024 | 1961 | 12.237 | 12.025 | 12.036 | 12.059 | model: -0.391 (-1.244 to +0.461) |
| Season 2025 | 1941 | 12.540 | 12.286 | 12.201 | 12.259 | + opponent: -0.343 (-0.544 to -0.141) |
| Season 2026 | 921 | 11.435 | 11.187 | 11.180 | 11.284 | + opponent: -0.282 (-0.727 to +0.164) |
| QB | 2111 | 9.646 | 9.328 | 9.355 | 9.339 | model: -0.431 (-0.748 to -0.114) |
| RB | 4657 | 13.022 | 12.847 | 12.807 | 12.848 | + opponent: -0.332 (-0.646 to -0.018) |

## Live 2026 forecasts: production's volume proxy times each rate (average miss)

| Stat | Rows | Line | Production | Baseline | Model | + opponent | + opponent and weather |
|---|---:|---:|---:|---:|---:|---:|---:|
| receiving_yards, all | 1432 | - | 21.24 | 21.42 | 21.53 | 21.47 | 21.44 |
| receiving_yards, with a line | 1196 | 21.27 | 22.26 | 22.39 | 22.53 | 22.45 | 22.43 |
| receptions, all | 1049 | - | 1.53 | 1.54 | 1.53 | 1.54 | 1.54 |
| receptions, with a line | 936 | 1.58 | 1.60 | 1.61 | 1.60 | 1.61 | 1.61 |
| rushing_yards, all | 650 | - | 20.76 | 20.92 | 21.20 | 21.28 | 21.20 |
| rushing_yards, with a line | 553 | 18.89 | 20.54 | 21.17 | 21.42 | 21.59 | 21.48 |

The opponent columns are in the training table but not in the live snapshot, and the weather columns are the game's own weather; either would need a live source before use.
