# NBA prop selector: paper picks

`nba-ml-selector-v1`. Research only: no stakes, no alerts. A pick is a FanDuel, DraftKings or theScore Bet points, rebounds or assists offer, seen 15-45 minutes before tip, that the model values at least 6% above its price; one pick per player and stat.

269 picks over 14 game days, 2026-10-24 to 2026-11-09.

The edge against Pinnacle is the fast test: a model that finds real value should show a positive number there. Results at $1 are noisy until there are far more picks.

| Picks | Count | Settled | Record | Result ($1 each) | Per bet | Model's EV | Edge vs Pinnacle at pick | Edge vs Pinnacle at close |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| All | 269 | 267 | 134-133 | -$22.02 | -8.2% | +8.0% | -4.8% (138) | -4.5% (143) |
| Replayed from saved quotes | 269 | 267 | 134-133 | -$22.02 | -8.2% | +8.0% | -4.8% (138) | -4.5% (143) |
| Book: draftkings | 68 | 67 | 40-27 | +$6.68 | +10.0% | +8.4% | -4.7% (33) | -3.8% (32) |
| Book: espnbet | 38 | 38 | 17-21 | -$6.73 | -17.7% | +7.8% | -4.8% (20) | -4.8% (20) |
| Book: fanduel | 163 | 162 | 77-85 | -$21.96 | -13.6% | +7.9% | -4.9% (85) | -4.7% (91) |
| Side: over | 87 | 86 | 48-38 | +$2.36 | +2.7% | +7.8% | -6.1% (37) | -6.5% (39) |
| Side: under | 182 | 181 | 86-95 | -$24.38 | -13.5% | +8.1% | -4.4% (101) | -3.8% (104) |
| Stat: assists | 43 | 43 | 19-24 | -$9.40 | -21.9% | +7.1% | -3.7% (20) | -4.0% (21) |
| Stat: points | 160 | 159 | 86-73 | -$0.70 | -0.4% | +8.4% | -5.6% (81) | -5.2% (83) |
| Stat: rebounds | 66 | 65 | 29-36 | -$11.91 | -18.3% | +7.6% | -3.8% (37) | -3.3% (39) |

Edge at the close across all picks: -4.5%, standard error 0.4% (whole days as draws).

Replayed picks come from quotes saved before this model existed, so the model had never seen those games. Picks that sit below zero against Pinnacle are paying the book's cut, not finding value.
