# NFL Sharp-Gap Strategy

Verdict: **trial: $1 stakes within limits** - $1 flat within the daily, weekly and per-game limits

Registration `nfl-sharp-gap-cash-v7`: fanduel, espnbet, draftkings at +3% or better against Pinnacle no-vig, seen 45-15 minutes before kickoff, receiving_yards, receptions, rushing_yards, total. One decision per player-game-stat (the best book when first seen), whatever its stake.

Decisions graded: 29 over 1 week(s).

## Gates

| Gate | Now | Rule |
|---|---:|---|
| Close capture (last 29) | 100.0% | pause below 90% once 10 decisions exist |
| Mean EV at the sharp close | +3.26% (29 graded) | pause at or below +0% after 50; scale review at +1.5% over 150+ and 3+ weeks |
| Loss pause | $+0.00 lowest | sticky at $-30 confirmed |
| Largest player / team share | 6.9% / 14.8% | <= 20% / 35% to scale |

## Evidence

| Measure (most reliable first) | Decisions | Earlier gaps | Later gaps | +1% to +3% gaps |
|---|---:|---:|---:|---:|
| Count | 29 | 23 | 22 | 85 |
| Sharp close coverage | 100.0% | 100.0% | 100.0% | 100.0% |
| Mean EV at the sharp close | +3.26% | +3.43% | +4.77% | +1.53% |
| Share still +EV at the sharp close | 86.2% | 91.3% | 100.0% | 92.9% |
| Beat the book's own exact close | 32.1% | 40.0% | 0.0% | 25.0% |
| EV claimed when seen | +4.29% | +4.37% | +4.78% | +1.87% |
| Simulated flat ROI (descriptive) | +14.58% | +7.68% | +26.34% | -5.92% |
| Settled | 27 | 22 | 20 | 78 |

Sharp close method: exact 13 at +3.65%; shifted 16 at +2.94%. 'shifted' moves Pinnacle's nearest closing line to the bet's line (yardage only).

### By book

| | Decisions | EV at sharp close | Still +EV | ROI |
|---|---:|---:|---:|---:|
| draftkings | 18 | +3.58% | 94.4% | +5.37% |
| espnbet | 9 | +2.32% | 66.7% | +17.14% |
| fanduel | 2 | +4.61% | 100.0% | +88.50% |

### By market

| | Decisions | EV at sharp close | Still +EV | ROI |
|---|---:|---:|---:|---:|
| receiving_yards | 11 | +1.86% | 72.7% | +15.17% |
| receptions | 10 | +3.45% | 90.0% | +35.42% |
| rushing_yards | 8 | +4.96% | 100.0% | -16.03% |

## Shadow model filter

Flags a gap when the model sees an injury tag, a limited-usage role, or strongly prefers the other side. Recorded only; it never skips a bet until skipping flagged gaps raises EV at the close.

| Verdict | Decisions | EV at sharp close | Still +EV | ROI |
|---|---:|---:|---:|---:|
| flag | 1 | -9.10% | 0.0% | - |
| no_forecast | 5 | +5.55% | 100.0% | +99.25% |
| pass | 23 | +3.30% | 87.0% | -0.14% |

Judged: not yet (needs 60 passed decisions with a close).

## Alternate ladders (research, never staked)

243 rungs priced from Pinnacle's main line by the fitted ladder model: EV when seen +31.19%, at the sharp close +31.40% (99.6% still +EV), flat ROI -8.04% on 213 settled. The close is valued with the same model, so ROI is the real test.

Legacy closes (diagnostic only):
- inferred_clv: 21 rows; mean +0.60%.
- inferred_ev_at_sharp_close: 29 rows; mean +1.25%.

Legacy alerts: 2292; captured decisions: 4716; excluded from primary: 6931; repeated eligible observations removed: 48.
Missing exact closes remain unknown. Weekly confidence bounds use equal week weights. Slippage is positive when the taken price is worse. Nothing here places a bet.
Quote observations do not establish that a bet was available or placed. Receipt-confirmed wagers, execution attempts, availability and actual-price slippage belong to the private wager_reconciliation report.

Raw observations by week (not independent decisions):

| Week | Tier | Count | Beat exact close | EV at sharp close |
|---|---|---:|---:|---:|
| 2026-4 | early | 7 | - | - |
| 2026-4 | logged | 30 | - | - |
| 2026-5 | alert | 13 | - | - |
| 2026-5 | early | 35 | - | - |
| 2026-5 | late | 8 | - | - |
| 2026-5 | logged | 148 | - | - |
| 2026-6 | early | 7 | - | - |
| 2026-6 | logged | 4 | - | - |
| 2026-7 | alert | 6995 | 24.0% | +19.59% |
| 2026-7 | early | 3328 | 37.8% | +19.95% |
| 2026-7 | late | 2185 | 0.0% | +19.60% |
| 2026-7 | logged | 2726 | 16.0% | +1.77% |
