# NFL Sharp-Gap Strategy

Verdict: **trial: $1 stakes within limits** - $1 flat within the daily, weekly and per-game limits

Registration `nfl-sharp-gap-cash-v6`: fanduel, espnbet, draftkings at +3% or better against Pinnacle no-vig, seen 45-15 minutes before kickoff, receiving_yards, receptions, rushing_yards. One decision per player-game-stat (the best book when first seen), whatever its stake.

Decisions graded: 2 over 1 week(s).

## Gates

| Gate | Now | Rule |
|---|---:|---|
| Close capture (last 2) | 100.0% | pause below 90% once 10 decisions exist |
| Mean EV at the sharp close | +0.36% (2 graded) | pause at or below +0% after 50; scale review at +1.5% over 150+ and 3+ weeks |
| Loss pause | $+0.00 lowest | sticky at $-30 confirmed |
| Largest player / team share | 50.0% / - | <= 20% / 35% to scale |

## Evidence

| Measure (most reliable first) | Decisions | Earlier gaps | Later gaps | +1% to +3% gaps |
|---|---:|---:|---:|---:|
| Count | 2 | 0 | 2 | 8 |
| Sharp close coverage | 100.0% | - | 100.0% | 100.0% |
| Mean EV at the sharp close | +0.36% | - | +4.36% | +1.42% |
| Share still +EV at the sharp close | 50.0% | - | 100.0% | 87.5% |
| Beat the book's own exact close | 50.0% | - | 0.0% | 12.5% |
| EV claimed when seen | +7.00% | - | +4.36% | +1.74% |
| Simulated flat ROI (descriptive) | - | - | - | - |
| Settled | 0 | - | 0 | 0 |

Sharp close method: exact 2 at +0.36%. 'shifted' moves Pinnacle's nearest closing line to the bet's line (yardage only).

### By book

| | Decisions | EV at sharp close | Still +EV | ROI |
|---|---:|---:|---:|---:|
| espnbet | 1 | -6.77% | 0.0% | - |
| fanduel | 1 | +7.50% | 100.0% | - |

### By market

| | Decisions | EV at sharp close | Still +EV | ROI |
|---|---:|---:|---:|---:|
| receptions | 1 | +7.50% | 100.0% | - |
| rushing_yards | 1 | -6.77% | 0.0% | - |

## Shadow model filter

Flags a gap when the model sees an injury tag, a limited-usage role, or strongly prefers the other side. Recorded only; it never skips a bet until skipping flagged gaps raises EV at the close.

| Verdict | Decisions | EV at sharp close | Still +EV | ROI |
|---|---:|---:|---:|---:|
| pass | 2 | +0.36% | 50.0% | - |

Judged: not yet (needs 60 passed decisions with a close).

Legacy closes (diagnostic only):
- inferred_clv: 4 rows; mean +1.81%.
- inferred_ev_at_sharp_close: 4 rows; mean +1.24%.

Legacy alerts: 17; captured decisions: 4; excluded from primary: 17; repeated eligible observations removed: 2.
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
| 2026-7 | alert | 8 | 25.0% | +1.24% |
| 2026-7 | late | 5 | 0.0% | +4.59% |
| 2026-7 | logged | 36 | 11.1% | +1.32% |
