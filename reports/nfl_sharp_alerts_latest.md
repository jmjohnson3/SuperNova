# NFL price-gap paper evidence

**Paper only. No cash authorization.** Actual sportsbook checks remain separate from API observations.

Observed reference closes below match the exact selected line. Estimated line conversions do not count as observed closes.

| Book | Decisions | Observed exact reference closes | Estimated conversions |
|---|---:|---:|---:|
| draftkings | 27 | 10 | 16 |
| espnbet | 14 | 4 | 10 |
| fanduel | 12 | 6 | 6 |

The legacy mixed-book diagnostic below includes estimated conversions in its sharp-close coverage and EV. Those figures are not exact-close coverage or evidence of obtainable profit.

---

# NFL Sharp-Gap Strategy

Verdict: **paper only: no new stakes** - paper checks continue; no new cash recommendations

Paused because: Paper-only collection: actual-price evidence and explicit cash review are still required

Registration `nfl-sharp-gap-cash-v10`: fanduel, espnbet, draftkings at +3% or better against Pinnacle no-vig, seen 45-15 minutes before kickoff, receiving_yards, receptions, rushing_yards, total. One decision per player-game-stat (the best book when first seen), whatever its stake.

Decisions graded: 53 over 2 week(s).

## Gates

| Gate | Now | Rule |
|---|---:|---|
| Close capture (last 30) | 96.7% | pause below 90% once 10 decisions exist |
| Mean EV at the sharp close | +3.56% (52 graded) | pause at or below +0% after 50; scale review at +1.5% over 150+ and 3+ weeks |
| Loss pause | $+0.00 lowest | sticky at $-30 confirmed |
| Largest player / team share | 3.8% / 9.8% | <= 20% / 35% to scale |

## Evidence

| Measure (most reliable first) | Decisions | Earlier gaps | Later gaps | +1% to +3% gaps |
|---|---:|---:|---:|---:|
| Count | 53 | 41 | 40 | 163 |
| Sharp close coverage | 98.1% | 97.6% | 100.0% | 98.8% |
| Mean EV at the sharp close | +3.56% | +3.44% | +4.67% | +1.38% |
| Share still +EV at the sharp close | 92.3% | 87.5% | 100.0% | 90.1% |
| Beat the book's own exact close | 28.6% | 40.6% | 5.0% | 21.2% |
| EV claimed when seen | +4.71% | +4.50% | +4.75% | +1.84% |
| Simulated flat ROI (descriptive) | +12.11% | +7.07% | +25.51% | +1.36% |
| Settled | 51 | 39 | 38 | 155 |

Sharp close method: exact 20 at +3.24%; shifted 32 at +3.75%. 'shifted' moves Pinnacle's nearest closing line to the bet's line (yardage only).

### By book

| | Decisions | EV at sharp close | Still +EV | ROI |
|---|---:|---:|---:|---:|
| draftkings | 27 | +4.05% | 96.2% | +4.44% |
| espnbet | 14 | +4.48% | 100.0% | +0.52% |
| fanduel | 12 | +1.42% | 75.0% | +41.62% |

### By market

| | Decisions | EV at sharp close | Still +EV | ROI |
|---|---:|---:|---:|---:|
| receiving_yards | 22 | +3.86% | 90.9% | +6.12% |
| receptions | 14 | +3.23% | 100.0% | +11.59% |
| rushing_yards | 17 | +3.42% | 87.5% | +20.78% |

## Shadow model filter

Flags a gap when the model sees an injury tag, a limited-usage role, or strongly prefers the other side. Recorded only; it never skips a bet until skipping flagged gaps raises EV at the close.

| Verdict | Decisions | EV at sharp close | Still +EV | ROI |
|---|---:|---:|---:|---:|
| flag | 3 | +1.31% | 66.7% | -37.17% |
| no_forecast | 9 | +4.81% | 100.0% | -13.97% |
| pass | 41 | +3.47% | 92.7% | +21.92% |

Judged: not yet (needs 60 passed decisions with a close).

Legacy closes (diagnostic only):
- inferred_clv: 79 rows; mean +0.19%.
- inferred_ev_at_sharp_close: 89 rows; mean +2.55%.

Legacy alerts: 2368; captured decisions: 4853; excluded from primary: 7089; repeated eligible observations removed: 79.
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
| 2026-7 | alert | 7002 | 22.7% | +19.57% |
| 2026-7 | early | 3330 | 36.2% | +19.94% |
| 2026-7 | late | 2190 | 0.0% | +19.57% |
| 2026-7 | logged | 2750 | 15.4% | +1.75% |
| 2026-8 | alert | 99 | 28.4% | +2.39% |
| 2026-8 | early | 38 | 32.1% | +1.49% |
| 2026-8 | late | 47 | 2.2% | +2.40% |
| 2026-8 | logged | 470 | 10.5% | +0.77% |
| 2026-9 | alert | 107 | 19.1% | +1.70% |
| 2026-9 | early | 45 | 37.1% | +2.57% |
| 2026-9 | late | 36 | 5.6% | +2.58% |
| 2026-9 | logged | 496 | 14.0% | +0.40% |
