# NFL Sharp-Edge Alerts

Verdict: **keep collecting**

Primary: registered FanDuel receiving-yard decisions with an exact Pinnacle reference. First eligible observation per book/player-game/stat; later prices do not add independent decisions.

Alerts graded: 0 over 0 week(s) (pass bar: 100 over 3). Early gaps graded: 0. Logged gaps graded: 0.

| Evidence (most reliable first) | Alerts (>= +3%, last 180 min, pinged) | Early (>= +3%, earlier) | Logged (+1% to +3%) | Pass bar (alerts) |
|---|---:|---:|---:|---:|
| Exact execution close coverage | - | - | - | >= 90% |
| Exact reference close coverage | - | - | - | >= 90% |
| Mean EV at exact reference close | - | - | - | >= +1% |
| Share positive at the sharp close | - | - | - | - |
| Beat exact FanDuel close | - | - | - | >= 55% |
| Mean FanDuel CLV (no-vig prob) | - | - | - | > 0 |
| EV claimed when seen | - | - | - | - |
| Simulated flat ROI | - | - | - | descriptive |
| Settled | - | - | - | - |

Per-book historical diagnostics, excluded from pooled approval:

| Book | Decisions | Exact reference-close EV | Beat exact close |
|---|---:|---:|---:|
| draftkings | 27 | - | - |
| fanduel | 12 | - | - |

Model-inferred movement and legacy closes (diagnostic only):
- inferred_clv: 0 rows; mean -.
- inferred_ev_at_sharp_close: 0 rows; mean -.

Legacy alerts: 47; captured candidate decisions: 0; excluded from primary: 47; repeated eligible alerts removed: 0.
Missing exact closes remain unknown. Weekly confidence bounds use equal week weights. No cash authorization is granted by this report. Slippage is positive when the taken price is worse.
Quote observations do not establish that a bet was available or placed. Receipt-confirmed wagers, execution attempts, availability and actual-price slippage belong to the private wager_reconciliation report.

Historical raw observations by week (not independent decisions):

| Week | Tier | Count | Beat exact close | Exact reference-close EV |
|---|---|---:|---:|---:|
| 2026-4 | alert | 1 | - | - |
| 2026-4 | early | 6 | - | - |
| 2026-4 | logged | 30 | - | - |
| 2026-5 | alert | 42 | - | - |
| 2026-5 | early | 14 | - | - |
| 2026-5 | logged | 148 | - | - |
| 2026-6 | alert | 4 | - | - |
| 2026-6 | early | 3 | - | - |
| 2026-6 | logged | 4 | - | - |
