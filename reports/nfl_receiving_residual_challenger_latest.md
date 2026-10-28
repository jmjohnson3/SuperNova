# FanDuel Receiving Residual Challenger

Status: collecting_residual_evidence
Independent captured rows: 3034; player-games: 828

| Policy cohort | Weeks | Player-games | Point trust | Conservative trust | Supported |
|---|---:|---:|---:|---:|---|
| benchmark-20260921T230305171980Z|nfl-20260918T143342Z|stable_ensemble_calibrated | 5 | 828 | 0.35 | 0.00 | False |

Paper-only. Production and bankroll approval are unchanged.

Only immutable true-paired FanDuel common receiving lines enter training.
Each test week uses labels available before its first lock; overlapping player-games are purged.
Duplicate offers receive inverse player-game weight.
The independent player curve is discounted for interval uncertainty before residual trust is applied.
The selector abstains unless conservative EV is positive; five is a cap, never a quota.
This is a paper challenger. It cannot deploy production or authorize cash.
