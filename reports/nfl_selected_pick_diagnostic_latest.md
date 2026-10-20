# NFL Selected Pick Diagnostic

Diagnostic only. No model deployment or betting approval.

| Population | Rows | Player-games | Weeks | Mean P | Win rate | Brier | Calibration |
|---|---:|---:|---:|---:|---:|---:|---:|

- Actual selections use validated exact ledger IDs, not a reranked historical top five.
- Eligibility is the stored pre-cap screen; no current rules are applied retrospectively.
- Release/date matching is not an exact publication-batch replay.
- Repeated offers are weighted by player-game/stat; weeks and player-games, not offer counts, measure independence.
- Small groups are descriptive, not evidence to deploy a calibrator or claims about causes.
- Pushes are excluded from binary probability scores; new probabilities are conditional on no push.

Full row details and breakdowns are in the adjacent JSON file.
