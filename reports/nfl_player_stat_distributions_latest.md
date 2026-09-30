# NFL Player Stat Distributions

This estimates projection uncertainty from player-game holdout residuals before live prop lines exist.
Yardage props test low/normal/spike workload mixtures and use them only when holdout line Brier improves; otherwise they keep the normal residual curve.

- Status: ready
- Training rows: 29073
- Trained at: 2026-09-17T16:07:04Z

| Stat | Kind | Rows | Model MAE | Best Baseline MAE | Sigma | Bias | Brier | Base Brier | Mixture Brier | Receiver Spike Brier | Receiver Spike MAE | QB Vol Brier | Accepted |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| passing_yards | normal_residual | 33 | 63.207 | 63.207 | 84.224 | -3.416 | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.250 | no |
| rushing_yards | normal_residual | 86 | 15.803 | 18.613 | 24.158 | +1.629 | 0.218 | 0.250 | 0.218 | 0.000 | 0.000 | 0.000 | yes |
| passing_tds | poisson_count | 33 | 0.899 | 0.921 | 1.149 | +0.239 | 0.185 | 0.182 | 0.000 | 0.000 | 0.000 | 0.000 | yes |
| rushing_tds | poisson_count | 53 | 0.436 | 0.445 | 1.000 | +0.189 | 0.173 | 0.193 | 0.000 | 0.000 | 0.000 | 0.000 | yes |
| receiving_yards | mixture_yardage_residual | 194 | 18.500 | 21.426 | 29.143 | +3.267 | 0.212 | 0.250 | 0.206 | 0.207 | 18.560 | 0.000 | yes |
| receiving_tds | poisson_count | 141 | 0.351 | 0.382 | 1.000 | +0.040 | 0.152 | 0.176 | 0.000 | 0.000 | 0.000 | 0.000 | yes |

## Receiver Spike Challenger

- Challenger accepted: False
- Current curve Brier on the same proxy lines: 0.20522614119539181
- Brier gain: -0.0018122398027998476
- Mean MAE gain: -0.05965883475406386
- Earlier validation rows: 1036
- Final holdout used for parameter selection: False
- Five components: low, normal, target spike, air-yards spike, YPT tail.
- Proxy line Brier uses 75%, 100%, and 125% of the pregame baseline, rounded to half yards. These are not sportsbook lines or CLV proof.
- Upstream projection/opportunity artifacts are held fixed. This evaluates the added mixture, not a nested retrain of the full model stack.
