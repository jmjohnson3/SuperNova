# NFL Player Projection Repair Queue

This turns the player projection audit into ranked model work. Higher scores mean the model is losing to baseline in a larger, more fixable group.

- Status: ready
- Audit status: ready
- Built at: 2026-09-15T15:37:04Z

| Rank | Stat | Group | Rows | Model MAE | Baseline MAE | Gain | Bias | Stat Pass | Fix Hint |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | receiving_yards | opportunity_high | 21 | 44.737 | 37.881 | -6.857 | -40.801 | yes | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 2 | receiving_yards | player_rate_or_efficiency | 133 | 13.255 | 16.786 | 3.531 | -4.271 | yes | route/target opportunity: route participation proxy, targets per route, WOPR, air yards, depth role, and matchup |
| 3 | receiving_yards | opportunity_low | 30 | 25.115 | 30.845 | 5.729 | 23.818 | yes | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 4 | rushing_yards | opportunity_high | 15 | 29.536 | 29.540 | 0.004 | -25.712 | yes | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 5 | passing_yards | opportunity_high | 8 | 59.713 | 59.713 | 0.000 | -44.479 | no | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 6 | rushing_yards | opportunity_low | 15 | 22.924 | 31.489 | 8.565 | 21.136 | yes | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 7 | passing_yards | opportunity_low | 4 | 149.323 | 149.323 | 0.000 | 74.073 | no | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 8 | rushing_yards | player_rate_or_efficiency | 52 | 9.909 | 11.426 | 1.518 | -2.950 | yes | carry-share opportunity: RB depth movement, starter confidence, rest risk, spread/game script, and snap share |
| 9 | passing_yards | player_rate_or_efficiency | 20 | 48.716 | 48.716 | 0.000 | 6.789 | no | QB game script/opportunity: attempts/dropbacks, rest/injury confidence, pace, pass rate, and opponent profile |
| 10 | receiving_yards | projected_full_workload_failed | 9 | 14.264 | 21.419 | 7.155 | 8.860 | yes | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 11 | rushing_yards | projected_full_workload_failed | 4 | 15.915 | 22.775 | 6.860 | 8.352 | yes | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 12 | passing_yards | projected_full_workload_failed | 1 | 36.501 | 36.501 | 0.000 | 36.501 | no | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 13 | passing_tds | opportunity_high | 7 | 1.307 | 1.120 | -0.187 | -1.211 | yes | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 14 | rushing_tds | opportunity_high | 8 | 1.243 | 1.119 | -0.124 | -1.192 | yes | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 15 | receiving_tds | opportunity_high | 6 | 0.735 | 0.658 | -0.077 | -0.682 | yes | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 16 | rushing_tds | td_rate_or_variance | 31 | 0.231 | 0.231 | -0.000 | -0.095 | yes | goal-line role: red-zone/goal-line carries, team implied points, game script, and RB depth role |
| 17 | receiving_yards | model_flagged_limited_usage | 1 | 4.200 | 10.430 | 6.230 | -4.200 | yes | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 18 | passing_tds | td_role_opportunity | 2 | 0.549 | 0.413 | -0.136 | -0.431 | yes | QB game script/opportunity: attempts/dropbacks, rest/injury confidence, pace, pass rate, and opponent profile |
| 19 | receiving_tds | td_rate_or_variance | 125 | 0.321 | 0.327 | 0.006 | -0.011 | yes | rare-event TD head: red-zone targets, goal-line targets, first-read/route role, QB TD tendency, team total, opponent red-zone defense |
| 20 | passing_tds | opportunity_low | 9 | 0.975 | 1.148 | 0.173 | 0.222 | yes | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 21 | passing_tds | td_rate_or_variance | 15 | 0.710 | 0.760 | 0.050 | -0.037 | yes | QB game script/opportunity: attempts/dropbacks, rest/injury confidence, pace, pass rate, and opponent profile |
| 22 | rushing_tds | opportunity_low | 7 | 0.375 | 0.578 | 0.203 | 0.375 | yes | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
| 23 | receiving_tds | td_role_opportunity | 6 | 0.592 | 0.831 | 0.239 | -0.227 | yes | rare-event TD head: red-zone targets, goal-line targets, first-read/route role, QB TD tendency, team total, opponent red-zone defense |
| 24 | rushing_tds | td_role_opportunity | 7 | 0.482 | 0.493 | 0.012 | -0.026 | yes | goal-line role: red-zone/goal-line carries, team implied points, game script, and RB depth role |
| 25 | receiving_tds | opportunity_low | 4 | 0.323 | 1.021 | 0.698 | 0.323 | yes | starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement |
