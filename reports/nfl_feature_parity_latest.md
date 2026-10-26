# NFL Live-vs-Training Feature Parity

Season 2026, weeks 5, 6, 7: live snapshots rebuilt with the production loaders at each slate's cutoff (1491 rows), joined to the training table on 902 player-games. A feature matches when its relative mean difference is at most 2%.

Summary: match 247, differs 83, missing_live 33

## Team role groups

Live counts 17.8 players per team: 4.7 who did not go on to play and 2.4 cold-start roster members with a position prior.

| Total | Live | Training | Live from players who played | ...who did not play | ...cold-start priors |
|---|---:|---:|---:|---:|---:|
| teammate carries (5-game avg sum) | 26.51 | 28.73 | 25.11 | 6.40 | 4.58 |
| teammate targets (5-game avg sum) | 33.16 | 36.16 | 31.44 | 4.96 | 4.43 |

## Features that differ (worst first)

| Feature | Status | Relative difference | Live mean | Training mean | Live coverage | Training coverage |
|---|---|---:|---:|---:|---:|---:|
| is_indoor | missing_live | - | - | - | - | - |
| wind_mph | missing_live | - | - | - | - | - |
| temp_f | missing_live | - | - | - | - | - |
| is_high_wind | missing_live | - | - | - | - | - |
| is_cold | missing_live | - | - | - | - | - |
| is_grass | missing_live | - | - | - | - | - |
| opp_allowed_receiving_yards_wr_avg_5 | missing_live | - | - | - | - | - |
| opp_allowed_receiving_yards_te_avg_5 | missing_live | - | - | - | - | - |
| opp_allowed_receiving_yards_rb_avg_5 | missing_live | - | - | - | - | - |
| opp_allowed_yards_per_target_wr_avg_5 | missing_live | - | - | - | - | - |
| opp_allowed_yards_per_target_te_avg_5 | missing_live | - | - | - | - | - |
| opp_allowed_yards_per_target_rb_avg_5 | missing_live | - | - | - | - | - |
| opp_allowed_air_yards_per_target_wr_avg_5 | missing_live | - | - | - | - | - |
| opp_allowed_air_yards_per_target_te_avg_5 | missing_live | - | - | - | - | - |
| opp_allowed_air_yards_per_target_rb_avg_5 | missing_live | - | - | - | - | - |
| opp_allowed_yards_per_carry_rb_avg_5 | missing_live | - | - | - | - | - |
| roster_status | differs | - | - | - | 0.00 | 0.00 |
| depth_chart_position | differs | - | - | - | 0.00 | 0.00 |
| depth_pos_abb | differs | - | - | - | 0.00 | 0.00 |
| same_week_teammate_receiver_out_count | differs | 0.102 | 0.76 | 0.75 | 1.00 | 1.00 |
| team_spread_line | differs | 0.044 | -0.08 | -0.07 | 1.00 | 1.00 |
| depth_pos_slot | differs | 0.028 | 7.37 | 7.24 | 0.99 | 1.00 |
| injury_report_status | differs | - | - | - | 0.00 | 0.00 |
| injury_practice_status | differs | - | - | - | 0.00 | 0.00 |
| same_week_teammate_receiver_injury_score | differs | 0.104 | 1.13 | 1.11 | 0.93 | 0.93 |
| same_week_teammate_skill_injury_score | differs | 0.073 | 1.57 | 1.56 | 0.97 | 0.97 |
| same_week_teammate_skill_injury_count | differs | 0.067 | 2.11 | 2.08 | 1.00 | 1.00 |
| routes_run_avg_3 | differs | - | - | - | 0.00 | 0.00 |
| routes_run_avg_5 | differs | - | - | - | 0.00 | 0.00 |
| routes_run_avg_10 | differs | - | - | - | 0.00 | 0.00 |
| route_participation_avg_3 | differs | - | - | - | 0.00 | 0.00 |
| route_participation_avg_5 | differs | - | - | - | 0.00 | 0.00 |
| route_participation_avg_10 | differs | - | - | - | 0.00 | 0.00 |
| targets_per_route_run_avg_3 | differs | - | - | - | 0.00 | 0.00 |
| targets_per_route_run_avg_5 | differs | - | - | - | 0.00 | 0.00 |
| targets_per_route_run_avg_10 | differs | - | - | - | 0.00 | 0.00 |
| yards_per_route_run_avg_3 | differs | - | - | - | 0.00 | 0.00 |
| yards_per_route_run_avg_5 | differs | - | - | - | 0.00 | 0.00 |
| yards_per_route_run_avg_10 | differs | - | - | - | 0.00 | 0.00 |
| first_read_targets_avg_3 | differs | - | - | - | 0.00 | 0.00 |
| first_read_targets_avg_5 | differs | - | - | - | 0.00 | 0.00 |
| first_read_targets_avg_10 | differs | - | - | - | 0.00 | 0.00 |
| first_read_target_share_avg_3 | differs | - | - | - | 0.00 | 0.00 |
| first_read_target_share_avg_5 | differs | - | - | - | 0.00 | 0.00 |
| first_read_target_share_avg_10 | differs | - | - | - | 0.00 | 0.00 |
| end_zone_target_share_avg_3 | differs | - | - | - | 0.00 | 0.00 |
| end_zone_target_share_avg_5 | differs | - | - | - | 0.00 | 0.00 |
| end_zone_target_share_avg_10 | differs | - | - | - | 0.00 | 0.00 |
| unexpected_partial_rate_avg_5 | missing_live | - | - | - | - | - |
| pass_attempts_full_role_avg_5 | missing_live | - | - | - | - | - |
| passing_yards_full_role_avg_5 | missing_live | - | - | - | - | - |
| carries_full_role_avg_5 | missing_live | - | - | - | - | - |
| rushing_yards_full_role_avg_5 | missing_live | - | - | - | - | - |
| targets_full_role_avg_5 | missing_live | - | - | - | - | - |
| receptions_full_role_avg_5 | missing_live | - | - | - | - | - |
| receiving_yards_full_role_avg_5 | missing_live | - | - | - | - | - |
| receiving_air_yards_full_role_avg_5 | missing_live | - | - | - | - | - |
| pass_attempts_robust_avg_5 | missing_live | - | - | - | - | - |
| passing_yards_robust_avg_5 | missing_live | - | - | - | - | - |
| carries_robust_avg_5 | missing_live | - | - | - | - | - |
