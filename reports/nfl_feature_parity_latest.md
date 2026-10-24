# NFL Live-vs-Training Feature Parity

Season 2026, weeks 5, 6: live snapshots rebuilt with the production loaders at each slate's cutoff (1027 rows), joined to the training table on 627 player-games. A feature matches when its relative mean difference is at most 2%.

Summary: match 214, differs 116, missing_live 33

## Team role groups

Live counts 17.7 players per team: 4.6 who did not go on to play and 2.4 cold-start roster members with a position prior.

| Total | Live | Training | Live from players who played | ...who did not play | ...cold-start priors |
|---|---:|---:|---:|---:|---:|
| teammate carries (5-game avg sum) | 35.54 | 28.44 | 24.77 | 6.17 | 4.60 |
| teammate targets (5-game avg sum) | 40.37 | 35.46 | 30.85 | 4.89 | 4.63 |

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
| same_week_teammate_receiver_out_count | differs | 0.658 | 0.80 | 0.71 | 1.00 | 1.00 |
| team_spread_line | differs | 0.047 | -0.03 | -0.03 | 1.00 | 1.00 |
| depth_pos_slot | differs | 0.031 | 7.37 | 7.23 | 0.99 | 1.00 |
| injury_report_status | differs | - | - | - | 0.00 | 0.00 |
| injury_practice_status | differs | - | - | - | 0.00 | 0.00 |
| same_week_teammate_receiver_injury_score | differs | 0.637 | 1.25 | 0.89 | 0.92 | 0.94 |
| same_week_teammate_skill_injury_count | differs | 0.592 | 2.25 | 1.46 | 1.00 | 1.00 |
| same_week_teammate_skill_injury_score | differs | 0.527 | 1.66 | 1.24 | 0.96 | 0.98 |
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
