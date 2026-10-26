# NFL Live-vs-Training Feature Parity

Season 2026, weeks 5, 6, 7: live snapshots rebuilt with the production loaders at each slate's cutoff (1452 rows), joined to the training table on 856 player-games. A feature matches when its relative mean difference is at most 2%.

Summary: match 226, differs 104, missing_live 33

## Team role groups

Live counts 17.7 players per team: 5.0 who did not go on to play and 2.4 cold-start roster members with a position prior.

| Total | Live | Training | Live from players who played | ...who did not play | ...cold-start priors |
|---|---:|---:|---:|---:|---:|
| teammate carries (5-game avg sum) | 26.49 | 27.88 | 25.00 | 6.41 | 4.53 |
| teammate targets (5-game avg sum) | 33.24 | 35.51 | 31.09 | 5.41 | 4.40 |

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
| same_week_teammate_receiver_out_count | differs | 0.537 | 0.77 | 0.68 | 1.00 | 1.00 |
| team_spread_line | differs | 0.045 | -0.02 | -0.02 | 1.00 | 1.00 |
| depth_pos_slot | differs | 0.029 | 7.36 | 7.22 | 0.99 | 1.00 |
| injury_report_status | differs | - | - | - | 0.00 | 0.00 |
| injury_practice_status | differs | - | - | - | 0.00 | 0.00 |
| same_week_teammate_receiver_injury_score | differs | 0.528 | 1.16 | 0.86 | 0.93 | 0.94 |
| same_week_teammate_skill_injury_count | differs | 0.497 | 2.12 | 1.45 | 1.00 | 1.00 |
| same_week_teammate_skill_injury_score | differs | 0.448 | 1.57 | 1.21 | 0.97 | 0.98 |
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
