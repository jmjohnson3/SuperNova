# MLB Prop Restart Readiness

Generated UTC: 2026-07-15T01:29:57Z
Status: **PASS**
Checkpoint status: `evaluation_ready_no_micro_buckets`
Micro-ready buckets: 0
Backdated prop-promotion slates: 4 (2026-07-02, 2026-07-03, 2026-07-04, 2026-07-05)

## Checks

| Check | Required | Pass | Detail |
|---|---|---|---|
| task_registered_MLB-Morning | True | True | registered under \SuperNovaBets\ |
| task_registered_MLB-PreGame-Day | True | True | registered under \SuperNovaBets\ |
| task_registered_MLB-PreGame-Evening | True | True | registered under \SuperNovaBets\ |
| task_registered_MLB-Prop-Targeted-Close | True | True | registered under \SuperNovaBets\ |
| task_registered_MLB-Close | True | True | registered under \SuperNovaBets\ |
| task_registered_MLB-Training | True | True | registered under \SuperNovaBets\ |
| scheduled_task_last_results | False | False | MLB-Training=1 |
| targeted_close_interval_pt10m | True | True | interval='PT10M' |
| targeted_close_duration_present | False | True | duration='PT15H' |
| targeted_close_limit_not_too_long | False | True | execution_time_limit='PT15M' |
| targeted_close_command_exists | True | True | command='C:\\Users\\josh\\Git\\SuperNovaBets\\scripts\\mlb_prop_targeted_close.bat' |
| database_connectivity | True | True | connected |
| table_exists_bets_mlb_bankroll_ledger | True | True | bets.mlb_bankroll_ledger |
| table_exists_bets_mlb_model_pick_ledger | True | True | bets.mlb_model_pick_ledger |
| table_exists_bets_mlb_daily_forecast_ledger | True | True | bets.mlb_daily_forecast_ledger |
| table_exists_odds_mlb_player_prop_line_snapshots | True | True | odds.mlb_player_prop_line_snapshots |
| table_exists_features_mlb_prop_market_training_examples | True | True | features.mlb_prop_market_training_examples |
| webhook_present_MLB_DISCORD_WEBHOOK_URL | True | True | set in process/user environment |
| webhook_present_MLB_RECORD_LEDGER_DISCORD_WEBHOOK_URL | True | True | set in process/user environment |
| optional_webhook_present_MLB_PROP_RESEARCH_DISCORD_WEBHOOK_URL | False | True | set in process/user environment |
| optional_webhook_present_MLB_OPS_DISCORD_WEBHOOK_URL | False | True | set in process/user environment |
| optional_webhook_present_MLB_ALERTS_DISCORD_WEBHOOK_URL | False | True | set in process/user environment |
| frozen_hitter_artifact_valid | True | True | sha256 matched |
| checkpoint_ran | True | True | 2026-07-15T00:55:46Z |
| micro_report_ran | True | True | 2026-07-15T00:55:46Z |
| backdated_audit_has_promotion_slates | False | True | 4 prop-promotion backdated slates |
| micro_buckets_not_forced | False | True | 0 micro-ready buckets |

## Scheduled Tasks

| Task | Exists | State | Last Result | Last Run | Next Run |
|---|---|---|---:|---|---|
| MLB-Morning | True | Ready | 0 | /Date(1784030400000)/ | /Date(1784116800000)/ |
| MLB-PreGame-Day | True | Ready | 0 | /Date(1784039430000)/ | /Date(1784125830000)/ |
| MLB-PreGame-Evening | True | Ready | 0 | /Date(1784061030000)/ | /Date(1784147430000)/ |
| MLB-Prop-Targeted-Close | True | Ready | 0 | /Date(1784078725000)/ | /Date(1784079335000)/ |
| MLB-Close | True | Ready | 0 | /Date(1784076345000)/ | /Date(1784079945000)/ |
| MLB-Training | True | Ready | 1 | /Date(1784003430000)/ | /Date(1784089830000)/ |
