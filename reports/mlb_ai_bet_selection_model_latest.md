# MLB AI Bet Selection Model

- Generated: 2026-08-26T09:45:35.166701+00:00
- Status: enabled enabled=True
- Candidate set: regularized_logistic
- Rows: 12000 after dedupe (283467 duplicates removed)
- Serious clean rows: 5755 true-paired rows used for real training
- Date range: 2026-07-25 to 2026-07-30
- FanDuel evidence: all={'fanduel_one_sided_display_only': 6245, 'fanduel_true_pair': 2421} serious={'fanduel_true_pair': 2421}

## Targets
- Good-bet classifier: status=enabled rows=4936 dates=6 selected=regularized_logistic brier_gain=0.075
- Win classifier: status=shadow_brier_not_better rows=5755 dates=6 selected=regularized_logistic brier_gain=-0.006
- CLV classifier: status=enabled rows=4936 dates=6 selected=regularized_logistic brier_gain=0.023
- Line-available classifier: status=enabled rows=5317 dates=6 selected=regularized_logistic brier_gain=0.074
- Daily rank classifier: status=enabled rows=5755 dates=6 selected=regularized_logistic brier_gain=0.142

## Good-Bet Metrics
- model_prob_side: Brier=0.256 AUC=0.582 CalErr=30.0% selected=- ROI=- win=-
- market_prob_side: Brier=0.235 AUC=0.610 CalErr=27.7% selected=- ROI=- win=-
- model_market_blend: Brier=0.246 AUC=0.597 CalErr=29.0% selected=- ROI=- win=-
- regularized_logistic: Brier=0.160 AUC=0.681 CalErr=6.0% selected=- ROI=- win=-

## Win Metrics
- model_prob_side: Brier=0.251 AUC=0.565 CalErr=8.8% selected=82 ROI=-18.5% win=40.2%
- market_prob_side: Brier=0.246 AUC=0.582 CalErr=5.9% selected=19 ROI=29.5% win=36.8%
- model_market_blend: Brier=0.248 AUC=0.574 CalErr=6.2% selected=19 ROI=29.5% win=36.8%
- regularized_logistic: Brier=0.252 AUC=0.593 CalErr=6.9% selected=341 ROI=-2.3% win=50.4%

## CLV Metrics
- global_mean: Brier=0.244 AUC=0.500 CalErr=0.3%
- regularized_logistic: Brier=0.222 AUC=0.679 CalErr=8.1%

## Market Families
- draftkings|pitcher_strikeouts|under|k_low: rows=27 dates=6 enabled=False targets=- good_status=insufficient_rows good_gain=-
- draftkings|pitcher_strikeouts|under|k_common: rows=75 dates=6 enabled=False targets=- good_status=insufficient_rows good_gain=-
- draftkings|batter_total_bases|over|tb_1.5: rows=527 dates=6 enabled=True targets=target_good_bet,target_line_available_at_close good_status=enabled good_gain=0.043
- fanduel|pitcher_strikeouts|under|k_low: rows=22 dates=6 enabled=False targets=- good_status=insufficient_rows good_gain=-
- fanduel|pitcher_strikeouts|under|k_common: rows=65 dates=6 enabled=False targets=- good_status=insufficient_rows good_gain=-
- fanduel|batter_total_bases|over|tb_1.5: rows=631 dates=6 enabled=True targets=target_good_bet,target_clv,target_line_available_at_close good_status=enabled good_gain=0.082
- fanduel|batter_hits|over|hits_0.5: rows=938 dates=6 enabled=True targets=target_good_bet,target_clv,target_line_available_at_close good_status=enabled good_gain=0.080
- draftkings|batter_hits|over|hits_0.5: rows=885 dates=6 enabled=True targets=target_good_bet,target_clv,target_line_available_at_close good_status=enabled good_gain=0.133

## Notes
- This model is trained only from pre-existing locked/graded rows.
- Serious real-money training uses clean true-paired rows; FanDuel synthetic/one-sided rows are display/watch evidence only.
- Good-bet means the row won and beat closing price on a valid close row.
- Daily rank learns which historical slate rows were among the strongest realized betting opportunities.
- It does not use closing line, final result, or actual stat values as live features.
- Enabled means the learned good-bet or win model beat the best simple baseline on grouped walk-forward Brier.
