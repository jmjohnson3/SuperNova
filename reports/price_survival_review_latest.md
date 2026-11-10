# Price survival and model contribution

Updated 2026-11-10T02:36:54.994701+00:00.
Paper research only. No filter promotion, sportsbook expansion or wagering authorization.

| Observer / market | Attempts | Available | Worse prices | Main offer unavailable | Checked exact closes | Positive at close | Model-value status |
|---|---:|---:|---:|---:|---:|---:|---|
| fanduel-direct-check-v1 / nba:assists | 6 | 3 | 1 | 0 | 3 | 2 | insufficient_public_evidence |
| fanduel-direct-check-v1 / nba:points | 16 | 7 | 1 | 0 | 7 | 7 | insufficient_public_evidence |
| fanduel-direct-check-v1 / nba:rebounds | 20 | 8 | 1 | 0 | 8 | 8 | insufficient_public_evidence |
| fanduel-direct-check-v1 / nba:threes | 8 | 5 | 0 | 0 | 5 | 4 | insufficient_public_evidence |
| fanduel-automatic-paper-v2 / nba:assists | 3 | 2 | 0 | 0 | 2 | 1 | insufficient_public_evidence |
| fanduel-automatic-paper-v2 / nba:points | 6 | 3 | 0 | 0 | 3 | 3 | insufficient_public_evidence |
| fanduel-automatic-paper-v2 / nba:rebounds | 8 | 5 | 1 | 0 | 4 | 3 | insufficient_public_evidence |
| fanduel-automatic-paper-v2 / nba:threes | 3 | 1 | 0 | 0 | 1 | 1 | insufficient_public_evidence |
| fanduel-automatic-paper-v2 / nfl:receiving_yards | 1 | 0 | 0 | 1 | 0 | 0 | insufficient_public_evidence |

## Delays, failures and paired comparisons

- fanduel-direct-check-v1 / nba:assists: reasons {'exact_quote': 3, 'server_response_stale': 3}; median delivery-to-check 24.703001 seconds; 6 paired offers, 1 model selections. Incremental close-valued EV game interval: -0.0337113931979367 to 0.07792272485921348 (unknown bounds remain unknown).
  Close-clock audit: {'observed_before_earliest_schedule': 6}; 5 originally reported exact closes excluded.
  Separately recovered exact closes from existing pregame quotes: 5.
- fanduel-direct-check-v1 / nba:points: reasons {'conflicting_exact_quotes': 5, 'exact_quote': 7, 'server_response_stale': 4}; median delivery-to-check 23.02328 seconds; 16 paired offers, 9 model selections. Incremental close-valued EV game interval: -0.03012150143419436 to -0.009129662028458996 (unknown bounds remain unknown).
  Close-clock audit: {'observed_before_earliest_schedule': 16}; 9 originally reported exact closes excluded.
  Separately recovered exact closes from existing pregame quotes: 9.
- fanduel-direct-check-v1 / nba:rebounds: reasons {'server_response_stale': 8, 'conflicting_exact_quotes': 4, 'exact_quote': 8}; median delivery-to-check 23.3127365 seconds; 20 paired offers, 8 model selections. Incremental close-valued EV game interval: -0.034212266605388654 to -0.013168300613702975 (unknown bounds remain unknown).
  Close-clock audit: {'observed_before_earliest_schedule': 19, 'no_exact_close': 1}; 15 originally reported exact closes excluded.
  Separately recovered exact closes from existing pregame quotes: 14.
- fanduel-direct-check-v1 / nba:threes: reasons {'exact_quote': 5, 'server_response_stale': 3}; median delivery-to-check 23.645262 seconds; 8 paired offers, 6 model selections. Incremental close-valued EV game interval: -0.03057349578230093 to 0.0 (unknown bounds remain unknown).
  Close-clock audit: {'observed_before_earliest_schedule': 8}; 5 originally reported exact closes excluded.
  Separately recovered exact closes from existing pregame quotes: 5.
- fanduel-automatic-paper-v2 / nba:assists: reasons {'exact_main_quote': 2, 'server_response_stale': 1}; median delivery-to-check 12.632562 seconds; 3 paired offers, 1 model selections. Incremental close-valued EV game interval: None to None (unknown bounds remain unknown).
  Close-clock audit: {'observed_before_earliest_schedule': 3}; 3 originally reported exact closes excluded.
  Separately recovered exact closes from existing pregame quotes: 3.
- fanduel-automatic-paper-v2 / nba:points: reasons {'server_response_stale': 3, 'exact_main_quote': 3}; median delivery-to-check 13.293392 seconds; 6 paired offers, 2 model selections. Incremental close-valued EV game interval: -0.032804286468291144 to 0.0 (unknown bounds remain unknown).
  Close-clock audit: {'observed_before_earliest_schedule': 6}; 6 originally reported exact closes excluded.
  Separately recovered exact closes from existing pregame quotes: 6.
- fanduel-automatic-paper-v2 / nba:rebounds: reasons {'exact_main_quote': 5, 'server_response_stale': 3}; median delivery-to-check 13.512354 seconds; 8 paired offers, 1 model selections. Incremental close-valued EV game interval: -0.031091523924789477 to 0.03858891803282016 (unknown bounds remain unknown).
  Close-clock audit: {'observed_before_earliest_schedule': 7, 'no_exact_close': 1}; 6 originally reported exact closes excluded.
  Separately recovered exact closes from existing pregame quotes: 5.
- fanduel-automatic-paper-v2 / nba:threes: reasons {'server_response_stale': 2, 'exact_main_quote': 1}; median delivery-to-check 13.453321 seconds; 3 paired offers, 3 model selections. Incremental close-valued EV game interval: None to None (unknown bounds remain unknown).
  Close-clock audit: {'observed_before_earliest_schedule': 3}; 3 originally reported exact closes excluded.
  Separately recovered exact closes from existing pregame quotes: 3.
- fanduel-automatic-paper-v2 / nfl:receiving_yards: reasons {'main_offer_absent_from_complete_tab': 1}; median delivery-to-check 6.672064 seconds; 1 paired offers, 0 model selections. Incremental close-valued EV game interval: None to None (unknown bounds remain unknown).
  Close-clock audit: {'registered_nfl_reader_enforces_identical_cutoff': 1}; 0 originally reported exact closes excluded.
  Separately recovered exact closes from existing pregame quotes: 0.

## Market-specific execution review

| Sport | Book | Market | Decisions | Audited exact closes | Checked prices | Review |
|---|---|---|---:|---:|---:|---|
| nba | draftkings | assists | 19 | 19 | 0 | collecting |
| nba | draftkings | points | 38 | 38 | 0 | collecting |
| nba | draftkings | rebounds | 22 | 22 | 0 | collecting |
| nba | draftkings | threes | 24 | 24 | 0 | collecting |
| nba | espnbet | assists | 4 | 4 | 0 | collecting |
| nba | espnbet | points | 10 | 10 | 0 | collecting |
| nba | espnbet | rebounds | 7 | 6 | 0 | collecting |
| nba | espnbet | threes | 5 | 5 | 0 | collecting |
| nba | fanduel | assists | 9 | 9 | 0 | collecting |
| nba | fanduel | points | 23 | 23 | 0 | collecting |
| nba | fanduel | rebounds | 25 | 24 | 0 | collecting |
| nba | fanduel | threes | 12 | 12 | 0 | collecting |

NFL comparisons remain separate; no additional sportsbook or market is enabled.

Unattempted current-observer alerts: {'nba': {'unattempted': 0, 'expired': 0}, 'nfl': {'unattempted': 0, 'expired': 0}}. Expired checks cannot be recovered from historical snapshots.

## Upcoming collection windows

- NBA NOP @ GSW: scan Nov 09 07:00 PM MST; start 08:00 PM MST.
- NFL WAS @ NYG: scan Nov 12 05:15 PM MST; start 06:15 PM MST.

## Interpretation

- Main-market-only v2 and original v1 observations are evaluated separately.
- Estimated closes, unknown checks, late checks and integer-line push assumptions cannot create checked-price EV.
- An original close after the earlier native/provider schedule cutoff is excluded here. Frozen report counts may still include it; those counts alone cannot establish pregame close coverage.
- A replacement is used only when an exact quote was already saved before the earlier cutoff. Recovered closes are identified separately; absent exact quotes remain missing. No historical API requests or synthetic conversions are made.
- Unavailable main offers contribute zero opportunity only when verified by a timely complete response; unknowns stay unknown.
- Paired diagnostics require the same original offer and a model decision captured before the automatic attempt.
- Confidence bounds describe observed subsets, not missing prices. Public evidence cannot clear the manual execution gate.
- NBA market reviews apply existing review thresholds separately by book and market; none grants expansion or cash approval.
