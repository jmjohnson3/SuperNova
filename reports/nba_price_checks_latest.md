# NBA paper price checks

Experiment `nba-prop-price-check-v1` — frozen.
Registered: 2026-10-29T00:48:15.833740+00:00. Prior/unregistered observations remain historical context.
Research only. Passing a review never enables stakes. Model forecasts remain a shadow comparison.

Each book needs 50 exact closes, 21 days spanning 3 weeks, 7 active days, 20 games, 90% exact-close coverage, at least +1.5% mean closing EV, 90% timely manual checks, 80% exact-offer availability, 50 checked prices with exact closes, and positive 95% lower bounds clustered by game and by day.

Checks must be made within 5 minutes of the alert and recorded within 5 minutes of checking, before tip. Missing checks are unknown; moved lines and disappeared offers are not treated as executable.

| Book | Decisions | Games | Days/span | Exact closes | Timely checks | Available | Checked-price EV | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fanduel | 69 | 39 | 11/13 | 69 | 0 | 0 | - | collecting |
| draftkings | 103 | 52 | 10/12 | 103 | 0 | 0 | - | collecting |
| espnbet | 26 | 20 | 9/13 | 26 | 0 | 0 | - | collecting |


fanduel close_by_game: 95% interval +0.11% to +2.86% across 39 clusters.

fanduel close_by_day: 95% interval +0.39% to +2.69% across 11 clusters.

fanduel outstanding: collection_span, weeks, manual_check_coverage, offer_availability, checked_prices, checked_close_coverage, checked_by_game, checked_by_day.

draftkings close_by_game: 95% interval +3.36% to +5.00% across 52 clusters.

draftkings close_by_day: 95% interval +3.41% to +4.91% across 10 clusters.

draftkings outstanding: collection_span, weeks, manual_check_coverage, offer_availability, checked_prices, checked_close_coverage, checked_by_game, checked_by_day.

espnbet close_by_game: 95% interval +0.84% to +3.45% across 20 clusters.

espnbet close_by_day: 95% interval +0.35% to +3.49% across 9 clusters.

espnbet outstanding: decisions, collection_span, weeks, manual_check_coverage, offer_availability, checked_prices, checked_close_coverage, checked_by_game, checked_by_day.
