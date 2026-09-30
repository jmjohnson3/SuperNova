# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-09T16:43:22Z
Slate: 2026-07-06
Evaluation: **FINAL**
Games final: 8 / 8
Strict clean slate: **False**

- Locked executable offers: 1532
- Valid exact closes: 1344 (87.7%)
- Stale closes: 0.0%
- T-120/T-60/T-20 complete events: 5 / 8

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 1344 |
| player_prop_unavailable_at_close | 152 |
| line_disappeared_at_close | 26 |
| player_market_unavailable_at_close | 5 |
| fallback_other_book_only | 5 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| 8708f0b1ad493caa280c8d963be2dc0b | fanduel | batter_total_bases | 68 | 12 | 82.4% | {"player_prop_unavailable_at_close": 12, "valid_close": 56} |
| caea4c3449e9f30ffe2c708a884e3296 | fanduel | batter_total_bases | 72 | 12 | 83.3% | {"player_prop_unavailable_at_close": 12, "valid_close": 60} |
| 92bad434cf32ec6dd5e2b856708a6596 | draftkings | batter_hits | 32 | 10 | 68.8% | {"line_disappeared_at_close": 6, "player_prop_unavailable_at_close": 4, "valid_close": 22} |
| 14986526f62254742f296b0eda680507 | draftkings | batter_hits | 34 | 10 | 70.6% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 92bad434cf32ec6dd5e2b856708a6596 | fanduel | batter_total_bases | 52 | 8 | 84.6% | {"player_prop_unavailable_at_close": 8, "valid_close": 44} |
| 14986526f62254742f296b0eda680507 | fanduel | batter_total_bases | 56 | 8 | 85.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 48} |
| e5295ada7ac0d16be6bf67d6a4a2923b | fanduel | batter_total_bases | 68 | 8 | 88.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 60} |
| e5295ada7ac0d16be6bf67d6a4a2923b | draftkings | pitcher_strikeouts | 8 | 6 | 25.0% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 2} |
| 8708f0b1ad493caa280c8d963be2dc0b | draftkings | batter_hits | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 8708f0b1ad493caa280c8d963be2dc0b | fanduel | batter_home_runs | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| caea4c3449e9f30ffe2c708a884e3296 | fanduel | batter_hits | 36 | 6 | 83.3% | {"player_prop_unavailable_at_close": 6, "valid_close": 30} |
| caea4c3449e9f30ffe2c708a884e3296 | fanduel | batter_home_runs | 36 | 6 | 83.3% | {"player_prop_unavailable_at_close": 6, "valid_close": 30} |
| caea4c3449e9f30ffe2c708a884e3296 | draftkings | batter_hits | 38 | 6 | 84.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 32} |
| 8708f0b1ad493caa280c8d963be2dc0b | fanduel | batter_hits | 60 | 6 | 90.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 54} |
| e5295ada7ac0d16be6bf67d6a4a2923b | fanduel | pitcher_strikeouts | 6 | 4 | 33.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 2} |
| 8708f0b1ad493caa280c8d963be2dc0b | draftkings | batter_total_bases | 14 | 4 | 71.4% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 10} |
| e5295ada7ac0d16be6bf67d6a4a2923b | draftkings | batter_total_bases | 18 | 4 | 77.8% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 14} |
| 92bad434cf32ec6dd5e2b856708a6596 | fanduel | batter_hits | 26 | 4 | 84.6% | {"player_prop_unavailable_at_close": 4, "valid_close": 22} |
| 92bad434cf32ec6dd5e2b856708a6596 | fanduel | batter_home_runs | 26 | 4 | 84.6% | {"player_prop_unavailable_at_close": 4, "valid_close": 22} |
| 14986526f62254742f296b0eda680507 | fanduel | batter_home_runs | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| ceda0420b6aec99d07f85f99472f8c68 | draftkings | batter_hits | 32 | 4 | 87.5% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 28} |
| e5295ada7ac0d16be6bf67d6a4a2923b | fanduel | batter_hits | 34 | 4 | 88.2% | {"player_prop_unavailable_at_close": 4, "valid_close": 30} |
| e5295ada7ac0d16be6bf67d6a4a2923b | fanduel | batter_home_runs | 34 | 4 | 88.2% | {"player_prop_unavailable_at_close": 4, "valid_close": 30} |
| 14986526f62254742f296b0eda680507 | fanduel | batter_hits | 52 | 4 | 92.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 48} |
| d1ba74caa5492c9e300650c9ba7bebab | fanduel | batter_total_bases | 52 | 4 | 92.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 48} |
| ceda0420b6aec99d07f85f99472f8c68 | fanduel | batter_total_bases | 60 | 4 | 93.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 56} |
| 14986526f62254742f296b0eda680507 | draftkings | pitcher_strikeouts | 4 | 2 | 50.0% | {"line_disappeared_at_close": 2, "valid_close": 2} |
| 8708f0b1ad493caa280c8d963be2dc0b | draftkings | pitcher_strikeouts | 6 | 2 | 66.7% | {"line_disappeared_at_close": 2, "valid_close": 4} |
| 8708f0b1ad493caa280c8d963be2dc0b | fanduel | pitcher_strikeouts | 6 | 2 | 66.7% | {"line_disappeared_at_close": 2, "valid_close": 4} |
| ce47831dfdb0f68ba311fd79cc19d68a | draftkings | pitcher_strikeouts | 6 | 2 | 66.7% | {"line_disappeared_at_close": 2, "valid_close": 4} |
| ceda0420b6aec99d07f85f99472f8c68 | draftkings | pitcher_strikeouts | 6 | 2 | 66.7% | {"line_disappeared_at_close": 2, "valid_close": 4} |
| 92bad434cf32ec6dd5e2b856708a6596 | draftkings | batter_total_bases | 10 | 2 | 80.0% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "valid_close": 8} |
| 14986526f62254742f296b0eda680507 | draftkings | batter_total_bases | 20 | 2 | 90.0% | {"player_prop_unavailable_at_close": 2, "valid_close": 18} |
| ce47831dfdb0f68ba311fd79cc19d68a | draftkings | batter_total_bases | 24 | 2 | 91.7% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "valid_close": 22} |
| d1ba74caa5492c9e300650c9ba7bebab | draftkings | batter_hits | 26 | 2 | 92.3% | {"player_prop_unavailable_at_close": 2, "valid_close": 24} |
| d1ba74caa5492c9e300650c9ba7bebab | fanduel | batter_hits | 26 | 2 | 92.3% | {"player_prop_unavailable_at_close": 2, "valid_close": 24} |
| d1ba74caa5492c9e300650c9ba7bebab | fanduel | batter_home_runs | 26 | 2 | 92.3% | {"player_prop_unavailable_at_close": 2, "valid_close": 24} |
| ceda0420b6aec99d07f85f99472f8c68 | fanduel | batter_home_runs | 30 | 2 | 93.3% | {"player_prop_unavailable_at_close": 2, "valid_close": 28} |
| ce47831dfdb0f68ba311fd79cc19d68a | draftkings | batter_hits | 34 | 2 | 94.1% | {"line_disappeared_at_close": 2, "valid_close": 32} |
| e5295ada7ac0d16be6bf67d6a4a2923b | draftkings | batter_hits | 34 | 2 | 94.1% | {"player_prop_unavailable_at_close": 2, "valid_close": 32} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| PHI @ KC | 2026-07-06T18:11:00+00:00 | yes | yes | yes | 15 |
| NYY @ TB | 2026-07-06T22:41:00+00:00 | yes | yes | yes | 28 |
| HOU @ WAS | 2026-07-06T22:46:00+00:00 | yes | yes | yes | 28 |
| ARI @ SD | 2026-07-07T01:41:00+00:00 | yes | yes | yes | 31 |
| COL @ LAD | 2026-07-07T02:11:00+00:00 | yes | yes | yes | 31 |
| NYM @ ATL | 2026-07-06T23:16:00+00:00 | no | yes | yes | 30 |
| MIL @ STL | 2026-07-06T23:46:00+00:00 | no | yes | yes | 29 |
| TOR @ SF | 2026-07-07T01:46:00+00:00 | no | yes | yes | 31 |
