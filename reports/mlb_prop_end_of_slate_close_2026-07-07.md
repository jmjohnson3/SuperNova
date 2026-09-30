# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-09T16:43:23Z
Slate: 2026-07-07
Evaluation: **FINAL**
Games final: 16 / 16
Strict clean slate: **False**

- Locked executable offers: 3099
- Valid exact closes: 2713 (87.5%)
- Stale closes: 0.3%
- T-120/T-60/T-20 complete events: 13 / 16

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2713 |
| player_prop_unavailable_at_close | 318 |
| line_disappeared_at_close | 39 |
| player_market_unavailable_at_close | 10 |
| fallback_other_book_only | 10 |
| stale_close_before_lock | 9 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| 8df69c8e501590af54db5363e5fdc3af | fanduel | batter_total_bases | 68 | 12 | 82.4% | {"player_prop_unavailable_at_close": 12, "valid_close": 56} |
| fa63ecda3e7fe4b049c7e9cfd633f461 | fanduel | batter_total_bases | 68 | 12 | 82.4% | {"player_prop_unavailable_at_close": 12, "valid_close": 56} |
| e3cd1f3a8c40c74e867b7713af7e6365 | fanduel | batter_total_bases | 72 | 12 | 83.3% | {"player_prop_unavailable_at_close": 12, "valid_close": 60} |
| 7b7ae2c411b9b8122144d502c0bd1757 | fanduel | batter_total_bases | 76 | 12 | 84.2% | {"player_prop_unavailable_at_close": 12, "valid_close": 64} |
| a250fe250560caa8b56cd4e061c8e7b0 | fanduel | batter_total_bases | 80 | 12 | 85.0% | {"player_prop_unavailable_at_close": 12, "valid_close": 68} |
| 6611ecd500747b1f1aebe1ea49f9d00a | fanduel | batter_hits | 41 | 11 | 73.2% | {"player_prop_unavailable_at_close": 4, "stale_close_before_lock": 7, "valid_close": 30} |
| 622a46f12d0dc3b057125b0b59f00507 | draftkings | batter_hits | 34 | 10 | 70.6% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 8df69c8e501590af54db5363e5fdc3af | draftkings | batter_hits | 36 | 10 | 72.2% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 8, "valid_close": 26} |
| a250fe250560caa8b56cd4e061c8e7b0 | draftkings | batter_hits | 46 | 10 | 78.3% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 6, "valid_close": 36} |
| 288b4514ed237def0df905ead634d530 | draftkings | batter_hits | 34 | 8 | 76.5% | {"line_disappeared_at_close": 6, "player_prop_unavailable_at_close": 2, "valid_close": 26} |
| 53966d4de4d1da8972c6564664409a9b | fanduel | batter_total_bases | 56 | 8 | 85.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 48} |
| 622a46f12d0dc3b057125b0b59f00507 | fanduel | batter_total_bases | 56 | 8 | 85.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 48} |
| c892fb37a2662fa901742b27c6b7f205 | fanduel | batter_total_bases | 60 | 8 | 86.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 52} |
| e38e52ed47e9148a67603013c94f7295 | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| 6611ecd500747b1f1aebe1ea49f9d00a | fanduel | batter_total_bases | 68 | 8 | 88.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 60} |
| a21ff3f3e886298451df6978c4be743c | fanduel | batter_total_bases | 68 | 8 | 88.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 60} |
| 7b7ae2c411b9b8122144d502c0bd1757 | fanduel | batter_hits | 70 | 7 | 90.0% | {"line_disappeared_at_close": 1, "player_prop_unavailable_at_close": 6, "valid_close": 63} |
| c892fb37a2662fa901742b27c6b7f205 | draftkings | batter_hits | 32 | 6 | 81.2% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 26} |
| 8df69c8e501590af54db5363e5fdc3af | fanduel | batter_hits | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 8df69c8e501590af54db5363e5fdc3af | fanduel | batter_home_runs | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| fa63ecda3e7fe4b049c7e9cfd633f461 | fanduel | batter_home_runs | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 6611ecd500747b1f1aebe1ea49f9d00a | draftkings | batter_hits | 36 | 6 | 83.3% | {"player_prop_unavailable_at_close": 4, "stale_close_before_lock": 2, "valid_close": 30} |
| 7b7ae2c411b9b8122144d502c0bd1757 | draftkings | batter_hits | 36 | 6 | 83.3% | {"player_prop_unavailable_at_close": 6, "valid_close": 30} |
| e38e52ed47e9148a67603013c94f7295 | draftkings | batter_hits | 36 | 6 | 83.3% | {"player_prop_unavailable_at_close": 6, "valid_close": 30} |
| e3cd1f3a8c40c74e867b7713af7e6365 | draftkings | batter_hits | 36 | 6 | 83.3% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 30} |
| e3cd1f3a8c40c74e867b7713af7e6365 | fanduel | batter_home_runs | 36 | 6 | 83.3% | {"player_prop_unavailable_at_close": 6, "valid_close": 30} |
| 7b7ae2c411b9b8122144d502c0bd1757 | fanduel | batter_home_runs | 38 | 6 | 84.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 32} |
| a250fe250560caa8b56cd4e061c8e7b0 | fanduel | batter_home_runs | 40 | 6 | 85.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 34} |
| fa63ecda3e7fe4b049c7e9cfd633f461 | fanduel | batter_hits | 59 | 6 | 89.8% | {"player_prop_unavailable_at_close": 6, "valid_close": 53} |
| e3cd1f3a8c40c74e867b7713af7e6365 | fanduel | batter_hits | 62 | 6 | 90.3% | {"player_prop_unavailable_at_close": 6, "valid_close": 56} |
| a250fe250560caa8b56cd4e061c8e7b0 | fanduel | batter_hits | 73 | 6 | 91.8% | {"player_prop_unavailable_at_close": 6, "valid_close": 67} |
| c892fb37a2662fa901742b27c6b7f205 | fanduel | batter_hits | 54 | 5 | 90.7% | {"line_disappeared_at_close": 1, "player_prop_unavailable_at_close": 4, "valid_close": 49} |
| a21ff3f3e886298451df6978c4be743c | draftkings | batter_total_bases | 10 | 4 | 60.0% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 6} |
| e3cd1f3a8c40c74e867b7713af7e6365 | draftkings | batter_total_bases | 16 | 4 | 75.0% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 12} |
| 8df69c8e501590af54db5363e5fdc3af | draftkings | batter_total_bases | 20 | 4 | 80.0% | {"player_prop_unavailable_at_close": 4, "valid_close": 16} |
| e38e52ed47e9148a67603013c94f7295 | draftkings | batter_total_bases | 20 | 4 | 80.0% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 16} |
| 0ae1e836fd92f0f98b83e2d7451f6f97 | draftkings | batter_hits | 24 | 4 | 83.3% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 20} |
| 53966d4de4d1da8972c6564664409a9b | fanduel | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 53966d4de4d1da8972c6564664409a9b | fanduel | batter_home_runs | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 622a46f12d0dc3b057125b0b59f00507 | fanduel | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| SEA @ MIA | 2026-07-07T22:41:00+00:00 | yes | yes | yes | 29 |
| ATH @ DET | 2026-07-07T22:41:00+00:00 | yes | yes | yes | 28 |
| ATL @ PIT | 2026-07-07T22:41:00+00:00 | yes | yes | yes | 28 |
| NYY @ TB | 2026-07-07T22:41:00+00:00 | yes | yes | yes | 28 |
| HOU @ WAS | 2026-07-07T22:46:00+00:00 | yes | yes | yes | 29 |
| KC @ NYM | 2026-07-07T23:11:00+00:00 | yes | yes | yes | 29 |
| PHI @ CIN | 2026-07-07T23:11:50+00:00 | yes | yes | yes | 28 |
| CHC @ BAL | 2026-07-07T23:36:00+00:00 | yes | yes | yes | 29 |
| CLE @ MIN | 2026-07-07T23:41:00+00:00 | yes | yes | yes | 29 |
| BOS @ CWS | 2026-07-07T23:41:00+00:00 | yes | yes | yes | 30 |
| LAA @ TEX | 2026-07-08T00:06:00+00:00 | yes | yes | yes | 30 |
| ARI @ SD | 2026-07-08T01:41:00+00:00 | yes | yes | yes | 30 |
| COL @ LAD | 2026-07-08T02:11:00+00:00 | yes | yes | yes | 31 |
| MIL @ STL | 2026-07-07T18:16:00+00:00 | no | yes | yes | 15 |
| MIL @ STL | 2026-07-07T23:46:00+00:00 | no | yes | yes | 22 |
| TOR @ SF | 2026-07-08T01:46:00+00:00 | no | yes | yes | 30 |
