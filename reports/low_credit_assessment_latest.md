# 500-credit collection assessment

Assessment only. No subscription, scheduled task, live polling rule or frozen registration was changed.

| Item | Maximum credits per cycle |
|---|---:|
| 62 games × 3 polls × 1 market × one group of books | 186 |
| One additional charged attempt per poll | 186 |
| Reserved credits | 100 |
| Total | 472 / 500 |

Proposed collection: NBA points, at most two games/day and 62 games/provider cycle. Poll at T−40, T−20 and T−5. Pick games by schedule before seeing odds or results. Use the earlier matched native/provider schedule; omit unverified games. Hard request limits and a verified credit-cycle boundary would be required before activation.

Recorded selected-watcher usage in the configured cycle: 1638 credits/attempt budget; this is not a complete account audit.
Counters include only these watchers; missing counters, other crawlers and manual requests are not zero. NFL spread counter counts attempts at the one-credit maximum; this may exceed actual charged credits.

## What changes and what remains unverified

- Only NBA points would receive paid odds collection in this proposed design.
- Points is a capacity example, not a finding that this market is more profitable.
- NFL/NCAAF odds and other NBA prop markets would not fit unchanged alongside this schedule.
- Fewer polls may miss brief edges and exact closes; neither coverage nor 50 qualifying observations is guaranteed.
- The free plan says most bookmakers; required reference-book availability is unverified.
- Original experiments keep their evidence. This new sampling schedule needs its own prospective registration.

## Example from the current cached schedule

Cache timestamp: 2026-11-09T21:35:09.847890+00:00. This is a proposal, not the live scanner schedule.
- Los Angeles Lakers @ Atlanta Hawks: Nov 09 04:20 PM MST, Nov 09 04:40 PM MST, Nov 09 04:55 PM MST.
- Cleveland Cavaliers @ Charlotte Hornets: Nov 09 04:20 PM MST, Nov 09 04:40 PM MST, Nov 09 04:55 PM MST.

Switch-review blockers: frozen_collection_completed, required_books_verified_on_free_plan, provider_credit_reset_verified, other_paid_consumers_controlled, new_prospective_protocol_registered.

Provider: [pricing](https://the-odds-api.com/) and [quota calculation](https://the-odds-api.com/liveapi/guides/v4/). The 500-credit free plan excludes historical odds; current event-list requests are free.
