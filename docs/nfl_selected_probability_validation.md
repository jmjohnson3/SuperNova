# Selected-Pick Validation and Receiving Result Repair

## Production Boundary

The receiving selection experiment never changes the active model, fixed research
selections, cash registration, or wagers. A forecast deployment review is separate
from cash eligibility. Sample thresholds trigger reviews, not automatic approval.

## Scoring Tests

`nfl_pipeline.modeling.receiving_selection_experiment` compares the existing EV
ranking with fixed disagreement discounts. The uncertainty-aware challenger uses
locked interval width relative to the projection; missing uncertainty gets the
full discount. It does not rewrite the displayed probability.

The later-week experiment verifies original scoring replay and exact same-book
paired prices. It includes the current paper-tier FanDuel receiving trial, not
only legacy micro labels. It retains pending and void selections in daily caps,
while excluding them from binary accuracy metrics. Fixed research selections
and exact legacy micro entries are shown separately from reselected candidates.

Calibration uses earlier fit/tune/gate weeks and a later untouched test week,
within the same production/scoring version. Labels must have been observed before
the first test lock; overlapping games are purged. Offers are weighted by player
game. A monotone central survival map preserves the 10th/90th-percentile anchors.
Integer lines remain unchanged until their push distributions can be validated.

Raw, context, heuristic, final, calibrated, and market probabilities are scored.
Locked projection intervals are not mislabeled as final-CDF coverage. Supplying
`--model <receiving-coherent-model.joblib>` additionally replays captured inputs
through all adjustments, constructs the coherent candidate curve, then checks
calibration, ranking, and coverage together. This is retrospective evidence,
never prospective credit. Missing replay inputs remain excluded.

## Result Repair

The settlement path now refreshes snap counts without schema DDL, grades existing
results, repairs verified receiving gaps, and regrades recovered rows. Active
near-kickoff close collection skips these slower settlement steps.

A missing receiving stat may be recovered only with:

1. An unambiguous PFR-to-player identity and positive offensive snaps for the
   exact game and team.
2. A completed play-by-play source ending in END GAME.
3. Complete receiving accounting, with unsupported lateral plays rejected.
4. Team passing attempts and yards reconciling against imported final totals.
   Sacks and two-point attempts do not count as official passing attempts.

A verified participant with no receiving production may then legitimately receive
zero yards. Absence of a stat row alone never creates a zero. Existing conflicting
stats are flagged, not overwritten. Two-way players are no longer rejected merely
because their roster position is defensive.

Exact-week inactive roster evidence plus published game participation and complete
play-by-play can establish nonparticipation. Supported FanDuel full-game receiving
markets are recorded as `void_nonparticipant`, with null actual yards and zero
unit profit. These are excluded from model labels and pending-result blockers.
No generic missing player is automatically voided. The rule reference is
[FanDuel Colorado house rules, American Football proposition bets](https://www.fanduel.com/fanduel-sportsbook-house-rules-co),
effective July 22, 2026. The application's record is not confirmation of an actual
bookmaker wager or account settlement.

Every repair saves an immutable evidence report in
`models/receiving_result_repair/`; repaired stat rows also retain source provenance.
Original forecasts, prices, timestamps, and scoring code remain unchanged.

## Cohort Accounting

Cash readiness now lists every fixed research selection as pre-policy, pending,
settled, verified void, wrong cohort, or missing evidence. Pre-policy daily slots
are preserved. A new date has its own cap. Wrong-cohort selected rows cannot
silently disappear from unresolved accounting.

Tests cover chronology, unavailable labels, pending caps, outcome-blind ranking,
tail coherence, verified zeros versus unknown participation, explicit inactives,
and post-registration selection entry. None of these checks automatically approves
real-money betting.
