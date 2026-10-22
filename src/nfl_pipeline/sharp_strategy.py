"""The NFL cash strategy: bet a soft book's price when it beats the sharp no-vig price.

This is the only NFL strategy that may carry money. The model forecasts stay research
(docs/nfl_market_calibration.md: they do not beat FanDuel's own line), so nothing here reads a
model probability to choose or size a bet. The model is recorded beside each decision as a
shadow filter, and only becomes a rule if it improves EV at the sharp close (see SHADOW_FILTER).

Kept apart from betting_preferences, which governs forecast tiers and Discord display and is
pinned by the model trials: an edit there must not drift this strategy, nor the reverse.
Changing anything here is a strategy change and needs a new registration (sharp_watch --register).
"""
from __future__ import annotations

# v6 (22 October): links and alert text moved to the unpinned sharp_links (theScore Bet links were all
# rejected); pricing, selection, staking and grading are unchanged from v5.
EXPERIMENT_NAME = "nfl-sharp-gap-cash-v6"
PREDECESSOR = "nfl-sharp-gap-cash-v5"

# Books with an account. Order breaks EV ties only; each poll takes the best price first.
BET_BOOKS = ("fanduel", "espnbet", "draftkings")
# Markets whose gaps were backtested against the sharp close. Other scanned markets are graded
# at stake 0 until they prove out on their own.
STAKED_STATS = frozenset({"receiving_yards", "rushing_yards", "receptions"})

# A gap must be worth at least this much against the sharp no-vig price when seen.
MIN_EV = 0.03
# Stake only gaps seen this many minutes before kickoff. Earlier gaps decay before the close
# (FanDuel had already moved on ~36% of +3% receiving gaps seen at T-90); later than 15 minutes
# leaves too little time to place the bet by hand.
STAKE_WINDOW_MINUTES = (15, 45)

# Risk limits, the same as the documented NFL cash policy (docs/nfl_cash_readiness.md).
BETS_ENABLED = True
STAKE = 1.0
MAX_BETS_PER_DAY = 5
MAX_STAKE_PER_WEEK = 20.0
MAX_BETS_PER_GAME = 2      # one game's script moves every prop in it; spread the exposure
LOSS_PAUSE = -30.0         # sticky: confirmed realized profit at or below this stops new stakes

# Readiness gates, evaluated by modeling/sharp_alert_report. Only these four apply to this
# strategy; the frozen-model gates in cash_readiness guard model probabilities it does not use.
GATES = dict(
    # Every decision must be gradable at the sharp close. Below this the scanner is broken, not
    # unlucky, and staking stops until it is fixed.
    min_close_coverage=0.90,
    close_coverage_min_decisions=10,
    # Closing EV: stop if the price edge is not surviving to the close...
    kill_min_decisions=50,
    kill_max_mean_close_ev=0.0,
    # ...and review raising stakes only on a sample that can show it is real.
    scale_min_decisions=150,
    scale_min_mean_close_ev=0.015,
    scale_min_weeks=3,
    # Concentration: no single player or team may carry the result.
    max_player_share=0.20,
    max_team_share=0.35,
)

# Shadow model filter: flag a gap when the model disagrees on playing time or role. Recorded on
# every decision and graded beside it; never used to skip or size a bet until it is shown to
# improve EV at the sharp close on enough decisions.
SHADOW_FILTER = dict(
    unavailable_injury=("out", "doubtful", "ir", "inactive"),
    over_risky_injury=("questionable",),
    over_max_limited_usage_risk=0.40,
    max_side_probability_gap=0.15,  # model's chance of the bet side this far below the sharp fair price
    min_decisions_to_judge=60,
)
