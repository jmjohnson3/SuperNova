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

from datetime import datetime, timedelta

# v9 (26 October): the user raised the daily cap to 10 and the weekly cap to $40 ($1 flat, 2 per game and the
# $30 sticky loss pause unchanged). On 25 October 29 decisions qualified and 5 were staked.
# v8 (26 October): the day's bets are spread across kickoff windows instead of first come; alternate
# ladders are no longer scanned (no edge: on 7,460 settled rungs the ladder model was no better than the
# books' own prices, its "+EV" rungs lost 8%, and cross-book rung gaps lost too); the shadow filter reads
# ESPN's game-day statuses so inactives announced after the T-120 forecast are seen (sharp_inactives).
# v7 (22 October): game totals carry stakes; alternate player-prop ladders are scanned and graded as
# research (never staked). v6 moved links and alert text to the unpinned sharp_links.
EXPERIMENT_NAME = "nfl-sharp-gap-cash-v9"
PREDECESSOR = "nfl-sharp-gap-cash-v8"

# Books with an account. Order breaks EV ties only; each poll takes the best price first.
BET_BOOKS = ("fanduel", "espnbet", "draftkings")
# Markets whose gaps were backtested against the sharp close. Other scanned markets are graded
# at stake 0 until they prove out on their own.
STAKED_STATS = frozenset({"receiving_yards", "rushing_yards", "receptions", "total"})
# Alternate player-prop ladders ("60+ receiving yards"): over-only at every book and absent at Pinnacle,
# so each rung is priced from Pinnacle's main line through modeling/alt_line_model. That model beats a
# normal curve on 2026 rungs but still misses long receiving rungs by 4-6 points of probability, more
# than any edge it would find, so ladders are scanned and graded at the close, never staked.
LADDER_MARKETS = {"player_reception_yds_alternate": "receiving_yards",
                  "player_rush_yds_alternate": "rushing_yards",
                  "player_receptions_alternate": "receptions"}
# v8: not polled. Kept so the parser and the report can still read v7's ladder evidence.
LADDER_SCAN = False

# A gap must be worth at least this much against the sharp no-vig price when seen.
MIN_EV = 0.03
# Stake only gaps seen this many minutes before kickoff. Earlier gaps decay before the close
# (FanDuel had already moved on ~36% of +3% receiving gaps seen at T-90); later than 15 minutes
# leaves too little time to place the bet by hand.
STAKE_WINDOW_MINUTES = (15, 45)

# Risk limits, the same as the documented NFL cash policy (docs/nfl_cash_readiness.md).
BETS_ENABLED = True
STAKE = 1.0
MAX_BETS_PER_DAY = 10
MAX_STAKE_PER_WEEK = 40.0
MAX_BETS_PER_GAME = 2      # one game's script moves every prop in it; spread the exposure
# The day's bets are shared across kickoff windows (kickoffs within WINDOW_MERGE_MINUTES of the window's
# first are one window): every window gets one, the rest go to the windows with the most games, and a
# window that stakes less hands the rest to later windows (the cap is cumulative). On 25 October the
# first-come rule spent all five by the 1 PM ET window; the afternoon and night games had none.
WINDOW_MERGE_MINUTES = 45
LOSS_PAUSE = -30.0         # sticky: confirmed realized profit at or below this stops new stakes

def kickoff_windows(kickoffs: list[datetime]) -> list[tuple[datetime, int]]:
    """[(window start, games)] for the day's kickoffs, earliest first."""
    windows: list[list] = []
    for k in sorted(kickoffs):
        if windows and k - windows[-1][0] <= timedelta(minutes=WINDOW_MERGE_MINUTES):
            windows[-1][1] += 1
        else:
            windows.append([k, 1])
    return [(w[0], w[1]) for w in windows]


def window_allocation(kickoffs: list[datetime], cap: int | None = None) -> list[tuple[datetime, int]]:
    """[(window start, bets)]: one per window, the rest by games (more games first, then earlier)."""
    cap = MAX_BETS_PER_DAY if cap is None else cap
    windows = kickoff_windows(kickoffs)
    if not windows or cap <= 0:
        return [(w, 0) for w, _ in windows]
    alloc = [0] * len(windows)
    order = sorted(range(len(windows)), key=lambda i: (-windows[i][1], i))
    for i in order[:cap]:
        alloc[i] = 1
    left = cap - sum(alloc)
    while left > 0:
        # next bet to the window with the most games per bet already given
        i = max(range(len(windows)), key=lambda j: (windows[j][1] / (alloc[j] + 1), -j))
        alloc[i] += 1
        left -= 1
    return [(windows[i][0], alloc[i]) for i in range(len(windows))]


def cumulative_cap(kickoffs: list[datetime], kickoff: datetime, cap: int | None = None) -> int:
    """Bets the day may hold once this kickoff's window is open: its allocation plus every earlier one."""
    cap = MAX_BETS_PER_DAY if cap is None else cap
    if not kickoffs:
        return cap
    total = 0
    for start, n in window_allocation(list(kickoffs) + ([] if kickoff in kickoffs else [kickoff]), cap):
        if start <= kickoff:
            total += n
    return min(total, cap)


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
