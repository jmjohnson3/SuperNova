# NFL paper decisions: props, spreads and totals

The NFL channel now has an **NFL PAPER DECISIONS** message for each game. It is
updated in place by `NFL-Sharp-Watch`. It shows the kickoff, scanning and selection
windows, latest quote observations by market, saved game forecast, and fresh
FanDuel paper selections or model vetoes. It distinguishes waiting for the scan,
missing/delayed collection, no qualifying offer and a closed selection window.

Only the existing **PAPER PRICE CHECK** alert starts a five-minute manual-check
window. The board does not reset that clock, renew an old price, select another
offer, or request a wager. Send Codex the original M ID, actual checking time,
exact line and displayed odds, including worse prices, disappeared or suspended
offers. Discord replies are not imported automatically.

The model-assisted strategy uses the saved player probability at the exact
FanDuel line/price, or the saved game margin/total, to select or veto the same
offer as the price-only control. There is no forced daily quota. Missing models
remain vetoes. These are prospective paper comparisons; no claim of a proven
betting advantage follows from a forecast or a small winning sample.

## Collection

The original frozen NFL scanner still gathers receiving yards, rushing yards,
receptions and alternate totals. The new, separately frozen
`nfl-spread-quotes-v1` collector supplies the previously missing spread quotes to
the existing comparison readers. It does not write cash alerts or decisions.

- Runs only in the final 60 minutes before kickoff, at most once every five
  minutes; actual invocation follows the existing NFL scheduled task cadence.
- One current spreads request covers all due games, with FanDuel, Pinnacle,
  DraftKings and BetMGM. A free events request checks remaining credits first.
- At most 48 paid requests per ET date, with a 1,000-credit floor. Failed paid
  attempts count toward the cap; there is no immediate retry after a timeout.
- Exact home/away identity and kickoff matching; both opposing handicap sides
  required. Conflicting duplicates are rejected, never resolved by order.
- Both book and market update times are preserved. The market timestamp takes
  precedence. Full response receipts are saved under `data/nfl_spread_quotes/`.
- No historical backfill, sportsbook login, or bet placement. API observations
  do not establish account-level availability or accepted wagers.

The collection registration pins its source and original parser registration.
The original receiving, sharp-price, model-assisted and market-consensus
registrations keep their identities and scoring rules. New spread data starts
prospectively; older missing spreads/closes are not reconstructed.

## Report repair and cash hold

The sharp evidence report repeatedly exceeded its 120-second task budget and
stopped refreshing after October 29. The reporting adapter now indexes quotes
by the same game/event/stat/player equality checks already used by the frozen
close reader, then calls that reader with the relevant quotes. It preserves
quote order and uses the same result join, grading and selection functions.
Equivalence regression tests cover identities, line changes, missing data and
timestamp ties. A full live-data refresh completed in about 72 seconds.

The repaired publication explicitly preserves **paper-only collection**. Its
old v10 cash verdict is stored under `frozen_readiness_diagnostic`, while the
published readiness remains paused pending actual-price evidence and explicit
cash review. Restoring a report must not silently authorize $1 recommendations.
Existing loss pauses remain binding.

The headline report separates books and observed exact reference closes from
estimated line conversions. On November 5, the active legacy cohort had 26
decisions in one NFL week, including seven FanDuel decisions: three observed
exact reference closes and four estimated conversions. Its older combined
sharp-close metric includes estimates and is labeled diagnostic. The separate
model-assisted comparison had three NFL offers and zero manual checks.

## Local commands

```powershell
# Saved-data preview; no Discord post or paid request.
.\.venv\Scripts\python.exe -B -m nfl_decision_board
# Update the NFL channel's existing per-game board messages.
.\.venv\Scripts\python.exe -B -m nfl_decision_board --post
# Read-only evidence refresh; preserves the paper-only cash hold.
.\.venv\Scripts\python.exe -B -m nfl_pipeline.modeling.sharp_evidence_report
# Verify the separate spread collector; once buys data only when due and budgeted.
.\.venv\Scripts\python.exe -B -m nfl_spread_collection verify
.\.venv\Scripts\python.exe -B -m nfl_spread_collection once
```

Board reports: `reports/nfl_decision_board_latest.md` and `.json`.
Discord message IDs: `reports/nfl_decision_board_discord_state.json`.
Report: `reports/nfl_sharp_alerts_latest.md` and `.json`.
Manual checks: the existing `model_research_checks` command; see
`docs/model_assisted_paper.md`.

The collection request/credit behavior is documented in the
[Odds API v4 guide](https://the-odds-api.com/liveapi/guides/v4/).
