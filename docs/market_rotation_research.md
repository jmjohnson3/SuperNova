# Market consensus and NBA rotation research

`market-rotation-paper-v1` is a separate, frozen paper experiment. It cannot send
wagers or authorize stakes. Existing price-only, model-assisted, count-probability,
NFL receiving and NBA workload experiments keep their original registrations.

## Scope

The same market-comparison principles apply across sports: match the exact offer,
preserve information timestamps, compare probabilities on identical offers, and
measure the price actually obtainable. Workload models must be specific to each sport.

This package reads existing NBA, NFL and NCAA football snapshots. It makes no paid
API requests. NBA gets the new joint rotation and workload-mixture challenger.
NFL retains its existing player models; each sport's game forecasts are saved as
separate comparisons. The current NFL observation archive has totals and selected
props but no spreads; the adapter supports spreads when collection supplies them.
MLB is not activated: its legacy stored lines do not preserve separate book and
market update clocks. Adding that evidence needs its own adapter and registration.

## Fixed market experiment

- FanDuel is the execution book and is excluded from the reference consensus.
- At least two of Pinnacle, DraftKings, BetMGM, Fanatics and ESPN Bet must supply
  fresh, two-sided prices at the identical half-point line. Each operator receives
  equal weight in log odds after removing its two-sided overround. Reference books
  are benchmarks, not ground truth or statistically independent observations.
- Preserve both update clocks. Prefer the market clock; book time is a fallback
  only when market time is absent. Source age is at most five minutes and source
  skew at most two minutes. No estimated line conversions.
- Select the FanDuel line nearest a 50% no-vig probability, with fixed tie breaks.
  Record the first paired offer 15–45 minutes before the game for each player and
  market, evaluate both offered sides, and require at least 3% estimated EV for a
  paper selection. Identity, participation and context checks remain separate.
- Incomplete reference observations are retained and may qualify on a later poll.
  An eligible decision cannot be rewritten or given a model forecast created later.

Compare consensus alone with consensus plus the existing production prediction,
and separately with consensus plus the rotation mixture. Save production-only,
collapsed-mean and mixture-only probabilities as diagnostics on those same offers.
Missing model forecasts remain missing; the consensus comparison still proceeds.

Weight fitting uses earlier game-weighted probability error, with zero in the
candidate grid. Each sport and market is fitted separately. At least 100 games
over 21 days are required for weight fitting; a separate calibration period needs
50 games over seven days. Current weights are all zero because earlier evidence
is insufficient, not because a well-powered comparison established zero as optimal.
The underlying player and game forecasts are still retained and graded.

Game forecast error distributions also need sufficient earlier pregame forecasts.
The current qualifying counts are NBA 0, NFL 46 and NCAA 0, below the 100-game
requirement. Point forecasts remain research comparisons; unsupported numerical
game probabilities are not manufactured.

## Joint NBA workload challenger

Sixteen joint rotations, each with zero/one/two-overtime scenarios, allocate exactly
240/265/290 team minutes. Verified absences receive no workload, known restrictions
cap minutes, and five starters are allocated jointly. Shot attempts, three-point
attempts, free throws, assists and rebounds are redistributed within team budgets
estimated from earlier results. Shooting percentages shrink toward longer-term
priors. Every player stores scenario minutes and production, not just their averages.

The early lineup feed sometimes contains only five expected starters. It is not a
full rotation: live projection waits for at least eight roster entries. Teams with
insufficient player history or capacity remain excluded. Uncertain teammate context
blocks selection. This is deliberately restrictive and its coverage is reported.

Price each workload scenario first, then combine probabilities using its weight.
The collapsed-mean comparison gets its own dispersion and calibration fits, so a
mixture is not credited merely for having more suitable uncertainty parameters.

Historical roster reconstruction uses prior appearances, not the current game's
eventual participants or final lineup. Historical injury evidence is incomplete,
so these results are development diagnostics. On 19,171 common player-games,
minutes MAE worsened from 4.9562 to 7.0177 and points MAE from 4.5364 to 4.8077.
Made-threes MAE improved slightly, 0.8921 to 0.8839, but attempts MAE worsened.
FanDuel points Brier was 0.249467 for the market and 0.260117 for the mixture.
The new rotation model has not earned promotion.

## Chronology and immutable evidence

- Rotation dispersion: out-of-sample forecasts for October 2024–January 2025.
- Rotation calibration: February–June 2025.
- Historical rotation evaluation: October 2025–June 2026, explicitly development
  because these historical periods have already been examined.
- Market weight fitting: October 1–27, 2026; separate calibration October 28–31;
  historical evaluation November 1–3. Game error fits use pre-October forecasts.
- New prospective evidence starts only after immutable registration. There is no
  backfill of prospective forecasts or assumed price checks.

Source files, dependency registrations, model binaries, manifests, weights and
runtime versions are checked before collection. Scheduled work never retrains
registered weights. A material repair or newly fitted weight needs a new registered
challenger and its own later evaluation period.

The separately registered `market-rotation-report-v1` reader parses timestamps
from saved JSON before official-result SQL interval arithmetic. The initial live
report exposed that storage-type issue for NCAA rows. The reader leaves the original
scorer, selections, probabilities, hashes and captured ledger unchanged. Scheduled
reporting uses this reader; the original package's `report` subcommand is superseded.

## Execution, review and economics

Observed exact-line closes remain separate from missing closes. Public FanDuel-feed
checks, user-entered live checks and actual wagers are distinct records. A check can
be reused only for the same original alert time, event, player, market, side, line
and price. Different or unmatched selections have unknown execution evidence.

The initial audit found 20 public-feed attempts: 11 available and nine unknown.
Eleven had exact reference closes, with mean reference-close EV of approximately
2.13% at the feed price. This is neither account-specific availability nor profit.
Historical snapshots cannot establish availability at checking time: the provider
returns a snapshot at or before the requested timestamp. See the
[Odds API documentation](https://the-odds-api.com/historical-odds-data/).

Use the existing guided recorder for an actual live observation:

```powershell
.venv\Scripts\python.exe -B -m nba_props.fanduel_check pending
.venv\Scripts\python.exe -B -m nba_props.fanduel_check record --alert 123
```

Replace 123 with the original alert ID and enter the time, exact line and displayed
odds, including worse prices, disappearance or suspension. Save immediately.
Discord replies are not automatically imported. See [price checks](nba_price_checks.md).

Review needs at least 21 days, three collection weeks, seven active days, 20 games,
50 checked prices paired with exact closes, 90% exact-close and manual-check
coverage, 80% availability, and positive uncertainty bounds clustered separately
by game and date. Model variants must also improve paired probability error overall
and on their selected subset. Missing outcomes remain in coverage denominators.
Counts and even passing review gates never automatically authorize betting.

Per-variant economics uses $1 comparison stakes, observed opportunity volume and
EV at manually checked prices against exact reference closes. Disappeared timely
checks contribute zero opportunities; unknown checks are not filled with zero.
Monthly estimates require at least 21 days and 90% priced opportunity coverage.
Subtract $30 once per alternative strategy; do not sum alternatives or increase
stakes to make them appear profitable. These estimates are not actual profit.
Confirmed wagers and actual costs continue through the separate finance ledger.

## Operations

```powershell
.venv\Scripts\python.exe -B -m edge_research verify
.venv\Scripts\python.exe -B -m edge_research project
.venv\Scripts\python.exe -B -m edge_research capture --sport nba ncaaf
.venv\Scripts\python.exe -B -m edge_research capture --sport nfl
.venv\Scripts\python.exe -B -m edge_research_report
.venv\Scripts\python.exe -B -m edge_research availability
```

NBA full/news runs project after the existing production posting and workload
projection. Full runs update the trial and availability reports. Shared and NFL
scanner wrappers capture their saved quotes after time-sensitive existing checks.
The new package sends no additional alerts and makes no extra Odds API requests.

Reports: `reports/market_rotation_trial_latest.json` and `.md`,
`reports/nba_rotation_development_latest.json` and `.md`,
`reports/market_consensus_development_latest.json`, and
`reports/price_survival_audit_latest.json`.

Separate calibration data follows the
[scikit-learn calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).
