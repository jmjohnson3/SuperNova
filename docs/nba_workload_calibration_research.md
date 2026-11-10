# NBA workload and calibration research

`nba-workload-calibration-v1` is a separate paper challenger. It forecasts minutes,
production conditional on minutes, three-point attempts and shooting percentage,
and game margin/total. Its probabilities select or veto the same FanDuel offers
captured by the existing price experiment. None of these outputs authorizes a
wager or changes an existing frozen experiment.

## What is fixed

The registration at
`src/model_research_experiments/nba-workload-calibration-v1/registration.json`
pins the four research modules, trained model, training manifest, runtime versions,
and parent experiment. It checks the parent's dependencies before scoring.
Scheduled jobs refresh inputs and produce forecasts; they never refit this model.
Material changes require a separate challenger ID, artifact, and prospective
evidence. A changed source or artifact stops capture instead of rolling over.

The candidate evaluates the original FanDuel side and executable price. It requires
the parent's exact-line, identity, quote freshness, book timestamp agreement and
15–45 minute decision window checks. Capture must finish within five minutes of
the original observation. The forecast must be completed before that observation
and at most 30 minutes old. Its calibrated probability must produce at least 3%
estimated EV, with positively verified active context for props. An uncertain or
missing availability status prevents selection, while retaining the conditional
stat forecast for evaluation. There is no daily quota.

One decision is retained per experiment/event/player/market. A missing prior
forecast is saved as a miss; a later forecast cannot rewrite that decision.
Game lines use a separate margin/total error distribution and correct home/away
spread signs. Only half-point lines are eligible, so this trial does not model
integer-line pushes for selection.

## Training chronology and limitations

All outcome-derived features have a two-day lag. Team schedule dates can inform
rest without using the previous game's outcome. Current-game actual starters,
minutes, participation and final lineups are never prediction inputs.

| Purpose | Forecast period | Weights trained before |
|---|---|---|
| Fit residual dispersion and minutes intervals | Oct 2024–Jan 2025 | Sep 29, 2024 |
| Fit distribution calibration | Feb–Jun 2025 | Jan 30, 2025 |
| Evaluate the fixed recipe | Oct 2025–Jun 2026 | Jun 29, 2025 |
| Fixed live artifact | Future observations after registration | Cutoff stored in training manifest |

Final weights use the same fixed recipe. The saved artifact was fitted before
registration; future results do not update its weights, dispersion or calibration.
Rolling features continue to incorporate newly verified completed games.

Historical NBA results were retrieved later. Their original retrieval times and
revision state cannot be proven. Final lineup snapshots supply past-game role
labels and lagged role history only, never purported contemporaneous availability.
The historical results are therefore reconstructed development evidence. The
untouched prospective test begins at the new registration time.

There are no trustworthy real-offer snapshots in the calibration period.
Calibration instead uses fixed half-integer stat thresholds on unseen outcomes,
weighting each player-game equally. Threshold pairs are **not** executable offers
or independent evidence counts. A positive logistic slope preserves alternate-line
monotonicity. Later real-offer evaluation is separated by sportsbook.

## Workload, attempts and verified inputs

An independent immutable input ledger stores official player and team game logs,
including `fg3a`. It retains raw receipts, observation times and hashes. It requires
two reciprocal final team logs, consistent winner/scores, player shooting identities,
agreeing player/team totals and full game minutes. Conflicting or incomplete games
are excluded. Missing players or minutes never become zero-stat wins or assumed
nonparticipant voids.

The minutes model predicts starter and bench scenarios. A lagged starter
probability mixes them; a timely confirmed lineup can set the role explicitly.
Verified minutes restrictions cap both scenarios. Per-minute models forecast
points, rebounds, assists and three-point attempts conditional on that role.
Three-point makes equal forecast attempts times shooting percentage, with recent
shooting shrunk toward a fixed 36% prior equivalent to 100 attempts.

Forecast rows retain minutes, role scenarios, attempts, shooting percentage,
features, context, input cutoff and completion time. Reports decompose stat error:

`forecast error = workload error + per-minute production error`

As-of queries require observations to precede the cutoff. Manual restrictions
also require their recording time to precede it. Example, using actual verified
source details rather than placeholders:

```powershell
.venv\Scripts\python.exe -B -m nba_props.research_trial cap --game GAME_ID --player PLAYER_ID --team BOS --minutes 24 --observed-at 2026-11-04T17:00:00-07:00 --source "source URL" --note "Verified announced restriction"
```

The command requires an explicit timezone offset. It records context only and
cannot rewrite an earlier forecast.

## Results so far

Historical evaluation covers 27,921 player-games. The candidate did **not** beat
the rolling baseline: minutes MAE 5.039 vs 5.019; points 4.596 vs 4.569; threes
0.896 vs 0.882. Calibration improved the candidate's probabilities but did not
beat FanDuel's no-vig baseline on the matched offers:

| Stat | Offers | Market Brier | Raw candidate | Calibrated candidate |
|---|---:|---:|---:|---:|
| Points | 2,311 | 0.24934 | 0.25840 | 0.25605 |
| Rebounds | 2,084 | 0.24928 | 0.25127 | 0.25048 |
| Assists | 1,333 | 0.24700 | 0.25726 | 0.25616 |

Lower Brier is better. Three-point forecasts have stat-error and distribution
calibration results, but no historical real-offer validation. They require future
offer evidence. On 372 matched FanDuel games, margin MAE was 11.831 vs market
10.966; total MAE was 14.853 vs 14.297. These results do not support promotion.

## Prospective comparison and execution

Reports compare calibrated, raw and frozen production probabilities on identical
offers, plus the selected subsets and per-market calibration. Games and props are
reported separately. Uncertainty resamples complete games and dates; correlated
props do not count as independent games. All offers, including missing forecasts
and unresolved results, remain in the coverage denominators.
The prospective reference probability is Pinnacle's exact-line no-vig estimate,
which must itself be validated. The historical table above uses FanDuel's own
no-vig probability; these are explicitly different comparison sources.

Only observed exact-line closes count. The original alert's manual check is reused
without resetting its deadline. A game check must match the same original decision,
side, line, price and timestamps. Worse prices and missing offers remain evidence;
automated FanDuel feed observations are not relabeled manual checks or execution.

Existing review requirements remain: at least three weeks, 20 games, 50 checked
prices with exact closes, strong close/check coverage and positive uncertainty
bounds. Counts alone grant no betting approval. Improvements also need favorable
paired probability evidence and human review. Actual wagers and the $30/month data
cost stay in the separate finance ledger; paper returns are not actual profit.

## Operation

`scripts/nba_props_live.bat` refreshes the research result ledger on full runs,
projects after full/news runs, and writes its report on full runs. The existing
Discord summary is sent before these additional steps. `scripts/sharp_multi_watch.bat`
captures the challenger after the existing automatic FanDuel price observation.
The challenger sends no extra Discord alerts and consumes no additional Odds API
credits. Official NBA result refreshes use the NBA stats endpoints.

```powershell
.venv\Scripts\python.exe -B -m nba_props.research_trial verify
.venv\Scripts\python.exe -B -m nba_props.research_trial ingest
.venv\Scripts\python.exe -B -m nba_props.research_trial audit
.venv\Scripts\python.exe -B -m nba_props.research_trial project
.venv\Scripts\python.exe -B -m nba_props.research_trial capture
.venv\Scripts\python.exe -B -m nba_props.research_trial report
```

`train` refuses to replace registered weights. Re-running `register` is idempotent
only for the exact same files and rules. A shared database advisory lock prevents
overlapping research mutations. Failed ingestion returns a nonzero exit code.
The lock waits up to 30 seconds before reporting an unsuccessful step.
The scheduled wrappers retain a failure exit status while allowing the existing
notification flow to finish.

Outputs:

- `reports/nba_research_development_latest.md` and `.json`: historical evaluation.
- `reports/nba_research_training_manifest.json`: splits, corpus and model hashes.
- `reports/nba_research_audit_latest.json`: input coverage and limitations.
- `reports/nba_research_trial_latest.md` and `.json`: prospective evidence.
- `data/nba_props/research/`: local raw receipts and reusable history caches.
- `src/nba_props/models/research_v1.joblib`: fixed model binary; retain in backups.
