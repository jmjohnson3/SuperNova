# NBA count-probability comparison

`nba-count-probability-v1` is a separate frozen prospective challenger. It does
not change the original price experiment, the model-assisted experiment, Discord
recommendations, stake rules, or model training. It sends no additional alerts
and makes no additional API calls.

The candidate changes only the mapping from the saved mean and dispersion to an
over/under probability. It uses an integer negative-binomial distribution with
variance `max(mu, a*mu + b*mu^2)`, using Poisson when variance equals the mean.
The original distribution uses scaled Poisson when `b=0`, including cases where
the scale is greater than one. That produces probability steps larger than one
unit of a player's statistic.

The offline development comparison used 6,917 matching player-game-stat offers
over 245 games and 46 dates in the 2025-26 period. Quotes had to have been recorded
before tip; inferred historical snapshot times were excluded. No thresholds were
searched and no models were refitted. Brier fell from 0.25726 to 0.25024, while the
DraftKings no-vig benchmark scored 0.24858. Lower is better.

At the fixed 3% model-EV development threshold, paper ROI changed from -4.38% to
+0.63%. The candidate's game-clustered 95% interval was -2.19% to +3.68%.
Historical offer availability was not checked, these quotes are DraftKings rather
than FanDuel, and costs are excluded. This supports further probability research,
not a claim of profitability. The artifacts and source/input hashes are linked
from `reports/nba_count_distribution_latest.json`.

## Prospective rules

- Only NBA points, rebounds, and assists at the parent's exact FanDuel offer.
- Same pregame forecast, participation checks, line, side, price, and first offer
  as `fanduel-model-assisted-paper-v1`.
- Capture after this registration, within five minutes of the original offer
  observation, before either known game start. Historical rows are never added.
- Save both probabilities, both selections, full original inputs, and registration
  identity in an append-only record.
- The paper selection test is positive model EV at that price, matching the parent
  comparison. The offline 3% threshold is not an execution policy.
- Reuse the original NBA price-check ID and deadline. No extra checks or wagers.

Review retains the parent's NBA evidence and execution gates: at least 21 days,
20 games, seven active dates, three week bins, 50 checked prices paired with exact
closes, strong coverage, and positive game/day uncertainty bounds. The challenger
also needs positive paired Brier improvement against both the frozen model and
market benchmark, and positive paired execution-value improvement. Economics use
the existing $1 scenario and $30 monthly cost. Reviews never authorize betting.

## Operations

The sharp-watch batch captures this comparison immediately after the original
model-assisted collector. Registration checks fail on source or parent drift.

```powershell
.\.venv\Scripts\python.exe -B -m nba_props.count_distribution_trial verify
.\.venv\Scripts\python.exe -B -m nba_props.count_distribution_trial capture
.\.venv\Scripts\python.exe -B -m nba_props.count_distribution_trial report
```

The live report is `reports/nba_count_trial_latest.md` and `.json`. Official
results and exact closing observations use the parent's unchanged rules.
Missing results, participation, live checks, or exact closes remain unknown.
