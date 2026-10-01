# Market calibration, bet scope and receptions (October 1)

## Why

Weeks 1-3 of 2026 (2,669 settled over/under prop forecasts, 1,276 paper picks):

| Evidence | Result |
|---|---|
| Paper picks | 48.0% won vs 52.7% breakeven, -8.7% ROI; week 3 worst |
| Brier, model vs FanDuel no-vig | 0.258 vs 0.250 |
| Stated vs realised | model said 56% on its side, won 49%; 60-80% buckets won 46-47% |
| corr(projection - line, actual - line) | negative for every stat (receiving -0.10) |
| Projection MAE vs line MAE | passing 63.0 vs 54.7, receiving 22.2 vs 20.8, rushing 19.8 vs 16.8 |
| p10-p90 coverage | receiving 70%, rushing 73% (target 80%) |
| CLV on picks with a valid close | 16% beat close, 34% same, 50% worse |
| Picked vs not picked | identical (both ~49%, ~-7% ROI) |

The model's disagreement with the line carried no information, and its probabilities turned that
noise into apparent edges. Rushing was worst where the model sat above the line (QB rushing +6 to
+13 yards, backup RBs +10, deep backups +24); recent averages early in a season are mostly stale
2025 games.

## What changed

1. **Bet scope** (`betting_preferences.BET_PROP_STATS`, enforced in `lock_ledger` and scoring).
   Only receiving yards may lock into a staked tier. Rushing/passing yards, passing TDs,
   receptions, spreads and totals are still forecast, graded, and shown in Discord with FanDuel
   links as research rows.
2. **Market calibration layer** in `predict_player_props._candidate_from_offer`, parameters in
   `models/player_props/market_calibration.json`, fitted by `modeling/fit_market_calibration.py`:
   - projection anchor: `projection = line + a * (model - line)`; the raw model value is kept in
     `probability_trace.model_projection`.
   - range widening: distribution evaluated at `p50 + (x - p50) / k`; displayed p10/p90 widened.
   - market blend: `logit(p) = logit(market) + w * (logit(model) - logit(market))`.
   Parameters travel inside the captured `metrics`, so replays are exact. No file = identity.
3. **Receptions** market (`player_receptions`; SportsGameOdds `receiving_receptions`), projection
   `_receptions_projection` (recent receptions mix + projected targets x shrunk catch rate; MAE 1.255
   on 6,369 player-games), Poisson distribution, grading and Discord sections. Paper only.
4. **Gamelog source of record**: official nflverse box-score values win; play-by-play fills gaps
   and owns red-zone/end-zone columns (previously ~950 rows flipped every close run).
5. **CLV-first scorecard** (`modeling/clv_scorecard.py`, close run): valid-close rate, CLV beat
   rate and average move, then Brier vs market, then ROI, by stat and calibration version, with
   player-games weighted equally.

## Fitted parameters (market-calibration-20261001T030458Z)

Fit on replayed weeks 2-3 (week 1 forecasts predate replay capture); leave-one-week-out folds.
A stat keeps a fitted probability trust only if it beat the market's log loss in every fold.

| Stat | a | k | w | Out-of-sample |
|---|---:|---:|---:|---|
| receiving_yards | 0.00 | 1.25 | 0.00 | equals market, beats raw model both weeks; coverage 81% |
| passing_yards | 0.00 | 1.00 | 0.00 | equals market, beats raw model both weeks |
| rushing_yards | 0.00 | 1.00 | 1.00 | beats market both weeks (LL 0.678 vs 0.694; 0.688 vs 0.692) |
| passing_tds | 1.00 | 1.00 | 0.00 | mixed (tie, then worse than market) -> market |
| receptions | 0.35 | 1.00 | 0.00 | mixed on 349 recovered FanDuel lines -> market |

Re-scoring all 2,192 captured offers: the share showing >2% EV falls from 35-63% to 0% in every
stat. **No prop currently qualifies as a $1 bet, receiving yards included.** That is the expected,
honest result until a model shows information the market does not have. Rushing keeps a small,
validated lean (mostly unders) but not enough to clear the vig; it remains research.

## Consequences

- The scoring fingerprint changed. The registered receiving cash trial excludes new forecasts as a
  different scoring cohort (no error); continuing it needs an explicit re-registration.
- 148 rushing and ~60 passing player-games is little evidence. Re-check before relying on any of it.

## Re-checking against later weeks

Tuesday training runs `fit_market_calibration` report-only (`reports/nfl_market_calibration_latest.json`).
Judge by `reports/nfl_clv_scorecard_latest.md` first. To install a refit after review:

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.fit_market_calibration --write
```

## October 1: season-aware rushing baseline

Rolling `rushing_yards_avg_5` crosses seasons, so early-season projections lean on stale 2025 games.
`_season_aware_rushing_estimate` shrinks this season's per-game average toward last season's
(k=3 games; k=1 after a team change) and mixes it 50/50 with this-season carry share x team carries
x shrunk YPC. Live rushing projection = 50% frozen model + 50% this estimate; players with no NFL
history keep the model value. The frozen model's features are unchanged.

| Projection (MAE, rushing yards) | Weeks 1-8, 2020-2025 + 2026 (5,584 games) | 2026 settled forecasts (208) |
|---|---:|---:|
| 5-game cross-season average | 19.36 | 20.69 |
| Season-aware estimate | 18.62 | 19.06 |
| Frozen model | - | 19.25 |
| 50/50 model + estimate (live) | - | 18.62 |
| FanDuel line | - | 16.28 |

Lead backs, weeks 1-3: bias +4.8 -> +1.6 yards. It still carries no information beyond the line
(correlation with the line's error <= 0), so priced rushing stays anchored to the line until a refit says otherwise.

## October 1: RB role-change detection

The carry-share half of the season-aware estimate now reacts to role changes (RBs only):
recency-weighted share (half-life 2 games) plus 25% of the season carry share of RB teammates listed
Out/Doubtful on the pregame injury report for that week, split by the remaining backs' shares.
Explicit depth-chart promotion bumps were tested and hurt (recency already captures promotions).

Fit on 2020-2022, judged on 2023-2024 (weekly pregame injury reports; 2025 reports are missing):

| 2023-2024 holdout | All RB games | Games with an RB teammate ruled out (277) | Bias, teammate out |
|---|---:|---:|---:|
| Season share (previous) | 21.72 | 22.80 | -11.1 |
| Recency + 25% vacated (live) | 21.47 | 21.53 | -5.1 |

Full redistribution of vacated carries over-projected (+9.6 yards): carries also go to QBs/receivers
and team volume shifts. On 149 as-of 2026 forecasts the change is neutral (19.71 -> 19.68); only 8
had a ruled-out teammate, so weeks 4+ will show more.
