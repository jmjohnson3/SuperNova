# NCAAF game models (spreads, totals, team totals)

Status (2026-10-24): **research only. No model-driven NCAAF bets.** NCAAF game-line bets come only from
the sharp-gap scanner (`sharp_pipeline`), which prices soft books against Pinnacle.

## Data
`python -m ncaaf_pipeline.ingest --seasons 2014-2026` loads collegefootballdata.com (key `CFBD_API_KEY`,
free tier 1,000 calls/month, 6 calls per season) into `ncaaf.games`, `ncaaf.lines` (every book, with
openers from 2022 on) and `ncaaf.team_game_stats` (per-play efficiency).

## Models
- `ncaaf_pipeline.ratings`: opponent-adjusted points ratings (decayed ridge, half-life 16 weeks). One
  fit for margin (light shrinkage), one for total (heavy), so spread, total and both team totals agree.
  Settings were tuned on 2016-2019 only.
- `ncaaf_pipeline.efficiency`: adjusted offense/defense ratings for PPA, success rate, explosiveness,
  rush/pass PPA, plays and stuff rate, then a second stage anchored to the market that predicts
  (result - closing line), trained on earlier seasons only.

## Evidence (out of sample, 2020-2025; reports/ncaaf_*_latest.md)
| Test | Result |
|---|---|
| Points ratings, margin MAE | 13.23 vs closing line 12.21 |
| Points ratings, total MAE | 12.99 vs 12.62 |
| Points ratings picks vs close | spreads 47-49% (worse than a coin), totals 51.5-52.3% |
| Efficiency model, gap to close | correlation +0.04 spread, +0.02 total; MAE unchanged |
| Predicting the opener-to-close total move (2023-25) | correlation +0.15, +0.5 pt average CLV, but 49-52% against the opener |

Reading: the closing line already contains everything in these public ratings. The model does see some
of where totals will move from the opener, but half a point of CFB total is worth roughly 1.5-2% of win
probability, below the 4.5% vig, so betting openers on it does not pay.

## What could change this
- Information the market prices slowly: injuries/QB changes, weather, coaching changes (not in this data).
- Derivative markets priced off the full-game line (team totals, halves) - these are scanned live by
  `sharp_pipeline` against Pinnacle, the correct test.
- Re-run `python -m ncaaf_pipeline.efficiency` after each season; adopt nothing unless picks clear 52.4%
  against the close for two straight out-of-sample seasons.
