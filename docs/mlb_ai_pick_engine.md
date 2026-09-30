# MLB AI Pick Engine

The AI pick engine is the canonical daily decision layer for MLB. It does not
train projections itself. It consumes the existing game and player-prop
predictions, checks market price/drift/link/CLV/external-agreement context, and
writes one unified pick table.

## Run

```powershell
python -m mlb_pipeline.modeling.ai_pick_engine --date 2026-07-28
```

Scheduled daily and pre-game runs call it after:

```text
external pick fetch
external pick import
game predictions
player prop predictions
external model comparison
```

## Outputs

```text
bets.mlb_ai_pick_engine
data/ai_picks/mlb/YYYY-MM-DD.csv
src/mlb_pipeline/modeling/models/player_props/ai_pick_engine.json
reports/mlb_ai_pick_engine_latest.md
```

## Tiers

```text
bankroll
starter
micro
watch
paper_common
paper_game
lottery
one_sided_fanduel
```

Only `bankroll`, `starter`, and `micro` can be `bettable_now`. Props must still
have a valid link, positive current EV, and a current price that meets or beats
the minimum acceptable American price.

## Why This Exists

The engine separates the stack into two questions:

```text
Projection layer: what does the model think happens?
Pick layer: is the current offered bet worth taking?
```

That keeps projected means like `2.70 TB` from becoming automatic bets without
checking the full price/market/drift context.
