# NFL Discord Matchup Cards

The daily runner now publishes combined matchup cards instead of separate
league-wide game-bet and prop lists. Each card contains the teams, date, kickoff
in the Discord reader's local timezone, and the matching saved selections.

Sections remain separate:

- Bankroll game picks and props, only when their original tier is ledger-locked.
- Current $1 micro tests. The five-play limit is across the entire slate, not
  five per game. The price/minimum-price/drift details are retained.
- Previously locked micro picks, explicitly not additional plays.
- Paper game picks and paper props, with individual bet links and available
  research-section FanDuel parlay links.

Paper prop shortlists retain the existing top-ten-per-stat/position selection
across the slate. Grouping happens after ranking; it does not widen eligibility
or change models. Game picks are grouped under their actual matchup. Games that
have started are excluded.

Long matchups use numbered continuation cards. Rows and links are never
truncated or mixed with another game's card. Discord rate-limit responses are
retried using the provided delay; unconfirmed network failures are not blindly
retried. Successful cards are recorded individually in the daily run report.

## Preview Without Sending

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.publish_forecasts --date 2026-09-20 --kind matchups
```

This only reads saved forecasts and writes the preview under
`reports/nfl_discord_matchups_2026-09-20.md` and `.json`. It does not fetch odds,
retrain, write ledger rows, or send to Discord. Add `--game-id <saved-game-id>`
to preview one game. Single-game filtering happens after slate-wide selection.

The normal `run_daily_and_notify` job sends the cards after forecasts and ledger
writes succeed, using the existing NFL webhook. No extra channel is required.
Production artifacts and forecast-scoring code are unchanged by this format.
