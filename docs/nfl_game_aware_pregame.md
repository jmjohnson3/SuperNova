# Game-aware NFL pregame runs

`NFL-PrimeTime-Refresh` now checks the database kickoff schedule every ten minutes,
every day. It starts one prediction/publication wave when games enter T-90 minutes.
The first check normally lands between T-90 and T-80. This supports Sunday waves,
Monday/Thursday evenings, unusual weekdays, and changing kickoff times without
hard-coded local start times. Timestamps are compared in UTC.

The morning `NFL-Daily` data refresh remains unchanged. The near-kickoff path
refreshes current context and FanDuel quotes, scores the existing frozen release,
locks paper/micro candidates, captures pinned challengers, checks the receiving
trial, and publishes only the requested matchups. It skips training and the
standalone schema setup step. Unrelated games' current forecasts are preserved. Daily micro caps remain
global, not five new picks per wave. Historic locks are never rewritten.

Idle checks do not fetch odds or send Discord messages. A due wave may fetch its
date's provider payload, but scoring and publication are restricted to due games.
This runner is separate from the existing near-kickoff close collector.

## Failures and duplicates

`reports/nfl_pregame_state.json` tracks attempts by game ID and kickoff. Successful
waves are not repeated. Known failures before publication can retry after ten
minutes, up to three attempts and only while more than twenty minutes remain.
An interrupted run, timeout, or uncertain/partial Discord delivery requires review
rather than automatically sending duplicate recommendations. The child process
tree times out before kickoff. Failed or missed windows return nonzero.

Read `reports/nfl_pregame_latest.json` and the referenced
`reports/nfl_daily_run_pregame_<id>.json` for diagnosis. Do not erase state to retry
without first checking whether Discord already delivered messages.

## Install and verify

From the repository, run in an administrator PowerShell if Windows denies access:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_nfl_pregame_task.ps1
.\.venv\Scripts\python.exe -m nfl_pipeline.run_pregame --dry-run
```

The installer updates the existing refresh task rather than creating a second
publisher. The task uses an interactive user token: the user must be logged in,
the machine available, and network/DB access working. A dry run reads schedule
and completion state only; it does not fabricate a future lock or send Discord.
