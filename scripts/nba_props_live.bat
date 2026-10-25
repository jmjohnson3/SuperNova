@echo off
:: SuperNovaBets - NBA projections (research context, not bets).
::   nba_props_live.bat          full run: game logs, lineups/injuries, refit, project, post if due
::   nba_props_live.bat --news   news watch: lineups/injuries; re-project on change, post if due
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v ODDS_API_KEY 2^>nul ^| findstr ODDS_API_KEY') do set "ODDS_API_KEY=%%B"
for %%V in (NBA_DISCORD_WEBHOOK_URL MLB_DISCORD_WEBHOOK_URL) do (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v %%V 2^>nul ^| findstr %%V') do set "%%V=%%B"
)
set LOGFILE=logs\nba_props_live_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log
echo ======================================== >> %LOGFILE% 2>&1
echo NBA props %1 started at %DATE% %TIME% >> %LOGFILE% 2>&1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_with_nfl_mutex.ps1 -LockName SuperNovaBets_NBA_Props -WaitSeconds 240 -CommandText ".venv\Scripts\python.exe -m nba_props.live %1" >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%
.venv\Scripts\python.exe -m nba_props.post >> %LOGFILE% 2>&1
echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
