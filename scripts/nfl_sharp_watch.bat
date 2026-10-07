@echo off
:: SuperNovaBets NFL - live sharp-gap scanner (book price vs Pinnacle/exchange no-vig fair price).
:: Runs on its own mutex so alerts never queue behind the model pipeline's long steps.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs

:: Always take the saved key: a scheduler-inherited environment can hold a replaced (old) key.
for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v ODDS_API_KEY 2^>nul ^| findstr ODDS_API_KEY') do set "ODDS_API_KEY=%%B"
if "%NFL_DISCORD_WEBHOOK_URL%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v NFL_DISCORD_WEBHOOK_URL 2^>nul ^| findstr NFL_DISCORD_WEBHOOK_URL') do set "NFL_DISCORD_WEBHOOK_URL=%%B"
)
if "%NFL_DISCORD_WEBHOOK_URL%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v MLB_DISCORD_WEBHOOK_URL 2^>nul ^| findstr MLB_DISCORD_WEBHOOK_URL') do set "NFL_DISCORD_WEBHOOK_URL=%%B"
)
for %%V in (NFL_FANDUEL_STATE NFL_SHARP_WATCH_MARKETS NFL_SHARP_CREDIT_FLOOR NFL_ODDS_API_RESET_DAY) do (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v %%V 2^>nul ^| findstr %%V') do set "%%V=%%B"
)

set LOGFILE=logs\nfl_sharp_watch_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

echo ======================================== >> %LOGFILE% 2>&1
echo NFL Sharp Watch started at %DATE% %TIME% >> %LOGFILE% 2>&1

powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_with_nfl_mutex.ps1 -LockName SuperNovaBets_NFL_SharpWatch -WaitSeconds 60 -CommandText ".venv\Scripts\python.exe -m nfl_pipeline.sharp_watch" >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%

echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
