@echo off
:: SuperNovaBets NFL - Daily lock/prediction pipeline.
:: Crawls current odds, parses, predicts, locks micro candidates, and posts to Discord.
:: Training runs separately so live runs stay fast.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs

if "%NFL_DISCORD_WEBHOOK_URL%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v NFL_DISCORD_WEBHOOK_URL 2^>nul ^| findstr NFL_DISCORD_WEBHOOK_URL') do set "NFL_DISCORD_WEBHOOK_URL=%%B"
)
if "%NFL_DISCORD_WEBHOOK_URL%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v MLB_DISCORD_WEBHOOK_URL 2^>nul ^| findstr MLB_DISCORD_WEBHOOK_URL') do set "NFL_DISCORD_WEBHOOK_URL=%%B"
)
if "%NFL_DISCORD_WEBHOOK_URL%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v DISCORD_WEBHOOK_URL 2^>nul ^| findstr DISCORD_WEBHOOK_URL') do set "NFL_DISCORD_WEBHOOK_URL=%%B"
)
if "%ODDS_API_KEY%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v ODDS_API_KEY 2^>nul ^| findstr ODDS_API_KEY') do set "ODDS_API_KEY=%%B"
)
if "%SPORTSGAMEODDS_API_KEY%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v SPORTSGAMEODDS_API_KEY 2^>nul ^| findstr SPORTSGAMEODDS_API_KEY') do set "SPORTSGAMEODDS_API_KEY=%%B"
)
if "%SPORTS_GAME_ODDS_API_KEY%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v SPORTS_GAME_ODDS_API_KEY 2^>nul ^| findstr SPORTS_GAME_ODDS_API_KEY') do set "SPORTS_GAME_ODDS_API_KEY=%%B"
)
if "%THERUNDOWN_API_KEY%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v THERUNDOWN_API_KEY 2^>nul ^| findstr THERUNDOWN_API_KEY') do set "THERUNDOWN_API_KEY=%%B"
)
if "%NFL_ODDS_PROVIDER_ORDER%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v NFL_ODDS_PROVIDER_ORDER 2^>nul ^| findstr NFL_ODDS_PROVIDER_ORDER') do set "NFL_ODDS_PROVIDER_ORDER=%%B"
)
:: Optional NFL settings (FanDuel state for betslip links; sharp-watch markets and credit floor).
for %%V in (NFL_FANDUEL_STATE NFL_SHARP_WATCH_MARKETS NFL_SHARP_CREDIT_FLOOR NFL_ODDS_API_RESET_DAY) do (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v %%V 2^>nul ^| findstr %%V') do set "%%V=%%B"
)

set RUN_DATE_ARG=
set NFL_RUN_MODULE=nfl_pipeline.run_daily_and_notify
set NFL_RUN_EXTRA=--skip-train
set NFL_MUTEX_WAIT=3600
set NFL_LOG_PREFIX=nfl_daily
if /I "%~1"=="--game-aware" (
    set NFL_RUN_MODULE=nfl_pipeline.run_pregame
    set NFL_RUN_EXTRA=
    set NFL_MUTEX_WAIT=30
    set NFL_LOG_PREFIX=nfl_pregame
) else (
if /I "%~1"=="--help" (
    set "RUN_DATE_ARG=--help"
) else (
    if not "%~1"=="" set "RUN_DATE_ARG=--date %~1"
)
)
set LOGFILE=logs\%NFL_LOG_PREFIX%_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

echo ======================================== >> %LOGFILE% 2>&1
echo NFL Daily run started at %DATE% %TIME% >> %LOGFILE% 2>&1
echo ======================================== >> %LOGFILE% 2>&1

powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_with_nfl_mutex.ps1 -LockName SuperNovaBets_NFL_Operational -WaitSeconds %NFL_MUTEX_WAIT% -CommandText ".venv\Scripts\python.exe -m %NFL_RUN_MODULE% %RUN_DATE_ARG% %NFL_RUN_EXTRA%" >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%

echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
