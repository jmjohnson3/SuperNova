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
:: FanDuel/DraftKings arbitrage alerts from the payloads just saved (no credits). Background, separate log.
start "" /b cmd /c ".venv\Scripts\python.exe -B -m sharp_pipeline.arb_watch >> logs\sharp_side_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%_arb_nfl.log 2>&1"
:: Independent prospective spread quotes; frozen props/totals scanner remains unchanged.
.venv\Scripts\python.exe -B -m nfl_spread_collection once >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
:: Paired paper comparison uses saved model forecasts; never reserves or confirms stakes.
.venv\Scripts\python.exe -B -m model_research capture --sport nfl --alerts >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
:: Current public FanDuel quotes for original M alerts; no wagers or manual-check entries.
.venv\Scripts\python.exe -B -m automatic_price_checks once --sport nfl >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
:: Separate frozen consensus comparison; original NFL experiment and stake guards stay fixed.
.venv\Scripts\python.exe -B -m edge_research capture --sport nfl >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
:: One status message per game, edited in place; original check alerts retain their deadlines.
.venv\Scripts\python.exe -B -m nfl_decision_board --post >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
:: Refresh the cross-sport prediction ledger from saved decisions/results only.
.venv\Scripts\python.exe -B -m prediction_ledger >> %LOGFILE% 2>&1
if errorlevel 1 (
    set EXITCODE=1
) else (
    .venv\Scripts\python.exe -B -m prediction_ledger_discord >> %LOGFILE% 2>&1
    if errorlevel 1 set EXITCODE=1
)

:: Read-only cross-sport price survival and exact-close audit after time-sensitive publication.
.venv\Scripts\python.exe -B -m price_survival_review >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1

:: Separate read-only health/news observations; original decisions and cash pause remain fixed.
.venv\Scripts\python.exe -B -m nfl_slate_readiness --refresh-news >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1

echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
