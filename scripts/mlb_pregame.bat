@echo off
:: SuperNovaBets MLB - Pre-game updates (~10:30 AM and ~4:30 PM ET)
:: Re-crawls injuries + odds, re-predicts props, posts to Discord.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if "%MLB_DISCORD_WEBHOOK_URL%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v MLB_DISCORD_WEBHOOK_URL 2^>nul ^| findstr MLB_DISCORD_WEBHOOK_URL') do set "MLB_DISCORD_WEBHOOK_URL=%%B"
)
set LOCK_PHASE=%~1
if "%LOCK_PHASE%"=="" set LOCK_PHASE=pregame_manual

set LOGFILE=logs\mlb_pregame_%LOCK_PHASE%_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

echo ======================================== >> %LOGFILE% 2>&1
echo MLB Pre-game %LOCK_PHASE% run started at %DATE% %TIME% >> %LOGFILE% 2>&1
echo ======================================== >> %LOGFILE% 2>&1

powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_with_mlb_mutex.ps1 -LockName SuperNovaBets_MLB_Operational -CommandText ".venv\Scripts\python.exe -m mlb_pipeline.run_daily_and_notify --pre-game --lock-phase %LOCK_PHASE%" >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%

echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
