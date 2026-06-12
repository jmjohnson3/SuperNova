@echo off
:: SuperNovaBets MLB - Morning operational pipeline (~8:00 AM ET)
:: Crawls, parses, predicts, shadow-locks, and posts to Discord.
:: Training runs separately overnight so it cannot delay today's slate.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if "%MLB_DISCORD_WEBHOOK_URL%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v MLB_DISCORD_WEBHOOK_URL 2^>nul ^| findstr MLB_DISCORD_WEBHOOK_URL') do set "MLB_DISCORD_WEBHOOK_URL=%%B"
)

set LOGFILE=logs\mlb_morning_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

echo ======================================== >> %LOGFILE% 2>&1
echo MLB Morning run started at %DATE% %TIME% >> %LOGFILE% 2>&1
echo ======================================== >> %LOGFILE% 2>&1

powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_with_mlb_mutex.ps1 -LockName SuperNovaBets_MLB_Operational -CommandText ".venv\Scripts\python.exe -m mlb_pipeline.run_daily_and_notify --skip-train" >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%

echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
