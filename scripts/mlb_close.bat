@echo off
:: SuperNovaBets MLB - Hourly closing line captures (~9:45 AM through late evening ET)
:: Re-crawls closing odds, grades outcomes + CLV. No Discord needed.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
if "%SUPERNOVA_DISABLE_MLB%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v SUPERNOVA_DISABLE_MLB 2^>nul ^| findstr SUPERNOVA_DISABLE_MLB') do set "SUPERNOVA_DISABLE_MLB=%%B"
)
if /I "%SUPERNOVA_DISABLE_MLB%"=="1" (
    echo MLB Close skipped because SUPERNOVA_DISABLE_MLB=1. >> logs\mlb_disabled_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log 2>&1
    exit /b 0
)

set LOGFILE=logs\mlb_close_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

echo ======================================== >> %LOGFILE% 2>&1
echo MLB Close run started at %DATE% %TIME%   >> %LOGFILE% 2>&1
echo ======================================== >> %LOGFILE% 2>&1

powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_with_mlb_mutex.ps1 -LockName SuperNovaBets_MLB_Operational -CommandText ".venv\Scripts\python.exe -m mlb_pipeline.run_daily --close-only" >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%

echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
