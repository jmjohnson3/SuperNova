@echo off
:: SuperNovaBets MLB - Overnight model refresh (~12:30 AM ET)
:: Training is isolated from the operational prediction and snapshot tasks.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
if "%SUPERNOVA_DISABLE_MLB%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v SUPERNOVA_DISABLE_MLB 2^>nul ^| findstr SUPERNOVA_DISABLE_MLB') do set "SUPERNOVA_DISABLE_MLB=%%B"
)
if /I "%SUPERNOVA_DISABLE_MLB%"=="1" (
    echo MLB Training skipped because SUPERNOVA_DISABLE_MLB=1. >> logs\mlb_disabled_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log 2>&1
    exit /b 0
)

set LOGFILE=logs\mlb_training_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

echo ======================================== >> %LOGFILE% 2>&1
echo MLB Training run started at %DATE% %TIME% >> %LOGFILE% 2>&1
echo ======================================== >> %LOGFILE% 2>&1

.venv\Scripts\python.exe -m mlb_pipeline.run_daily --skip-crawl --skip-parse --skip-predict >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%

echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
