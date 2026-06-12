@echo off
:: SuperNovaBets MLB - Overnight model refresh (~12:30 AM ET)
:: Training is isolated from the operational prediction and snapshot tasks.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8

set LOGFILE=logs\mlb_training_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

echo ======================================== >> %LOGFILE% 2>&1
echo MLB Training run started at %DATE% %TIME% >> %LOGFILE% 2>&1
echo ======================================== >> %LOGFILE% 2>&1

.venv\Scripts\python.exe -m mlb_pipeline.run_daily --skip-crawl --skip-parse --skip-predict >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%

echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
