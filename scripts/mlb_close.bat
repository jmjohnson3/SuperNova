@echo off
:: SuperNovaBets MLB — Closing line run (~6:30 PM ET)
:: Re-crawls closing odds, grades outcomes + CLV. No Discord needed.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8

set LOGFILE=logs\mlb_close_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

echo ======================================== >> %LOGFILE% 2>&1
echo MLB Close run started at %DATE% %TIME%   >> %LOGFILE% 2>&1
echo ======================================== >> %LOGFILE% 2>&1

.venv\Scripts\python.exe -m mlb_pipeline.run_daily --close-only >> %LOGFILE% 2>&1

echo Exit code: %ERRORLEVEL% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
