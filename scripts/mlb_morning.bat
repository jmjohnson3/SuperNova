@echo off
:: SuperNovaBets MLB — Morning full pipeline (~8:00 AM ET)
:: Crawls, parses, trains, predicts, posts to Discord.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8

set LOGFILE=logs\mlb_morning_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

echo ======================================== >> %LOGFILE% 2>&1
echo MLB Morning run started at %DATE% %TIME% >> %LOGFILE% 2>&1
echo ======================================== >> %LOGFILE% 2>&1

.venv\Scripts\python.exe -m mlb_pipeline.run_daily_and_notify >> %LOGFILE% 2>&1

echo Exit code: %ERRORLEVEL% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
