@echo off
:: SuperNovaBets MLB — Pre-game update (~4:30 PM ET)
:: Re-crawls injuries + odds, re-predicts props, posts to Discord.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8

set LOGFILE=logs\mlb_pregame_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

echo ======================================== >> %LOGFILE% 2>&1
echo MLB Pre-game run started at %DATE% %TIME% >> %LOGFILE% 2>&1
echo ======================================== >> %LOGFILE% 2>&1

.venv\Scripts\python.exe -m mlb_pipeline.run_daily_and_notify --pre-game >> %LOGFILE% 2>&1

echo Exit code: %ERRORLEVEL% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
