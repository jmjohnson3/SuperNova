@echo off
:: SuperNovaBets - refresh NBA game logs (stats.nba.com) and store the day's prop projections.
:: Research: the logs grade the sharp-gap decisions; the projections are the recorded shadow filter.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
set LOGFILE=logs\nba_props_live_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log
echo ======================================== >> %LOGFILE% 2>&1
echo NBA props live started at %DATE% %TIME% >> %LOGFILE% 2>&1
.venv\Scripts\python.exe -m nba_props.live >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%
echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
