@echo off
:: SuperNovaBets - Two-weekly rerun of the price-gap validity test (stored prices only, no credits, no Discord).
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
set LOGFILE=logs\price_gap_validity_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

echo Price-gap validity run started at %DATE% %TIME% >> %LOGFILE% 2>&1
.venv\Scripts\python.exe -m sharp_pipeline.gap_validity >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%
echo Exit code: %EXITCODE% at %TIME% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
