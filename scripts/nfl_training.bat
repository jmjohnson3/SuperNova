@echo off
:: SuperNovaBets NFL - Offline model refresh after the NFL week completes.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs

set RUN_SEASON_ARG=
if /I "%~1"=="--help" (
    set "RUN_SEASON_ARG=--help"
) else (
    if not "%~1"=="" set "RUN_SEASON_ARG=--season %~1"
)
set LOGFILE=logs\nfl_training_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

echo ======================================== >> %LOGFILE% 2>&1
echo NFL Training run started at %DATE% %TIME% >> %LOGFILE% 2>&1
echo ======================================== >> %LOGFILE% 2>&1

powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_with_nfl_mutex.ps1 -LockName SuperNovaBets_NFL_Training -CommandText ".venv\Scripts\python.exe -m nfl_pipeline.run_training %RUN_SEASON_ARG%" >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%

echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
