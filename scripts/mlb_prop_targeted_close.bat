@echo off
:: Lightweight T-120/T-60/T-20 prop close capture. The Python command skips
:: without an API call when no uncaptured game window is due.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
if "%SUPERNOVA_DISABLE_MLB%"=="" (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v SUPERNOVA_DISABLE_MLB 2^>nul ^| findstr SUPERNOVA_DISABLE_MLB') do set "SUPERNOVA_DISABLE_MLB=%%B"
)
if /I "%SUPERNOVA_DISABLE_MLB%"=="1" (
    echo MLB Prop targeted close skipped because SUPERNOVA_DISABLE_MLB=1. >> logs\mlb_disabled_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log 2>&1
    exit /b 0
)

set LOGFILE=logs\mlb_prop_targeted_close_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

echo ======================================== >> %LOGFILE% 2>&1
echo MLB targeted prop close started at %DATE% %TIME% >> %LOGFILE% 2>&1
echo ======================================== >> %LOGFILE% 2>&1

powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_with_mlb_mutex.ps1 -LockName SuperNovaBets_MLB_Operational -WaitSeconds 300 -CommandText ".venv\Scripts\python.exe -m mlb_pipeline.run_daily --prop-close-capture-only" >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% EQU 75 (
  echo Targeted close missed because the MLB operational mutex stayed busy for 300s. >> %LOGFILE% 2>&1
  echo This is a real close-coverage failure, not scheduler success. >> %LOGFILE% 2>&1
)

echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
