@echo off
:: SuperNovaBets - NBA props/lines and NCAAF lines sharp-gap RESEARCH scanner (no stakes, no pings).
:: Scans, stores and grades gaps against Pinnacle; posts one morning digest per sport.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs

:: Always take the saved key: a scheduler-inherited environment can hold a replaced (old) key.
for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v ODDS_API_KEY 2^>nul ^| findstr ODDS_API_KEY') do set "ODDS_API_KEY=%%B"
for %%V in (NFL_ODDS_API_RESET_DAY MLB_DISCORD_WEBHOOK_URL NBA_DISCORD_WEBHOOK_URL NCAAF_DISCORD_WEBHOOK_URL) do (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v %%V 2^>nul ^| findstr %%V') do set "%%V=%%B"
)

set LOGFILE=logs\sharp_multi_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log

echo ======================================== >> %LOGFILE% 2>&1
echo Sharp Multi Watch started at %DATE% %TIME% >> %LOGFILE% 2>&1

powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_with_nfl_mutex.ps1 -LockName SuperNovaBets_Sharp_Multi -WaitSeconds 60 -CommandText ".venv\Scripts\python.exe -m sharp_pipeline.scan" >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%
.venv\Scripts\python.exe -m sharp_pipeline.report --digest >> %LOGFILE% 2>&1

echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
