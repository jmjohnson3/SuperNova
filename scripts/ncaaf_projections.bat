@echo off
:: SuperNovaBets - daily NCAAF projections (research context, not bets) to #ncaa.
:: Refreshes this season's games/lines from CFBD (2 calls), refits ratings, posts the day's slate once.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
for %%V in (CFBD_API_KEY NCAAF_DISCORD_WEBHOOK_URL MLB_DISCORD_WEBHOOK_URL) do (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v %%V 2^>nul ^| findstr %%V') do set "%%V=%%B"
)
set LOGFILE=logs\ncaaf_projections_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log
echo ======================================== >> %LOGFILE% 2>&1
echo NCAAF projections started at %DATE% %TIME% >> %LOGFILE% 2>&1
.venv\Scripts\python.exe -m ncaaf_pipeline.projections >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%
echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
