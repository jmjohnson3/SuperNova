@echo off
:: SuperNovaBets - NBA projections (research context, not bets).
::   nba_props_live.bat          full run: game logs, lineups/injuries, refit, project, post if due
::   nba_props_live.bat --news   news watch: lineups/injuries; re-project on change, post if due
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v ODDS_API_KEY 2^>nul ^| findstr ODDS_API_KEY') do set "ODDS_API_KEY=%%B"
for %%V in (NBA_DISCORD_WEBHOOK_URL MLB_DISCORD_WEBHOOK_URL) do (
    for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v %%V 2^>nul ^| findstr %%V') do set "%%V=%%B"
)
set LOGFILE=logs\nba_props_live_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log
echo ======================================== >> %LOGFILE% 2>&1
echo NBA props %1 started at %DATE% %TIME% >> %LOGFILE% 2>&1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_with_nfl_mutex.ps1 -LockName SuperNovaBets_NBA_Props -WaitSeconds 240 -CommandText ".venv\Scripts\python.exe -m nba_props.live %1" >> %LOGFILE% 2>&1
set EXITCODE=%ERRORLEVEL%
:: Official final scores for the separate prospective comparison, after normal result ingestion.
if /I not "%1"=="--news" (
    .venv\Scripts\python.exe -B -m model_research results >> %LOGFILE% 2>&1
    if errorlevel 1 set EXITCODE=1
    .venv\Scripts\python.exe -B -m nba_props.paper_recap --refresh-results >> %LOGFILE% 2>&1
    if errorlevel 1 set EXITCODE=1
)
.venv\Scripts\python.exe -m nba_props.post >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
:: Independent frozen workload/calibration research. Keep normal posting ahead of extra work.
:: NBA stats requests consume no Odds API credits; registered weights are never refitted here.
if /I not "%1"=="--news" (
    .venv\Scripts\python.exe -B -m nba_props.research_trial ingest >> %LOGFILE% 2>&1
    if errorlevel 1 set EXITCODE=1
)
.venv\Scripts\python.exe -B -m nba_props.research_trial project >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
:: Separate registered joint rotations; no production model replacement or automatic refit.
.venv\Scripts\python.exe -B -m edge_research project >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
if /I not "%1"=="--news" (
    .venv\Scripts\python.exe -B -m nba_props.research_trial report >> %LOGFILE% 2>&1
    if errorlevel 1 set EXITCODE=1
    .venv\Scripts\python.exe -B -m edge_research_numeric_report >> %LOGFILE% 2>&1
    if errorlevel 1 set EXITCODE=1
    .venv\Scripts\python.exe -B -m edge_research availability >> %LOGFILE% 2>&1
    if errorlevel 1 set EXITCODE=1
    :: Refresh free schedule/quota metadata and the cost audit; never buys historical odds.
    .venv\Scripts\python.exe -B -m collection_budget --refresh-events >> %LOGFILE% 2>&1
    if errorlevel 1 set EXITCODE=1
    :: Offline alternative only: assess a 500-credit design without changing collection.
    .venv\Scripts\python.exe -B -m low_credit_assessment >> %LOGFILE% 2>&1
    if errorlevel 1 set EXITCODE=1
    :: Read-only right/wrong ledger after final-result refresh and research reports.
    .venv\Scripts\python.exe -B -m prediction_ledger >> %LOGFILE% 2>&1
    if errorlevel 1 (
        set EXITCODE=1
    ) else (
        .venv\Scripts\python.exe -B -m prediction_ledger_discord >> %LOGFILE% 2>&1
        if errorlevel 1 set EXITCODE=1
    )
)
echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
