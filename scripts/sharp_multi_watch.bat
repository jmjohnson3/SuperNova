@echo off
:: SuperNovaBets - NBA props/lines and NCAAF lines sharp-gap RESEARCH scanner (no stakes).
:: Frozen NBA props send PAPER PRICE CHECK requests; one per book/player/game/market.
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
:: Side jobs on the snapshots just saved. Background and separate logs, so they never delay the timed checks below.
:: FanDuel/DraftKings arbitrage alerts (no credits); DraftKings re-quote a minute after each alert (1 credit each).
set SIDELOG=logs\sharp_side_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%
start "" /b cmd /c ".venv\Scripts\python.exe -B -m sharp_pipeline.arb_watch >> %SIDELOG%_arb.log 2>&1"
start "" /b cmd /c ".venv\Scripts\python.exe -B -m sharp_pipeline.draftkings_checks >> %SIDELOG%_draftkings.log 2>&1"
:: Separate frozen FanDuel model comparison. Original price experiment remains unchanged.
.venv\Scripts\python.exe -B -m model_research capture --sport nba --alerts >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
:: First-attempt current public quotes; distinct from account/manual evidence. Run before slower reports.
.venv\Scripts\python.exe -B -m automatic_price_checks once --sport nba >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
:: Separate probability challenger reuses the same forecasts/checks; no additional alerts or API calls.
.venv\Scripts\python.exe -B -m nba_props.count_distribution_trial capture >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
:: Separate current FanDuel-feed observations; no Odds API credits or manual-check writes.
.venv\Scripts\python.exe -B -m nba_props.fanduel_live once >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
:: Fixed workload/calibration challenger; uses forecasts saved before the original offer.
:: Run after the time-sensitive direct check; no additional alerts or Odds API calls.
.venv\Scripts\python.exe -B -m nba_props.research_trial capture >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
:: Frozen multi-book consensus comparison; only saved fresh quotes, no extra alerts or API calls.
.venv\Scripts\python.exe -B -m edge_research capture --sport nba ncaaf >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
:: Refresh frozen reviews directly so the legacy all-history report cannot starve their updates.
.venv\Scripts\python.exe -B -m price_survival_review >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
.venv\Scripts\python.exe -B -m paper_collection_review --refresh >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1
.venv\Scripts\python.exe -m sharp_pipeline.report --digest >> %LOGFILE% 2>&1
if errorlevel 1 set EXITCODE=1

echo Exit code: %EXITCODE% >> %LOGFILE% 2>&1
echo Finished at %TIME% >> %LOGFILE% 2>&1
exit /b %EXITCODE%
