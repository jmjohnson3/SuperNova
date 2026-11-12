@echo off
:: SuperNovaBets - keeps the Sports Betting Sniper dashboard running at http://localhost:8517
:: Started at logon and re-checked every 5 minutes by the Sniper-Dashboard task (through sniper_dashboard.vbs,
:: so no console window opens). It listens on this PC only (127.0.0.1); reach it from other devices through a
:: private network such as Tailscale ("tailscale serve --bg 8517"), never by opening the port.
cd /d C:\Users\josh\Git\SuperNovaBets
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs

:: Already up (this task's copy or one started by hand): nothing to do.
netstat -ano | findstr /c:":8517 " | findstr LISTENING >nul && exit /b 0

set LOGFILE=logs\sniper_dashboard_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log
echo ======================================== >> %LOGFILE% 2>&1
echo Dashboard started at %DATE% %TIME% >> %LOGFILE% 2>&1
.venv\Scripts\python.exe -m streamlit run src\sports_sniper\dashboard\app.py --server.port 8517 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false >> %LOGFILE% 2>&1
echo Dashboard stopped at %DATE% %TIME% with code %ERRORLEVEL% >> %LOGFILE% 2>&1
