' Runs sniper_dashboard.bat with no console window and waits for it, so the scheduled task stays "Running"
' while the dashboard is up and a second copy is never started.
CreateObject("WScript.Shell").Run "cmd /c ""C:\Users\josh\Git\SuperNovaBets\scripts\sniper_dashboard.bat""", 0, True
