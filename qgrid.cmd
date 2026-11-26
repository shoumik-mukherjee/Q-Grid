@echo off
start "Q-Grid backend" /D "D:\QGrid" cmd /k ".venv\Scripts\python.exe -m uvicorn qgrid.api:app --app-dir backend --port 8000"
start "Q-Grid website" /D "D:\QGrid\frontend" cmd /k "npm run dev"
timeout /t 8 /nobreak >nul
start "" "http://localhost:5173/?mode=operate"
