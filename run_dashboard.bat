@echo off
REM Uses the active environment's Python (venv or PATH) - no hard-coded interpreter.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python not found in PATH & pause & exit /b 1)
python launch_dashboard.py
pause
