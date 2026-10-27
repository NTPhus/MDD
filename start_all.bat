@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found!
    echo Please run setup first.
    pause
    exit /b 1
)

.venv\Scripts\python.exe scripts\run_system.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] An error occurred while executing the command.
    pause
    exit /b %errorlevel%
)
