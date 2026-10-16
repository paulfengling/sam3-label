@echo off
cd /d "%~dp0"
if exist ".sam3_label\Scripts\python.exe" (
    ".sam3_label\Scripts\python.exe" main.py %*
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py %*
) else (
    echo Missing virtual environment: .sam3_label or .venv. See RUN_SAM3.md.
    pause
    exit /b 1
)
if errorlevel 1 pause
