@echo off
setlocal

cd /d "%~dp0scripts"
python wfs4.py

if errorlevel 1 (
    echo.
    echo Workflow Studio exited with an error.
    pause
)
