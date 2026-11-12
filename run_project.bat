
@echo off
title Elbow Rehabilitation Monitoring System

cd /d "%~dp0"

echo ==========================================
echo   Elbow Rehabilitation Monitoring System
echo ==========================================
echo.

set "PYTHON=.venv\Scripts\python.exe"
if exist ".venv-ui\Scripts\python.exe" set "PYTHON=.venv-ui\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Virtual environment not found.
    echo Please create .venv and install requirements first.
    pause
    exit /b 1
)

echo Checking Python environment...
"%PYTHON%" --version

echo.
echo Starting the Streamlit application...
echo Keep this window open while using the application.
echo.

"%PYTHON%" -m streamlit run app.py

echo.
echo The application has stopped.
pause
