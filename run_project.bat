
@echo off
title Elbow Rehabilitation Monitoring System

cd /d "%~dp0"

echo ==========================================
echo   Elbow Rehabilitation Monitoring System
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo Please create .venv and install requirements first.
    pause
    exit /b 1
)

echo Checking Python environment...
".venv\Scripts\python.exe" --version

echo.
echo Starting the system...
echo Keep this window open while using the application.
echo.

".venv\Scripts\python.exe" app.py

echo.
echo The application has stopped.
pause
