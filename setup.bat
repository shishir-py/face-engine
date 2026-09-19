@echo off
title FaceMatch AI — Setup
echo.
echo  ╔══════════════════════════════════════════╗
echo  ║     FaceMatch AI — First-Time Setup      ║
echo  ╚══════════════════════════════════════════╝
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo  Please install Python 3.10+ from https://python.org
    echo.
    pause
    exit /b 1
)

echo  [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo  [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo  [2/3] Activating environment...
call venv\Scripts\activate.bat

echo  [3/3] Installing dependencies (this may take a few minutes)...
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo  [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║        Setup Complete!                   ║
echo  ║  Run start.bat to launch FaceMatch AI    ║
echo  ╚══════════════════════════════════════════╝
echo.
pause
