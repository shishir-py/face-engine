@echo off
title FaceMatch AI — Server
echo.
echo  ╔══════════════════════════════════════════╗
echo  ║      FaceMatch AI — Starting Server      ║
echo  ╚══════════════════════════════════════════╝
echo.

:: Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo  [OK] Virtual environment activated.
) else (
    echo  [!] No virtual environment found. Run setup.bat first,
    echo      or ensure dependencies are installed globally.
    echo.
)

:: Create data directories if needed
if not exist data\people mkdir data\people
if not exist data\index mkdir data\index
if not exist data\db mkdir data\db
if not exist data\uploads mkdir data\uploads

echo.
echo  Starting FaceMatch AI server...
echo  Open http://localhost:8000 in your browser.
echo  Press Ctrl+C to stop the server.
echo.

python run.py
if errorlevel 1 (
    echo.
    echo  [ERROR] Server failed to start.
    echo  Make sure you ran setup.bat first.
    pause
)
