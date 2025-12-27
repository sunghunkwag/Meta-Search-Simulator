@echo off
cd /d "%~dp0"
title RSI Infinite Loop Monitor (L2 Meta-Logic)

echo ====================================================
echo Starting Infinite Recursive Self-Improvement Loop...
echo ====================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python (check "Add to PATH" during installation).
    echo.
    pause
    exit /b
)

REM Check if the script exists in the same folder
if not exist "L2_UNIFIED_RSI.py" (
    echo.
    echo [ERROR] L2_UNIFIED_RSI.py is MISSING!
    echo Please ensure this BAT file is in the EXACT SAME FOLDER as L2_UNIFIED_RSI.py.
    echo (If you downloaded only the BAT file, you must also download the .py file)
    echo.
    pause
    exit /b
)

echo Mode: L2_UNIFIED_RSI (Turing Complete + Meta-Engine)
echo Task: poly2 (Self-Optimization)
echo.
python L2_UNIFIED_RSI.py rsi-loop --generations 500 --rounds 100
echo.
echo RSI Loop stopped.
pause
