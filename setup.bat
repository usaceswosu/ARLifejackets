@echo off
REM ========================================
REM ARLifejackets Windows Setup Script
REM ========================================

REM --- 1. Check if Python is installed ---
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python not found. Installing Python 3.11...
    REM Install latest Python 3.11 via winget
    winget install --id Python.Python.3.11 -e --source winget
) ELSE (
    echo Python is already installed:
    python --version
)

REM --- 2. Upgrade pip ---
echo Upgrading pip...
python -m pip install --upgrade pip

REM --- 3. Install dependencies ---
echo Installing required packages from requirements.txt...
pip install -r requirements.txt

REM --- 4. Run the project ---
echo Running main.py...
python main.py

pause
