@echo off
REM ========================================
REM ARLifejackets Windows Setup Script
REM ========================================

python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python not found. Installing Python 3.11 (64-bit)...
    winget install --id Python.Python.3.11 -e
) ELSE (
    echo Python is already installed:
    python --version
)

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing required packages...
python -m pip install -r requirements.txt

echo Running main.py...
python main.py

pause
