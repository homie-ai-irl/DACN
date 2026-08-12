@echo off
echo ============================================================
echo   Setup: Credit Card Fraud Detection Pipeline
echo ============================================================
echo.

REM Cai dat tat ca thu vien
echo [1/3] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Checking environment...
python check_env.py

echo.
echo [3/3] Creating required folders...
if not exist "data"    mkdir data
if not exist "outputs" mkdir outputs
if not exist "logs"    mkdir logs

echo.
echo ============================================================
echo   Setup complete!
echo   Next steps:
echo     1. Copy creditcard.csv vao folder: data/
echo     2. Run: python train.py
echo ============================================================
pause
