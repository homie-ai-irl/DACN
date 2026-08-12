@echo off
echo ============================================================
echo   Credit Card Fraud Detection – Starting Training
echo ============================================================
echo.

REM Kiem tra file data
if not exist "data\creditcard.csv" (
    echo ERROR: data\creditcard.csv khong tim thay!
    echo Download tu: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    echo Roi copy vao folder: data\
    pause
    exit /b 1
)

echo Running pipeline...
echo.
python train.py --config configs/config.yaml

echo.
echo ============================================================
echo   Training complete! Xem ket qua trong thu muc: outputs/
echo ============================================================
pause
