@echo off
REM Download CIFAR-100 dataset before running evolution
REM Run this ONCE before the first evolution run

echo ============================================================
echo CIFAR-100 Dataset Download Script
echo ============================================================
echo.

cd /d "%~dp0"
echo Current directory: %CD%
echo.

echo Downloading CIFAR-100 dataset...
echo This may take a few minutes (~160MB)...
echo.

python download_dataset.py

echo.
echo ============================================================
echo If download successful, you can now run: python evolve.py
echo ============================================================
pause
