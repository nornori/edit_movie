@echo off
echo ========================================
echo Enhanced Cut Selection Training V2
echo With Data Augmentation and Deeper Network
echo ========================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Set PYTHONPATH
set PYTHONPATH=%CD%

REM Run training with V2 config
python src/cut_selection/train_cut_selection_kfold_enhanced_v2.py --config configs/config_cut_selection_kfold_enhanced_v2.yaml

echo.
echo ========================================
echo Training Complete!
echo ========================================
pause
