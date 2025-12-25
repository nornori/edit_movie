@echo off
echo ========================================
echo Training Enhanced Cut Selection Model
echo ========================================
echo.

python src/cut_selection/train_cut_selection_kfold_enhanced.py ^
    --config configs/config_cut_selection_kfold_enhanced.yaml

pause
