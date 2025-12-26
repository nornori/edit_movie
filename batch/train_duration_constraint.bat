@echo off
echo ========================================
echo Enhanced Cut Selection Model Training
echo With Duration Constraint Learning
echo ========================================
echo.
echo Configuration: config_cut_selection_kfold_enhanced_reset.yaml
echo Target Duration: 180 seconds (3 minutes)
echo Duration Penalty Weight: 1.0
echo.
echo Starting training...
echo.

set PYTHONPATH=%CD%
python src/cut_selection/train_cut_selection_kfold_enhanced.py ^
    --config configs/config_cut_selection_kfold_enhanced_reset.yaml

echo.
echo ========================================
echo Training completed!
echo ========================================
echo.
echo Check results:
echo   - Graphs: checkpoints_cut_selection_kfold_enhanced_reset/
echo   - Summary: checkpoints_cut_selection_kfold_enhanced_reset/kfold_summary.csv
echo   - HTML Viewer: checkpoints_cut_selection_kfold_enhanced_reset/view_training.html
echo.
pause
