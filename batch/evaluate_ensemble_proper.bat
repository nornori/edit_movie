@echo off
REM Proper Ensemble Evaluation (V1) - Validation Data Only
REM This script evaluates the ensemble using ONLY validation data to avoid training data leakage

echo ========================================
echo Proper Ensemble Evaluation (V1)
echo ========================================
echo.
echo This evaluation uses ONLY validation data from each fold
echo to ensure no training data leakage.
echo.

call .venv\Scripts\activate.bat

python src/cut_selection/evaluate_ensemble_proper.py ^
    --checkpoint_dir checkpoints_cut_selection_kfold_enhanced ^
    --data_path preprocessed_data/combined_sequences_cut_selection_enhanced.npz ^
    --n_folds 5 ^
    --device cuda

echo.
echo ========================================
echo Evaluation Complete!
echo ========================================
echo.
echo Results saved to:
echo   - checkpoints_cut_selection_kfold_enhanced/ensemble_comparison_proper.csv
echo   - checkpoints_cut_selection_kfold_enhanced/ensemble_comparison_proper.png
echo.

pause
