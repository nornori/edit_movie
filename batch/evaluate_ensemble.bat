@echo off
echo ========================================
echo Ensemble Model Evaluation
echo ========================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Set PYTHONPATH
set PYTHONPATH=%CD%

REM Run ensemble evaluation
python src/cut_selection/evaluate_ensemble.py --checkpoint_dir checkpoints_cut_selection_kfold_enhanced --data_path preprocessed_data/combined_sequences_cut_selection_enhanced.npz --device cuda

echo.
echo ========================================
echo Evaluation Complete!
echo ========================================
pause
