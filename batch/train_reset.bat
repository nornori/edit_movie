@echo off
echo ========================================
echo Enhanced Cut Selection Model - RESET
echo ========================================
echo.
echo Simple baseline configuration:
echo - Standard Cross Entropy Loss (no focal loss)
echo - No temporal variation penalty
echo - No adoption rate penalty
echo - Standard dropout: 0.1
echo - Standard learning rate: 0.0001
echo - Standard weight decay: 0.01
echo.
echo This will establish a clean baseline.
echo.
pause

python -m src.cut_selection.train_cut_selection_kfold_enhanced --config configs/config_cut_selection_kfold_enhanced_reset.yaml

echo.
echo ========================================
echo Training Complete!
echo ========================================
pause
