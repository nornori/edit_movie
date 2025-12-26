@echo off
echo ========================================
echo Enhanced Cut Selection Model - RETRAIN
echo ========================================
echo.
echo Improvements:
echo - Increased dropout: 0.15 -^> 0.20
echo - Increased focal_alpha: 0.5 -^> 0.75
echo - Increased focal_gamma: 2.0 -^> 3.0
echo - DOUBLED adoption_penalty: 10.0 -^> 20.0
echo - Target adoption rate: 23%% (matches data)
echo.
echo This should reduce false positives significantly.
echo.
pause

python -m src.cut_selection.train_cut_selection_kfold_enhanced --config configs/config_cut_selection_kfold_enhanced_retrain.yaml

echo.
echo ========================================
echo Training Complete!
echo ========================================
pause
