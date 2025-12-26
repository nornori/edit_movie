@echo off
echo ========================================
echo Training Full Video Cut Selection Model
echo 1 VIDEO = 1 SAMPLE (no sequence splitting)
echo Applies 90s constraint PER VIDEO
echo ========================================
echo.

python src/cut_selection/train_cut_selection_fullvideo.py --config configs/config_cut_selection_fullvideo.yaml

pause
