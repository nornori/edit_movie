@echo off
REM 動画編集AI - 学習実行スクリプト

REM 引数チェック
set CONFIG_PATH=%~1
if "%CONFIG_PATH%"=="" set CONFIG_PATH=configs\config_multimodal_experiment.yaml

echo ================================================================================
echo 動画編集AI - 学習実行
echo ================================================================================
echo.
echo 設定ファイル: %CONFIG_PATH%
echo.
echo ================================================================================
echo.

echo 学習を開始します...
python -m src.training.train --config "%CONFIG_PATH%"

if errorlevel 1 (
    echo.
    echo エラー: 学習に失敗しました
    exit /b 1
)

echo.
echo ================================================================================
echo 学習完了！
echo ================================================================================
echo.
echo 学習済みモデルは checkpoints\ に保存されました。
echo.

exit /b 0
