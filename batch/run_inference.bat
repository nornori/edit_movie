@echo off
REM 動画編集AI - 推論実行スクリプト

REM Pythonパスを設定
set PYTHONPATH=%PYTHONPATH%;%CD%\src

REM 引数チェック
if "%~1"=="" (
    echo Usage: run_inference.bat ^<video_path^> [model_path] [output_path]
    echo.
    echo Example:
    echo   run_inference.bat "D:\videos\my_video.mp4"
    echo   run_inference.bat "D:\videos\my_video.mp4" "models\checkpoints_50epochs\best_model.pth"
    exit /b 1
)

REM デフォルト値を設定
set VIDEO_PATH=%~1
set MODEL_PATH=%~2
set OUTPUT_PATH=%~3

if "%MODEL_PATH%"=="" set MODEL_PATH=models\checkpoints_50epochs\best_model.pth
if "%OUTPUT_PATH%"=="" set OUTPUT_PATH=outputs\inference_results\result.xml

echo ================================================================================
echo 動画編集AI - 推論実行
echo ================================================================================
echo.
echo 入力動画: %VIDEO_PATH%
echo モデル: %MODEL_PATH%
echo 出力XML: %OUTPUT_PATH%
echo.
echo ================================================================================
echo.

echo [1/2] 推論を実行中...
python src\inference\inference_pipeline.py "%VIDEO_PATH%" --model "%MODEL_PATH%" --output "%OUTPUT_PATH%" --telop_config configs\config_telop_disabled.yaml

if errorlevel 1 (
    echo.
    echo エラー: 推論に失敗しました
    exit /b 1
)

REM テロップをグラフィックに変換する処理は無効化
REM echo.
REM echo [2/3] テロップをグラフィックに変換中...
REM python src\inference\fix_telop_simple.py "%TEMP_XML%" "%OUTPUT_PATH%"
REM 
REM if errorlevel 1 (
REM     echo.
REM     echo エラー: テロップ変換に失敗しました
REM     exit /b 1
REM )

echo.
echo [2/2] 完了！
echo.
echo ================================================================================
echo 出力XMLファイル: %OUTPUT_PATH%
echo ================================================================================
echo.
echo Premiere Proで上記のXMLファイルを開いてください。
echo.

exit /b 0
