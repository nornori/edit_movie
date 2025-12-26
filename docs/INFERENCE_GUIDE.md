# Full Video Model 推論ガイド

このドキュメントでは、Full Video Modelを使用した動画からXML生成までの完全な推論プロセスについて説明します。

## 概要

Full Video Modelは、動画全体を解析して重要なシーンを自動抽出し、Premiere Pro用のXMLファイルを生成します。

## クイックスタート

### 1. ワンコマンド実行（推奨）

動画パスを指定するだけで、特徴量抽出→推論→XML生成まで自動実行します：

```bash
python scripts/video_to_xml.py "動画ファイルのパス"
```

**例**:
```bash
python scripts/video_to_xml.py "D:\videos\sample.mp4"
```

### 2. 目標秒数の指定

デフォルトは180秒ですが、`--target` オプションで変更できます：

```bash
# 60秒目標（範囲: 30～80秒）
python scripts/video_to_xml.py "動画パス" --target 60

# 120秒目標（範囲: 60～140秒）
python scripts/video_to_xml.py "動画パス" --target 120
```

**範囲の計算**:
- 最小値 = 目標 ÷ 2
- 最大値 = 目標 + 20

### 3. 出力先の指定

```bash
python scripts/video_to_xml.py "動画パス" --output "custom_output.xml"
```

## 処理フロー

### STEP 1: 特徴量抽出

動画から以下の特徴を10FPSでサンプリング：

**音声特徴（235次元）**:
- 音声エネルギー、発話検出、無音時間
- 話者ID、話者埋め込み（192次元）
- ピッチ、スペクトル、MFCC等の音響特徴
- 音声の時系列変化

**映像特徴（543次元）**:
- シーン変化、動き検出
- 顔検出（位置、サイズ、数）
- CLIP埋め込み（512次元）
- 映像の時系列変化

**時系列特徴（6次元）**:
- カット間隔、動画内位置
- カット密度、累積採用率

**合計**: 784次元の特徴量

### STEP 2: モデル読み込み

Full Video Modelを読み込みます：
- モデルパス: `checkpoints_cut_selection_fullvideo/best_model.pth`
- アーキテクチャ: Transformer Encoder
- 入力: 784次元（音声235 + 映像543 + 時系列6）
- 出力: 各フレームのactive/inactive確率

### STEP 3: 推論実行

各フレームに対してactive/inactiveを予測：
- バッチサイズ: 1（動画ごとに処理）
- 正規化: 学習時のスケーラーを使用
- 出力: 信頼度スコア（active確率 - inactive確率）

### STEP 4: 閾値最適化

目標秒数に合わせて最適な閾値を自動決定：

1. 動画の長さが最小値未満の場合 → 全フレームを採用
2. 100個の閾値候補を生成
3. 各閾値で予測時間を計算
4. 目標範囲内（最小～最大）に収まる閾値を選択
5. 範囲内に収まらない場合 → 目標に最も近い閾値を選択

**例**:
```
Video duration: 601.0s
Target: 180.0s (range: 90.0s - 200.0s)
✅ Optimal threshold: -0.5781
   Duration: 188.8s
   Active ratio: 31.4%
```

### STEP 5: クリップ抽出

閾値を超えたフレームをクリップとして抽出：

**結合処理**:
- クリップ間の隙間が1秒未満 → 隙間を含めて結合
- 例: [0-10s] + [10.5-20s] → [0-20s]

**フィルタリング**:
- 結合後に3秒未満のクリップ → 除外

**パラメータ**:
- `min_clip_duration`: 3.0秒（最小クリップ長）
- `max_gap`: 1.0秒（結合する最大隙間）

### STEP 6: XML生成

Premiere Pro互換のXMLファイルを生成：

**特徴**:
- 動画ファイルのフルパスを自動取得・埋め込み
- 元動画のFPS・解像度を自動検出
- クリップを連続して配置
- 音声トラックも同時に生成

**出力先**: `outputs/{動画名}_output.xml`

## 出力例

```
🚀 Video to XML Pipeline
   Video: D:\videos\sample.mp4
   Output: outputs/sample_output.xml
   Target duration: 180.0s

================================================================================
STEP 1: Extract Features
================================================================================
✅ Features extracted: 6010 timesteps, 740 features

================================================================================
STEP 2: Load Features
================================================================================
✅ Loaded CSV: 6010 frames

================================================================================
STEP 3: Load Model
================================================================================
✅ Model loaded (Epoch 9)

================================================================================
STEP 4: Run Inference
================================================================================
   Running inference on 6010 frames...
✅ Inference completed

================================================================================
STEP 5: Optimize Threshold
================================================================================
   Video duration: 601.0s
   Target: 180.0s (range: 90.0s - 200.0s)
✅ Optimal threshold: -0.5781
   Duration: 188.8s
   Active ratio: 31.4%

================================================================================
STEP 6: Extract Clips
================================================================================
✅ Extracted 29 clips (merged clips with gaps ≤ 1.0s)
   Total duration: 242.6s

================================================================================
STEP 7: Generate XML
================================================================================
   Generating XML for 29 clips...
   Video properties: 1920x1080 @ 59.99fps
✅ XML generated: outputs\sample_output.xml

================================================================================
✅ COMPLETED
================================================================================
   Clips: 29
   Duration: 188.8s
   XML: outputs\sample_output.xml
```

## 詳細設定

### 既存の特徴量を使用する場合

特徴量が既に抽出されている場合は、`generate_xml_from_inference.py` を使用：

```bash
python scripts/generate_xml_from_inference.py "動画パス"
```

**注意**: 特徴量ファイル（`temp_features/{動画名}_features_enhanced.csv`）が必要です。

### カスタムパラメータ

クリップ抽出のパラメータを変更したい場合は、スクリプトを直接編集：

```python
# scripts/video_to_xml.py の extract_clips 関数
clips = extract_clips(
    result['active_binary'], 
    min_clip_duration=3.0,  # 最小クリップ長（秒）
    max_gap=1.0             # 結合する最大隙間（秒）
)
```

## トラブルシューティング

### 特徴量抽出でエラーが出る

**症状**: `Error tokenizing data. C error: Expected X fields in line Y, saw Z`

**原因**: CSVファイルの列数が途中で変わってしまった

**解決策**: 
1. 一時ファイルを削除:
   ```bash
   del "動画パス.temp_visual.csv"
   del "動画パス.temp_audio.csv"
   ```
2. 再実行

### XMLで動画が紐づかない

**症状**: Premiere Proで「メディアがオフライン」と表示される

**原因**: 動画ファイルのパスが正しく埋め込まれていない

**解決策**: 
- `video_to_xml.py` を使用（自動的にフルパスを取得）
- 動画ファイルを移動していないか確認

### クリップが少なすぎる/多すぎる

**症状**: 抽出されるクリップ数が期待と異なる

**解決策**:
1. 目標秒数を調整: `--target` オプションで変更
2. 閾値を手動調整: スクリプト内の `optimize_threshold` 関数を編集

## パフォーマンス

### 処理時間の目安

- **特徴量抽出**: 約3～5分（10分の動画）
- **推論**: 約2秒（6000フレーム）
- **XML生成**: 約0.1秒

**合計**: 約3～5分（初回のみ、2回目以降は特徴量を再利用可能）

### メモリ使用量

- **GPU**: 約2GB（CLIP + 話者認識）
- **RAM**: 約4GB（特徴量の一時保存）

## 関連ドキュメント

- [README.md](../README.md) - プロジェクト全体の概要
- [QUICK_START.md](QUICK_START.md) - クイックスタートガイド
- [PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md) - 技術仕様
