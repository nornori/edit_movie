# 🚀 クイックスタートガイド

## 📋 前提条件

- Python 3.8以上がインストールされている
- 必要なライブラリがインストールされている（`pip install -r requirements.txt`）
- 学習済みモデルがある（または学習を実行する）

---

## 🎯 新しい動画を自動編集する（推論）

### 方法1: バッチファイルを使う（簡単！）

```bash
run_inference.bat "D:\videos\my_video.mp4"
```

これだけで完了！Premiere Pro用のXMLが `outputs/inference_results/result.xml` に生成されます。

### 方法2: 手動で実行

```bash
# Pythonパスを設定
set PYTHONPATH=%PYTHONPATH%;%CD%\src

# 1. 推論実行
python src/inference/inference_pipeline.py "D:\videos\my_video.mp4" ^
    --model models/checkpoints_50epochs/best_model.pth ^
    --output outputs/inference_results/temp.xml

# 2. テロップをグラフィックに変換
python src/inference/fix_telop_simple.py ^
    outputs/inference_results/temp.xml ^
    outputs/inference_results/final.xml
```

### 3. Premiere Proで開く

生成された `final.xml` をPremiere Proで開いてください。

---

## 📚 学習用データを準備する

### 方法1: バッチファイルを使う（簡単！）

```bash
run_data_preparation.bat
```

### 方法2: 手動で実行

```bash
# Pythonパスを設定
set PYTHONPATH=%PYTHONPATH%;%CD%\src

# 1. XMLからラベル抽出
python src/data_preparation/premiere_xml_parser.py

# 2. 動画から特徴量抽出
python src/data_preparation/extract_video_features_parallel.py

# 3. データ統合
python src/data_preparation/data_preprocessing.py
```

**注意**: 編集済み動画とXMLを `data/raw/editxml/` に配置してください。

---

## 🎓 モデルを学習する

### 方法1: バッチファイルを使う（簡単！）

```bash
run_training.bat
```

### 方法2: 手動で実行

```bash
# Pythonパスを設定
set PYTHONPATH=%PYTHONPATH%;%CD%\src

# 学習実行
python src/training/training.py --config configs/config_multimodal.yaml
```

学習済みモデルは `models/checkpoints_50epochs/` に保存されます。

---

## 📁 ファイル配置

### 推論前に必要なもの
```
models/
└── checkpoints_50epochs/
    ├── best_model.pth
    ├── audio_preprocessor.pkl
    └── visual_preprocessor.pkl
```

### データ準備前に必要なもの
```
data/
└── raw/
    └── editxml/
        ├── video1.mp4
        ├── video1.xml
        ├── video2.mp4
        └── video2.xml
```

---

## 🔧 トラブルシューティング

### エラー: ModuleNotFoundError

**原因**: Pythonパスが設定されていない

**解決策**:
```bash
set PYTHONPATH=%PYTHONPATH%;%CD%\src
```

または、バッチファイル（`run_inference.bat`など）を使用してください。

### エラー: FileNotFoundError: checkpoints_50epochs/best_model.pth

**原因**: 学習済みモデルがない

**解決策**:
1. データ準備を実行: `run_data_preparation.bat`
2. 学習を実行: `run_training.bat`

### XMLが読み込めない

**原因**: テロップ変換が実行されていない

**解決策**:
```bash
python src/inference/fix_telop_simple.py temp.xml final.xml
```

または、`run_inference.bat`を使用してください（自動で実行されます）。

---

## 📊 実行時間の目安

- **データ準備**: 約30分〜1時間（動画の数と長さによる）
- **学習**: 約2〜3時間（50エポック、GPU使用時）
- **推論**: 約30秒/動画（特徴量抽出含む）

---

## 💡 ヒント

### カット数を調整したい

`src/inference/inference_pipeline.py` の閾値を変更してください：

```python
# 現在の設定（約500カット）
active_frames = track_params['active'] > 0.29

# カット数を増やす場合（閾値を下げる）
active_frames = track_params['active'] > 0.20

# カット数を減らす場合（閾値を上げる）
active_frames = track_params['active'] > 0.40
```

### GPUを使用したい

学習時に自動的にGPUが使用されます（CUDA対応GPUがある場合）。

推論時にGPUを使用する場合:
```bash
python src/inference/inference_pipeline.py "video.mp4" ^
    --model models/checkpoints_50epochs/best_model.pth ^
    --device cuda ^
    --output result.xml
```

---

## 📖 詳細なドキュメント

- [プロジェクト全体の流れ](docs/guides/PROJECT_WORKFLOW_GUIDE.md)
- [必要なファイル一覧](docs/guides/REQUIRED_FILES_BY_PHASE.md)
- [音声カット & テロップ変換](docs/summaries/AUDIO_CUT_AND_TELOP_GRAPHICS_SUMMARY.md)

---

## 🎉 成功例

推論が成功すると、以下のようなメッセージが表示されます：

```
================================================================================
✅ Inference complete!
Output XML: outputs/inference_results/final.xml
================================================================================

Premiere Proで上記のXMLファイルを開いてください。
```

Premiere Proで開くと、自動的に編集されたタイムラインが表示されます！
