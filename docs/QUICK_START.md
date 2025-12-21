# 🚀 クイックスタートガイド - カット選択モデル

## 📋 前提条件

- Python 3.8以上がインストールされている
- 必要なライブラリがインストールされている（`pip install -r requirements.txt`）
- 学習済みモデルがある（または学習を実行する）

**注意**: 本プロジェクトは現在**カット選択（Cut Selection）に特化**しています。グラフィック配置やテロップ生成は精度が低いため、今後の課題となっています。

---

## 🎯 新しい動画を自動編集する（推論）

### 方法1: バッチファイルを使う（簡単！）

```bash
run_inference.bat "path\to\your_video.mp4"
```

これだけで完了！Premiere Pro用のXMLが `outputs/inference_results/result.xml` に生成されます。

### 方法2: 手動で実行

```bash
# 推論実行
python -m src.inference.inference_pipeline "path\to\your_video.mp4" ^
    --output outputs/inference_results/result.xml
```

### 3. Premiere Proで開く

生成された `result.xml` をPremiere Proで開いてください。

---

## 📚 カット選択用データを準備する

### データ準備

```bash
# カット選択用データの作成
python scripts/create_cut_selection_data.py
```

**注意**: 
- 編集済み動画とXMLを `data/raw/editxml/` に配置してください
- 動画から特徴量を抽出済みであることが前提です（`data/processed/source_features/`）
- アクティブラベルを抽出済みであることが前提です（`data/processed/active_labels/`）

---

## 🎓 カット選択モデルを学習する

### 学習実行

```bash
# バッチファイルで実行（推奨）
train_cut_selection.bat
```

### 学習状況の確認

ブラウザで `checkpoints_cut_selection/view_training.html` を開くと、2秒ごとに自動更新されるグラフで学習の様子をリアルタイム確認できます。

**可視化される情報**:
- 損失関数（Train/Val Loss）
- 損失の内訳（CE Loss vs TV Loss）
- 分類性能（Accuracy & F1 Score）
- Precision, Recall, Specificity
- 最適閾値の推移
- 予測の採用/不採用割合

学習済みモデルは `checkpoints_cut_selection/` に保存されます。

---

## 📁 ファイル配置

### 推論前に必要なもの
```
checkpoints_cut_selection/
├── best_model.pth
├── inference_params.yaml
└── (その他のチェックポイント)

preprocessed_data/
├── audio_scaler_cut_selection.pkl
└── visual_scaler_cut_selection.pkl
```

### データ準備前に必要なもの
```
data/
├── processed/
│   ├── source_features/
│   │   ├── video1_features.csv
│   │   └── video2_features.csv
│   └── active_labels/
│       ├── video1_active.csv
│       └── video2_active.csv
```

---

## 🔧 トラブルシューティング

### エラー: ModuleNotFoundError

**原因**: Pythonパスが設定されていない

**解決策**:
```bash
set PYTHONPATH=%PYTHONPATH%;%CD%
```

または、バッチファイル（`train_cut_selection.bat`など）を使用してください。

### エラー: FileNotFoundError: best_model.pth

**原因**: 学習済みモデルがない

**解決策**:
1. データ準備を実行: `python scripts/create_cut_selection_data.py`
2. 学習を実行: `train_cut_selection.bat`

### 学習が進まない

**原因**: データが不足している、または設定が不適切

**解決策**:
- データ数を確認: 最低でも50本以上の動画が推奨
- 設定を確認: `configs/config_cut_selection.yaml`
- ログを確認: 学習中のメッセージをチェック

---

## 📊 実行時間の目安

- **データ準備**: 約5-10分（動画の数と長さによる）
- **学習**: 約1-2時間（50エポック、GPU使用時）
- **推論**: 約5-10分/動画（特徴量抽出含む）

---

## 💡 ヒント

### カット数を調整したい

学習時に最適閾値が自動計算されますが、手動で調整することも可能です：

`checkpoints_cut_selection/inference_params.yaml` を編集：

```yaml
confidence_threshold: -0.200  # デフォルト値
target_duration: 90.0         # 目標合計時間（秒）
max_duration: 150.0           # 最大合計時間（秒）
```

閾値を下げる → カット数が増える  
閾値を上げる → カット数が減る

### GPUを使用したい

学習時に自動的にGPUが使用されます（CUDA対応GPUがある場合）。

---

## 📖 詳細なドキュメント

- [README](../README.md) - プロジェクト概要
- [PROJECT_SPECIFICATION](PROJECT_SPECIFICATION.md) - 詳細仕様

---

## 🎉 成功例

学習が成功すると、以下のようなメッセージが表示されます：

```
✅ Training complete!
Best Val F1: 0.5630
📊 Training visualization saved: checkpoints_cut_selection/training_final.png
```

推論が成功すると：

```
================================================================================
✅ Inference complete!
Output XML: outputs/inference_results/result.xml
================================================================================
```

Premiere Proで開くと、自動的にカット編集されたタイムラインが表示されます！
