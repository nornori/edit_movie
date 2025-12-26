# プロジェクト構造

## 📁 フォルダ構成（2025-12-26更新）

```
xmlai/
├── .git/                                      # Git管理
├── .kiro/                                     # Kiro AI設定
├── .pytest_cache/                             # Pytestキャッシュ
├── .venv/                                     # Python仮想環境
├── .vscode/                                   # VSCode設定
│
├── archive/                                   # 🗄️ アーカイブ（古いファイル）
│   ├── checkpoints_cut_selection_kfold/       # 旧K-Foldモデル
│   ├── checkpoints_cut_selection_kfold_enhanced_reset/
│   ├── checkpoints_cut_selection_kfold_enhanced_retrain/
│   ├── checkpoints_cut_selection_kfold_enhanced_v2/
│   ├── experiment_log_advanced.csv
│   └── (その他の古いファイル)
│
├── backups/                                   # 💾 バックアップ
│   └── 2025-12-26_01-20-31_ensemble_60_80_percent/
│
├── batch/                                     # 🔧 バッチファイル
│   ├── retrain_model.bat
│   ├── train_cut_selection_kfold_enhanced.bat
│   ├── train_duration_constraint.bat
│   ├── train_fullvideo.bat
│   ├── train_reset.bat
│   └── (その他のバッチファイル)
│
├── checkpoints_cut_selection_fullvideo/       # 🎯 Full Videoモデル（最新）
│   ├── best_model.pth                         # Epoch 9, F1=52.90%
│   ├── training_history.csv
│   ├── training_progress.png
│   ├── training_final.png
│   └── view_training.html
│
├── checkpoints_cut_selection_kfold_enhanced/  # 🎯 K-Fold拡張モデル（最新）
│   ├── fold_1_best_model.pth                  # F1=49.42%（最良）
│   ├── fold_2_best_model.pth                  # F1=41.22%
│   ├── fold_3_best_model.pth                  # F1=43.10%
│   ├── fold_4_best_model.pth                  # F1=45.57%
│   ├── fold_5_best_model.pth                  # F1=32.20%
│   ├── kfold_summary.csv
│   ├── kfold_comparison.png
│   ├── kfold_realtime_progress.png
│   ├── inference_params.yaml
│   └── view_training.html
│
├── configs/                                   # ⚙️ 設定ファイル
│   ├── config_cut_selection_fullvideo.yaml    # Full Video設定
│   ├── config_cut_selection_kfold_enhanced.yaml # K-Fold設定
│   └── (その他の設定ファイル)
│
├── data/                                      # 📊 データ
│   ├── processed/                             # 処理済みデータ
│   │   ├── source_features/                   # 特徴量CSV
│   │   └── active_labels/                     # アクティブラベル
│   ├── raw/                                   # 生データ
│   │   └── editxml/                           # Premiere Pro XML
│   ├── reports/                               # レポート
│   └── temp/                                  # 一時ファイル
│
├── docs/                                      # 📚 ドキュメント
│   ├── guides/                                # ガイド
│   │   ├── FCPXML_EXTRACTION_GUIDE.md
│   │   ├── PROJECT_WORKFLOW_GUIDE.md
│   │   ├── REQUIRED_FILES_BY_PHASE.md
│   │   ├── SPEAKER_IDENTIFICATION.md
│   │   └── VIDEO_FEATURE_EXTRACTION_GUIDE.md
│   ├── summaries/                             # サマリー
│   │   ├── AUDIO_CUT_AND_TELOP_GRAPHICS_SUMMARY.md
│   │   ├── FINAL_PROGRESS.md
│   │   ├── INFERENCE_PIPELINE_SUMMARY.md
│   │   ├── PREMIERE_XML_EXTRACTION_SUMMARY.md
│   │   ├── PREMIERE_XML_PARSER_UPDATE.md
│   │   ├── PROGRESS.md
│   │   └── PROJECT_COMPLETE.md
│   ├── COMPLETE_METRICS_SUMMARY.md            # 完全メトリクスサマリー
│   ├── CUT_SELECTION_MODEL.md
│   ├── CUT_SELECTION_REORGANIZATION.md        # カット選択モジュール整理レポート
│   ├── FINAL_RESULTS.md                       # 最終結果レポート
│   ├── INFERENCE_TEST_RESULTS.md              # 推論テスト結果
│   ├── K_FOLD_CROSS_VALIDATION.md
│   ├── K_FOLD_FINAL_RESULTS.md
│   ├── PROJECT_SPECIFICATION.md
│   ├── QUICK_START.md                         # クイックスタート
│   ├── TRAINING_GRAPHS_UPDATE.md              # グラフ更新
│   └── TRAINING_REPORT.md
│
├── models/                                    # 🧠 モデル定義（空）
│
├── outputs/                                   # 📤 出力ファイル
│   └── bandicam 2025-05-11 19-25-14-768_output.xml
│
├── preprocessed_data/                         # 🔄 前処理済みデータ
│   ├── combined_sequences_cut_selection_enhanced.npz  # K-Fold用データ
│   ├── train_fullvideo_cut_selection_enhanced.npz     # Full Video訓練データ
│   ├── val_fullvideo_cut_selection_enhanced.npz       # Full Video検証データ
│   ├── audio_scaler_cut_selection_enhanced.pkl
│   ├── visual_scaler_cut_selection_enhanced.pkl
│   ├── temporal_scaler_cut_selection_enhanced.pkl
│   ├── audio_scaler_cut_selection_enhanced_fullvideo.pkl
│   ├── visual_scaler_cut_selection_enhanced_fullvideo.pkl
│   └── temporal_scaler_cut_selection_enhanced_fullvideo.pkl
│
├── scripts/                                   # 🔨 スクリプト
│   ├── add_temporal_features.py               # 時系列特徴量追加
│   ├── combine_sequences_enhanced.py          # シーケンス結合
│   ├── create_combined_data_for_kfold.py      # K-Fold用データ作成
│   ├── create_cut_selection_data_enhanced.py  # データ作成
│   ├── create_cut_selection_data_enhanced_fullvideo.py  # Full Video用
│   ├── generate_xml_from_inference.py         # XML生成
│   └── (その他のスクリプト)
│
├── src/                                       # 💻 ソースコード
│   ├── cut_selection/                         # カット選択モジュール（整理済み）
│   │   ├── __init__.py                        # モジュールエクスポート
│   │   ├── models/                            # モデル定義
│   │   │   ├── __init__.py
│   │   │   └── cut_model_enhanced.py          # 拡張モデル（現行）
│   │   ├── datasets/                          # データセットクラス
│   │   │   ├── __init__.py
│   │   │   ├── cut_dataset_enhanced.py        # K-Fold用データセット
│   │   │   └── cut_dataset_enhanced_fullvideo.py  # Full Video用データセット
│   │   ├── training/                          # 訓練スクリプト
│   │   │   ├── __init__.py
│   │   │   ├── train_cut_selection_kfold_enhanced.py  # K-Fold訓練
│   │   │   ├── train_cut_selection_fullvideo.py       # Full Video訓練
│   │   │   └── train_cut_selection_fullvideo_v2.py    # Full Video訓練V2（現行）
│   │   ├── inference/                         # 推論モジュール
│   │   │   ├── __init__.py
│   │   │   ├── inference_cut_selection.py     # 基本推論
│   │   │   └── inference_enhanced.py          # 拡張推論
│   │   ├── evaluation/                        # 評価スクリプト
│   │   │   ├── __init__.py
│   │   │   ├── ensemble_predictor.py          # アンサンブル予測
│   │   │   ├── evaluate_ensemble_proper.py    # アンサンブル評価
│   │   │   └── evaluate_ensemble_no_leakage.py  # リーク防止評価
│   │   ├── utils/                             # ユーティリティ
│   │   │   ├── __init__.py
│   │   │   ├── losses.py                      # 損失関数
│   │   │   ├── positional_encoding.py         # 位置エンコーディング
│   │   │   ├── fusion.py                      # モダリティ融合
│   │   │   ├── temporal_loss.py               # 時系列損失
│   │   │   └── time_series_augmentation.py    # 時系列拡張
│   │   └── archive/                           # アーカイブ（旧バージョン）
│   │       ├── __init__.py
│   │       ├── cut_model.py                   # 旧モデル
│   │       ├── cut_model_enhanced_v2.py       # 旧V2モデル
│   │       ├── cut_dataset.py                 # 旧データセット
│   │       ├── cut_dataset_enhanced_v2.py     # 旧V2データセット
│   │       └── (その他の旧ファイル)
│   ├── data_preparation/                      # データ準備
│   │   ├── extract_active_labels.py
│   │   ├── extract_video_features_parallel.py
│   │   └── (その他)
│   ├── inference/                             # 推論
│   │   ├── direct_xml_generator.py
│   │   ├── inference_pipeline.py
│   │   └── (その他)
│   ├── model/                                 # モデル（旧）
│   ├── training/                              # 訓練（旧）
│   └── utils/                                 # ユーティリティ
│
├── temp_features/                             # 📁 一時特徴量
│   ├── bandicam 2025-04-29 18-51-06-891_features_enhanced.csv
│   ├── bandicam 2025-05-11 19-25-14-768_features_enhanced.csv
│   └── (その他の特徴量CSV)
│
├── tests/                                     # 🧪 テストコード
│   ├── test_inference_fullvideo.py            # Full Video推論テスト
│   ├── test_inference_simple.py               # シンプル推論テスト
│   ├── check_model.py                         # モデルチェック
│   └── (その他のテスト)
│
├── .gitignore                                 # Git除外設定
├── CHANGELOG.md                               # 変更履歴
├── CLEANUP_SUMMARY.md                         # 整理サマリー
├── FEATURE_ENHANCEMENT_README.md              # 機能拡張README
├── LICENSE                                    # ライセンス
├── PROJECT_STRUCTURE.md                       # このファイル
├── README.md                                  # メインREADME
└── requirements.txt                           # 依存パッケージ
```

---

## 📊 フォルダの役割と重要度

### ⭐⭐⭐ 最重要フォルダ

| フォルダ | 役割 | サイズ目安 |
|---------|------|-----------|
| `src/` | ソースコード | 小 |
| `configs/` | 設定ファイル | 小 |
| `docs/` | ドキュメント | 小 |
| `checkpoints_cut_selection_fullvideo/` | Full Videoモデル | 中 |
| `checkpoints_cut_selection_kfold_enhanced/` | K-Fold拡張モデル | 中 |
| `data/` | 生データ | 大 |
| `preprocessed_data/` | 前処理済みデータ | 大 |

### ⭐⭐ 重要フォルダ

| フォルダ | 役割 | サイズ目安 |
|---------|------|-----------|
| `scripts/` | 実行スクリプト | 小 |
| `tests/` | テストコード | 小 |
| `batch/` | バッチファイル | 小 |
| `temp_features/` | 一時特徴量 | 大 |
| `outputs/` | 出力ファイル | 中 |

### ⭐ 補助フォルダ

| フォルダ | 役割 | サイズ目安 |
|---------|------|-----------|
| `backups/` | バックアップ | 大 |
| `archive/` | アーカイブ | 大 |
| `models/` | モデル定義（空） | 小 |

---

## 🎯 現在のアクティブなモデル

### Full Video Model（推奨）

**パス**: `checkpoints_cut_selection_fullvideo/best_model.pth`

**性能**:
- Epoch: 9
- F1: 52.90%
- Recall: 80.65%
- Precision: 38.94%

**用途**:
- per-video制約（90-200秒）推論
- 目標180秒に最適化
- 推論テスト結果: 181.9秒（完璧）

### K-Fold Enhanced Model

**パス**: `checkpoints_cut_selection_kfold_enhanced/fold_1_best_model.pth`

**性能**:
- Epoch: 4
- F1: 49.42%
- Recall: 74.65%
- Precision: 36.94%

**用途**:
- K-Fold CV評価
- 汎化性能測定

---

## 📝 重要なファイル

### ドキュメント

| ファイル | 説明 |
|---------|------|
| `README.md` | プロジェクト概要 |
| `docs/QUICK_START.md` | クイックスタート |
| `docs/FINAL_RESULTS.md` | 最終結果レポート |
| `docs/INFERENCE_TEST_RESULTS.md` | 推論テスト結果 |
| `docs/COMPLETE_METRICS_SUMMARY.md` | 完全メトリクス |
| `CHANGELOG.md` | 変更履歴 |
| `CLEANUP_SUMMARY.md` | 整理サマリー |

### 設定ファイル

| ファイル | 説明 |
|---------|------|
| `configs/config_cut_selection_fullvideo.yaml` | Full Video設定 |
| `configs/config_cut_selection_kfold_enhanced.yaml` | K-Fold設定 |
| `requirements.txt` | 依存パッケージ |

### スクリプト

| ファイル | 説明 |
|---------|------|
| `scripts/generate_xml_from_inference.py` | XML生成 |
| `scripts/add_temporal_features.py` | 時系列特徴量追加 |
| `scripts/combine_sequences_enhanced.py` | シーケンス結合 |

### テスト

| ファイル | 説明 |
|---------|------|
| `tests/test_inference_fullvideo.py` | Full Video推論テスト |
| `tests/test_inference_simple.py` | シンプル推論テスト |

### バッチファイル

| ファイル | 説明 |
|---------|------|
| `batch/train_fullvideo.bat` | Full Video学習 |
| `batch/train_cut_selection_kfold_enhanced.bat` | K-Fold学習 |

---

## 🔍 .gitignoreの設定

以下のフォルダ/ファイルはGit管理から除外されています：

```gitignore
# データ
data/
preprocessed_data/
temp_features/

# モデル
checkpoints*/
*.pth
*.pkl

# 出力
outputs/
archive/
backups/

# Python
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd

# Jupyter
.ipynb_checkpoints/

# その他
.pytest_cache/
.vscode/
.DS_Store
```

---

## 📈 ディスク使用量の目安

| フォルダ | サイズ目安 | 説明 |
|---------|-----------|------|
| `data/` | 5-10 GB | 動画ファイル、XML |
| `temp_features/` | 2-5 GB | 特徴量CSV |
| `preprocessed_data/` | 500 MB - 1 GB | NPZファイル |
| `checkpoints_*/` | 200-500 MB | モデルファイル |
| `archive/` | 1-5 GB | 古いファイル |
| `backups/` | 500 MB - 2 GB | バックアップ |
| **合計** | **10-25 GB** | プロジェクト全体 |

---

## 🚀 よく使うコマンド

### データ準備

```bash
# 特徴量抽出
python -m src.data_preparation.extract_video_features_parallel --video_dir videos --output_dir data/processed/source_features --n_jobs 4

# ラベル抽出
python -m src.data_preparation.extract_active_labels --xml_dir data/raw/editxml --feature_dir data/processed/source_features --output_dir data/processed/active_labels

# 時系列特徴量追加
python scripts/add_temporal_features.py

# K-Fold用データ作成
python scripts/combine_sequences_enhanced.py
```

### 学習

```bash
# Full Video学習
batch/train_fullvideo.bat

# K-Fold学習
batch/train_cut_selection_kfold_enhanced.bat
```

### 推論

```bash
# Full Video推論テスト
python tests/test_inference_fullvideo.py "video_name"

# XML生成
python scripts/generate_xml_from_inference.py "path/to/video.mp4"
```

---

## 📞 トラブルシューティング

### ファイルが見つからない

1. `archive/` フォルダを確認
2. `backups/` フォルダを確認
3. `.gitignore` で除外されていないか確認

### スクリプトが動かない

1. パスを確認（相対パスが変わった可能性）
2. `tests/` または `scripts/` フォルダを確認
3. Python環境を確認（`.venv` がアクティブか）

### モデルが見つからない

1. `checkpoints_cut_selection_fullvideo/` を確認
2. `checkpoints_cut_selection_kfold_enhanced/` を確認
3. `archive/` フォルダを確認

---

**最終更新**: 2025-12-26  
**バージョン**: 2.0.0  
**ステータス**: ✅ 整理完了

