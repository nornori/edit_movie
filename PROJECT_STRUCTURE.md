# プロジェクト構造

## 📁 整理後のディレクトリ構造

```
xmlai/
├── 📂 src/                          # ソースコード
│   ├── cut_selection/               # カット選択モデル
│   │   ├── cut_model_enhanced.py    # V1モデル（現在の最良）
│   │   ├── cut_model_enhanced_v2.py # V2モデル（開発中）
│   │   ├── cut_dataset_enhanced.py  # V1データセット
│   │   ├── cut_dataset_enhanced_v2.py # V2データセット（データ拡張付き）
│   │   ├── ensemble_predictor.py    # アンサンブル予測器
│   │   ├── evaluate_ensemble.py     # アンサンブル評価
│   │   ├── time_series_augmentation.py # データ拡張
│   │   ├── train_cut_selection_kfold_enhanced.py # V1トレーニング
│   │   └── train_cut_selection_kfold_enhanced_v2.py # V2トレーニング
│   ├── model/                       # 共通モデルコンポーネント
│   └── ...
│
├── 📂 scripts/                      # データ準備スクリプト
│   ├── add_temporal_features.py     # 時系列特徴量追加
│   ├── create_cut_selection_data_enhanced.py # 拡張データ作成
│   └── combine_sequences_enhanced.py # K-Fold用データ結合
│
├── 📂 configs/                      # 設定ファイル
│   ├── config_cut_selection_kfold_enhanced.yaml # V1設定（現在の最良）
│   └── config_cut_selection_kfold_enhanced_v2.yaml # V2設定
│
├── 📂 batch/                        # バッチファイル（整理済み）
│   ├── train_cut_selection_enhanced.bat # V1トレーニング
│   ├── train_cut_selection_enhanced_v2.bat # V2トレーニング
│   ├── evaluate_ensemble.bat        # アンサンブル評価
│   └── run_inference.bat            # 推論実行
│
├── 📂 checkpoints_cut_selection_kfold_enhanced/ # V1モデルチェックポイント
│   ├── fold_1_best_model.pth        # Fold 1最良モデル（F1: 49.52%）
│   ├── fold_2_best_model.pth        # Fold 2最良モデル（F1: 41.22%）
│   ├── fold_3_best_model.pth        # Fold 3最良モデル（F1: 40.69%）
│   ├── fold_4_best_model.pth        # Fold 4最良モデル（F1: 40.43%）
│   ├── fold_5_best_model.pth        # Fold 5最良モデル（F1: 34.27%）
│   ├── kfold_summary.csv            # K-Fold統計
│   ├── kfold_comparison.png         # 比較グラフ
│   ├── ensemble_comparison.csv      # アンサンブル比較
│   ├── ensemble_comparison.png      # アンサンブルグラフ
│   └── view_training.html           # リアルタイムビューアー
│
├── 📂 preprocessed_data/            # 前処理済みデータ
│   └── combined_sequences_cut_selection_enhanced.npz # 拡張特徴量データ
│       - 289 sequences
│       - 67 unique videos
│       - 784 features (235 audio + 543 visual + 6 temporal)
│
├── 📂 docs/                         # ドキュメント
│   ├── ENSEMBLE_RESULTS.md          # アンサンブル結果詳細
│   ├── K_FOLD_CROSS_VALIDATION.md   # K-Fold詳細
│   └── ...
│
├── 📂 archive/                      # 古いファイル（整理済み）
│   ├── old_experiments/             # 古い実験スクリプト
│   ├── old_logs/                    # 古い実験ログ
│   ├── old_batch_files/             # 古いバッチファイル
│   └── old_checkpoints/             # 古いチェックポイント
│
├── 📄 README.md                     # プロジェクト概要
├── 📄 CHANGELOG.md                  # 変更履歴
├── 📄 FEATURE_ENHANCEMENT_README.md # 特徴量拡張ガイド
├── 📄 PROJECT_STRUCTURE.md          # このファイル
├── 📄 requirements.txt              # 依存パッケージ
└── 📄 .gitignore                    # Git除外設定
```

---

## 🎯 現在の最良モデル

### V1モデル（アンサンブル）

**チェックポイント**: `checkpoints_cut_selection_kfold_enhanced/`

**性能**:
- 個別モデル平均: 41.23% F1
- **アンサンブル: 60.80% F1** ✨
- 改善: +19.57ポイント (+47.47%)

**設定**: `configs/config_cut_selection_kfold_enhanced.yaml`

**特徴**:
- 784次元入力（235 audio + 543 visual + 6 temporal）
- 6層Transformerエンコーダー
- 8個のAttentionヘッド
- Focal Loss + TV Loss + Adoption Penalty
- 5-Fold Cross Validation

---

## 🚀 使用方法

### 1. V1モデルのトレーニング（既に完了）

```bash
batch/train_cut_selection_enhanced.bat
```

### 2. アンサンブル評価

```bash
batch/evaluate_ensemble.bat
```

### 3. V2モデルのトレーニング（開発中）

```bash
batch/train_cut_selection_enhanced_v2.bat
```

---

## 📊 データフロー

```
1. 元データ
   ↓
2. 時系列特徴量追加 (scripts/add_temporal_features.py)
   ↓
3. トレーニングデータ作成 (scripts/create_cut_selection_data_enhanced.py)
   ↓
4. K-Fold用データ結合 (scripts/combine_sequences_enhanced.py)
   ↓
5. K-Foldトレーニング (src/cut_selection/train_cut_selection_kfold_enhanced.py)
   ↓
6. アンサンブル評価 (src/cut_selection/evaluate_ensemble.py)
```

---

## 🗑️ 整理されたファイル

### archive/old_experiments/
- `advanced_auto_experiment.py`
- `auto_experiment.py`
- `simple_auto_experiment.py`

### archive/old_logs/
- `experiment_log.csv`
- `experiment_log_v2.csv`
- `experiment_log_advanced.csv`

### archive/old_batch_files/
- `train_cut_selection.bat`
- `train_cut_selection_kfold.bat`
- `enhance_features.bat`

### archive/old_checkpoints/
- `checkpoints/` (古いマルチモーダルモデル)
- `checkpoints_cut_selection/` (古いカット選択モデル)
- `checkpoints_cut_selection_kfold/` (古いK-Foldモデル)

---

## 📝 重要なファイル

### 設定ファイル
- `configs/config_cut_selection_kfold_enhanced.yaml` - V1設定（現在の最良）
- `configs/config_cut_selection_kfold_enhanced_v2.yaml` - V2設定（開発中）

### ドキュメント
- `docs/ENSEMBLE_RESULTS.md` - アンサンブル結果の詳細分析
- `FEATURE_ENHANCEMENT_README.md` - 特徴量拡張の詳細ガイド
- `CHANGELOG.md` - 全ての変更履歴

### バッチファイル
- `batch/train_cut_selection_enhanced.bat` - V1トレーニング
- `batch/evaluate_ensemble.bat` - アンサンブル評価
- `batch/train_cut_selection_enhanced_v2.bat` - V2トレーニング

---

## 🎯 次のステップ

1. **V2モデルのトレーニング**
   - データ拡張（ノイズ、時間シフト、スケーリング）
   - より深いネットワーク（8層、16ヘッド）
   - 期待値: 65%+ F1

2. **V2モデルでのアンサンブル評価**
   - V1との比較
   - 最終的な性能評価

3. **本番環境への展開**
   - 推論パイプラインの最適化
   - APIの実装
