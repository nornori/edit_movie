# 🚀 クイックスタートガイド - カット選択モデル

## 📋 前提条件

- Python 3.8以上がインストールされている
- 必要なライブラリがインストールされている（`pip install -r requirements.txt`）
- 学習済みモデルがある（または学習を実行する）
- **GPU推奨**: NVIDIA GPU（CUDA対応）、VRAM 8GB以上

**注意**: 本プロジェクトは現在**カット選択（Cut Selection）に特化**しています。グラフィック配置やテロップ生成は精度が低いため、今後の課題となっています。

---

## 🎯 新しい動画を自動編集する（推論）

### 方法1: Full Video Model（推奨）

**per-video制約（90-200秒）を満たす最適閾値を自動探索**

```bash
# XML生成（推論 + クリップ抽出 + XML生成）
python generate_xml_from_inference.py "D:\path\to\video.mp4"
```

**出力**: `outputs/video_name_output.xml`

**性能**:
- モデル: `checkpoints_cut_selection_fullvideo/best_model.pth` (Epoch 9, F1=0.5290)
- 制約: 90-200秒（目標180秒）
- 最適化: F1最大化
- テスト結果: 181.9秒（目標180秒に完璧）、10クリップ抽出

**詳細**: [推論テスト結果レポート](INFERENCE_TEST_RESULTS.md)

### 方法2: K-Fold Model（バッチファイル）

```bash
run_inference.bat "path\to\your_video.mp4"
```

これだけで完了！Premiere Pro用のXMLが `outputs/inference_results/result.xml` に生成されます。

### 方法3: 手動で実行

```bash
# 推論実行
python -m src.inference.inference_pipeline "path\to\your_video.mp4" ^
    --output outputs/inference_results/result.xml
```

### 3. Premiere Proで開く

生成された `result.xml` をPremiere Proで開いてください。自動的にカット編集されたタイムラインが表示されます。

**推論時間**: 約5-10分/動画（10分の動画、特徴量抽出含む）

---

## 📚 カット選択用データを準備する

### ステップ1: 編集済み動画とXMLを用意

```
editxml/
├── video1.mp4
├── video1.xml  (Premiere ProからエクスポートしたXML)
├── video2.mp4
└── video2.xml
```

**最低30本以上**の動画を用意することを推奨

### ステップ2: 動画から特徴量を抽出

```bash
python -m src.data_preparation.extract_video_features_parallel \
    --video_dir videos \
    --output_dir data/processed/source_features \
    --n_jobs 4
```

**処理時間**: 5-10分/動画（10分の動画、GPU使用時）

**抽出される特徴量**:
- **音声**: 235次元（RMS, VAD, 話者埋め込み192次元, MFCC, ピッチ等）
- **映像**: 543次元（シーン変化, 動き, 顔検出, CLIP 512次元等）
- **時系列**: 6次元（カットタイミング, 位置, 密度等）
- **合計**: 784次元

### ステップ3: XMLからアクティブラベルを抽出

```bash
python -m src.data_preparation.extract_active_labels \
    --xml_dir editxml \
    --feature_dir data/processed/source_features \
    --output_dir data/processed/active_labels
```

### ステップ4: 時系列特徴量を追加

```bash
python scripts/add_temporal_features.py
```

**追加される特徴量（83個）**:
- 移動統計量: MA5, MA10, MA30, MA60, MA120, STD5, STD30, STD120
- 変化率: DIFF1, DIFF2, DIFF30
- カットタイミング: time_since_prev, time_to_next, cut_duration等
- CLIP類似度: clip_sim_prev, clip_sim_next, clip_sim_mean5
- 音声・映像変化: audio_change_score, visual_motion_change等

### ステップ5: K-Fold用データを作成

```bash
python scripts/create_combined_data_for_kfold.py
```

**処理内容**:
- 特徴量とアクティブラベルをマージ
- シーケンス分割（長さ1000フレーム、オーバーラップ500）
- 動画単位でグループ化（GroupKFold用、データリーク防止）
- 特徴量の正規化（StandardScaler）

**出力**:
- `preprocessed_data/combined_sequences_cut_selection_enhanced.npz` (289シーケンス、67動画)
- `preprocessed_data/audio_scaler_cut_selection_enhanced.pkl`
- `preprocessed_data/visual_scaler_cut_selection_enhanced.pkl`
- `preprocessed_data/temporal_scaler_cut_selection_enhanced.pkl`

---

## 🎓 カット選択モデルを学習する

### K-Fold Cross Validationで学習（推奨）

```bash
# バッチファイルで実行（推奨）
train_cut_selection_kfold_enhanced.bat
```

**学習設定**:
- K-Fold: 5分割（GroupKFold）
- エポック数: 50/Fold
- バッチサイズ: 16
- 学習率: 0.0001
- Early Stopping: 15エポック
- Mixed Precision: 有効
- Random Seed: 42（再現性確保）

**学習時間**: 約2-3時間（250エポック = 50 × 5 Folds、GPU使用時）

**Early Stopping効果**:
- 平均収束: 7.4エポック
- 実際の学習: 37エポック（85%削減）

### 学習状況の確認

ブラウザで `checkpoints_cut_selection_kfold_enhanced/view_training.html` を開くと、2秒ごとに自動更新されるグラフで学習の様子をリアルタイム確認できます。

**全体進捗（kfold_realtime_progress.png）**:
1. F1スコアの推移（各Fold）
2. Validation Lossの推移（各Fold）
3. 現在のF1スコア（棒グラフ）
4. 進捗状況（テキスト）
5. 最良F1の推移（各Fold）
6. 採用/不採用の正確性（Recall & Specificity）

**各Fold詳細（fold_X/training_progress.png）**:
1. 損失関数（Train/Val Loss）
2. 損失の内訳（CE Loss vs TV Loss）
3. 分類性能（Accuracy & F1 Score）
4. Precision, Recall, Specificity
5. 最適閾値の推移
6. 予測の採用/不採用割合

学習済みモデルは `checkpoints_cut_selection_kfold_enhanced/` に保存されます。

---

## 📁 ファイル配置

### 推論前に必要なもの
```
checkpoints_cut_selection_kfold_enhanced/
├── fold_1_best_model.pth          # 最良モデル (F1: 49.42%)
├── fold_2_best_model.pth
├── fold_3_best_model.pth
├── fold_4_best_model.pth
├── fold_5_best_model.pth
├── inference_params.yaml          # 推論用パラメータ
└── (その他のチェックポイント)

preprocessed_data/
├── audio_scaler_cut_selection_enhanced.pkl
├── visual_scaler_cut_selection_enhanced.pkl
└── temporal_scaler_cut_selection_enhanced.pkl
```

### データ準備前に必要なもの
```
editxml/
├── video1.mp4
├── video1.xml  (Premiere Proで編集したXML)
├── video2.mp4
└── video2.xml

data/processed/
├── source_features/
│   ├── video1_features_enhanced.csv
│   └── video2_features_enhanced.csv
└── active_labels/
    ├── video1_active.csv
    └── video2_active.csv
```

---

## 🔧 トラブルシューティング

### エラー: ModuleNotFoundError

**原因**: Pythonパスが設定されていない

**解決策**:
```bash
set PYTHONPATH=%PYTHONPATH%;%CD%
```

または、バッチファイル（`train_cut_selection_kfold_enhanced.bat`など）を使用してください。

### エラー: FileNotFoundError: fold_1_best_model.pth

**原因**: 学習済みモデルがない

**解決策**:
1. データ準備を実行: ステップ1-5を完了
2. 学習を実行: `train_cut_selection_kfold_enhanced.bat`

### エラー: CUDA out of memory

**原因**: VRAM不足

**解決策**:
```yaml
# configs/config_cut_selection_kfold_enhanced.yaml を編集
batch_size: 8  # 16から削減
```

または並列処理数を減らす:
```bash
python extract_video_features_parallel.py --n_jobs 2
```

### エラー: MediaPipe initialization failed

**原因**: プロジェクトパスに日本語などの非ASCII文字が含まれている

**解決策**:
```
プロジェクトをASCII文字のみのパスに移動
例: D:\切り抜き\xmlai → C:\projects\xmlai
```

### 学習が進まない

**原因**: データが不足している、または設定が不適切

**解決策**:
- データ数を確認: 最低でも30本以上の動画が推奨
- 設定を確認: `configs/config_cut_selection_kfold_enhanced.yaml`
- ログを確認: 学習中のメッセージをチェック
- GPU使用を確認: `nvidia-smi`

---

## 📊 実行時間の目安

- **特徴量抽出**: 5-10分/動画（10分の動画、GPU使用時）
  - 30本の動画: 約2.5-5時間（4並列処理）
- **学習**: 約2-3時間（250エポック、5 Folds、GPU使用時）
  - Early Stoppingで実際は37エポック（約30-45分）
- **推論**: 約5-10分/動画（特徴量抽出含む）

---

## 💡 ヒント

### カット数を調整したい

学習時に最適閾値が自動計算されますが、手動で調整することも可能です：

`checkpoints_cut_selection_kfold_enhanced/inference_params.yaml` を編集：

```yaml
confidence_threshold: -0.558  # Fold 1の最適閾値
min_clip_duration: 3.0        # 最小クリップ継続時間（秒）
max_gap_duration: 2.0         # ギャップ結合の最大長（秒）
target_duration: 90.0         # 目標合計時間（秒）
max_duration: 150.0           # 最大合計時間（秒）
```

閾値を下げる（例: -0.7） → カット数が増える  
閾値を上げる（例: -0.4） → カット数が減る

### GPUを使用したい

学習時に自動的にGPUが使用されます（CUDA対応GPUがある場合）。

確認方法:
```bash
nvidia-smi
```

### データリークを防ぐには

- **GroupKFold**を使用（同じ動画は同じFoldに）
- 各Foldで完全に未見のデータで評価
- アンサンブル評価は避ける（データが少ない場合）

### 学習データの品質を上げるには

1. **多様性を確保**
   - 異なる話者、異なるトピック、異なる長さの動画を含める
   - 「盛り上がるシーン」「静かなシーン」の両方を含める

2. **編集の一貫性**
   - 「採用する基準」を明確にする（例: 笑い声がある、重要な発言がある）
   - 一貫性のない編集はモデルを混乱させる

3. **データ量**
   - 最低30本、理想は50-100本以上
   - 少ないデータでは過学習のリスク

---

## 📖 詳細なドキュメント

- [README](../README.md) - プロジェクト概要
- [PROJECT_WORKFLOW_GUIDE](guides/PROJECT_WORKFLOW_GUIDE.md) - 全体ワークフロー
- [VIDEO_FEATURE_EXTRACTION_GUIDE](guides/VIDEO_FEATURE_EXTRACTION_GUIDE.md) - 特徴量抽出詳細
- [FINAL_RESULTS](FINAL_RESULTS.md) - 最終結果レポート
- [TRAINING_REPORT](TRAINING_REPORT.md) - 詳細学習レポート

---

## 🎉 成功例

学習が成功すると、以下のようなメッセージが表示されます：

```
================================================================================
K-FOLD CROSS VALIDATION SUMMARY
================================================================================

Mean F1 Score: 0.4230 ± 0.0575
Mean Accuracy: 0.5024 ± 0.1492
Mean Precision: 0.2983 ± 0.0580
Mean Recall: 0.7610 ± 0.0519
Mean Optimal Threshold: -0.533 ± 0.036
================================================================================

✅ Training complete!
Best Model: Fold 1 (F1: 49.42%)
📊 Training visualization saved: checkpoints_cut_selection_kfold_enhanced/kfold_comparison.png
```

推論が成功すると：

```
================================================================================
✅ Inference complete!
Output XML: outputs/inference_results/result.xml
Clips: 10 clips, Total duration: 95.3 seconds
================================================================================
```

Premiere Proで開くと、自動的にカット編集されたタイムラインが表示されます！

---

## 📊 期待される性能

### K-Fold Cross Validation結果（学習性能）

| 指標 | 平均値 | 標準偏差 | 最良（Fold 1） |
|------|--------|----------|----------------|
| **F1 Score** | **42.30%** | ±5.75% | **49.42%** |
| **Accuracy** | 50.24% | ±14.92% | 73.63% |
| **Precision** | 29.83% | ±5.80% | 36.94% |
| **Recall** | **76.10%** | ±5.19% | 74.65% |

**推奨モデル**: Fold 1（F1: 49.42%、最も安定した性能）

### Full Video Model結果（推論性能）

**モデル**: `checkpoints_cut_selection_fullvideo/best_model.pth` (Epoch 9)

**学習時性能**:
- F1: 0.5290
- Recall: 0.8065
- Avg Duration: 101.3秒

**推論テスト結果**（bandicam 2025-05-11 19-25-14-768.mp4）:
- 動画長: 1000.1秒（約16.7分）
- **最適閾値**: 0.8952（F1最大化、90-200秒制約内）
- **予測時間**: 181.9秒（目標180秒に完璧）
- **採用率**: 18.2%（1,819 / 10,001フレーム）
- **抽出クリップ数**: 10個（合計138.3秒）
- **XML生成**: 成功

**制約満足度**:
- ✅ 90-200秒制約を満たす
- ✅ 目標180秒にほぼ完璧に一致（+1.9秒）
- ✅ per-video最適化が正しく動作

**詳細**: [推論テスト結果レポート](INFERENCE_TEST_RESULTS.md)

### 目標達成状況

| 項目 | 目標 | 達成 | 状況 |
|------|------|------|------|
| F1スコア | 55% | 42.30% | ❌ 未達成 (-12.70pt) |
| Recall | 71% | 76.10% | ✅ 達成 (+5.10pt) |

**強み**:
- 高いRecall（76.10%）: 採用すべきカットを見逃さない
- 安定した再現性: Random Seed 42で固定

**弱み**:
- 低いPrecision（29.83%）: 誤検出が多い、後処理フィルタリングが必要
- Fold間のばらつき: 最良49.42% vs 最悪32.20%（17.22pt差）

---

**最終更新**: 2025-12-26  
**バージョン**: 3.0.0（K-Fold CV + 時系列特徴量版）
