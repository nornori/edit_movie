# 🚀 クイックスタートガイド - カット選択モデル

## 📋 前提条件

- Python 3.8以上がインストールされている
- 必要なライブラリがインストールされている（`pip install -r requirements.txt`）
- 学習済みモデルがある（または学習を実行する）
- **GPU推奨**: NVIDIA GPU（CUDA対応）、VRAM 8GB以上

**注意**: 本プロジェクトは現在**カット選択（Cut Selection）に特化**しています。グラフィック配置やテロップ生成は精度が低いため、今後の課題となっています。

---

## 🎯 新しい動画を自動編集する（推論）

### ワンコマンド実行（推奨）✅

動画パスを指定するだけで、特徴量抽出→推論→XML生成まで自動実行：

```bash
python scripts/video_to_xml.py "D:\path\to\video.mp4"
```

**出力**: `outputs/video_name_output.xml`

**目標秒数の指定**:
```bash
# 60秒目標（範囲: 30～80秒）
python scripts/video_to_xml.py "動画パス" --target 60

# デフォルトは180秒（範囲: 90～200秒）
python scripts/video_to_xml.py "動画パス" --target 180
```

**処理時間**: 約3～5分/動画（10分の動画、特徴量抽出含む）

### 既存の特徴量を使用する場合

特徴量が既に抽出されている場合は、こちらを使用：

```bash
python scripts/generate_xml_from_inference.py "D:\path\to\video.mp4"
```

**注意**: 特徴量ファイル（`temp_features/{動画名}_features_enhanced.csv`）が必要です。

### Premiere Proで開く

生成された XML をPremiere Proで開いてください。自動的にカット編集されたタイムラインが表示されます。

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

### ステップ5: Full Video用データを作成

```bash
python scripts/create_cut_selection_data_enhanced_fullvideo.py
```

**出力**:
- `preprocessed_data/combined_sequences_cut_selection_enhanced.npz` (289シーケンス、67動画)
- `preprocessed_data/audio_scaler_cut_selection_enhanced.pkl`
- `preprocessed_data/visual_scaler_cut_selection_enhanced.pkl`
- `preprocessed_data/temporal_scaler_cut_selection_enhanced.pkl`

---

## 🎓 カット選択モデルを学習する

### Full Video学習（推奨）✅

```bash
# バッチファイルで実行（推奨）
batch/train_fullvideo.bat
```

**学習設定**:
- 1動画=1サンプル（per-video最適化）
- エポック数: 50
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

ブラウザで `checkpoints_cut_selection_fullvideo/view_training.html` を開くと、2秒ごとに自動更新されるグラフで学習の様子をリアルタイム確認できます。

**学習グラフ**:
1. F1スコアの推移
2. Validation Lossの推移
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

学習済みモデルは `checkpoints_cut_selection_fullvideo/` に保存されます。

---

## 📁 ファイル配置

### 推論前に必要なもの
```
checkpoints_cut_selection_fullvideo/
├── best_model.pth                 # 学習済みモデル
└── inference_params.yaml          # 推論パラメータ
```
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

または、バッチファイル（`batch/train_fullvideo.bat`など）を使用してください。

### エラー: FileNotFoundError: best_model.pth

**原因**: 学習済みモデルがない

**解決策**:
1. データ準備を実行: ステップ1-5を完了
2. 学習を実行: `batch/train_fullvideo.bat`

### エラー: CUDA out of memory

**原因**: VRAM不足

**解決策**:
```yaml
# configs/config_cut_selection_fullvideo.yaml を編集
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
- 設定を確認: `configs/config_cut_selection_fullvideo.yaml`
- ログを確認: 学習中のメッセージをチェック
- GPU使用を確認: `nvidia-smi`

---

## 📊 実行時間の目安

- **特徴量抽出**: 3-5分/動画（10分の動画、GPU使用時）
  - 30本の動画: 約1.5-2.5時間（4並列処理）
- **学習**: 約2-3時間（250エポック、5 Folds、GPU使用時）
  - Early Stoppingで実際は37エポック（約30-45分）
- **推論**: 約3-5分/動画（特徴量抽出含む）

---

## 💡 ヒント

### カット数を調整したい

推論時に `--target` オプションで目標秒数を指定できます：

```bash
# 60秒目標（範囲: 30～80秒）
python scripts/video_to_xml.py "動画パス" --target 60

# 120秒目標（範囲: 60～140秒）
python scripts/video_to_xml.py "動画パス" --target 120

# デフォルトは180秒（範囲: 90～200秒）
python scripts/video_to_xml.py "動画パス" --target 180
```

**範囲の計算**:
- 最小値 = 目標 ÷ 2
- 最大値 = 目標 + 20

**クリップの結合・フィルタリング**:
- クリップ間の隙間が1秒未満 → 隙間を含めて結合
- 結合後に3秒未満のクリップ → 除外

### GPUを使用したい

学習時に自動的にGPUが使用されます（CUDA対応GPUがある場合）。

確認方法:
```bash
nvidia-smi
```

### データリークを防ぐには

- **動画単位で分割**（同じ動画は同じsplitに）
- 各splitで完全に未見のデータで評価

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
- [KFOLD_TRAINING_REPORT](KFOLD_TRAINING_REPORT.md) - K-Fold学習レポート（改善中）

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

### Full Video Model（推奨）✅

| 指標 | 学習時 | 推論テスト |
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
- **最適閾値**: 0.8952（制約満足、90-200秒制約内）
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
**バージョン**: 4.0.0（Full Video Model推奨版）
