# カット選択モデル - 最終結果レポート

## 📊 最終性能（検証済み）

### 真の汎化性能

```
平均F1スコア: 42.30% ± 5.75%
平均Accuracy: 50.24% ± 14.92%
平均Precision: 29.83% ± 5.80%
平均Recall: 76.10% ± 5.19%
```

**評価方法**: K-Fold Cross Validation（5-Fold）  
**データセット**: 67動画、289シーケンス  
**評価日**: 2025-12-26

---

## 🎯 目標達成状況

| 項目 | 目標 | 達成 | 状況 |
|------|------|------|------|
| F1スコア | 55% | 42.30% | ❌ 未達成 (-12.70pt) |
| Recall | 71% | 76.10% | ✅ 達成 (+5.10pt) |

---

## 📈 各Foldの詳細結果

| Fold | Best Epoch | F1 Score | Accuracy | Precision | Recall | Threshold |
|------|-----------|----------|----------|-----------|--------|-----------|
| 1 | 4 | **49.42%** | 73.63% | 36.94% | 74.65% | -0.558 |
| 2 | 1 | 41.22% | 36.44% | 27.85% | 79.24% | -0.474 |
| 3 | 20 | 43.10% | 48.45% | 30.94% | 71.02% | -0.573 |
| 4 | 9 | 45.57% | 59.42% | 33.54% | 71.03% | -0.509 |
| 5 | 3 | 32.20% | 33.26% | 19.89% | 84.54% | -0.550 |
| **平均** | **7.4±6.8** | **42.30±5.75%** | **50.24±14.92%** | **29.83±5.80%** | **76.10±5.19%** | **-0.533±0.036** |

### 最良モデル

- **Fold 1**: 49.42% F1（Epoch 4）
- 最も安定した性能を示した

---

## 📈 学習結果の可視化

### 全Fold比較

![K-Fold Comparison](../checkpoints_cut_selection_kfold_enhanced/kfold_comparison.png)

**グラフの説明:**
- **左上**: 各FoldのF1スコア比較
- **右上**: 各Foldの精度（Accuracy）比較
- **左下**: Precision vs Recall のトレードオフ
- **右下**: 最適閾値の分布

### リアルタイム学習進捗

![Realtime Progress](../checkpoints_cut_selection_kfold_enhanced/kfold_realtime_progress.png)

**グラフの説明:**
- 全250エポック（5 Folds × 50 Epochs）の学習進捗
- 各Foldの最良F1スコアをリアルタイム表示
- Early Stoppingによる効率的な学習

### Fold 1 詳細（最良モデル）

![Fold 1 Training](../checkpoints_cut_selection_kfold_enhanced/fold_1/training_final.png)

**6つのグラフの説明:**
1. **左上 - 損失関数**: Train/Val Lossの推移
2. **右上 - 損失の内訳**: CE Loss（左軸）とTV Loss（右軸）
3. **中左 - 分類性能**: Accuracy & F1 Score
4. **中右 - 詳細メトリクス**: Precision, Recall, Specificity
5. **下左 - 最適閾値**: エポックごとの最適閾値の変化
6. **下右 - 予測分布**: 採用/不採用の予測割合

---

## 🎯 性能分析

### 強み ✅

1. **高いRecall（76.10%）**
   - 採用すべきカットの76%を正しく検出
   - False Negative（見逃し）が少ない
   - ハイライト動画に重要なシーンを含められる

2. **安定した再現性**
   - ランダムシード42で一貫した結果
   - 標準偏差: ±5.75%（比較的小さい）

3. **Early Stopping効果**
   - 平均7.4エポックで収束
   - 過学習を防止
   - 効率的な学習

### 弱み ❌

1. **低いPrecision（29.83%）**
   - 予測の約70%が誤検出（False Positive）
   - 不要なカットを多く含んでしまう
   - 後処理でのフィルタリングが必要

2. **Fold間のばらつき**
   - 最良（Fold 1: 49.42%）と最悪（Fold 5: 32.20%）の差: 17.22pt
   - データの偏りの影響を受けやすい
   - より多くのデータが必要

3. **目標未達成**
   - 目標F1: 55%
   - 達成F1: 42.30%
   - 差分: -12.70pt

---

## 🔬 評価の信頼性

### ✅ 正しい評価方法

```
各Foldで完全に未見のデータで評価
├─ Fold 1: 12動画で検証（他55動画で訓練）
├─ Fold 2: 14動画で検証（他53動画で訓練）
├─ Fold 3: 14動画で検証（他53動画で訓練）
├─ Fold 4: 14動画で検証（他53動画で訓練）
└─ Fold 5: 13動画で検証（他54動画で訓練）

結果: データリークなし、真の汎化性能
```

### ❌ アンサンブル評価の問題

以前報告した**60.80% F1**には以下の問題がありました：

```
問題: データリークを含む評価
- 全データ（訓練+検証）で評価
- 各モデルが自分の訓練データも評価
- 過大評価（楽観的な数値）

結論: 60.80%は信頼できない
```

---

## 🏗️ モデルアーキテクチャ

### Transformer設定

```yaml
d_model: 256                    # モデル次元数
nhead: 8                        # Attentionヘッド数
num_encoder_layers: 6           # エンコーダー層数
dim_feedforward: 1024           # フィードフォワード層の次元数
dropout: 0.15                   # ドロップアウト率
```

### 入力特徴量

```yaml
audio_features: 235             # 音声特徴量次元
visual_features: 543            # 映像特徴量次元
temporal_features: 6            # 時系列特徴量次元
total_features: 784             # 合計入力次元数
```

---

## 🎓 訓練設定

### 基本設定

```yaml
num_epochs: 50                  # 最大エポック数
batch_size: 16                  # バッチサイズ
learning_rate: 0.0001           # 学習率
weight_decay: 0.0001            # 重み減衰
max_grad_norm: 1.0              # 勾配クリッピング
use_amp: true                   # 混合精度訓練
random_state: 42                # 乱数シード（再現性）
```

### K-Fold Cross Validation

```yaml
n_folds: 5                      # Fold数
early_stopping_patience: 15     # Early Stopping待機エポック数
```

---

## 📉 損失関数

### Focal Loss

```yaml
use_focal_loss: true            # Focal Loss使用
focal_alpha: 0.5                # クラスバランス重み
focal_gamma: 2.0                # フォーカスパラメータ
label_smoothing: 0.0            # ラベル平滑化
```

### 正則化項

```yaml
tv_weight: 0.02                 # Total Variation重み
adoption_penalty_weight: 10.0   # 採用率ペナルティ重み
target_adoption_rate: 0.23      # 目標採用率
```

---

## 🔧 時系列特徴量（83個追加）

### 1. 移動統計量
- MA5, MA10, MA30, MA60, MA120（移動平均）
- STD5, STD30, STD120（移動標準偏差）

### 2. 変化率
- DIFF1, DIFF2, DIFF30（差分）

### 3. カットタイミング
- time_since_prev（前のカットからの時間）
- time_to_next（次のカットまでの時間）
- cut_duration（カット長）
- position_in_video（動画内位置）
- cut_density_10s（10秒間のカット密度）

### 4. CLIP類似度
- clip_sim_prev（前フレームとの類似度）
- clip_sim_next（次フレームとの類似度）
- clip_sim_mean5（5フレーム平均類似度）

### 5. 音声変化
- audio_change_score（音声変化スコア）
- silence_to_speech（無音→発話）
- speech_to_silence（発話→無音）
- speaker_change（話者変化）
- pitch_change（ピッチ変化）

### 6. 映像変化
- visual_motion_change（動き変化）
- face_count_change（顔数変化）
- saliency_movement（顕著性移動）

### 7. 累積統計
- cumulative_position（累積位置）
- cumulative_adoption_rate（累積採用率）

---

## 📁 ファイル構成

### チェックポイント

```
checkpoints_cut_selection_kfold_enhanced/
├── fold_1_best_model.pth       # Fold 1最良モデル (F1: 49.42%)
├── fold_2_best_model.pth       # Fold 2最良モデル (F1: 41.22%)
├── fold_3_best_model.pth       # Fold 3最良モデル (F1: 43.10%)
├── fold_4_best_model.pth       # Fold 4最良モデル (F1: 45.57%)
├── fold_5_best_model.pth       # Fold 5最良モデル (F1: 32.20%)
├── kfold_summary.csv           # 各Foldの結果サマリー
├── kfold_comparison.png        # Fold間の比較グラフ
├── kfold_realtime_progress.png # リアルタイム進捗グラフ
├── inference_params.yaml       # 推論用パラメータ
└── view_training.html          # 訓練結果ビューア
```

### 設定ファイル

```
configs/config_cut_selection_kfold_enhanced.yaml
```

### データセット

```
preprocessed_data/combined_sequences_cut_selection_enhanced.npz
- 289シーケンス
- 67動画
- 784特徴量/フレーム
```

---

## 🚀 実行方法

### トレーニング

```bash
python src/cut_selection/train_cut_selection_kfold_enhanced.py \
    --config configs/config_cut_selection_kfold_enhanced.yaml
```

### 推論

```bash
python src/cut_selection/inference_cut_selection.py \
    --checkpoint checkpoints_cut_selection_kfold_enhanced/fold_1_best_model.pth \
    --input_video path/to/video.mp4 \
    --output_json path/to/output.json
```

---

## 📊 性能分析

### 強み

✅ **高いRecall（76.10%）**
- 採用すべきカットを見逃さない
- False Negativeが少ない

✅ **安定した再現性**
- ランダムシード42で一貫した結果
- 平均F1: 42.30%（標準偏差: 5.75%）

### 弱み

❌ **低いPrecision（29.83%）**
- 不採用カットを誤って採用してしまう
- False Positiveが多い

❌ **Fold間のばらつき**
- 最良（Fold 1: 49.42%）と最悪（Fold 5: 32.20%）の差が大きい
- データの偏りの影響を受けやすい

---

## 💡 改善の方向性

### 1. データ拡張

```
現状: 67動画、289シーケンス
目標: より多くのデータで訓練

方法:
- 追加の動画データ収集
- データ拡張技術（時間シフト、ノイズ追加）
```

### 2. クラス不均衡への対応強化

```
現状: 採用23% vs 不採用77%
問題: Precisionが低い（29.83%）

方法:
- より強力なFocal Loss設定
- SMOTE等のサンプリング手法
- コスト考慮型学習
```

### 3. 特徴量エンジニアリング

```
追加候補:
- より長期的な時系列パターン（MA240, MA480）
- シーン境界検出特徴量
- 編集スタイル特徴量
- 視聴者注目度予測
```

### 4. モデルアーキテクチャ改善

```
試行候補:
- より深いTransformer（8-12層）
- Temporal Convolution追加
- Multi-scale Attention
- Ensemble学習（正しい方法で）
```

---

## 📅 開発履歴

- **2025-12-25**: 時系列特徴量追加（83個）
- **2025-12-25**: K-Fold Cross Validation実装
- **2025-12-25**: 初回トレーニング完了（平均F1: 41.23%）
- **2025-12-26**: データリーク問題発見
- **2025-12-26**: 正しい評価方法で再検証
- **2025-12-26**: 再トレーニング完了（平均F1: 42.30%）

---

## 🎯 結論

### 達成したこと

✅ 安定した機械学習パイプライン構築  
✅ 正しい評価方法の確立  
✅ 再現性の確保（ランダムシード固定）  
✅ 高いRecall（76.10%）の達成  

### 今後の課題

❌ F1スコア目標（55%）未達成（現在42.30%）  
❌ Precisionの改善が必要（現在29.83%）  
❌ より多くのデータが必要  

### 推奨事項

1. **本番環境での使用**: Fold 1モデル（F1: 49.42%）を推奨
2. **閾値調整**: Recall優先なら-0.558、Precision優先なら調整が必要
3. **継続的改善**: データ収集とモデル改善を並行して実施

---

## 📝 備考

- 全ての実験結果は再現可能（random_state=42）
- チェックポイントは全て保存済み
- バックアップ: `backups/2025-12-26_01-20-31_ensemble_60_80_percent/`

---

**最終更新**: 2025-12-26  
**作成者**: AI開発チーム
