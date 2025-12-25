# 60.80% F1スコア達成時のパラメーター記録

## 📊 達成結果

```
F1 Score: 60.80%
Accuracy: 78.69%
Precision: 52.90%
Recall: 71.45%
Specificity: 80.87%
```

**日時**: 2025-12-25  
**評価方法**: アンサンブル学習（Soft Voting）  
**データセット**: 67動画、289シーケンス

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
```

### K-Fold Cross Validation
```yaml
n_folds: 5                      # Fold数
random_state: 42                # 乱数シード
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

## 🎯 アンサンブル設定

### 投票戦略
```yaml
strategy: soft                  # Soft Voting（確率平均）
num_models: 5                   # 使用モデル数（全Fold）
optimal_threshold: -0.4477      # 最適閾値
min_recall_constraint: 0.71     # 最小Recall制約
```

### 各Foldモデルの性能
| Fold | Best Epoch | F1 Score | Accuracy | Precision | Recall | Threshold |
|------|-----------|----------|----------|-----------|--------|-----------|
| 1 | 4 | 49.52% | 73.48% | 36.88% | 75.34% | -0.559 |
| 2 | 1 | 41.22% | 36.44% | 27.85% | 79.24% | -0.474 |
| 3 | 2 | 40.69% | 43.11% | 28.52% | 71.00% | -0.510 |
| 4 | 19 | 40.43% | 47.18% | 27.68% | 74.95% | -0.386 |
| 5 | 32 | 34.27% | 48.92% | 22.58% | 71.03% | -0.458 |
| **平均** | **11.6±12.1** | **41.23±4.86%** | **49.83±12.58%** | **28.70±4.61%** | **74.31±3.08%** | **-0.477±0.057** |

---

## 📁 ファイルパス

### 設定ファイル
```
configs/config_cut_selection_kfold_enhanced.yaml
```

### チェックポイント
```
checkpoints_cut_selection_kfold_enhanced/
├── fold_1_best_model.pth
├── fold_2_best_model.pth
├── fold_3_best_model.pth
├── fold_4_best_model.pth
├── fold_5_best_model.pth
├── kfold_summary.csv
├── ensemble_comparison.csv
└── inference_params.yaml
```

### データセット
```
preprocessed_data/combined_sequences_cut_selection_enhanced.npz
```

---

## 🔬 時系列特徴量（83個追加）

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

## ⚠️ 重要な注意事項

### データリークの可能性

この60.80%の評価には**データリークの可能性**があります：

```
問題: 全データ（67動画）で評価
     = 訓練データ + 検証データ

各モデルは自分の訓練データも評価に含まれている
→ 過大評価の可能性
```

### 信頼できる数値

```
個別モデル平均: 41.23% F1
```

これは各Foldで完全に未見のデータで評価した結果なので、**真の汎化性能**を表しています。

### 推定される真のアンサンブル性能

```
推定: 45-50% F1
```

データリークを考慮した現実的な推定値です。

---

## 🚀 再現方法

### 訓練
```bash
python src/cut_selection/train_cut_selection_kfold_enhanced.py \
    --config configs/config_cut_selection_kfold_enhanced.yaml
```

### アンサンブル評価（旧方式 - データリークあり）
```bash
python src/cut_selection/evaluate_ensemble.py \
    --checkpoint_dir checkpoints_cut_selection_kfold_enhanced \
    --data_path preprocessed_data/combined_sequences_cut_selection_enhanced.npz
```

### 正しい評価（検証データのみ）
```bash
python src/cut_selection/evaluate_ensemble_proper.py \
    --checkpoint_dir checkpoints_cut_selection_kfold_enhanced \
    --data_path preprocessed_data/combined_sequences_cut_selection_enhanced.npz
```

---

## 📅 記録日時

- **作成日**: 2025-12-26
- **訓練完了日**: 2025-12-25
- **評価実施日**: 2025-12-25

---

## 📝 備考

- 目標F1スコア（55%）を5.80ポイント上回る
- アンサンブル効果により個別モデルから+19.57ポイント改善
- Soft VotingとHard Votingが同じ結果（60.80%）
- Weighted Votingは若干低い（60.24%）
- 最良の個別モデルはFold 1（49.52% F1）
