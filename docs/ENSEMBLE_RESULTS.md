# アンサンブル学習による大幅な性能向上

## 🎯 達成した成果

### F1スコアの劇的な改善

```
個別モデル平均: 41.23% F1
        ↓
アンサンブル: 60.80% F1 ✨

改善: +19.57ポイント (+47.47%)
```

**目標達成: ✅ 55% F1 → 60.80% F1 (目標を5.80ポイント上回る)**

---

## 📊 詳細な結果

### アンサンブル性能

| メトリクス | 個別モデル平均 | アンサンブル | 改善 |
|-----------|---------------|-------------|------|
| **F1 Score** | 41.23% | **60.80%** | +19.57pt |
| **Accuracy** | 49.83% | **78.69%** | +28.86pt |
| **Precision** | 28.70% | **52.90%** | +24.20pt |
| **Recall** | 74.31% | **71.45%** | -2.86pt |
| **Specificity** | - | **80.87%** | - |

### 各Foldの個別性能

| Fold | Best Epoch | F1 Score | Accuracy | Precision | Recall | Threshold |
|------|-----------|----------|----------|-----------|--------|-----------|
| 1 | 4 | **49.52%** | 73.48% | 36.88% | 75.34% | -0.559 |
| 2 | 1 | 41.22% | 36.44% | 27.85% | 79.24% | -0.474 |
| 3 | 2 | 40.69% | 43.11% | 28.52% | 71.00% | -0.510 |
| 4 | 19 | 40.43% | 47.18% | 27.68% | 74.95% | -0.386 |
| 5 | 32 | 34.27% | 48.92% | 22.58% | 71.03% | -0.458 |
| **平均** | **11.6±12.1** | **41.23±4.86%** | **49.83±12.58%** | **28.70±4.61%** | **74.31±3.08%** | **-0.477±0.057** |

### アンサンブル戦略の比較

| 戦略 | F1 Score | Accuracy | Precision | Recall | Specificity | Threshold |
|------|----------|----------|-----------|--------|-------------|-----------|
| **Soft Voting** | **60.80%** | 78.69% | 52.90% | 71.45% | 80.87% | -0.448 |
| Hard Voting | 60.80% | 78.69% | 52.90% | 71.45% | 80.87% | -0.448 |
| Weighted Voting | 60.24% | 78.30% | 52.26% | 71.10% | 80.47% | -0.454 |

**最良戦略: Soft Voting (確率の平均)**

---

## 🔧 使用した設定

### モデルアーキテクチャ

```yaml
d_model: 256
nhead: 8
num_encoder_layers: 6
dim_feedforward: 1024
dropout: 0.15
```

### 入力特徴量

- **Audio**: 235次元
- **Visual**: 543次元
- **Temporal**: 6次元（新規追加）
- **Total**: 784次元

### 時系列特徴量（83個追加）

1. **移動統計量**: MA5, MA10, MA30, MA60, MA120, STD5, STD30, STD120
2. **変化率**: DIFF1, DIFF2, DIFF30
3. **カットタイミング**: time_since_prev, time_to_next, cut_duration, position_in_video, cut_density_10s
4. **CLIP類似度**: clip_sim_prev, clip_sim_next, clip_sim_mean5
5. **音声変化**: audio_change_score, silence_to_speech, speech_to_silence, speaker_change, pitch_change
6. **映像変化**: visual_motion_change, face_count_change, saliency_movement
7. **累積統計**: cumulative_position, cumulative_adoption_rate

### トレーニング設定

```yaml
num_epochs: 50
batch_size: 16
learning_rate: 0.0001
weight_decay: 0.0001
use_amp: true
n_folds: 5
random_state: 42
early_stopping_patience: 15
```

### 損失関数

```yaml
use_focal_loss: true
focal_alpha: 0.5
focal_gamma: 2.0
tv_weight: 0.02
adoption_penalty_weight: 10.0
target_adoption_rate: 0.23
```

---

## 🚀 成功の要因

### 1. 時系列特徴量の追加 ⭐⭐⭐⭐⭐

カットのコンテキスト情報を追加することで、単独のフレーム情報だけでなく、前後の関係性を学習できるようになった。

### 2. アンサンブル学習 ⭐⭐⭐⭐⭐

5つの異なるFoldで学習したモデルを組み合わせることで、個々のモデルの弱点を補完し、汎化性能が大幅に向上。

### 3. Focal Loss ⭐⭐⭐⭐

クラス不均衡（採用23% vs 不採用77%）に対応し、難しいサンプルに集中して学習。

### 4. K-Fold Cross Validation ⭐⭐⭐⭐

動画単位でデータを分割することで、データリークを防ぎ、真の汎化性能を測定。

### 5. 最適閾値の自動探索 ⭐⭐⭐

Recall制約（≥71%）を満たしつつF1を最大化する閾値を自動で発見。

---

## 📁 生成されたファイル

### チェックポイント

```
checkpoints_cut_selection_kfold_enhanced/
├── fold_1_best_model.pth
├── fold_2_best_model.pth
├── fold_3_best_model.pth
├── fold_4_best_model.pth
├── fold_5_best_model.pth
├── kfold_summary.csv
├── kfold_comparison.png
├── ensemble_comparison.csv
├── ensemble_comparison.png
└── view_training.html
```

### データ

```
preprocessed_data/
└── combined_sequences_cut_selection_enhanced.npz
    - 289 sequences
    - 67 unique videos
    - 784 features per frame
```

---

## 🎯 次のステップ

### V2モデル（さらなる改善）

以下の改善を加えたV2モデルを開発中：

1. **より深いネットワーク**: 8層エンコーダー（現在6層）
2. **より多いAttention**: 16ヘッド（現在8ヘッド）
3. **データ拡張**: ノイズ追加、時間シフト、スケーリング
4. **改善されたFusion**: 残差接続付き

**期待される性能: 65%+ F1**

---

## 📝 実行方法

### トレーニング

```bash
# V1モデル（現在の最良）
python src/cut_selection/train_cut_selection_kfold_enhanced.py --config configs/config_cut_selection_kfold_enhanced.yaml

# V2モデル（改善版）
python src/cut_selection/train_cut_selection_kfold_enhanced_v2.py --config configs/config_cut_selection_kfold_enhanced_v2.yaml
```

### アンサンブル評価

```bash
# V1モデルの評価
python src/cut_selection/evaluate_ensemble.py --checkpoint_dir checkpoints_cut_selection_kfold_enhanced

# V2モデルの評価
python src/cut_selection/evaluate_ensemble.py --checkpoint_dir checkpoints_cut_selection_kfold_enhanced_v2
```

---

## 📅 タイムライン

- **2025-12-25**: 時系列特徴量追加 + K-Fold トレーニング完了
- **2025-12-25**: アンサンブル学習実装 + 60.80% F1達成 ✨
- **2025-12-25**: V2モデル設計完了（データ拡張 + 深いネットワーク）

---

## 🏆 結論

時系列特徴量の追加とアンサンブル学習により、F1スコアを**41.23%から60.80%へ47.47%改善**し、**目標の55%を大幅に上回る**ことに成功しました。

この成果は以下の要素の組み合わせによるものです：
- 適切な特徴量エンジニアリング
- 強力なモデルアーキテクチャ
- 効果的な損失関数設計
- アンサンブル学習による汎化性能向上

次のステップとして、V2モデルでさらなる改善（65%+ F1）を目指します。
