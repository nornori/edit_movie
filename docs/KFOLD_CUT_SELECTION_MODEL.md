# カット選択モデル - 技術詳細

⚠️ **注意**: このドキュメントには旧K-Foldモデルの技術詳細が含まれています。現在は**Full Video Model**の使用を推奨しています。

## 概要

カット選択モデルは、動画から自動的に**最適なカット位置**を予測するTransformerベースのモデルです。

**目的**: 10分程度の動画を約2分（90秒〜150秒）のハイライト動画に自動編集

**現在の性能（Full Video Model - 推奨）**:
- F1スコア: 52.90%（学習時）
- Recall: 80.65%（採用すべきカットを見逃さない）
- Precision: 38.94%
- 推論テスト: 181.9秒（目標180秒に完璧に一致）

**旧K-Fold Model（改善中）**:
- 平均F1スコア: 42.30% ± 5.75%
- シーケンス分割の問題により改善中

---

## モデルアーキテクチャ

### 全体構造

```
入力: 音声（235次元）+ 映像（543次元）+ 時系列（6次元）= 784次元
  ↓
[Modality Embedding]
  - Audio → 256次元
  - Visual → 256次元
  - Temporal → 256次元
  ↓
[Three-Modality Gated Fusion]
  - 動的な重み付けで3つのモダリティを融合
  - 各モダリティにゲート機構を適用
  - 出力: 256次元
  ↓
[Positional Encoding]
  - 正弦波ベースの位置エンコーディング
  - 最大長: 5000フレーム
  ↓
[Transformer Encoder]
  - 層数: 6
  - ヘッド数: 8
  - 隠れ層次元: 256
  - FFN次元: 1024
  - Dropout: 0.15
  ↓
[Active Head]
  - 出力: (batch, seq_len, 2)
  - クラス0: 不採用
  - クラス1: 採用
```

### パラメータ数

約 **8.5M** パラメータ

---

## 特徴量

### 音声特徴量（235次元）

1. **基本音声特徴量（4次元）**
   - `audio_energy_rms`: RMSエネルギー（音量）
   - `audio_is_speaking`: 発話検出（VAD、0/1）
   - `silence_duration_ms`: 無音区間の長さ（ミリ秒）
   - `speaker_id`: 話者ID

2. **話者埋め込み（192次元）**
   - pyannote.audioによる話者埋め込み
   - 音声の特徴で自動的に話者を識別
   - 話者の変化を検出

3. **音響特徴量（16次元）**
   - `pitch`: 基本周波数（音の高さ）
   - `spectral_centroid`: スペクトル重心（音色の明るさ）
   - `spectral_bandwidth`: スペクトル帯域幅
   - `mfcc_0~12`: メル周波数ケプストラム係数（13次元）

4. **テキスト・テロップ（3次元）**
   - `text_is_active`: Whisperによる音声認識フラグ（0/1）
   - `text_word`: 単語数
   - `telop_is_active`: テロップ検出フラグ（未実装、0/1）

5. **時系列音声特徴量（20次元）**
   - 移動平均: MA5, MA10, MA30（RMS energy）
   - 変化率: DIFF1, DIFF2（RMS energy）
   - 音声変化: audio_change_score, silence_to_speech, speech_to_silence, speaker_change, pitch_change
   - その他: 累積統計等

### 映像特徴量（543次元）

1. **基本映像特徴量（10次元）**
   - `scene_change`: シーン変化検出（0/1）
   - `visual_motion`: モーション量（0-1）
   - `saliency_x`, `saliency_y`: 顕著性マップの中心座標（0-1）
   - `face_count`: 検出された顔の数
   - `face_center_x`, `face_center_y`: 顔の中心座標（0-1）
   - `face_size`: 顔のサイズ（0-1）
   - `face_mouth_open`: 口の開き具合（0-1）
   - `face_eyebrow_raise`: 眉の上がり具合（0-1）

2. **CLIP特徴量（512次元）**
   - CLIP ViT-B/32の視覚的意味表現
   - シーンの内容を高次元ベクトルで表現
   - 視覚的な類似性を捉える

3. **時系列視覚特徴量（21次元）**
   - 移動平均: MA5, MA10, MA30（visual_motion）
   - 変化率: DIFF1, DIFF2（visual_motion）
   - CLIP類似度: clip_sim_prev, clip_sim_next, clip_sim_mean5
   - 映像変化: visual_motion_change, face_count_change, saliency_movement
   - その他: 累積統計等

### 時系列特徴量（6次元）

**カットタイミング特徴量**:
- `time_since_prev`: 前のカットからの時間（秒）
- `time_to_next`: 次のカットまでの時間（秒）
- `cut_duration`: カット長（秒）
- `position_in_video`: 動画内位置（0-1）
- `cut_density_10s`: 10秒間のカット密度
- `cumulative_adoption_rate`: 累積採用率

---

## 損失関数

### Focal Loss

```python
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

**パラメータ**:
- `α (alpha)`: 0.5
  - クラスバランスの重み
  - α = 0.5 → 両クラスを均等に扱う
- `γ (gamma)`: 2.0
  - 難しいサンプルに集中
  - γ = 2.0 → 確信度の低いサンプルを重視

**効果**:
- クラス不均衡に対応（採用23%、不採用77%）
- 確信度の低いサンプルを重点的に学習
- 簡単なサンプルの影響を抑制

### Total Variation Loss

```python
TV_Loss = Σ |active[t+1] - active[t]|
```

**重み**: 0.02

**効果**:
- 時間的な滑らかさを保証
- チャタリング（細かい振動）を防止
- 自然なカット遷移を実現

### Adoption Penalty

```python
Adoption_Penalty = |adoption_rate - target_rate|^2
```

**重み**: 10.0  
**目標採用率**: 0.23（23%）

**効果**:
- 採用率を目標値に近づける
- 過剰な採用/不採用を防止
- バランスの取れた予測

### 総合損失

```python
Total_Loss = Focal_Loss + tv_weight * TV_Loss + adoption_penalty_weight * Adoption_Penalty
```

---

## 学習設定

### ハイパーパラメータ

```yaml
# モデル
audio_features: 235
visual_features: 543
temporal_features: 6
d_model: 256
nhead: 8
num_encoder_layers: 6
dim_feedforward: 1024
dropout: 0.15

# 学習
batch_size: 16
num_epochs: 50
learning_rate: 0.0001
weight_decay: 0.0001
max_grad_norm: 1.0
use_amp: true  # Mixed Precision

# K-Fold
n_folds: 5
random_state: 42  # 再現性確保

# 損失関数
use_focal_loss: true
focal_alpha: 0.5
focal_gamma: 2.0
label_smoothing: 0.0
tv_weight: 0.02
adoption_penalty_weight: 10.0
target_adoption_rate: 0.23

# Early Stopping
early_stopping_patience: 15
```

### オプティマイザ

- **AdamW**
- 学習率: 0.0001
- Weight Decay: 0.0001（L2正則化）
- Gradient Clipping: 1.0（勾配爆発防止）

### 学習時間

- **250エポック（50 × 5 Folds）**: 約2-3時間（GPU使用時）
- **Early Stopping**: 平均7.4エポックで収束
- **実際の学習**: 37エポック（85%削減）

---

## データ準備

### データ分割（K-Fold CV）

**GroupKFoldで動画単位に分割**（データリーク防止）:
- 67本の動画を使用
- 5-Fold Cross Validation
- 各Foldで12-14動画を検証用に使用
- 同じ動画のシーケンスは必ず同じFoldに配置

### シーケンス生成

- **シーケンス長**: 1000フレーム（約100秒 @ 10FPS）
- **オーバーラップ**: 500フレーム
- **総シーケンス数**: 289

### 正規化

- **StandardScaler**を使用
- 音声・映像・時系列特徴量を個別に正規化
- 平均0、分散1に変換
- 学習時のScalerを保存して推論時に使用

---

## 性能指標

### K-Fold Cross Validation結果

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
- 推論時の推奨モデル

### クラス分布

- **全体**: 採用23.12%、不採用76.88%
- **Fold 1検証データ**: 採用17%、不採用83%

### 推論性能

- **推論時間**: 5~10分/動画（特徴量抽出含む）
- **カット数**: 約8〜12個のクリップ
- **出力動画長**: 約2分（90秒〜150秒）

---

## クリップフィルタリング

### 処理フロー

1. **Active判定**
   - 最適閾値（例: -0.558）でフィルタリング
   - Confidence Score = logits[1] - logits[0]
   - 正の値: 採用傾向、負の値: 不採用傾向

2. **最小継続時間フィルタ**
   - 3.0秒未満のクリップを除外
   - 短すぎるクリップによる音声・動画の飛び飛びを防止

3. **ギャップ結合**
   - クリップ間のギャップが2.0秒以内なら自動的に埋めて結合
   - 自然なカット遷移を実現
   - 細かい不採用区間を無視

4. **優先順位付け**
   - Confidence Score順にソート
   - 高確信度のクリップを優先選択

5. **合計時間制限**
   - 目標: 90秒
   - 最大: 150秒
   - 上位クリップから順に選択

---

## 推論パラメータ

### 自動最適化

学習時に検証データで最適値を自動計算：

```yaml
# checkpoints_cut_selection_kfold_enhanced/inference_params.yaml
confidence_threshold: -0.558  # Fold 1の最適閾値
min_clip_duration: 3.0        # 最小クリップ継続時間（秒）
max_gap_duration: 2.0         # ギャップ結合の最大長（秒）
target_duration: 90.0         # 目標合計時間（秒）
max_duration: 150.0           # 最大合計時間（秒）
```

### 手動調整

閾値を調整してカット数を変更可能：
- 閾値を下げる（例: -0.7） → カット数が増える
- 閾値を上げる（例: -0.4） → カット数が減る

---

## 学習可視化

### リアルタイムグラフ

**全体進捗（kfold_realtime_progress.png）**:
1. F1スコアの推移（各Fold）
2. Validation Lossの推移（各Fold）
3. 現在のF1スコア（棒グラフ）
4. 進捗状況（テキスト）
5. 最良F1の推移（各Fold）
6. 採用/不採用の正確性（Recall & Specificity）

**各Fold詳細（fold_X/training_progress.png）**:
1. **損失関数（Loss）**
   - Train Loss vs Val Loss
   - 過学習の検出

2. **損失の内訳（CE vs TV）**
   - Classification Loss（Focal Loss）
   - Total Variation Loss
   - バランスの確認

3. **分類性能（Accuracy & F1）**
   - 全体的な精度
   - F1スコアの推移

4. **Precision, Recall, Specificity**
   - 採用の検出率（Recall）
   - 採用の精度（Precision）
   - 不採用の検出率（Specificity）

5. **最適閾値の推移**
   - エポックごとの最適閾値
   - 収束の確認

6. **予測の採用/不採用割合**
   - 予測分布
   - 正解分布との比較

### 自動更新

- ブラウザで `checkpoints_cut_selection_kfold_enhanced/view_training.html` を開く
- 2秒ごとに自動更新
- リアルタイムで学習状況を監視

---

## 技術的な工夫

### Three-Modality Gated Fusion

音声・映像・時系列の3つのモダリティを動的に融合：

```python
# 各モダリティにゲート機構を適用
audio_gate = sigmoid(W_audio_gate * audio_emb)
visual_gate = sigmoid(W_visual_gate * visual_emb)
temporal_gate = sigmoid(W_temporal_gate * temporal_emb)

audio_gated = audio_emb * audio_gate
visual_gated = visual_emb * visual_gate
temporal_gated = temporal_emb * temporal_gate

# 連結して融合
concat = [audio_gated; visual_gated; temporal_gated]
fused = LayerNorm(Linear(concat))
```

**効果**:
- 各フレームで最適なモダリティの重み付け
- 音声が重要なシーンでは音声を重視
- 映像が重要なシーンでは映像を重視
- カットタイミングが重要なシーンでは時系列を重視

### Positional Encoding

時間的な位置情報を埋め込み：

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**効果**:
- シーケンス内の相対的な位置を学習
- 長期的な依存関係を捉える
- 動画の前半・中盤・後半の違いを認識

### Gradient Clipping

勾配爆発を防止：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**効果**:
- 学習の安定化
- 大きな勾配による発散を防止

### Mixed Precision Training

計算の高速化とメモリ削減：

```python
with torch.amp.autocast('cuda'):
    outputs = model(audio, visual, temporal)
    loss, loss_dict = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**効果**:
- 学習速度が約1.5-2倍向上
- VRAM使用量が約30-40%削減

---

## 性能分析

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
   - 計算資源の節約

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

## 今後の改善予定

### 短期（1-2ヶ月）
- [ ] データ拡張（時間シフト、ノイズ追加）
- [ ] より強力なFocal Loss設定
- [ ] SMOTE等のサンプリング手法

### 中期（3-6ヶ月）
- [ ] より長期的な時系列パターン（MA240, MA480）
- [ ] シーン境界検出特徴量
- [ ] 編集スタイル特徴量
- [ ] より深いTransformer（8-12層）

### 長期（6ヶ月以上）
- [ ] Temporal Convolution追加
- [ ] Multi-scale Attention
- [ ] Attention可視化
- [ ] モデルの軽量化

---

## 参考文献

- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- **Transformer**: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
- **pyannote.audio**: Bredin et al., "pyannote.audio: neural building blocks for speaker diarization", ICASSP 2020

---

**最終更新**: 2025-12-26  
**バージョン**: 4.0.0（Full Video Model推奨版）  
**注意**: このドキュメントは主に旧K-Foldモデルの技術詳細を記載しています
