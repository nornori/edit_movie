# カット選択モデル - 技術詳細

## 概要

カット選択モデルは、動画から自動的に**最適なカット位置**を予測するTransformerベースのモデルです。

**目的**: 10分程度の動画を約2分（90秒〜150秒）のハイライト動画に自動編集

---

## モデルアーキテクチャ

### 全体構造

```
入力: 音声特徴量（215次元）+ 映像特徴量（522次元）
  ↓
[Modality Embedding]
  - Audio → 256次元
  - Visual → 256次元
  ↓
[Gated Fusion]
  - 動的な重み付けで2つのモダリティを融合
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

約 **5.2M** パラメータ

---

## 特徴量

### 音声特徴量（215次元）

1. **基本音声特徴量（4次元）**
   - `audio_energy_rms`: RMSエネルギー（音量）
   - `audio_is_speaking`: 発話検出（0/1）
   - `silence_duration_ms`: 無音区間の長さ（ミリ秒）
   - `speaker_id`: 話者ID

2. **話者埋め込み（192次元）**
   - pyannote.audioによる256次元の話者埋め込み
   - 音声の特徴で自動的にキャラクターを判別
   - 100人以上のキャラクターに対応

3. **感情表現特徴（16次元）**
   - `pitch_f0`: 基本周波数
   - `pitch_std`: ピッチの変動
   - `spectral_centroid`: 音色の明るさ
   - `zcr`: ゼロ交差率
   - `mfcc_0~12`: メル周波数ケプストラム係数（13次元）

4. **テキスト・テロップ（3次元）**
   - `text_is_active`: テキストがアクティブか（0/1）
   - `telop_active`: テロップがアクティブか（0/1）

### 映像特徴量（522次元）

1. **基本映像特徴量（10次元）**
   - `scene_change`: シーン変化検出（0/1）
   - `visual_motion`: モーション量（0-1）
   - `saliency_x`, `saliency_y`: 顕著性マップの中心座標
   - `face_count`: 検出された顔の数
   - `face_center_x`, `face_center_y`: 顔の中心座標
   - `face_size`: 顔のサイズ
   - `face_mouth_open`: 口の開き具合（0-1）
   - `face_eyebrow_raise`: 眉の上がり具合（0-1）

2. **CLIP特徴量（512次元）**
   - CLIP ViT-B/32の視覚的意味表現
   - シーンの内容を高次元ベクトルで表現

---

## 損失関数

### Focal Loss

```python
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

**パラメータ**:
- `α (alpha)`: 0.70
  - 採用見逃し（False Negative）に2.3倍のペナルティ
  - α > 0.5 → 採用クラスを重視
- `γ (gamma)`: 2.0
  - 難しいサンプルに集中

**効果**:
- クラス不均衡に対応（採用26.93%、不採用73.07%）
- 確信度の低いサンプルを重点的に学習

### Total Variation Loss

```python
TV_Loss = Σ |active[t+1] - active[t]|
```

**重み**: 0.05

**効果**:
- 時間的な滑らかさを保証
- チャタリング（細かい振動）を防止
- 自然なカット遷移

### 総合損失

```python
Total_Loss = Focal_Loss + tv_weight * TV_Loss
```

---

## 学習設定

### ハイパーパラメータ

```yaml
# モデル
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
grad_clip: 1.0

# 損失関数
focal_alpha: 0.70
focal_gamma: 2.0
tv_weight: 0.05

# Early Stopping
early_stopping_patience: 20
```

### オプティマイザ

- **Adam**
- 学習率: 0.0001
- Weight Decay: 0.0001（L2正則化）

### 学習時間

- **50エポック**: 約1-2時間（GPU使用時）
- **Early Stopping**: 通常30-40エポックで収束

---

## データ準備

### データ分割

**動画単位で分割**（データリーク防止）:
- 68本の動画を使用
- 学習: 54動画（210シーケンス）
- 検証: 14動画（91シーケンス）
- 同じ動画のシーケンスは必ず同じセットに配置

### シーケンス生成

- **シーケンス長**: 1000フレーム（約100秒 @ 10FPS）
- **オーバーラップ**: 500フレーム
- **総シーケンス数**: 301

### 正規化

- **StandardScaler**を使用
- 音声・映像特徴量を個別に正規化
- 平均0、分散1に変換

---

## 性能指標

### 学習結果

- **Best F1スコア**: 0.5630（Epoch 33）
- **最適閾値**: -0.200
- **Accuracy**: 約0.88
- **Precision**: 約0.52
- **Recall**: 約0.60
- **Specificity**: 約0.95

### クラス分布

- **学習データ**: 採用28.19%、不採用71.81%
- **検証データ**: 採用12.14%、不採用87.86%

### 推論性能

- **推論時間**: 5~10分/動画（特徴量抽出含む）
- **カット数**: 約8〜12個のクリップ
- **出力動画長**: 約2分（90秒〜150秒）

---

## クリップフィルタリング

### 処理フロー

1. **Active判定**
   - 最適閾値（-0.200）でフィルタリング
   - Confidence Score = Active確率 - Inactive確率

2. **最小継続時間フィルタ**
   - 3.0秒未満のクリップを除外
   - 短すぎるクリップによる音声・動画の飛び飛びを防止

3. **ギャップ結合**
   - クリップ間のギャップが2.0秒以内なら自動的に埋めて結合
   - 自然なカット遷移を実現

4. **優先順位付け**
   - Active確率（確信度）順にソート
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
# checkpoints_cut_selection/inference_params.yaml
confidence_threshold: -0.200  # 最適閾値
target_duration: 90.0         # 目標合計時間（秒）
max_duration: 150.0           # 最大合計時間（秒）
```

### 手動調整

閾値を調整してカット数を変更可能：
- 閾値を下げる（例: -0.3） → カット数が増える
- 閾値を上げる（例: -0.1） → カット数が減る

---

## 学習可視化

### リアルタイムグラフ（6つ）

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

- ブラウザで `checkpoints_cut_selection/view_training.html` を開く
- 2秒ごとに自動更新
- リアルタイムで学習状況を監視

---

## 技術的な工夫

### Gated Fusion

音声と映像のモダリティを動的に融合：

```python
gate = sigmoid(W_gate * [audio; visual])
fused = gate * audio_emb + (1 - gate) * visual_emb
```

**効果**:
- 各フレームで最適なモダリティの重み付け
- 音声が重要なシーンでは音声を重視
- 映像が重要なシーンでは映像を重視

### Positional Encoding

時間的な位置情報を埋め込み：

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**効果**:
- シーケンス内の相対的な位置を学習
- 長期的な依存関係を捉える

### Gradient Clipping

勾配爆発を防止：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

---

## 今後の改善予定

### 短期（1-2ヶ月）
- [ ] K-Fold Cross Validation実装
- [ ] データ拡張（時間シフト、ノイズ追加）
- [ ] アンサンブル学習

### 中期（3-6ヶ月）
- [ ] Attention可視化
- [ ] モデルの軽量化
- [ ] 推論速度の向上

### 長期（6ヶ月以上）
- [ ] グラフィック配置モデルの統合
- [ ] テロップ生成モデルの統合
- [ ] エンドツーエンド学習

---

## 参考文献

- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- **Transformer**: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021

---

**最終更新**: 2025-12-22
**バージョン**: 1.0.0
